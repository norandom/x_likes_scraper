"""Compiled DSPy program persistence and the optimizer entry point.

Implements the ``compiled`` component of the synthesis-report design
(Requirements 6.3, 6.4, 6.5):

* :func:`compiled_path` — resolve the per-shape JSON path under a
  configurable root (default ``output/synthesis_compiled/``).
* :func:`load_compiled` — load a previously compiled DSPy program for
  ``shape``; return ``None`` when the file is missing or corrupt so the
  orchestrator can transparently fall back to the un-optimized
  signature.
* :func:`save_compiled` — write a compiled program atomically (temp +
  ``flush`` + ``fsync`` + :func:`os.replace`), mirroring the on-disk
  contract used by ``url_cache.py``.
* :func:`prepare_demo` — sanitize and fence one labeled example so a
  malicious tweet in the demo set cannot smuggle prompt injection past
  the optimizer.
* :func:`run_optimizer` — build the un-optimized base program, prepare
  every demo via :func:`prepare_demo`, run ``BootstrapFewShot``, and
  persist the compiled artifact.

The module deliberately treats DSPy's serialization as opaque (it
delegates to ``Module.save`` / ``Module.load``) so a future DSPy point
release that changes the on-disk shape does not require a code change
here. A corrupt or schema-mismatched file collapses to ``None`` from
:func:`load_compiled`; the next ``--report-optimize`` run rewrites it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]

from ..sanitize import fence_for_llm, sanitize_text
from .dspy_modules import make_synthesizer
from .shapes import ReportShape

__all__ = [
    "LabeledExample",
    "compiled_path",
    "load_compiled",
    "prepare_demo",
    "run_optimizer",
    "save_compiled",
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabeledExample:
    """One demo for the DSPy optimizer.

    Attributes:
        query: The user-supplied query for the demo. Treated as the
            intent surface — sanitized but not fenced (queries are the
            only legitimate instruction channel into the LM).
        fenced_context_raw: The raw, untrusted tweet / URL body blob
            that this module sanitizes and fences via
            :func:`prepare_demo` before any optimizer sees it.
        expected_outputs: Operator-supplied gold answers. Field names
            must match the per-shape signature (e.g. ``claims``,
            ``top_entities`` for BRIEF).
    """

    query: str
    fenced_context_raw: str
    expected_outputs: dict[str, object]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def compiled_path(shape: ReportShape, root: Path) -> Path:
    """Return the on-disk path for ``shape``'s compiled program.

    Always ``root / f"{shape.value}.json"``. The orchestrator passes
    its configured ``output/synthesis_compiled/`` root.
    """

    return root / f"{shape.value}.json"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_compiled(shape: ReportShape, root: Path) -> dspy.Module | None:
    """Return the compiled DSPy program for ``shape``, or ``None``.

    ``None`` is returned in three cases:

    * The file does not exist.
    * The file exists but cannot be parsed (corrupt JSON, schema drift
      across DSPy versions, etc.).
    * Any other I/O failure during the load.

    The orchestrator interprets ``None`` as "no compiled artifact
    available" and falls back to the un-optimized signature. This
    keeps the call site free of exception handling for the common
    "first run, never optimized" path.
    """

    path = compiled_path(shape, root)
    if not path.exists():
        return None

    program = make_synthesizer(shape)
    try:
        program.load(str(path))
    except Exception as exc:
        # Treat any load failure (corrupt JSON, schema drift, file
        # vanished mid-call) as a miss; the orchestrator falls back to
        # the un-optimized signature and the next ``--report-optimize``
        # run rewrites the file.
        _LOGGER.warning(
            "load_compiled: failed to load %s for shape %s: %s",
            path,
            shape.value,
            exc,
        )
        return None
    return program


# ---------------------------------------------------------------------------
# Save (atomic; mirrors url_cache.py)
# ---------------------------------------------------------------------------


def save_compiled(module: dspy.Module, shape: ReportShape, root: Path) -> Path:
    """Atomically write ``module`` to ``compiled_path(shape, root)``.

    Behavior mirrors the URL-cache atomic-write contract:

    1. Lazy-mkdir ``root`` (``parents=True, exist_ok=True``).
    2. Ask DSPy to serialize the program to a ``NamedTemporaryFile``
       in ``root`` (same directory so :func:`os.replace` stays a
       same-volume rename — atomic on POSIX and Windows).
    3. ``flush`` + ``fsync`` the temp file before promoting it.
    4. Promote with :func:`os.replace`.
    5. On any exception, best-effort unlink the temp file so failed
       writes do not leave debris behind.

    Returns the final path so callers can log it.
    """

    root.mkdir(parents=True, exist_ok=True)
    final_path = compiled_path(shape, root)

    # DSPy's ``Module.save`` validates that the destination path ends
    # with ``.json`` (state-only) or ``.pkl`` (full pickle). Use the
    # ``.json`` suffix on the temp file so DSPy accepts the call;
    # ``dir=root`` keeps the temp on the same filesystem as the
    # destination so :func:`os.replace` is atomic.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(root),
        prefix=".compiled.",
        suffix=".json",
    )
    # Close the OS-level handle immediately — DSPy's ``Module.save``
    # opens the path itself. We re-open later to ``fsync``.
    os.close(fd)

    try:
        try:
            module.save(tmp_name)
            with open(tmp_name, "rb") as handle:
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)
            raise

        try:
            os.replace(tmp_name, final_path)
        except BaseException:
            # Promotion failure: drop the orphaned temp so a crashed
            # rename does not leave ``.compiled.*.json`` debris. The
            # caller's exception (the original error from
            # ``os.replace``) is preserved.
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)
            raise
    finally:
        # Defensive cleanup: if the rename succeeded the temp is gone;
        # if it failed we already unlinked above. ``missing_ok=True``
        # keeps this idempotent in either case.
        if Path(tmp_name).exists() and tmp_name != str(final_path):
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)

    return final_path


# ---------------------------------------------------------------------------
# Demo preparation (Req 6.5)
# ---------------------------------------------------------------------------


def prepare_demo(example: LabeledExample) -> dspy.Example:
    """Sanitize + fence a :class:`LabeledExample` before the optimizer sees it.

    Two-channel discipline (matches the live-request pipeline):

    * ``query`` — the only legitimate instruction surface. Sanitize so
      ANSI / BiDi / control codepoints don't sneak into the LM prompt,
      but do not fence (fencing the query would tell the LM to ignore
      its own intent input).
    * ``fenced_context_raw`` — untrusted tweet / URL body blob.
      Sanitize, then wrap in :func:`fence_for_llm`. The fence helper
      neutralizes every fence-marker family inside the body so a
      crafted demo cannot prematurely close one fence and resume
      prompt control.

    Output fields from ``expected_outputs`` are flattened onto the
    DSPy ``Example``. ``with_inputs("query", "fenced_context")`` tells
    the optimizer which fields are inputs (everything else is treated
    as a gold output during demo replay).
    """

    sanitized_query = sanitize_text(example.query)
    fenced_context = fence_for_llm(example.fenced_context_raw)

    fields: dict[str, Any] = {
        "query": sanitized_query,
        "fenced_context": fenced_context,
    }
    fields.update(example.expected_outputs)
    return dspy.Example(**fields).with_inputs("query", "fenced_context")


# ---------------------------------------------------------------------------
# Optimizer entry point (Req 6.4)
# ---------------------------------------------------------------------------


# Allowlist of optimizer names. v1 ships only ``BootstrapFewShot``
# (per design.md "Open Questions"); MIPROv2 is an explicit follow-up.
# Reject other names with a clear error rather than silently fall
# through to a dspy attribute that may or may not exist.
_OPTIMIZERS: dict[str, str] = {
    "BootstrapFewShot": "BootstrapFewShot",
}


def run_optimizer(
    shape: ReportShape,
    labeled_examples: list[LabeledExample],
    *,
    root: Path,
    optimizer: str = "BootstrapFewShot",
) -> dspy.Module:
    """Run the optimizer for ``shape`` and persist the compiled program.

    Pipeline:

    1. Build the un-optimized base program via
       :func:`make_synthesizer`.
    2. Convert each :class:`LabeledExample` to a ``dspy.Example`` via
       :func:`prepare_demo` (so the optimizer never sees raw
       fenced-context input — Req 6.5).
    3. Instantiate the requested optimizer class (only
       ``BootstrapFewShot`` is in scope for v1).
    4. ``compiled = optimizer.compile(program, trainset=demos)``.
    5. Persist via :func:`save_compiled` and return the compiled
       module.

    Raises:
        ValueError: ``optimizer`` is not in the v1 allowlist.
    """

    if optimizer not in _OPTIMIZERS:
        allowed = ", ".join(sorted(_OPTIMIZERS))
        raise ValueError(f"Unknown optimizer {optimizer!r}; allowed values: {allowed}")

    program = make_synthesizer(shape)

    # Resolve ``prepare_demo`` through the module so test monkeypatches
    # (Req 6.5 verification) take effect.
    import sys

    current_module = sys.modules[__name__]
    prepare = current_module.prepare_demo
    demos = [prepare(example) for example in labeled_examples]

    optimizer_cls = getattr(dspy, _OPTIMIZERS[optimizer])
    optimizer_obj = optimizer_cls()
    compiled = optimizer_obj.compile(program, trainset=demos)

    save_compiled(compiled, shape, root)
    return compiled
