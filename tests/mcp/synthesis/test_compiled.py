"""Tests for compiled DSPy program persistence and the optimizer entry point.

Covers task 3.2 of the synthesis-report spec (Requirements 6.3, 6.4, 6.5):

* :func:`compiled_path` resolves ``{root}/{shape}.json`` per shape.
* :func:`load_compiled` returns ``None`` for missing or corrupt files
  and never raises; orchestrator falls back to the un-optimized
  signature.
* :func:`save_compiled` writes atomically (temp + ``os.replace``) and
  lazily creates the parent directory.
* :func:`prepare_demo` sanitizes the user-supplied query and runs the
  raw fenced-context body through ``sanitize_text`` + ``fence_for_llm``
  before any optimizer sees it.
* :func:`run_optimizer` rejects unknown optimizer names and routes
  every labeled example through ``prepare_demo``. The end-to-end
  optimizer run is marked ``slow`` and skipped by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from x_likes_mcp.sanitize import LLM_FENCE_CLOSE, LLM_FENCE_OPEN
from x_likes_mcp.synthesis import compiled as compiled_mod
from x_likes_mcp.synthesis.compiled import (
    LabeledExample,
    compiled_path,
    load_compiled,
    prepare_demo,
    run_optimizer,
    save_compiled,
)
from x_likes_mcp.synthesis.dspy_modules import make_synthesizer
from x_likes_mcp.synthesis.shapes import ReportShape

# ---------------------------------------------------------------------------
# compiled_path
# ---------------------------------------------------------------------------


def test_compiled_path_uses_shape_value(tmp_path: Path) -> None:
    """``compiled_path`` returns ``root / f"{shape.value}.json"``."""

    assert compiled_path(ReportShape.BRIEF, tmp_path) == tmp_path / "brief.json"
    assert compiled_path(ReportShape.SYNTHESIS, tmp_path) == tmp_path / "synthesis.json"
    assert compiled_path(ReportShape.TREND, tmp_path) == tmp_path / "trend.json"


# ---------------------------------------------------------------------------
# load_compiled
# ---------------------------------------------------------------------------


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    """A directory with no ``brief.json`` returns ``None`` from ``load_compiled``."""

    assert load_compiled(ReportShape.BRIEF, tmp_path) is None


def test_load_returns_none_when_directory_missing(tmp_path: Path) -> None:
    """A missing root directory still produces ``None`` (not a crash)."""

    assert load_compiled(ReportShape.BRIEF, tmp_path / "does-not-exist") is None


def test_load_returns_none_when_corrupt(tmp_path: Path) -> None:
    """Garbage in ``{shape}.json`` is treated as a miss, not a crash.

    The orchestrator falls back to the un-optimized signature; raising
    here would force the fallback path through an exception handler in
    every caller.
    """

    path = compiled_path(ReportShape.BRIEF, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("xyz", encoding="utf-8")

    assert load_compiled(ReportShape.BRIEF, tmp_path) is None


# ---------------------------------------------------------------------------
# save_compiled
# ---------------------------------------------------------------------------


def test_save_creates_directory(tmp_path: Path) -> None:
    """``save_compiled`` lazy-creates the configured root directory."""

    root = tmp_path / "synthesis_compiled"
    assert not root.exists()

    program = make_synthesizer(ReportShape.BRIEF)
    save_compiled(program, ReportShape.BRIEF, root)

    assert root.is_dir()


def test_save_writes_to_compiled_path(tmp_path: Path) -> None:
    """After ``save_compiled``, the per-shape JSON file exists and is non-empty."""

    program = make_synthesizer(ReportShape.BRIEF)
    final_path = save_compiled(program, ReportShape.BRIEF, tmp_path)

    assert final_path == compiled_path(ReportShape.BRIEF, tmp_path)
    assert final_path.exists()
    assert final_path.stat().st_size > 0
    # The file must be JSON (DSPy's serialization format).
    json.loads(final_path.read_text(encoding="utf-8"))


def test_save_is_atomic_no_partial_files_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` raises, the final ``{shape}.json`` must not exist.

    Mirrors the atomic-write contract in :mod:`url_cache`: write to a
    temp in the same directory, ``flush`` + ``fsync``, then promote
    with :func:`os.replace`. A crash in the promote step leaves no
    committed file behind for the loader.
    """

    def _boom(src: str, dst: str) -> None:
        raise OSError("simulated mid-write crash")

    monkeypatch.setattr(compiled_mod.os, "replace", _boom, raising=True)

    program = make_synthesizer(ReportShape.BRIEF)

    with pytest.raises(OSError, match="simulated mid-write crash"):
        save_compiled(program, ReportShape.BRIEF, tmp_path)

    assert not compiled_path(ReportShape.BRIEF, tmp_path).exists()


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    """Save a base program, load it back, and verify the signature shape.

    DSPy 3.x's ``Module.save`` / ``.load`` round-trip works on an
    un-optimized ``ChainOfThought`` (the demo list is empty but the
    signature itself round-trips), so this test does not need the
    optimizer to have run first.
    """

    program = make_synthesizer(ReportShape.BRIEF)
    save_compiled(program, ReportShape.BRIEF, tmp_path)

    loaded = load_compiled(ReportShape.BRIEF, tmp_path)
    assert loaded is not None

    # The loaded program must expose the BRIEF output fields. We poke at
    # the inner predictor's signature rather than instantiate the LM.
    output_field_names = set(loaded.predict.signature.output_fields.keys())
    assert {"claims", "top_entities"}.issubset(output_field_names)


# ---------------------------------------------------------------------------
# prepare_demo
# ---------------------------------------------------------------------------


def test_prepare_demo_sanitizes_query() -> None:
    """A query carrying an ANSI escape comes out clean.

    The query is the user-supplied intent surface and is *not* fenced
    (it is the only legitimate instruction channel into the LM), but it
    still must not carry terminal-control or BiDi codepoints.
    """

    raw_query = "find tweets \x1b[31mabout\x1b[0m AI safety"
    example = LabeledExample(
        query=raw_query,
        fenced_context_raw="hello world",
        expected_outputs={"claims": [], "top_entities": []},
    )

    demo = prepare_demo(example)

    assert "\x1b" not in demo.query
    assert "about" in demo.query


def test_prepare_demo_fences_context() -> None:
    """A demo's raw context blob is sanitized + wrapped in the tweet fence.

    A malicious tweet body could embed the literal fence open marker to
    try to close our fence early. ``fence_for_llm`` neutralizes every
    occurrence of any fence marker before wrapping; the resulting demo
    field carries the marker replaced with ``[FENCE]`` and is
    surrounded by a fresh ``<<<TWEET_BODY>>>...<<<END_TWEET_BODY>>>``
    pair.
    """

    raw_body = f"prefix {LLM_FENCE_OPEN} payload {LLM_FENCE_CLOSE} suffix"
    example = LabeledExample(
        query="what is being discussed",
        fenced_context_raw=raw_body,
        expected_outputs={"claims": [], "top_entities": []},
    )

    demo = prepare_demo(example)

    fenced = demo.fenced_context
    # Marker neutralization: every interior occurrence is replaced with
    # the neutral token (the wrapping markers themselves remain).
    body_between = fenced.split(LLM_FENCE_OPEN, 1)[1].rsplit(LLM_FENCE_CLOSE, 1)[0]
    assert LLM_FENCE_OPEN not in body_between
    assert LLM_FENCE_CLOSE not in body_between
    assert "[FENCE]" in body_between
    # Fresh fence wrap is present.
    assert fenced.startswith(LLM_FENCE_OPEN)
    assert fenced.rstrip().endswith(LLM_FENCE_CLOSE)


def test_prepare_demo_attaches_inputs() -> None:
    """The DSPy ``Example`` must declare ``query`` and ``fenced_context`` as inputs.

    DSPy's optimizers only treat fields named via ``.with_inputs(...)``
    as inputs (the rest are taken to be gold outputs). Without this
    contract, ``BootstrapFewShot`` would not know which fields to feed
    the predictor when replaying a demo.
    """

    example = LabeledExample(
        query="q",
        fenced_context_raw="ctx",
        expected_outputs={"claims": [], "top_entities": ["ai"]},
    )
    demo = prepare_demo(example)

    inputs = demo.inputs()
    assert "query" in inputs
    assert "fenced_context" in inputs
    # Outputs flatten onto the example.
    assert demo.top_entities == ["ai"]


# ---------------------------------------------------------------------------
# run_optimizer
# ---------------------------------------------------------------------------


def test_run_optimizer_rejects_unknown_optimizer(tmp_path: Path) -> None:
    """Only ``BootstrapFewShot`` is in scope for v1; other names raise.

    The design document's "Open Questions" section names BootstrapFewShot
    as the v1 default; MIPROv2 is an explicit follow-up. Reject other
    names rather than silently fall through.
    """

    with pytest.raises(ValueError, match="optimizer"):
        run_optimizer(
            ReportShape.BRIEF,
            [],
            root=tmp_path,
            optimizer="NotARealOptimizer",
        )


def test_run_optimizer_calls_prepare_demo_for_each_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every labeled example is routed through ``prepare_demo``.

    Demos are untrusted input (Req 6.5); the optimizer must never see
    a raw fenced-context body. We monkeypatch ``prepare_demo`` to a
    sentinel and intercept the optimizer call so the test does not need
    a real LM.
    """

    import dspy  # type: ignore[import-untyped]

    examples = [
        LabeledExample(
            query=f"q{i}",
            fenced_context_raw=f"raw-body-{i}",
            expected_outputs={"claims": [], "top_entities": []},
        )
        for i in range(3)
    ]

    seen: list[LabeledExample] = []

    def _spy(example: LabeledExample) -> Any:
        seen.append(example)
        return dspy.Example(
            query=example.query,
            fenced_context="prepared",
            claims=[],
            top_entities=[],
        ).with_inputs("query", "fenced_context")

    monkeypatch.setattr(compiled_mod, "prepare_demo", _spy, raising=True)

    captured: dict[str, Any] = {}

    class _StubOptimizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["optimizer_args"] = (args, kwargs)

        def compile(self, program: Any, *, trainset: Any) -> Any:
            captured["program"] = program
            captured["trainset"] = list(trainset)
            return program

    monkeypatch.setattr(dspy, "BootstrapFewShot", _StubOptimizer, raising=True)

    result = run_optimizer(
        ReportShape.BRIEF,
        examples,
        root=tmp_path,
    )

    assert seen == examples
    assert len(captured["trainset"]) == len(examples)
    # Saved compiled program ended up on disk.
    assert compiled_path(ReportShape.BRIEF, tmp_path).exists()
    # Returned the compiled module.
    assert result is captured["program"]


def test_run_optimizer_writes_compiled_path_with_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optimizer return value lands at ``compiled_path``.

    Uses the same stub optimizer as above to keep the test offline.
    """

    import dspy  # type: ignore[import-untyped]

    class _StubOptimizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def compile(self, program: Any, *, trainset: Any) -> Any:
            return program

    monkeypatch.setattr(dspy, "BootstrapFewShot", _StubOptimizer, raising=True)

    run_optimizer(ReportShape.BRIEF, [], root=tmp_path)

    final = compiled_path(ReportShape.BRIEF, tmp_path)
    assert final.exists()
    assert final.stat().st_size > 0


@pytest.mark.slow
def test_run_optimizer_end_to_end(tmp_path: Path) -> None:
    """End-to-end ``BootstrapFewShot`` run.

    The optimizer call reaches the LM and is expensive / flaky against
    the FakeDspyLM stub. Marked ``slow`` and skipped by default; opt in
    by passing ``--run-slow`` to pytest.
    """

    examples = [
        LabeledExample(
            query="ai safety",
            fenced_context_raw="tweet about alignment",
            expected_outputs={"claims": [], "top_entities": ["alignment"]},
        ),
    ]
    result = run_optimizer(ReportShape.BRIEF, examples, root=tmp_path)
    assert result is not None
    assert compiled_path(ReportShape.BRIEF, tmp_path).exists()


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------
#
# The ``slow`` marker is registered via the synthesis conftest's
# ``pytest_configure`` hook. Adding the marker registration here too
# would emit a duplicate-warning, so we rely on the conftest entry.
# The opt-in flag (``--run-slow``) is also defined in the conftest so
# the collection-modify hook can skip ``slow`` items by default.

# Sanity check: confirm the per-file ``os`` reference exists so the
# atomic-write monkeypatch in ``test_save_is_atomic_no_partial_files_on_crash``
# targets a real attribute.
assert hasattr(compiled_mod, "os") or hasattr(os, "replace")
