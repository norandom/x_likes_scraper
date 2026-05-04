"""Test infrastructure for the ``x_likes_mcp.synthesis`` subpackage.

This conftest layers two synthesis-specific autouse guards on top of the
guards already declared in ``tests/conftest.py`` (real network via
``responses``, no real ``cookies.json`` reads) and ``tests/mcp/conftest.py``
(walker chat-completions seam, embeddings seam):

1. :class:`RealUrlFetchAttempted` + the autouse ``_block_real_url_fetch``
   fixture monkeypatches :func:`httpx.Client.send` so any unmocked URL
   fetch from the synthesis fetcher (or any other code path that constructs
   an ``httpx.Client``) fails loudly instead of reaching the network. The
   parent conftest blocks ``requests``-based traffic via ``responses``;
   this guard catches the ``httpx`` transport that crawl4ai-style fetchers
   and the OpenAI / DSPy LM client use.

2. :class:`RealDspyCallAttempted` and the autouse ``_stub_dspy_lm`` fixture
   install a :class:`FakeDspyLM` instance via ``dspy.configure(lm=...)``
   for every synthesis test. DSPy 3.x accepts a duck-typed callable as
   the LM, so :class:`FakeDspyLM` is a plain class (not a subclass of
   :class:`dspy.LM`); this avoids brittleness against DSPy's internal
   base-class surface. Tests can seed canned responses by attaching
   ``@pytest.mark.dspy_canned({...})`` to the test function.

The :class:`FakeDspyLM` records every call into ``self.calls`` so tests
can assert on prompts, message lists, or signature names.

The ``real_lm`` pytest marker is collected only when ``--run-real-lm`` is
passed on the command line; by default any test carrying the marker is
skipped so the suite never accidentally hits a live LM endpoint. This
mirrors the ``_block_real_llm`` pattern in ``tests/mcp/conftest.py``.

Per design.md ("Test seam") the synthesizer is stubbed at the LM level,
not at the signature level, so signature tests can exercise the real
DSPy machinery while staying offline.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest


class RealUrlFetchAttempted(AssertionError):
    """Raised when an unmocked URL fetch escapes test isolation.

    The synthesis fetcher will sit on top of ``httpx`` (talking to a
    crawl4ai container). Tests that legitimately exercise a fetcher path
    must monkeypatch the relevant ``httpx`` seam (or the fetcher itself);
    anything that reaches ``httpx.Client.send`` without a stub surfaces
    this error rather than emitting a real HTTP request.
    """


class RealDspyCallAttempted(AssertionError):
    """Raised when an unmocked DSPy LM call slips out.

    Reserved for future per-call assertions. The autouse
    ``_stub_dspy_lm`` fixture currently installs a benign
    :class:`FakeDspyLM`; tests that want to trip on *any* LM call (rather
    than collect benign ones) can re-patch ``dspy.settings.lm`` with a
    stub that raises this error.
    """


class FakeDspyLM:
    """Test-only DSPy LM stub.

    Returns canned responses keyed by signature name and input hash so
    signature tests stay offline. DSPy 3.x calls the configured LM as a
    plain callable (``lm(prompt, **kwargs)`` or
    ``lm(messages=..., **kwargs)``) and accepts duck-typed objects via
    :func:`dspy.configure`, so this class deliberately does **not**
    subclass :class:`dspy.LM`. Subclassing the real LM base introduces
    brittleness against DSPy point releases and forces a
    ``litellm``-style ``model`` argument we do not need in tests.

    Parameters
    ----------
    canned:
        Optional mapping of cache keys to pre-canned responses. The
        default lookup key is the prompt string; signature-aware tests
        can populate richer keys (e.g. ``"SynthesizeBrief:<hash>"``) and
        post-process via :attr:`calls`.

    Attributes
    ----------
    calls:
        Append-only list of every invocation as ``(prompt, kwargs)``
        tuples. Tests use this to assert which signature fired and with
        what fenced inputs.
    """

    def __init__(self, canned: dict[str, object] | None = None) -> None:
        self.canned: dict[str, object] = dict(canned or {})
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(
        self,
        prompt: str | list[dict[str, object]] = "",
        **kwargs: object,
    ) -> list[str]:
        # Record the call before any lookup so partial assertions still
        # see the invocation if a canned-response lookup is added later.
        prompt_key = prompt if isinstance(prompt, str) else repr(prompt)
        self.calls.append((prompt_key, dict(kwargs)))

        if prompt_key in self.canned:
            response = self.canned[prompt_key]
            if isinstance(response, list):
                return [str(item) for item in response]
            return [str(response)]

        # Benign default — empty completion. Tests that need a structured
        # response should seed ``canned`` via the ``dspy_canned`` marker.
        return [""]


@pytest.fixture(autouse=True)
def _block_real_url_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch ``httpx.Client.send`` to fail loudly.

    ``Client.send`` is the stable seam every higher-level helper
    (``get``/``post``/``request``) routes through, so patching it once
    blocks every flavor of HTTP call without touching the public API.
    Tests that legitimately want HTTP re-monkeypatch the same attribute
    with a stub.
    """

    def _raise(*_args: object, **_kwargs: object) -> httpx.Response:
        raise RealUrlFetchAttempted(
            "httpx.Client.send was invoked without a test-supplied stub; "
            "mock the synthesis fetcher (or httpx.Client.send directly) "
            "in the test."
        )

    monkeypatch.setattr(httpx.Client, "send", _raise, raising=True)


@pytest.fixture(autouse=True)
def _stub_dspy_lm(request: pytest.FixtureRequest) -> Iterator[FakeDspyLM]:
    """Install a :class:`FakeDspyLM` for every synthesis test.

    DSPy is heavy to import, so the import is deferred to the fixture
    body — modules that never call DSPy do not pay the import cost just
    by living in this test package.

    A test can seed canned responses by attaching the ``dspy_canned``
    marker:

    .. code-block:: python

        @pytest.mark.dspy_canned({"SynthesizeBrief:abc": ["..."]})
        def test_thing() -> None: ...

    The marker's first positional argument is forwarded straight into
    :class:`FakeDspyLM`'s ``canned`` argument.
    """

    import dspy  # local import — DSPy is heavy at module import time.

    canned_marker = request.node.get_closest_marker("dspy_canned")
    canned: dict[str, Any] = {}
    if canned_marker and canned_marker.args:
        first = canned_marker.args[0]
        if isinstance(first, dict):
            canned = first

    fake = FakeDspyLM(canned)
    dspy.configure(lm=fake)
    try:
        yield fake
    finally:
        dspy.configure(lm=None)


# ---------------------------------------------------------------------------
# real_lm marker plumbing
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-real-lm`` and ``--run-slow`` opt-in flags.

    Without ``--run-real-lm``, any test carrying ``@pytest.mark.real_lm``
    is skipped at collection time so the suite never accidentally hits a
    live LM endpoint. Without ``--run-slow``, any test carrying
    ``@pytest.mark.slow`` is skipped — these are the optimizer
    end-to-end tests that exercise ``BootstrapFewShot`` against the
    fake LM and are slow / flaky enough to keep out of the default
    suite (Req 6.4).
    """

    parser.addoption(
        "--run-real-lm",
        action="store_true",
        default=False,
        help=(
            "Collect tests marked @pytest.mark.real_lm and let them hit a "
            "real LM endpoint. Default: skip them."
        ),
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help=(
            "Collect tests marked @pytest.mark.slow (e.g. DSPy optimizer "
            "end-to-end runs). Default: skip them."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``real_lm``, ``dspy_canned``, and ``slow`` markers.

    Registering the markers up front keeps unknown-marker warnings out
    of the pytest output for the synthesis package.
    """

    config.addinivalue_line(
        "markers",
        "real_lm: opt-in marker; only runs when --run-real-lm is passed.",
    )
    config.addinivalue_line(
        "markers",
        "dspy_canned(canned): seed FakeDspyLM with the given canned-responses dict.",
    )
    config.addinivalue_line(
        "markers",
        "slow: opt-in marker for slow tests (e.g. DSPy optimizer end-to-end); "
        "only runs when --run-slow is passed.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip opt-in markers unless their flags are passed."""

    if not config.getoption("--run-real-lm"):
        skip_real_lm = pytest.mark.skip(
            reason="needs --run-real-lm to opt in to a real LM endpoint",
        )
        for item in items:
            if "real_lm" in item.keywords:
                item.add_marker(skip_real_lm)

    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(
            reason="needs --run-slow to opt in to slow tests",
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
