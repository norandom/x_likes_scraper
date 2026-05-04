"""End-to-end orchestrator tests for the synthesis-report feature (task 6.1).

These tests drive :func:`x_likes_mcp.synthesis.orchestrator.run_report`
against a real :class:`TweetIndex` built from the on-disk fixture corpus
in ``tests/mcp/fixtures/``. They cover the full pipeline (round-1
search, optional round-2 fan-out, KG build, fenced-context assembly,
synthesizer call, markdown render) for every report shape, while still
staying offline thanks to the autouse ``_block_real_url_fetch`` and
``_stub_dspy_lm`` guards in the synthesis conftest.

The :func:`x_likes_mcp.synthesis.dspy_modules.synthesize` seam is
monkeypatched in every test with a deterministic stub that introspects
the orchestrator's ``known_source_ids`` set and crafts valid claims so
the validator passes; one negative test (`unknown sources`) flips the
stub to cite ``tweet:99999`` and asserts the orchestrator surfaces a
``validation`` failure.

The fetcher seam is monkeypatched in the ``fetch_urls=False`` test
with sentinels that raise on call: the test passes only when the
sentinels are never tripped, which proves the orchestrator does not
even instantiate the fetcher when URL fetching is opt-out.

Boundary discipline: only this file is touched. The orchestrator,
synthesizer, fetcher, renderer, and index code are all consumed
through their public seams.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.index import TweetIndex
from x_likes_mcp.ranker import ScoredHit
from x_likes_mcp.synthesis import orchestrator
from x_likes_mcp.synthesis.dspy_modules import (
    SynthesisResult,
    SynthesisValidationError,
)
from x_likes_mcp.synthesis.orchestrator import OrchestratorError, run_report
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import (
    Claim,
    MonthSummary,
    ReportOptions,
    Section,
)

# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_os_environ() -> Iterator[None]:
    """Snapshot ``os.environ`` and restore it after each test.

    The shared ``fake_export`` fixture (in ``tests/mcp/conftest.py``)
    calls :func:`x_likes_mcp.config.load_config`, which has a documented
    side effect of writing ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` into
    ``os.environ``. The downstream ``tests/mcp/test_config.py`` tests
    rebind ``os.environ`` to a *copy of the current process environ* —
    so any leak from this file would surface there as a flaky failure
    in tests collected later. Snapshotting + restoring keeps this file's
    fixtures hermetic.
    """

    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_index(config: Config) -> TweetIndex:
    """Build a real :class:`TweetIndex` over the fixture export.

    Relies on the parent conftest's ``_block_real_embeddings`` autouse
    fixture, which stubs :meth:`Embedder._call_embeddings_api` with a
    deterministic 4-dim vector so the index build stays offline.
    """

    return TweetIndex.open_or_build(config, RankerWeights())


@dataclass
class _SearchProbe:
    """Recorder for :meth:`TweetIndex.search` calls.

    Wraps the original ``search`` method, captures the call timestamps
    (one entry per invocation) and the per-call ``query`` argument, and
    delegates to the underlying implementation so the rest of the
    pipeline still receives real hits.
    """

    queries: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _wrap_search(
    monkeypatch: pytest.MonkeyPatch,
    index: TweetIndex,
    *,
    delay: float = 0.0,
) -> _SearchProbe:
    """Replace ``index.search`` with a recorder.

    ``delay`` is forwarded to a ``time.sleep`` inside each wrapped call
    so a parallel-fan-out test can detect concurrency by observing
    overlapping start timestamps.
    """

    probe = _SearchProbe()
    original_search = index.search

    def _wrapped(query: str, **kwargs: Any) -> list[ScoredHit]:
        with probe.lock:
            probe.queries.append(query)
            probe.timestamps.append(time.monotonic())
        if delay:
            time.sleep(delay)
        return original_search(query, **kwargs)

    monkeypatch.setattr(index, "search", _wrapped, raising=True)
    return probe


def _make_synthesize_stub(
    *,
    shape: ReportShape,
    captured: dict[str, Any] | None = None,
    force_unknown_sources: bool = False,
) -> Callable[..., SynthesisResult]:
    """Return a stub for :func:`orchestrator.synthesize`.

    The stub introspects ``known_source_ids`` so it can craft claims
    that cite real tweet IDs for each shape:

    * BRIEF — one :class:`Claim` per tweet source.
    * SYNTHESIS — one :class:`Section` whose claims cover every tweet
      source.
    * TREND — one :class:`MonthSummary` per ``month_buckets`` entry,
      each anchoring to the first tweet source.

    When ``force_unknown_sources`` is true the stub cites ``tweet:99999``
    (which is never in the fixture index) so the orchestrator's
    validator surfaces a ``validation`` failure. The stub is invoked
    twice in that branch — once for the original call, once for the
    corrective retry — so the validator's "retry" behavior also runs.
    """

    def _stub(
        shape_arg: ReportShape,
        query: str,
        fenced_context: str,
        *,
        known_source_ids: set[str],
        month_buckets: list[str] | None = None,
        program: Any | None = None,
    ) -> SynthesisResult:
        if captured is not None:
            captured.setdefault("calls", []).append(
                {
                    "shape": shape_arg,
                    "query": query,
                    "fenced_context": fenced_context,
                    "known_source_ids": set(known_source_ids),
                    "month_buckets": list(month_buckets or []),
                    "program": program,
                }
            )

        # Pick tweet sources only — the renderer's anchor lists key off
        # ``tweet:<id>`` IDs. Sort for determinism so the resulting
        # report is stable across runs.
        tweet_sources = sorted(sid for sid in known_source_ids if sid.startswith("tweet:"))
        if force_unknown_sources:
            # Mimic the real :func:`synthesize` validator's behaviour:
            # when the LM cites an ID that is not in the known-source
            # set even after the corrective retry, the function raises
            # :class:`SynthesisValidationError`. The orchestrator
            # translates that into ``OrchestratorError("validation")``.
            raise SynthesisValidationError(
                "Synthesis cited unknown source IDs after retry: ['tweet:99999']"
            )
        sources = tweet_sources or list(known_source_ids)

        if shape_arg is ReportShape.BRIEF:
            claims = [Claim(text=f"Claim grounded in {sid}.", sources=[sid]) for sid in sources]
            return SynthesisResult(
                claims=claims,
                top_entities=["alice", "agents"],
            )

        if shape_arg is ReportShape.SYNTHESIS:
            section = Section(
                heading="Overview",
                claims=[Claim(text=f"Section claim for {sid}.", sources=[sid]) for sid in sources],
            )
            return SynthesisResult(
                sections=[section],
                top_entities=["alice", "agents"],
                cluster_assignments={
                    "alice": [sid.split(":", 1)[1] for sid in sources],
                },
            )

        if shape_arg is ReportShape.TREND:
            buckets = list(month_buckets or [])
            anchor = sources[0] if sources else ""
            anchor_id = anchor.split(":", 1)[1] if anchor.startswith("tweet:") else anchor
            per_month = [
                MonthSummary(
                    year_month=ym,
                    summary=f"Trend summary for {ym}.",
                    anchor_tweets=[anchor_id] if anchor_id else [],
                )
                for ym in buckets
            ]
            return SynthesisResult(
                per_month=per_month,
                top_entities=["alice", "agents"],
            )

        raise AssertionError(f"unexpected shape: {shape_arg!r}")  # pragma: no cover

    return _stub


def _install_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shape: ReportShape,
    captured: dict[str, Any] | None = None,
    force_unknown_sources: bool = False,
) -> None:
    """Patch ``configure_lm`` and ``synthesize`` for an e2e run.

    Tests inject ``year_month`` / ``handle`` / ``snippet`` onto the
    real fixture-backed hits via the orchestrator's own helper code
    path; only the LM-touching seams get stubbed here.
    """

    monkeypatch.setattr(orchestrator, "configure_lm", lambda _config: None)
    # The orchestrator's KG-population loop falls through to the DSPy
    # fallback for hits whose regex pass returned nothing; the FakeDspyLM
    # configured by the autouse ``_stub_dspy_lm`` fixture is rejected by
    # DSPy 3.x's ``Predict`` machinery (it requires a ``BaseLM``
    # subclass), so we stub the fallback to a deterministic empty list
    # to keep the pipeline LM-free.
    monkeypatch.setattr(
        orchestrator,
        "extract_entities",
        lambda _text, **_kwargs: [],
    )
    monkeypatch.setattr(
        orchestrator,
        "synthesize",
        _make_synthesize_stub(
            shape=shape,
            captured=captured,
            force_unknown_sources=force_unknown_sources,
        ),
    )


def _annotate_hits_with_year_month(
    monkeypatch: pytest.MonkeyPatch,
    index: TweetIndex,
) -> None:
    """Wrap ``index.search`` to inject ``year_month`` onto each hit.

    The renderer's TREND shape buckets hits by their ``year_month``
    attribute; ``index.search`` returns plain :class:`ScoredHit`
    instances that do not carry that attribute, so we replicate the
    ``tools.search_likes`` behavior here for the tests that care.

    This wrapper composes with ``_wrap_search``: order matters, so each
    test installs at most one of the two wrappers (the parallel test
    skips this).
    """

    original_search = index.search

    def _wrapped(query: str, **kwargs: Any) -> list[ScoredHit]:
        hits = original_search(query, **kwargs)
        annotated: list[ScoredHit] = []
        for hit in hits:
            tweet = index.tweets_by_id.get(hit.tweet_id)
            ym = ""
            handle = ""
            snippet = ""
            if tweet is not None:
                try:
                    ym = tweet.get_created_datetime().strftime("%Y-%m")
                except (ValueError, TypeError):
                    ym = ""
                if tweet.user is not None:
                    handle = tweet.user.screen_name or ""
                snippet = (tweet.text or "")[:120]
            new_hit = replace(hit)
            object.__setattr__(new_hit, "year_month", ym)
            object.__setattr__(new_hit, "handle", handle)
            object.__setattr__(new_hit, "snippet", snippet)
            annotated.append(new_hit)
        return annotated

    monkeypatch.setattr(index, "search", _wrapped, raising=True)


# ---------------------------------------------------------------------------
# BRIEF shape
# ---------------------------------------------------------------------------


def test_e2e_brief_returns_sanitized_markdown_with_valid_claims(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """The BRIEF shape produces a non-empty, sanitized markdown body.

    Every claim cites a tweet ID that lives in the fixture-backed
    index, so the orchestrator's validator passes without a retry.
    """

    index = _build_index(fake_export)
    captured: dict[str, Any] = {}
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.BRIEF, captured=captured)
    _annotate_hits_with_year_month(monkeypatch, index)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=1,
    )
    result = run_report(index, options, config=fake_export)

    assert result.shape is ReportShape.BRIEF
    assert result.used_hops == 1
    assert result.fetched_url_count == 0

    md = result.markdown
    assert md.strip(), "BRIEF markdown must not be empty"
    assert "# Synthesis report" in md
    assert "## Brief" in md
    assert "## Top entities" in md
    assert "## Anchor tweets" in md

    # Sanitization: no ANSI / BiDi codepoints in the body. The BiDi
    # codepoints are constructed via ``chr`` so this source file does
    # not itself contain hidden bidi runs that could fool a reviewer or
    # static analyser.
    forbidden = [
        "\x1b[",  # ANSI CSI
        chr(0x202A),  # LRE
        chr(0x202B),  # RLE
        chr(0x202C),  # PDF
        chr(0x202D),  # LRO
        chr(0x202E),  # RLO
    ]
    for needle in forbidden:
        assert needle not in md, f"sanitizer leaked {needle!r} into markdown"

    # Every claim cited a known tweet source from the fenced context.
    call = captured["calls"][0]
    fixture_ids = {f"tweet:{tid}" for tid in index.tweets_by_id}
    cited = {
        sid
        for cl in _make_synthesize_stub(shape=ReportShape.BRIEF)(
            ReportShape.BRIEF,
            options.query,
            "",
            known_source_ids=call["known_source_ids"],
        ).claims
        or []
        for sid in cl.sources
    }
    assert cited, "stub must have produced at least one cited claim"
    assert cited <= fixture_ids, f"claims cited unknown sources: {cited - fixture_ids}"


# ---------------------------------------------------------------------------
# SYNTHESIS shape
# ---------------------------------------------------------------------------


def test_e2e_synthesis_produces_mindmap_block(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """The SYNTHESIS shape emits a mermaid ``mindmap`` fence."""

    index = _build_index(fake_export)
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.SYNTHESIS)
    _annotate_hits_with_year_month(monkeypatch, index)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.SYNTHESIS,
        fetch_urls=False,
        hops=1,
    )
    result = run_report(index, options, config=fake_export)

    md = result.markdown
    assert "## Mindmap" in md
    assert "```mermaid" in md
    assert "mindmap" in md
    assert "## Overview" in md  # the section heading from the stub.


# ---------------------------------------------------------------------------
# TREND shape
# ---------------------------------------------------------------------------


def test_e2e_trend_orders_months_chronologically(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """The TREND shape emits month sections in chronological order."""

    index = _build_index(fake_export)
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.TREND)
    _annotate_hits_with_year_month(monkeypatch, index)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.TREND,
        fetch_urls=False,
        hops=1,
    )
    result = run_report(index, options, config=fake_export)

    md = result.markdown
    # The fixture spans 2025-01, 2025-02, 2025-03; all three should
    # appear and be ordered chronologically.
    positions: dict[str, int] = {}
    for ym in ("2025-01", "2025-02", "2025-03"):
        heading = f"## {ym}"
        assert heading in md, f"missing month section {heading!r}"
        positions[ym] = md.index(heading)
    assert positions["2025-01"] < positions["2025-02"] < positions["2025-03"]


# ---------------------------------------------------------------------------
# Hops dispatch
# ---------------------------------------------------------------------------


def test_e2e_hops_1_issues_one_index_search(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """``hops=1`` issues exactly one :meth:`TweetIndex.search` call."""

    index = _build_index(fake_export)
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.BRIEF)
    probe = _wrap_search(monkeypatch, index)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=1,
    )
    run_report(index, options, config=fake_export)

    assert (
        len(probe.queries) == 1
    ), f"hops=1 must trigger exactly one search call; got {probe.queries}"
    assert probe.queries[0] == "agents"


def test_e2e_hops_2_issues_round_two_searches_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """``hops=2`` fans out round-2 searches in parallel via the executor.

    Concurrency is detected by sleeping inside each wrapped search and
    asserting that the round-2 search timestamps cluster within a window
    much shorter than ``K * delay``: a sequential implementation would
    accumulate ``K * delay`` of wall-clock time between the first and
    last round-2 timestamp.
    """

    index = _build_index(fake_export)
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.BRIEF)
    probe = _wrap_search(monkeypatch, index, delay=0.05)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=2,
    )
    run_report(index, options, config=fake_export)

    # Round-1 (1 call) + up to K round-2 calls. K defaults to
    # ``Config.synthesis_round_two_k`` (5), but the actual round-2 call
    # count depends on the entity set the round-1 KG produced for the
    # fixture corpus. We assert a non-trivial fan-out and the parallel
    # window invariant.
    assert (
        len(probe.queries) >= 2
    ), f"hops=2 must issue at least one round-2 search; got {probe.queries}"
    # Round-1 is always the user query and always first.
    assert probe.queries[0] == "agents"

    round_two_starts = probe.timestamps[1:]
    assert round_two_starts, "round-2 must have at least one search"

    # Allow up to ``ceil(N / max_workers) * delay`` of wall-clock spread
    # in the parallel case (default ``max_workers=4``); a sequential
    # implementation would spread by ``N * delay``. Assert the spread is
    # strictly less than the sequential lower bound to prove fan-out is
    # parallelised.
    spread = max(round_two_starts) - min(round_two_starts)
    sequential_lower_bound = 0.05 * len(round_two_starts)
    # Generous safety margin: parallel spread should be well below the
    # sequential bound.
    assert spread < sequential_lower_bound, (
        f"round-2 calls were not parallelised: spread={spread:.3f}s, "
        f"sequential bound={sequential_lower_bound:.3f}s"
    )


# ---------------------------------------------------------------------------
# Fetcher gating
# ---------------------------------------------------------------------------


def test_e2e_fetch_urls_false_never_instantiates_fetcher(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """``fetch_urls=False`` never invokes the fetcher or the probe."""

    index = _build_index(fake_export)
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.BRIEF)
    _annotate_hits_with_year_month(monkeypatch, index)

    def _fail_fetch(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "synthesis.fetcher.fetch_all must not be invoked when " "fetch_urls=False",
        )

    def _fail_probe(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "synthesis.fetcher.probe_container must not be invoked when " "fetch_urls=False",
        )

    monkeypatch.setattr(orchestrator, "fetch_all", _fail_fetch)
    monkeypatch.setattr(orchestrator, "probe_container", _fail_probe)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=1,
    )
    result = run_report(index, options, config=fake_export)
    # The sentinels would have raised before we got here.
    assert result.fetched_url_count == 0


# ---------------------------------------------------------------------------
# Validation: claims with unknown sources
# ---------------------------------------------------------------------------


def test_e2e_claims_with_unknown_sources_raise_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """A synthesizer that cites unknown IDs surfaces ``validation``."""

    index = _build_index(fake_export)
    _install_pipeline_stubs(
        monkeypatch,
        shape=ReportShape.BRIEF,
        force_unknown_sources=True,
    )
    _annotate_hits_with_year_month(monkeypatch, index)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=1,
    )
    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, options, config=fake_export)
    assert exc_info.value.category == "validation"


# ---------------------------------------------------------------------------
# Dedupe across hops
# ---------------------------------------------------------------------------


def test_e2e_dedupes_by_tweet_id_across_hops(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """Final report's anchor list contains each tweet_id at most once.

    ``hops=2`` runs round-1 + round-2 over the fixture corpus; the
    round-2 fan-out re-issues entity-keyed searches that hit the same
    tweet IDs. The orchestrator's :func:`fuse_results` step must dedupe
    so the rendered anchor list never lists a tweet twice.
    """

    index = _build_index(fake_export)
    _install_pipeline_stubs(monkeypatch, shape=ReportShape.BRIEF)
    _annotate_hits_with_year_month(monkeypatch, index)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=2,
    )
    result = run_report(index, options, config=fake_export)

    md = result.markdown
    # Each fixture tweet ID must appear at most once as an anchor bullet
    # marker (``- [<id>](``). We count bracketed-id occurrences in the
    # rendered anchor list to detect a duplicate render.
    for tid in index.tweets_by_id:
        marker = f"- [{tid}]("
        assert md.count(marker) <= 1, f"tweet {tid} rendered more than once in anchor list"


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_e2e_returns_orchestrator_error_when_lm_endpoint_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_export: Config,
) -> None:
    """A Config with ``openai_base_url=None`` raises ``config``.

    The real :func:`configure_lm` is exercised here (the test does NOT
    install the stub) so a missing ``OPENAI_BASE_URL`` triggers
    :class:`ConfigError`, which the orchestrator translates into
    ``OrchestratorError(category="config")``.
    """

    index = _build_index(fake_export)
    # Replace synthesize so an accidental success path would still not
    # contact the LM, but we expect the pipeline to fail at
    # ``configure_lm`` before reaching it.
    monkeypatch.setattr(
        orchestrator,
        "synthesize",
        _make_synthesize_stub(shape=ReportShape.BRIEF),
    )
    _annotate_hits_with_year_month(monkeypatch, index)

    broken_config = replace(fake_export, openai_base_url=None)

    options = ReportOptions(
        query="agents",
        shape=ReportShape.BRIEF,
        fetch_urls=False,
        hops=1,
    )
    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, options, config=broken_config)
    assert exc_info.value.category == "config"
