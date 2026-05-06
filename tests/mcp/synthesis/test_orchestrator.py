"""Tests for the synthesis-report orchestrator (task 5.1).

Pin the orchestrator service interface:

* :func:`run_report` validates ``options.shape`` and ``options.hops``
  before any side effect and translates downstream exceptions into
  :class:`OrchestratorError` with a structured ``category``.
* The pipeline order is: round-1 search → optional round-2 fan-out →
  optional URL fetch (only when ``fetch_urls=True``) → KG build →
  fenced context → DSPy synthesis → markdown render.
* Empty round-1 hits short-circuit to the empty-report markdown without
  configuring the LM, fetching URLs, or calling the synthesizer (Req
  9.4).
* ``fetch_urls=False`` makes the orchestrator never reach the
  fetcher / probe (Req 12.4).
* The DSPy entity-extraction fallback fires only for hits whose regex
  pass returned nothing (Req 5.2).
* Compiled programs load through :func:`load_compiled` and feed into
  :func:`synthesize` as the ``program`` kwarg (Req 6.3).
* The orchestrator never writes anything to disk under the configured
  ``output_dir`` for the happy path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.ranker import ScoredHit
from x_likes_mcp.synthesis import orchestrator
from x_likes_mcp.synthesis.dspy_modules import (
    SynthesisError,
    SynthesisResult,
    SynthesisValidationError,
)
from x_likes_mcp.synthesis.fetcher import ContainerUnreachable
from x_likes_mcp.synthesis.kg import KG
from x_likes_mcp.synthesis.orchestrator import OrchestratorError, run_report
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import (
    Claim,
    Entity,
    EntityKind,
    FetchedUrl,
    ReportOptions,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeUser:
    """Minimal user duck-type for the Tweet stub (carries a screen_name)."""

    def __init__(self, screen_name: str = "") -> None:
        self.screen_name = screen_name


class _FakeTweet:
    """Minimal Tweet duck-type the orchestrator reads ``text`` / ``urls`` off."""

    def __init__(
        self,
        *,
        tweet_id: str,
        text: str = "",
        urls: list[str] | None = None,
        screen_name: str = "alice",
    ) -> None:
        self.id = tweet_id
        self.text = text
        self.urls = list(urls or [])
        self.user = _FakeUser(screen_name)


class _FakeIndex:
    """Stub TweetIndex carrying just the ``tweets_by_id`` map.

    The orchestrator's pipeline is monkeypatched at the leaf-function
    boundary, so the index does not need a working ``search``: it
    suffices that ``tweets_by_id`` returns the right :class:`_FakeTweet`
    for each hit.
    """

    def __init__(self, tweets: dict[str, _FakeTweet] | None = None) -> None:
        self.tweets_by_id: dict[str, Any] = dict(tweets or {})


def _scored_hit(
    tweet_id: str,
    *,
    snippet: str = "",
    handle: str = "alice",
) -> ScoredHit:
    """Build a :class:`ScoredHit` carrying the orchestrator-injected fields.

    ``ScoredHit`` is frozen, so we attach optional fields via
    ``object.__setattr__`` after construction (mirrors the production
    code path where the search seam injects them post-rank).
    """

    hit = ScoredHit(
        tweet_id=tweet_id,
        score=1.0,
        walker_relevance=0.0,
        why="",
        feature_breakdown={},
    )
    object.__setattr__(hit, "snippet", snippet)
    object.__setattr__(hit, "handle", handle)
    return hit


def _config(tmp_path: Path) -> Config:
    """Build a minimal :class:`Config` rooted at ``tmp_path``.

    The OPENAI_* fields are populated so :func:`configure_lm` (which is
    monkeypatched in the suite) does not even need the env vars — but
    we still set them so an accidental real call would point at a
    non-routable host instead of a real endpoint.
    """

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        output_dir=output_dir,
        by_month_dir=output_dir / "by_month",
        likes_json=output_dir / "likes.json",
        cache_path=output_dir / "tweet_tree_cache.pkl",
        ranker_weights=RankerWeights(),
        openai_base_url="http://fake/v1",
        openai_api_key="EMPTY",
        openai_model="fake-model",
        url_cache_dir=output_dir / "url_cache",
    )


def _options(
    *,
    query: str = "ai security",
    shape: ReportShape = ReportShape.BRIEF,
    fetch_urls: bool = False,
    hops: int = 1,
    filter_entities: bool = False,
) -> ReportOptions:
    return ReportOptions(
        query=query,
        shape=shape,
        fetch_urls=fetch_urls,
        hops=hops,
        filter_entities=filter_entities,
    )


# ---------------------------------------------------------------------------
# Pipeline-stage stubs installed via monkeypatch
# ---------------------------------------------------------------------------


def _install_default_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    round_one: list[ScoredHit] | None = None,
    round_two: list[ScoredHit] | None = None,
    synthesis_result: SynthesisResult | None = None,
    fetched_urls: list[FetchedUrl] | None = None,
    compiled_program: Any | None = None,
    extract_regex_impl: Callable[..., list[Entity]] | None = None,
) -> dict[str, list[Any]]:
    """Replace each pipeline-stage entry point with a recording stub.

    Returns a dict of call recorders (``round_one_calls``,
    ``round_two_calls``, ``configure_lm_calls``, ``synthesize_calls``,
    ``probe_calls``, ``fetch_calls``, ``load_compiled_calls``,
    ``extract_regex_calls``, ``dspy_extract_calls``) so individual tests
    can assert on what fired.
    """

    state: dict[str, list[Any]] = {
        "round_one_calls": [],
        "round_two_calls": [],
        "configure_lm_calls": [],
        "synthesize_calls": [],
        "probe_calls": [],
        "fetch_calls": [],
        "load_compiled_calls": [],
        "extract_regex_calls": [],
        "dspy_extract_calls": [],
    }

    def _stub_round_one(index: Any, options: Any) -> list[ScoredHit]:
        state["round_one_calls"].append((index, options))
        return list(round_one or [])

    def _stub_round_two(
        index: Any,
        options: Any,
        kg: KG,
        *,
        k: int = 5,
    ) -> list[ScoredHit]:
        # Snapshot the KG node-id set at call time so later mutations
        # (e.g. extending the KG with round-2 entities) do not retroactively
        # change what round_two saw.
        snapshot = set(kg._nodes.keys())
        state["round_two_calls"].append((index, options, kg, k, snapshot))
        return list(round_two or [])

    def _stub_configure_lm(config: Any) -> None:
        state["configure_lm_calls"].append(config)

    def _stub_synthesize(
        shape: Any,
        query: str,
        fenced_context: str,
        *,
        known_source_ids: set[str],
        month_buckets: list[str] | None = None,
        program: Any | None = None,
    ) -> SynthesisResult:
        state["synthesize_calls"].append(
            {
                "shape": shape,
                "query": query,
                "fenced_context": fenced_context,
                "known_source_ids": set(known_source_ids),
                "month_buckets": list(month_buckets or []),
                "program": program,
            }
        )
        if synthesis_result is None:
            return SynthesisResult(
                claims=[Claim(text="stub claim", sources=[])],
                top_entities=[],
            )
        return synthesis_result

    def _stub_probe(base_url: str, **_: Any) -> None:
        state["probe_calls"].append(base_url)

    def _stub_fetch_all(urls: Any, **_: Any) -> list[FetchedUrl]:
        state["fetch_calls"].append(list(urls))
        return list(fetched_urls or [])

    def _stub_load_compiled(shape: Any, root: Path) -> Any:
        state["load_compiled_calls"].append((shape, root))
        return compiled_program

    def _wrap_extract_regex(text: str, url_bodies: list[str]) -> list[Entity]:
        state["extract_regex_calls"].append(text)
        if extract_regex_impl is not None:
            return extract_regex_impl(text, url_bodies)
        # Default: the regex pass finds at least one entity so the DSPy
        # fallback never fires.
        return [Entity(EntityKind.HANDLE, "alice", 1.0)]

    def _stub_dspy_extract(text: str, **_: Any) -> list[Entity]:
        state["dspy_extract_calls"].append(text)
        return [Entity(EntityKind.CONCEPT, "fallback_concept", 1.0)]

    monkeypatch.setattr(orchestrator, "run_round_one", _stub_round_one)
    monkeypatch.setattr(orchestrator, "run_round_two", _stub_round_two)
    monkeypatch.setattr(orchestrator, "configure_lm", _stub_configure_lm)
    monkeypatch.setattr(orchestrator, "synthesize", _stub_synthesize)
    monkeypatch.setattr(orchestrator, "probe_container", _stub_probe)
    monkeypatch.setattr(orchestrator, "fetch_all", _stub_fetch_all)
    monkeypatch.setattr(orchestrator, "load_compiled", _stub_load_compiled)
    monkeypatch.setattr(orchestrator, "extract_regex", _wrap_extract_regex)
    monkeypatch.setattr(orchestrator, "extract_entities", _stub_dspy_extract)
    return state


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_unknown_shape_raises_invalid_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-:class:`ReportShape` shape value raises ``invalid_input``."""

    state = _install_default_stubs(monkeypatch)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi @alice")})

    # Bypass the dataclass type check by constructing with object.__new__
    # then setting ``shape`` to a bogus string.
    bad_options = object.__new__(ReportOptions)
    object.__setattr__(bad_options, "query", "x")
    object.__setattr__(bad_options, "shape", "not-a-shape")
    object.__setattr__(bad_options, "fetch_urls", False)
    object.__setattr__(bad_options, "hops", 1)
    object.__setattr__(bad_options, "year", None)
    object.__setattr__(bad_options, "month_start", None)
    object.__setattr__(bad_options, "month_end", None)
    object.__setattr__(bad_options, "limit", 50)

    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, bad_options, config=config)

    assert exc_info.value.category == "invalid_input"
    # The pipeline must not have fired at all.
    assert state["round_one_calls"] == []
    assert state["configure_lm_calls"] == []


def test_validate_hops_3_raises_invalid_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``hops=3`` is rejected before any search call."""

    state = _install_default_stubs(monkeypatch)
    config = _config(tmp_path)
    index = _FakeIndex()

    options = _options(hops=3)
    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, options, config=config)

    assert exc_info.value.category == "invalid_input"
    assert state["round_one_calls"] == []


# ---------------------------------------------------------------------------
# Empty corpus shortcut
# ---------------------------------------------------------------------------


def test_empty_corpus_skips_lm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty round-1 returns the empty report; no LM / fetch / synth call."""

    state = _install_default_stubs(monkeypatch, round_one=[])
    config = _config(tmp_path)
    index = _FakeIndex()

    result = run_report(index, _options(), config=config)

    assert result.fetched_url_count == 0
    assert result.synthesis_token_count == 0
    assert result.used_hops == 1
    assert result.shape is ReportShape.BRIEF
    assert "No matching tweets" in result.markdown
    assert state["configure_lm_calls"] == []
    assert state["synthesize_calls"] == []
    assert state["probe_calls"] == []
    assert state["fetch_calls"] == []


# ---------------------------------------------------------------------------
# Hops dispatch
# ---------------------------------------------------------------------------


def test_hops_1_skips_round_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``hops=1`` runs round-1 only and renders without round-2."""

    hits = [_scored_hit("1", snippet="hello @alice")]
    state = _install_default_stubs(monkeypatch, round_one=hits)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hello @alice")})

    result = run_report(index, _options(hops=1), config=config)

    assert state["round_one_calls"], "round-1 must run"
    assert state["round_two_calls"] == [], "round-2 must NOT run for hops=1"
    assert result.used_hops == 1
    assert state["synthesize_calls"], "synthesizer must fire when hits exist"


def test_hops_2_calls_round_two_with_round_one_kg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``hops=2`` runs round-2 with the round-1 KG as input."""

    hits1 = [_scored_hit("1", snippet="hello @alice")]
    hits2 = [_scored_hit("2", snippet="bye @bob", handle="bob")]
    state = _install_default_stubs(monkeypatch, round_one=hits1, round_two=hits2)
    config = _config(tmp_path)
    index = _FakeIndex(
        {
            "1": _FakeTweet(tweet_id="1", text="hello @alice"),
            "2": _FakeTweet(tweet_id="2", text="bye @bob", screen_name="bob"),
        }
    )

    result = run_report(index, _options(hops=2), config=config)

    assert len(state["round_two_calls"]) == 1
    _, _, _kg_passed, _, snapshot = state["round_two_calls"][0]
    # The KG handed to round_two must contain only round-1 nodes
    # (tweet:1, the query root, and round-1 entities), not tweet:2.
    assert "tweet:1" in snapshot
    assert "tweet:2" not in snapshot
    assert result.used_hops == 2


# ---------------------------------------------------------------------------
# Fetch behavior
# ---------------------------------------------------------------------------


def test_fetch_urls_false_skips_fetcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fetch_urls=False`` never reaches the probe or the fetcher."""

    hits = [_scored_hit("1", snippet="hello @alice")]
    state = _install_default_stubs(monkeypatch, round_one=hits)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi", urls=["https://a"])})

    result = run_report(index, _options(fetch_urls=False), config=config)

    assert state["probe_calls"] == [], "probe must not fire when fetch_urls=False"
    assert state["fetch_calls"] == [], "fetch_all must not fire when fetch_urls=False"
    assert state["configure_lm_calls"], "LM config still runs (synthesis fires)"
    assert result.fetched_url_count == 0


def test_fetch_urls_true_calls_probe_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fetch_urls=True`` probes the container before fetching."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    fetched = FetchedUrl(
        url="https://a",
        final_url="https://a",
        content_type="text/html",
        sanitized_markdown="body",
        size_bytes=4,
    )
    state = _install_default_stubs(
        monkeypatch,
        round_one=hits,
        fetched_urls=[fetched],
    )
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi", urls=["https://a"])})

    result = run_report(index, _options(fetch_urls=True), config=config)

    assert state["probe_calls"], "probe must fire when fetch_urls=True"
    assert state["fetch_calls"], "fetch_all must run after the probe"
    assert state["fetch_calls"][0] == ["https://a"]
    assert result.fetched_url_count == 1


def test_probe_failure_surfaces_as_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe-time ``ContainerUnreachable`` becomes ``OrchestratorError("upstream")``."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    state = _install_default_stubs(monkeypatch, round_one=hits)

    def _raise_probe(base_url: str, **_: Any) -> None:
        state["probe_calls"].append(base_url)
        raise ContainerUnreachable("nope")

    monkeypatch.setattr(orchestrator, "probe_container", _raise_probe)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi")})

    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, _options(fetch_urls=True), config=config)
    assert exc_info.value.category == "upstream"
    assert state["fetch_calls"] == [], "fetcher must not run after a probe failure"


# ---------------------------------------------------------------------------
# Compiled program
# ---------------------------------------------------------------------------


def test_compiled_program_loaded_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-``None`` :func:`load_compiled` return reaches :func:`synthesize`."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    sentinel_program = object()
    state = _install_default_stubs(
        monkeypatch,
        round_one=hits,
        compiled_program=sentinel_program,
    )
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi")})

    run_report(index, _options(), config=config)

    assert state["synthesize_calls"]
    assert state["synthesize_calls"][0]["program"] is sentinel_program


def test_compiled_program_none_passes_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing compiled artifact → ``program=None`` flows into ``synthesize``."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    state = _install_default_stubs(monkeypatch, round_one=hits, compiled_program=None)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi")})

    run_report(index, _options(), config=config)

    assert state["synthesize_calls"][0]["program"] is None


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_synthesis_validation_error_surfaces_as_validation_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:class:`SynthesisValidationError` → ``OrchestratorError("validation")``."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    _install_default_stubs(monkeypatch, round_one=hits)

    def _raise_validation(*_args: Any, **_kwargs: Any) -> SynthesisResult:
        raise SynthesisValidationError("cited unknown ids")

    monkeypatch.setattr(orchestrator, "synthesize", _raise_validation)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi")})

    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, _options(), config=config)
    assert exc_info.value.category == "validation"


def test_synthesis_error_surfaces_as_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic :class:`SynthesisError` → ``OrchestratorError("upstream")``."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    _install_default_stubs(monkeypatch, round_one=hits)

    def _raise_synth(*_args: Any, **_kwargs: Any) -> SynthesisResult:
        raise SynthesisError("LM blew up")

    monkeypatch.setattr(orchestrator, "synthesize", _raise_synth)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi")})

    with pytest.raises(OrchestratorError) as exc_info:
        run_report(index, _options(), config=config)
    assert exc_info.value.category == "upstream"


# ---------------------------------------------------------------------------
# Disk-write & metadata
# ---------------------------------------------------------------------------


def _snapshot(root: Path) -> set[Path]:
    """Return every relative file path under ``root``."""

    if not root.exists():
        return set()
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def test_orchestrator_does_not_write_to_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path with ``fetch_urls=False`` writes nothing under ``output_dir``."""

    hits = [_scored_hit("1", snippet="hi @alice")]
    _install_default_stubs(monkeypatch, round_one=hits)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="hi")})

    before = _snapshot(config.output_dir)
    run_report(index, _options(fetch_urls=False), config=config)
    after = _snapshot(config.output_dir)

    assert before == after, f"orchestrator wrote unexpected files: {after - before}"


def test_run_report_returns_report_result_with_correct_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ReportResult`` carries the right ``shape``, ``used_hops``, and counts."""

    hits1 = [_scored_hit("1", snippet="hi @alice")]
    hits2 = [_scored_hit("2", snippet="ok @bob", handle="bob")]
    _install_default_stubs(monkeypatch, round_one=hits1, round_two=hits2)
    config = _config(tmp_path)
    index = _FakeIndex(
        {
            "1": _FakeTweet(tweet_id="1", text="hi"),
            "2": _FakeTweet(tweet_id="2", text="ok", screen_name="bob"),
        }
    )

    result = run_report(
        index,
        _options(shape=ReportShape.SYNTHESIS, hops=2),
        config=config,
    )
    assert result.shape is ReportShape.SYNTHESIS
    assert result.used_hops == 2
    assert result.fetched_url_count == 0
    # Markdown is non-empty for the happy path.
    assert result.markdown.strip()


# ---------------------------------------------------------------------------
# Entity extractor fallback (Req 5.2)
# ---------------------------------------------------------------------------


def test_entity_fallback_only_for_empty_regex_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DSPy fallback fires only for hits whose regex returned ``[]``."""

    hits = [
        _scored_hit("1", snippet="hi @alice"),  # regex returns entities
        _scored_hit("2", snippet="opaque body"),  # regex returns []
    ]

    def _selective_regex(text: str, _bodies: list[str]) -> list[Entity]:
        if "@alice" in text:
            return [Entity(EntityKind.HANDLE, "alice", 1.0)]
        return []

    state = _install_default_stubs(
        monkeypatch,
        round_one=hits,
        extract_regex_impl=_selective_regex,
    )
    config = _config(tmp_path)
    index = _FakeIndex(
        {
            "1": _FakeTweet(tweet_id="1", text="hi @alice"),
            "2": _FakeTweet(tweet_id="2", text="opaque body"),
        }
    )

    run_report(index, _options(), config=config)

    # The DSPy fallback fires exactly once — only for hit "2".
    assert len(state["dspy_extract_calls"]) == 1
    fallback_text = state["dspy_extract_calls"][0]
    assert "opaque body" in fallback_text
    assert "@alice" not in fallback_text


# ---------------------------------------------------------------------------
# Round-2 KG seed
# ---------------------------------------------------------------------------


def test_round_2_kg_uses_round_1_entities_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KG handed to round-2 contains only round-1 nodes."""

    hits1 = [_scored_hit("1", snippet="alpha @alice")]
    hits2 = [_scored_hit("2", snippet="beta @bob", handle="bob")]
    state = _install_default_stubs(
        monkeypatch,
        round_one=hits1,
        round_two=hits2,
    )
    config = _config(tmp_path)
    index = _FakeIndex(
        {
            "1": _FakeTweet(tweet_id="1", text="alpha @alice"),
            "2": _FakeTweet(tweet_id="2", text="beta @bob", screen_name="bob"),
        }
    )

    run_report(index, _options(hops=2), config=config)

    assert state["round_two_calls"]
    _, _, _kg, _, snapshot = state["round_two_calls"][0]
    # tweet:1 must be present (round-1), tweet:2 must NOT be (it would
    # be a future round-2 hit).
    assert "tweet:1" in snapshot
    assert "tweet:2" not in snapshot
    # The handle entity from round-1 ("alice") is present; the round-2
    # handle "bob" is not yet in the graph.
    assert "handle:alice" in snapshot
    assert "handle:bob" not in snapshot


# ---------------------------------------------------------------------------
# LM-backed entity-relevance filter (Layer 2)
# ---------------------------------------------------------------------------


def test_filter_entities_off_by_default_does_not_call_lm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``filter_entities=False`` (the default) leaves the KG untouched
    and does not call ``filter_entities_by_relevance``."""

    hits1 = [_scored_hit("1", snippet="alpha @alice")]
    state = _install_default_stubs(monkeypatch, round_one=hits1)
    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="alpha @alice")})

    filter_calls: list[tuple[str, list[str]]] = []

    def _spy_filter(query: str, candidates: list[str]) -> list[str]:
        filter_calls.append((query, list(candidates)))
        return list(candidates)

    monkeypatch.setattr(orchestrator, "filter_entities_by_relevance", _spy_filter)

    rc = run_report(index, _options(filter_entities=False), config=config)
    assert rc is not None
    assert filter_calls == []
    assert state["round_one_calls"]


def test_filter_entities_true_drops_excluded_entity_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the LM omits ``elonmusk`` from the relevant set, the KG
    handed to the synthesis phase has no ``handle:elonmusk`` node."""

    # Two hits: one mentions @alice (relevant), the other @elonmusk
    # (the LM will mark off-topic).
    hits1 = [
        _scored_hit("1", snippet="quant alpha @alice"),
        _scored_hit("2", snippet="off topic @elonmusk", handle="elonmusk"),
    ]

    # Install standard stubs but override extract_regex so each hit
    # contributes its own handle entity.
    def _hit_specific_extract(text: str, _url_bodies: list[str]) -> list[Entity]:
        if "alice" in text:
            return [Entity(EntityKind.HANDLE, "alice", 1.0)]
        if "elonmusk" in text:
            return [Entity(EntityKind.HANDLE, "elonmusk", 1.0)]
        return []

    state = _install_default_stubs(
        monkeypatch,
        round_one=hits1,
        extract_regex_impl=_hit_specific_extract,
    )

    # Filter LM keeps only "alice".
    def _filter(query: str, candidates: list[str]) -> list[str]:
        return [c for c in candidates if c == "alice"]

    monkeypatch.setattr(orchestrator, "filter_entities_by_relevance", _filter)

    config = _config(tmp_path)
    index = _FakeIndex(
        {
            "1": _FakeTweet(tweet_id="1", text="quant alpha @alice"),
            "2": _FakeTweet(tweet_id="2", text="off topic @elonmusk", screen_name="elonmusk"),
        }
    )

    run_report(index, _options(filter_entities=True), config=config)

    # The synthesis stub recorded its inputs; the fenced-context blob
    # is what we assert against. We check the ENTITY fence specifically
    # because the tweet body itself ("off topic @elonmusk") is still in
    # the TWEET_BODY fence — only the entity-node surface is filtered.
    assert state["synthesize_calls"], "synthesize was never called"
    blob = state["synthesize_calls"][0]["fenced_context"]
    assert "<<<ENTITY>>>alice<<<END_ENTITY>>>" in blob
    assert "<<<ENTITY>>>elonmusk<<<END_ENTITY>>>" not in blob


def test_filter_entities_true_with_no_entities_skips_lm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty entity set short-circuits to a no-op filter call.

    The hit's handle is empty AND the regex extractor returns nothing
    AND the DSPy fallback is stubbed to ``[]``, so no HANDLE / HASHTAG
    / DOMAIN / CONCEPT node ever lands in the KG.
    """

    hits1 = [_scored_hit("1", snippet="just text no entities", handle="")]

    def _no_entities(_text: str, _url_bodies: list[str]) -> list[Entity]:
        return []

    state = _install_default_stubs(
        monkeypatch,
        round_one=hits1,
        extract_regex_impl=_no_entities,
    )

    filter_calls: list[tuple[str, list[str]]] = []

    def _spy_filter(query: str, candidates: list[str]) -> list[str]:
        filter_calls.append((query, list(candidates)))
        return list(candidates)

    monkeypatch.setattr(orchestrator, "filter_entities_by_relevance", _spy_filter)
    # Stub the DSPy fallback to also return [] so the orchestrator
    # cannot inject entities the filter would have to handle.
    monkeypatch.setattr(orchestrator, "extract_entities", lambda _t, **_kw: [])

    config = _config(tmp_path)
    index = _FakeIndex({"1": _FakeTweet(tweet_id="1", text="just text", screen_name="")})

    run_report(index=index, options=_options(filter_entities=True), config=config)

    # No filterable entities -> no LM call.
    assert filter_calls == []
    assert state["synthesize_calls"]
