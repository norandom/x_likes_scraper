"""Tests for the round-2 entity fan-out (synthesis-report task 4.3).

Pin the multihop service interface:

- ``validate_hops`` accepts the in-range values (1, 2) and rejects
  anything outside the documented range with a
  :class:`HopsOutOfRange` error that names the offending value and the
  allowed maximum.
- ``run_round_one`` calls ``index.search`` with the query and the same
  ``year`` / ``month_start`` / ``month_end`` filters from the
  :class:`ReportOptions`, and returns the search result unchanged.
- ``run_round_two`` skips entirely when ``options.hops == 1`` (no
  searches issued); otherwise it picks the top-K boosted entities from
  the round-1 KG, runs K parallel ``index.search`` calls (the same
  filters flow through), and returns the concatenated hits.
- Entity values are normalized for the search query: handles drop the
  ``handle:`` prefix, hashtags drop ``hashtag:``, domains keep their
  literal value, and concept slugs replace underscores with spaces.
- ``fuse_results`` dedupes by ``tweet_id``, preserves round-1 ordering,
  and round-1 wins for shared IDs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from x_likes_mcp.ranker import ScoredHit
from x_likes_mcp.synthesis.kg import KG, Node, NodeKind
from x_likes_mcp.synthesis.multihop import (
    DEFAULT_MAX_HOPS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_ROUND_TWO_K,
    ENTITY_BOOSTS,
    HopsOutOfRange,
    fuse_results,
    run_round_one,
    run_round_two,
    validate_hops,
)
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import ReportOptions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(tweet_id: str, score: float = 0.5) -> ScoredHit:
    """Build a minimal ScoredHit sentinel for fusion / search returns."""

    return ScoredHit(
        tweet_id=tweet_id,
        score=score,
        walker_relevance=0.0,
        why="",
        feature_breakdown={},
    )


class FakeIndex:
    """Stub TweetIndex whose ``search`` records call args and returns canned hits.

    ``canned_by_query`` maps a query string to the list of hits the next
    call with that query should return. Unknown queries return ``[]``.
    The optional ``sleep_seconds`` lets the parallel-execution test
    gauge wall-clock concurrency.
    """

    def __init__(
        self,
        canned_by_query: dict[str, list[ScoredHit]] | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.canned_by_query = canned_by_query or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sleep_seconds = sleep_seconds
        self._lock = threading.Lock()

    def search(
        self,
        query: str,
        year: int | None = None,
        month_start: str | None = None,
        month_end: str | None = None,
        top_n: int = 50,
    ) -> list[ScoredHit]:
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        with self._lock:
            self.calls.append(
                (
                    query,
                    {
                        "year": year,
                        "month_start": month_start,
                        "month_end": month_end,
                        "top_n": top_n,
                    },
                )
            )
        return list(self.canned_by_query.get(query, []))


def _options(
    *,
    query: str = "ai security",
    hops: int = 1,
    year: int | None = None,
    month_start: str | None = None,
    month_end: str | None = None,
    limit: int = 50,
) -> ReportOptions:
    return ReportOptions(
        query=query,
        shape=ReportShape.BRIEF,
        hops=hops,
        year=year,
        month_start=month_start,
        month_end=month_end,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# validate_hops
# ---------------------------------------------------------------------------


class TestValidateHops:
    def test_accepts_1(self) -> None:
        # No-op: returns None and does not raise.
        assert validate_hops(1) is None

    def test_accepts_2(self) -> None:
        assert validate_hops(2) is None

    def test_rejects_3(self) -> None:
        with pytest.raises(HopsOutOfRange) as excinfo:
            validate_hops(3)
        msg = str(excinfo.value)
        assert "3" in msg
        assert str(DEFAULT_MAX_HOPS) in msg

    def test_rejects_0(self) -> None:
        with pytest.raises(HopsOutOfRange):
            validate_hops(0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(HopsOutOfRange):
            validate_hops(-1)

    def test_hops_out_of_range_is_value_error(self) -> None:
        # Documented base class so callers can catch via the stdlib type.
        assert issubclass(HopsOutOfRange, ValueError)

    def test_max_hops_override_widens_range(self) -> None:
        # Test-specific override must be respected.
        assert validate_hops(3, max_hops=3) is None


# ---------------------------------------------------------------------------
# run_round_one
# ---------------------------------------------------------------------------


class TestRunRoundOne:
    def test_calls_index_search_with_query_and_filters(self) -> None:
        index = FakeIndex()
        options = _options(
            query="ai security",
            year=2025,
            month_start="01",
            month_end="03",
        )

        run_round_one(index, options)  # type: ignore[arg-type]

        assert len(index.calls) == 1
        query, kwargs = index.calls[0]
        assert query == "ai security"
        assert kwargs["year"] == 2025
        assert kwargs["month_start"] == "01"
        assert kwargs["month_end"] == "03"

    def test_returns_search_result_unchanged(self) -> None:
        hit_a = _hit("1", score=0.9)
        hit_b = _hit("2", score=0.8)
        index = FakeIndex(canned_by_query={"ai security": [hit_a, hit_b]})
        options = _options(query="ai security")

        result = run_round_one(index, options)  # type: ignore[arg-type]

        assert result == [hit_a, hit_b]

    def test_passes_no_filters_when_options_have_none(self) -> None:
        index = FakeIndex()
        options = _options(query="ai")

        run_round_one(index, options)  # type: ignore[arg-type]

        _, kwargs = index.calls[0]
        assert kwargs["year"] is None
        assert kwargs["month_start"] is None
        assert kwargs["month_end"] is None


# ---------------------------------------------------------------------------
# run_round_two
# ---------------------------------------------------------------------------


class TestRunRoundTwo:
    def test_returns_empty_when_hops_is_1(self) -> None:
        # Even when the KG is populated, hops=1 means no fan-out.
        kg = KG()
        kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label="foo", weight=2.0))
        index = FakeIndex()
        options = _options(hops=1)

        result = run_round_two(index, options, kg)  # type: ignore[arg-type]

        assert result == []
        assert index.calls == []

    def test_picks_top_k_entities(self) -> None:
        kg = KG()
        # 10 handles with descending weight; 5 hashtags with high weight that
        # get scaled down by the 0.7 boost. With K=5 the top-5 boosted
        # entities should all be handles.
        for i in range(10):
            kg.add_node(
                Node(
                    id=f"handle:h{i}",
                    kind=NodeKind.HANDLE,
                    label=f"h{i}",
                    weight=10.0 - i,  # 10..1
                )
            )
        for i in range(5):
            kg.add_node(
                Node(
                    id=f"hashtag:tag{i}",
                    kind=NodeKind.HASHTAG,
                    label=f"tag{i}",
                    weight=5.0 - i,  # 5..1; * 0.7 = 3.5..0.7
                )
            )
        index = FakeIndex()
        options = _options(hops=2)

        run_round_two(index, options, kg)  # type: ignore[arg-type]

        assert len(index.calls) == DEFAULT_ROUND_TWO_K  # K = 5 by default
        queries = {call[0] for call in index.calls}
        # Boosts make handles win across the board.
        assert queries == {"h0", "h1", "h2", "h3", "h4"}

    def test_uses_entity_value_as_query(self) -> None:
        kg = KG()
        kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label="foo", weight=10.0))
        kg.add_node(
            Node(
                id="concept:ai_pentesting",
                kind=NodeKind.CONCEPT,
                label="ai pentesting",
                weight=10.0,
            )
        )
        kg.add_node(
            Node(
                id="hashtag:bar",
                kind=NodeKind.HASHTAG,
                label="bar",
                weight=10.0,
            )
        )
        kg.add_node(
            Node(
                id="domain:example.com",
                kind=NodeKind.DOMAIN,
                label="example.com",
                weight=10.0,
            )
        )
        index = FakeIndex()
        options = _options(hops=2)

        run_round_two(index, options, kg, k=4)  # type: ignore[arg-type]

        queries = {call[0] for call in index.calls}
        assert "foo" in queries
        # Snake_case concept slugs decode back to spaced phrases.
        assert "ai pentesting" in queries
        assert "bar" in queries
        # Domains keep their literal value.
        assert "example.com" in queries

    def test_passes_same_filters_as_round_one(self) -> None:
        kg = KG()
        for i in range(3):
            kg.add_node(
                Node(
                    id=f"handle:h{i}",
                    kind=NodeKind.HANDLE,
                    label=f"h{i}",
                    weight=10.0 - i,
                )
            )
        index = FakeIndex()
        options = _options(
            hops=2,
            year=2025,
            month_start="01",
            month_end="06",
        )

        run_round_two(index, options, kg, k=3)  # type: ignore[arg-type]

        assert len(index.calls) == 3
        for _query, kwargs in index.calls:
            assert kwargs["year"] == 2025
            assert kwargs["month_start"] == "01"
            assert kwargs["month_end"] == "06"

    def test_runs_in_parallel(self) -> None:
        kg = KG()
        for i in range(5):
            kg.add_node(
                Node(
                    id=f"handle:h{i}",
                    kind=NodeKind.HANDLE,
                    label=f"h{i}",
                    weight=10.0 - i,
                )
            )
        # Each search sleeps 0.1s. Serial would be 0.5s; parallel with
        # max_workers >= 5 should finish well under 0.3s on any modern CI.
        index = FakeIndex(sleep_seconds=0.1)
        options = _options(hops=2)

        start = time.perf_counter()
        run_round_two(
            index,
            options,
            kg,
            k=5,
            max_workers=5,
        )  # type: ignore[arg-type]
        elapsed = time.perf_counter() - start

        assert len(index.calls) == 5
        assert elapsed < 0.3, f"expected concurrent execution; took {elapsed:.3f}s"

    def test_empty_kg_returns_empty(self) -> None:
        index = FakeIndex()
        options = _options(hops=2)

        result = run_round_two(index, options, KG())  # type: ignore[arg-type]

        assert result == []
        assert index.calls == []

    def test_concatenates_hits_across_searches(self) -> None:
        kg = KG()
        kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label="foo", weight=2.0))
        kg.add_node(Node(id="handle:bar", kind=NodeKind.HANDLE, label="bar", weight=1.0))
        canned = {
            "foo": [_hit("10"), _hit("11")],
            "bar": [_hit("20")],
        }
        index = FakeIndex(canned_by_query=canned)
        options = _options(hops=2)

        result = run_round_two(index, options, kg, k=2)  # type: ignore[arg-type]

        ids = sorted(hit.tweet_id for hit in result)
        assert ids == ["10", "11", "20"]

    def test_hops_2_with_default_k_issues_k_searches(self) -> None:
        # Pins Req 2.2 wording: "exactly K parallel searches" by default.
        kg = KG()
        for i in range(DEFAULT_ROUND_TWO_K + 3):
            kg.add_node(
                Node(
                    id=f"handle:h{i}",
                    kind=NodeKind.HANDLE,
                    label=f"h{i}",
                    weight=10.0 - i,
                )
            )
        index = FakeIndex()
        options = _options(hops=2)

        run_round_two(index, options, kg)  # type: ignore[arg-type]

        assert len(index.calls) == DEFAULT_ROUND_TWO_K


# ---------------------------------------------------------------------------
# fuse_results
# ---------------------------------------------------------------------------


class TestFuseResults:
    def test_dedupes_by_tweet_id(self) -> None:
        round_one = [_hit("1"), _hit("2")]
        round_two = [_hit("2"), _hit("3")]

        fused = fuse_results(round_one, round_two)

        assert [h.tweet_id for h in fused] == ["1", "2", "3"]

    def test_preserves_round_one_order(self) -> None:
        round_one = [_hit("b"), _hit("a")]
        round_two = [_hit("a"), _hit("c")]

        fused = fuse_results(round_one, round_two)

        assert [h.tweet_id for h in fused] == ["b", "a", "c"]

    def test_round_one_wins_for_shared_id(self) -> None:
        r1 = _hit("1", score=0.9)
        r2 = replace(_hit("1", score=0.5), why="round-two")
        fused = fuse_results([r1], [r2])

        assert len(fused) == 1
        assert fused[0].score == 0.9
        assert fused[0].why == ""  # round-1 wins, round-2 metadata ignored.

    def test_empty_round_one_returns_round_two_in_order(self) -> None:
        round_two = [_hit("3"), _hit("4")]

        fused = fuse_results([], round_two)

        assert [h.tweet_id for h in fused] == ["3", "4"]

    def test_empty_round_two_returns_round_one_unchanged(self) -> None:
        round_one = [_hit("1"), _hit("2")]

        fused = fuse_results(round_one, [])

        assert fused == round_one

    def test_round_two_internal_dupes_collapse(self) -> None:
        round_one: list[ScoredHit] = []
        round_two = [_hit("1"), _hit("1"), _hit("2")]

        fused = fuse_results(round_one, round_two)

        assert [h.tweet_id for h in fused] == ["1", "2"]


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_constants_match_design(self) -> None:
        assert DEFAULT_ROUND_TWO_K == 5
        assert DEFAULT_MAX_WORKERS == 4
        assert DEFAULT_MAX_HOPS == 2

    def test_entity_boosts_match_design(self) -> None:
        assert ENTITY_BOOSTS[NodeKind.HANDLE] == 1.0
        assert ENTITY_BOOSTS[NodeKind.HASHTAG] == 0.7
        assert ENTITY_BOOSTS[NodeKind.DOMAIN] == 0.5
        assert ENTITY_BOOSTS[NodeKind.CONCEPT] == 0.3
