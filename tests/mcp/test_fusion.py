"""Unit tests for the Reciprocal Rank Fusion helper.

Covers requirements 6.1, 6.2, 6.4 and absorbs the scope of task 5.4.
"""

from __future__ import annotations

import pytest

from x_likes_mcp.fusion import (
    DEFAULT_FUSED_TOP,
    DEFAULT_K_RRF,
    reciprocal_rank_fusion,
)


class TestConstants:
    def test_default_k_rrf_is_60(self) -> None:
        assert DEFAULT_K_RRF == 60

    def test_default_fused_top_is_300(self) -> None:
        assert DEFAULT_FUSED_TOP == 300


class TestTwoRankingsOverlapping:
    """r1=[a,b,c], r2=[b,a,d] — a and b tie on score; insertion order wins."""

    def test_fused_order_with_overlap(self) -> None:
        # a: 1/61 + 1/62, b: 1/62 + 1/61 (tied), c: 1/63, d: 1/63
        # tie on score → insertion order: a (counter=0) before b (counter=1)
        # c added at counter=2 in r1, d at counter=3 in r2; both score 1/63 (tied)
        # so c before d by insertion order.
        result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
        assert result == ["a", "b", "c", "d"]


class TestTwoRankingsNonOverlapping:
    """r1=[a,b], r2=[c,d] — rank-1 ids tie, rank-2 ids tie."""

    def test_fused_order_non_overlapping(self) -> None:
        # a: 1/61, b: 1/62, c: 1/61, d: 1/62.
        # Sorted desc by score: {a, c} tied at rank 1, {b, d} tied at rank 2.
        # Insertion order: a(0), b(1), c(2), d(3).
        # Final order: a, c (rank-1 tier), b, d (rank-2 tier).
        result = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        assert result == ["a", "c", "b", "d"]


class TestSingleMethodInput:
    """If one ranking is empty, the other ranking's order is preserved."""

    def test_second_ranking_empty(self) -> None:
        result = reciprocal_rank_fusion([["a", "b", "c"], []])
        assert result == ["a", "b", "c"]

    def test_first_ranking_empty(self) -> None:
        result = reciprocal_rank_fusion([[], ["x", "y", "z"]])
        assert result == ["x", "y", "z"]


class TestEmptyInputs:
    def test_both_rankings_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []

    def test_no_rankings_at_all(self) -> None:
        assert reciprocal_rank_fusion([]) == []


class TestTruncation:
    def test_top_truncates_results(self) -> None:
        result = reciprocal_rank_fusion(
            [["a", "b", "c", "d", "e"]],
            top=3,
        )
        assert result == ["a", "b", "c"]

    def test_top_larger_than_corpus_returns_all(self) -> None:
        result = reciprocal_rank_fusion(
            [["a", "b"]],
            top=100,
        )
        assert result == ["a", "b"]


class TestKRrfParameter:
    def test_k_rrf_does_not_break_ordering_for_small_value(self) -> None:
        # With k_rrf=1: rank 1 → 1/2, rank 2 → 1/3, rank 3 → 1/4 (still monotonic).
        result = reciprocal_rank_fusion([["a", "b", "c"]], k_rrf=1)
        assert result == ["a", "b", "c"]

    def test_k_rrf_does_not_break_ordering_for_large_value(self) -> None:
        result = reciprocal_rank_fusion([["a", "b", "c"]], k_rrf=1000)
        assert result == ["a", "b", "c"]

    def test_k_rrf_affects_relative_score_gap(self) -> None:
        # Small k_rrf widens the gap between rank 1 and rank 2.
        # We can't observe scores from the public API, but we can confirm
        # ties resolve identically regardless of k_rrf when the rank-shape
        # is symmetric across rankings.
        result_k60 = reciprocal_rank_fusion(
            [["a"], ["b"]], k_rrf=60
        )
        result_k1 = reciprocal_rank_fusion(
            [["a"], ["b"]], k_rrf=1
        )
        # Both rank-1 in their respective ranking → tied → insertion order wins.
        assert result_k60 == ["a", "b"]
        assert result_k1 == ["a", "b"]


class TestValidation:
    def test_k_rrf_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="k_rrf"):
            reciprocal_rank_fusion([["a"]], k_rrf=0)

    def test_k_rrf_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="k_rrf"):
            reciprocal_rank_fusion([["a"]], k_rrf=-1)

    def test_top_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="top"):
            reciprocal_rank_fusion([["a"]], top=0)

    def test_top_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="top"):
            reciprocal_rank_fusion([["a"]], top=-1)


class TestDeterminism:
    def test_same_inputs_produce_same_output(self) -> None:
        rankings = [["a", "b", "c"], ["b", "a", "d"], ["c", "d", "e"]]
        first = reciprocal_rank_fusion(rankings)
        second = reciprocal_rank_fusion(rankings)
        assert first == second


class TestThreeRankingsComplexOverlap:
    """r1=[a,b,c], r2=[b,c,d], r3=[c,d,e]. Numeric ordering: c > b > d > a > e."""

    def test_three_rankings_complex_order(self) -> None:
        # Hand-computed scores:
        # a: 1/61                  ≈ 0.01639
        # b: 1/62 + 1/61           ≈ 0.01613 + 0.01639 = 0.03252
        # c: 1/63 + 1/62 + 1/61    ≈ 0.01587 + 0.01613 + 0.01639 = 0.04839
        # d: 1/63 + 1/62           ≈ 0.01587 + 0.01613 = 0.03200
        # e: 1/63                  ≈ 0.01587
        # Order (descending): c > b > d > a > e.
        result = reciprocal_rank_fusion(
            [["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"]]
        )
        assert result == ["c", "b", "d", "a", "e"]

    def test_three_rankings_score_math_explicitly(self) -> None:
        # Sanity: confirm the algorithm matches the published RRF formula by
        # computing expected scores by hand and verifying ordering.
        k = 60
        scores = {
            "a": 1.0 / (k + 1),
            "b": 1.0 / (k + 2) + 1.0 / (k + 1),
            "c": 1.0 / (k + 3) + 1.0 / (k + 2) + 1.0 / (k + 1),
            "d": 1.0 / (k + 3) + 1.0 / (k + 2),
            "e": 1.0 / (k + 3),
        }
        expected = sorted(scores.keys(), key=lambda d: -scores[d])
        result = reciprocal_rank_fusion(
            [["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"]],
            k_rrf=k,
        )
        assert result == expected
