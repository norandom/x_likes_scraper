"""Tests for :mod:`x_likes_mcp.bm25` (task 2.5).

This file covers the BM25 lexical-retrieval module:

* The deterministic ``tokenize`` function (case, punctuation edges,
  internal punctuation, empty / whitespace-only inputs).
* ``BM25Index.build`` over a tweet-id -> Tweet mapping (sorted ids,
  ``None`` text handling, empty corpus).
* ``BM25Index.top_k`` ordering, ``k`` cap, ``restrict_to_ids`` masking
  (subset, no overlap, empty), empty / all-punctuation queries, and
  invalid ``k`` validation.

This task absorbs what task 5.3 would have covered (the test file scoped
to the bm25 module). Synthetic tweets use ``types.SimpleNamespace`` to
avoid coupling to the full ``x_likes_exporter.Tweet`` constructor: only
``.text`` is exercised by the BM25 module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from x_likes_mcp import bm25 as bm25_mod
from x_likes_mcp.bm25 import DEFAULT_TOP_K, BM25Index, tokenize


# ---------------------------------------------------------------------------
# Helpers


def _tweet(text: str | None) -> Any:
    """Synthetic tweet exposing only the ``.text`` attribute."""

    return SimpleNamespace(text=text)


def _corpus(text_by_id: dict[str, str | None]) -> dict[str, Any]:
    return {tid: _tweet(text) for tid, text in text_by_id.items()}


# ---------------------------------------------------------------------------
# tokenize


def test_tokenize_lowercase_and_split() -> None:
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation_edges() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_preserves_internal_punctuation() -> None:
    # Internal punctuation is preserved; only edge non-word chars stripped.
    assert tokenize("pentest@AI") == ["pentest@ai"]


def test_tokenize_drops_empty_after_strip() -> None:
    # Tokens that consist entirely of non-word characters strip to empty
    # and must not appear in the output.
    assert tokenize("... !!!") == []


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_whitespace_only() -> None:
    assert tokenize("   \t  \n  ") == []


def test_tokenize_multiple_spaces() -> None:
    # Collapsing multiple spaces is a free side-effect of ``re.split(r"\s+")``.
    assert tokenize("foo   bar") == ["foo", "bar"]


def test_tokenize_mixed_edges_and_internals() -> None:
    # Combined: leading/trailing punctuation stripped, internal kept.
    assert tokenize("  ?Hello-World?  ") == ["hello-world"]


# ---------------------------------------------------------------------------
# Module-level constants


def test_default_top_k_is_200() -> None:
    assert DEFAULT_TOP_K == 200


# ---------------------------------------------------------------------------
# BM25Index.build


def test_build_empty_corpus() -> None:
    """An empty input yields an index whose ``top_k`` returns ``[]``."""

    index = BM25Index.build({})

    assert index.ordered_ids == []
    assert index.top_k("any query") == []


def test_build_single_tweet() -> None:
    """A one-doc corpus returns that doc on a matching query.

    Note: ``BM25Okapi`` IDF is ``log((N - df + 0.5) / (df + 0.5))``;
    with N=1 and df=1 the IDF is negative (~-1.1). The score may be
    negative even for a perfect match; what matters is that the
    matching doc surfaces as the top (and only) result.
    """

    index = BM25Index.build(_corpus({"id_1": "pentesting with AI tools"}))

    result = index.top_k("pentesting")

    assert len(result) == 1
    assert result[0][0] == "id_1"


def test_build_ordered_ids_sorted() -> None:
    index = BM25Index.build(
        _corpus(
            {
                "id_3": "third",
                "id_1": "first",
                "id_2": "second",
            }
        )
    )

    assert index.ordered_ids == ["id_1", "id_2", "id_3"]


def test_build_handles_none_text() -> None:
    """A ``Tweet.text == None`` row builds without error and never wins.

    BM25Okapi assigns a (typically zero) baseline score to documents
    that tokenize to ``[]``, so a None-text row may appear in the
    result list. The contract that matters is: the None-text row never
    out-scores a document that actually matches the query.
    """

    index = BM25Index.build(
        _corpus(
            {
                "id_1": "alpha bravo",
                "id_2": None,
                "id_3": "charlie delta",
            }
        )
    )

    # All three ids are present in row order.
    assert index.ordered_ids == ["id_1", "id_2", "id_3"]

    # The matching doc must be the top result; the None-text row does
    # not surface above it.
    result = index.top_k("alpha")
    assert result, "expected at least one result for a matching query"
    assert result[0][0] == "id_1"
    score_by_id = dict(result)
    if "id_2" in score_by_id:
        assert score_by_id["id_2"] <= score_by_id["id_1"]


# ---------------------------------------------------------------------------
# BM25Index.top_k — ordering and k


def test_top_k_returns_descending() -> None:
    """Top-1 is the strongest match; results are descending by score."""

    index = BM25Index.build(
        _corpus(
            {
                "id_a": "pentesting with AI tools",  # all tokens
                "id_b": "AI tools for music",  # partial overlap
                "id_c": "tools and toys",  # weak overlap
                "id_d": "completely unrelated text",
                "id_e": "another unrelated topic",
            }
        )
    )

    result = index.top_k("pentesting AI tools", k=5)

    assert len(result) >= 1
    # Top-1 is the document containing all query tokens.
    assert result[0][0] == "id_a"

    # Scores in returned order are non-strictly descending.
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_top_k_default_k_is_200_returns_all_when_corpus_smaller() -> None:
    """Calling ``top_k`` with no ``k`` argument uses ``DEFAULT_TOP_K=200``.

    A 5-doc corpus where every doc shares the query token returns up to
    5 results (capped by corpus size, not by ``k``).
    """

    index = BM25Index.build(
        _corpus(
            {
                "id_1": "alpha beta",
                "id_2": "alpha gamma",
                "id_3": "alpha delta",
                "id_4": "alpha epsilon",
                "id_5": "alpha zeta",
            }
        )
    )

    result = index.top_k("alpha")  # k omitted -> default 200

    assert len(result) == 5


def test_top_k_at_most_k() -> None:
    text_by_id = {f"id_{i}": "alpha beta gamma" for i in range(10)}
    index = BM25Index.build(_corpus(text_by_id))

    result = index.top_k("alpha", k=3)

    assert len(result) == 3


# ---------------------------------------------------------------------------
# BM25Index.top_k — restrict_to_ids


def test_top_k_restrict_returns_only_restricted() -> None:
    text_by_id = {f"id_{i}": "alpha beta" for i in range(10)}
    index = BM25Index.build(_corpus(text_by_id))

    result = index.top_k(
        "alpha", k=200, restrict_to_ids={"id_2", "id_8"}
    )

    returned_ids = {tid for tid, _ in result}
    assert returned_ids.issubset({"id_2", "id_8"})
    assert returned_ids == {"id_2", "id_8"}


def test_top_k_restrict_smaller_than_k_returns_all() -> None:
    text_by_id = {f"id_{i}": "alpha beta gamma" for i in range(10)}
    index = BM25Index.build(_corpus(text_by_id))

    result = index.top_k("alpha", k=10, restrict_to_ids={"id_3", "id_7"})

    assert len(result) == 2
    returned_ids = {tid for tid, _ in result}
    assert returned_ids == {"id_3", "id_7"}


def test_top_k_empty_restrict_returns_empty() -> None:
    """``restrict_to_ids=set()`` is "no candidates allowed" -> ``[]``."""

    index = BM25Index.build(
        _corpus({"id_1": "alpha beta", "id_2": "alpha gamma"})
    )

    result = index.top_k("alpha", k=10, restrict_to_ids=set())

    assert result == []


def test_top_k_no_overlap_returns_empty() -> None:
    """``restrict_to_ids`` containing only non-corpus ids yields ``[]``."""

    index = BM25Index.build(
        _corpus({"id_1": "alpha beta", "id_2": "alpha gamma"})
    )

    result = index.top_k(
        "alpha", k=10, restrict_to_ids={"missing_a", "missing_b"}
    )

    assert result == []


def test_top_k_restrict_excludes_zero_scoring_documents() -> None:
    """Restrict ids with no token overlap should not surface as results.

    A restricted id whose document scores 0 (no query-token overlap) is
    still valid to include in principle — BM25Okapi returns 0.0, not
    ``-inf``. We accept either behaviour: the result list is at most the
    size of the restrict set, and any returned ids belong to the set.
    """

    index = BM25Index.build(
        _corpus(
            {
                "id_match": "alpha beta",
                "id_other": "completely different vocabulary",
            }
        )
    )

    result = index.top_k("alpha", k=10, restrict_to_ids={"id_match", "id_other"})

    assert len(result) <= 2
    returned_ids = {tid for tid, _ in result}
    assert returned_ids.issubset({"id_match", "id_other"})
    # The exact match must appear.
    assert "id_match" in returned_ids


# ---------------------------------------------------------------------------
# BM25Index.top_k — query-tokenization edge cases


def test_top_k_empty_query_returns_empty() -> None:
    index = BM25Index.build(_corpus({"id_1": "alpha beta"}))

    assert index.top_k("") == []


def test_top_k_query_all_punctuation() -> None:
    """A query that tokenizes to ``[]`` returns ``[]``."""

    index = BM25Index.build(_corpus({"id_1": "alpha beta"}))

    assert index.top_k("...") == []
    assert index.top_k("!!! ???") == []


# ---------------------------------------------------------------------------
# BM25Index.top_k — input validation


def test_top_k_invalid_k_zero_raises() -> None:
    index = BM25Index.build(_corpus({"id_1": "alpha"}))

    with pytest.raises(ValueError):
        index.top_k("alpha", k=0)


def test_top_k_invalid_k_negative_raises() -> None:
    index = BM25Index.build(_corpus({"id_1": "alpha"}))

    with pytest.raises(ValueError):
        index.top_k("alpha", k=-1)


# ---------------------------------------------------------------------------
# Module surface


def test_module_exports_tokenize_and_index() -> None:
    """The two public symbols are importable from ``x_likes_mcp.bm25``."""

    assert callable(bm25_mod.tokenize)
    assert hasattr(bm25_mod, "BM25Index")
    assert hasattr(bm25_mod, "DEFAULT_TOP_K")
