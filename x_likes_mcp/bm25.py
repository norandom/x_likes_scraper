"""BM25 lexical-retrieval index for the X Likes MCP server.

This module owns the lexical-retrieval seam. It is the only place in the
package that imports :mod:`rank_bm25`. The dense-retrieval counterpart
lives in :mod:`x_likes_mcp.embeddings`; both modules expose a top-k seam
that the fusion layer (``fusion.py``) combines via Reciprocal Rank
Fusion.

Design notes:

* :func:`tokenize` is a deterministic, dependency-free tokenizer
  (lowercase, split on whitespace, strip leading/trailing non-word
  characters, drop empties). It is used both at index-build time over
  ``Tweet.text`` and at query time over the user's query string so the
  two sides tokenize identically (Requirement 5.2).
* :class:`BM25Index` wraps :class:`rank_bm25.BM25Okapi` plus an
  ``ordered_ids`` list so row index ``N`` of the BM25 matrix maps to
  ``ordered_ids[N]``. Build is O(N) and sub-second at the corpus sizes
  this server handles; the index is rebuilt in-memory at startup and
  never persisted (Requirement 5.5).
* :meth:`BM25Index.top_k` accepts an optional ``restrict_to_ids`` set;
  when provided, non-restricted positions are masked to ``-inf`` before
  the top-k selection so they cannot surface. The mask is what the
  candidate-id-set filter (Requirement 5.4) hooks into.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus_text import tweet_index_text

if TYPE_CHECKING:
    from x_likes_exporter import Tweet


# ---------------------------------------------------------------------------
# Module-level constants
#
# ``DEFAULT_TOP_K`` mirrors :data:`x_likes_mcp.embeddings.DEFAULT_TOP_K`
# (200) so the lexical and dense recall layers feed equally-sized inputs
# to the fusion step. Keeping it duplicated rather than re-exported
# avoids a cross-module import in the lexical path that has no other
# reason to depend on the dense path.

DEFAULT_TOP_K: int = 200


__all__ = [
    "DEFAULT_TOP_K",
    "BM25Index",
    "tokenize",
]


# ---------------------------------------------------------------------------
# Tokenizer

_WS_RE = re.compile(r"\s+")
_EDGE_NON_WORD_RE = re.compile(r"^\W+|\W+$")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip non-word edges, drop empties.

    Algorithm:

    1. Lowercase the entire string first so case is irrelevant for any
       downstream comparison.
    2. Split on runs of whitespace (``re.split(r"\\s+")``).
    3. For each token, strip leading/trailing non-word characters
       (``re.sub(r"^\\W+|\\W+$", "", tok)``). Internal punctuation is
       preserved — e.g. ``"pentest@AI"`` becomes ``["pentest@ai"]``.
    4. Drop empty tokens (whitespace-only inputs, all-punctuation
       tokens like ``"..."``, leading/trailing splits).

    The same function is used at index-build time over each tweet's text
    and at query time over the search string so the two sides tokenize
    identically (Requirement 5.2).
    """

    if not text:
        return []

    raw_tokens = _WS_RE.split(text.lower())
    cleaned: list[str] = []
    for tok in raw_tokens:
        stripped = _EDGE_NON_WORD_RE.sub("", tok)
        if stripped:
            cleaned.append(stripped)
    return cleaned


# ---------------------------------------------------------------------------
# Index


@dataclass
class BM25Index:
    """In-memory BM25 index over a fixed corpus of tweets.

    ``bm25`` is the underlying :class:`rank_bm25.BM25Okapi` instance, or
    ``None`` for the empty-corpus sentinel (rank_bm25 cannot be
    constructed over a zero-document corpus). ``ordered_ids[N]`` is the
    tweet id whose tokenized text occupies row ``N`` of the BM25 matrix.
    """

    bm25: BM25Okapi | None
    ordered_ids: list[str]

    @classmethod
    def build(cls, tweets_by_id: dict[str, "Tweet"]) -> "BM25Index":
        """Build a BM25 index from a tweet-id -> Tweet mapping.

        The id list is sorted lexicographically so two builds over the
        same input produce byte-identical indices (helpful for tests
        and for any future caching layer). ``Tweet.text`` may be
        ``None``; we coerce to an empty string before tokenizing, which
        results in an empty token list and a row that scores 0 on every
        query.

        For an empty input we return a sentinel index with
        ``bm25=None`` and ``ordered_ids=[]``; :meth:`top_k` short-
        circuits on this case.
        """

        ordered_ids = sorted(tweets_by_id.keys())
        if not ordered_ids:
            return cls(bm25=None, ordered_ids=[])

        tokenized = [
            tokenize(tweet_index_text(tweets_by_id[tid])) for tid in ordered_ids
        ]

        # ``BM25Okapi`` divides by the average document length; if every
        # document tokenizes to ``[]`` the average is 0 and the
        # constructor raises ``ZeroDivisionError``. In that case we fall
        # back to the empty-corpus sentinel: any query will tokenize and
        # find no matches anyway.
        if all(not toks for toks in tokenized):
            return cls(bm25=None, ordered_ids=ordered_ids)

        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, ordered_ids=ordered_ids)

    def top_k(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        restrict_to_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(tweet_id, bm25_score)`` pairs, descending.

        Algorithm:

        1. Validate ``k`` (must be >= 1; ``ValueError`` otherwise).
        2. Tokenize the query; if it yields no tokens, return ``[]``.
        3. Short-circuit on the empty-corpus sentinel.
        4. Short-circuit on ``restrict_to_ids == set()`` (the literal
           empty set means "no candidates allowed", distinct from
           ``None`` which means "no restriction").
        5. Compute scores via :meth:`BM25Okapi.get_scores`; mask out
           non-restricted positions to ``-inf`` if ``restrict_to_ids``
           is given. Masking with ``-inf`` ensures masked rows can
           never appear in the top-k regardless of how rank_bm25
           scores them.
        6. Take the top-k indices by descending score, drop any
           ``-inf`` survivors (the "fewer than k candidates after
           masking" case), and return ``(ordered_ids[i],
           float(scores[i]))`` in descending score order.

        When the restricted scope is smaller than ``k`` we return only
        the surviving candidates, never padded to ``k`` (Requirement
        5.3 / 5.4).
        """

        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")

        tokens = tokenize(query)
        if not tokens:
            return []

        if not self.ordered_ids or self.bm25 is None:
            return []

        # ``restrict_to_ids=set()`` is distinct from ``None``: the user
        # asked for "no candidates allowed", which always returns [].
        if restrict_to_ids is not None and not restrict_to_ids:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)

        if restrict_to_ids is not None:
            masked = np.full_like(scores, -np.inf)
            for i, tid in enumerate(self.ordered_ids):
                if tid in restrict_to_ids:
                    masked[i] = scores[i]
            scores = masked

            # If no corpus id overlapped with the restrict set, every
            # entry is -inf -> return [] without touching argpartition.
            if not np.any(np.isfinite(scores)):
                return []

        # Take the top-k indices by descending score. ``argpartition``
        # is O(N), then we sort just the top-k slice.
        n = scores.shape[0]
        take = min(k, n)
        if take == n:
            top_idx = np.argsort(-scores, kind="stable")
        else:
            # ``argpartition`` puts the top-``take`` indices in the
            # first ``take`` positions (in arbitrary order); sort that
            # slice descending.
            partition = np.argpartition(-scores, take - 1)[:take]
            order = np.argsort(-scores[partition], kind="stable")
            top_idx = partition[order]

        results: list[tuple[str, float]] = []
        for i in top_idx:
            score = scores[i]
            if not np.isfinite(score):
                # Drop masked-out survivors (when the restrict set is
                # smaller than ``k``).
                continue
            results.append((self.ordered_ids[int(i)], float(score)))

        return results
