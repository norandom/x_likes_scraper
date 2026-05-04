"""Reciprocal Rank Fusion (RRF) over N ranked id-lists.

Pure-python, stdlib-only. Used by the hybrid recall layer to combine
the BM25 and dense rankings into a single fused id-list before the
heavy ranker runs.

The algorithm follows the published RRF formulation:

    score(d) = sum over rankings r_i of  1 / (k_rrf + rank_i(d))

where rank_i(d) is the 1-indexed position of `d` in ranking r_i.
Ids absent from a ranking contribute nothing for that ranking.

Tie-breaking is deterministic: ids that scored equal are ordered by
the position at which they were first inserted across all rankings,
then by lexicographic id as a final fallback for synthetic data.
"""

from __future__ import annotations

DEFAULT_K_RRF: int = 60
DEFAULT_FUSED_TOP: int = 300


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k_rrf: int = DEFAULT_K_RRF,
    top: int = DEFAULT_FUSED_TOP,
) -> list[str]:
    """Fuse N ranked id-lists into a single descending id-list of size <= top.

    For each ranking r_i (1-indexed) and each id d in r_i,
    `fused_score[d] += 1.0 / (k_rrf + rank_i(d))`. Ids absent from a
    given ranking contribute nothing for that ranking.

    Tie-breaking on equal fused score:
        1. Insertion order across the rankings (the id that first
           appeared in any ranking earlier wins).
        2. Lexicographic id as a final fallback.

    Empty rankings (`rankings[i] == []`) are silently ignored. If
    every ranking is empty, returns `[]`.

    Args:
        rankings: list of ranked id-lists. Each id is a string; rank 1
            is the first id, rank 2 is the second, and so on.
        k_rrf: the RRF dampening constant. Must be positive. Larger
            values flatten the contribution gap between rank positions.
        top: maximum number of ids to return. Must be positive.

    Returns:
        Fused id-list in descending fused-score order, length <= top.

    Raises:
        ValueError: if `k_rrf <= 0` or `top <= 0`.
    """
    if k_rrf <= 0:
        raise ValueError(f"k_rrf must be positive, got {k_rrf}")
    if top <= 0:
        raise ValueError(f"top must be positive, got {top}")

    fused_score: dict[str, float] = {}
    insertion_order: dict[str, int] = {}
    counter = 0
    for ranking in rankings:
        for rank_index, doc_id in enumerate(ranking, start=1):
            fused_score[doc_id] = fused_score.get(doc_id, 0.0) + 1.0 / (k_rrf + rank_index)
            if doc_id not in insertion_order:
                insertion_order[doc_id] = counter
                counter += 1

    items = list(fused_score.items())
    # Sort: descending fused_score, then ascending insertion_order, then lex id.
    items.sort(key=lambda kv: (-kv[1], insertion_order[kv[0]], kv[0]))

    return [doc_id for doc_id, _ in items[:top]]
