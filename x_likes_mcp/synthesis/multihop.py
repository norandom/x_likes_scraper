"""Round-2 entity fan-out for the synthesis-report feature.

Implements the multihop search expansion documented in the
``multihop`` design.md component:

- ``run_round_one`` issues the user's query against the existing
  :class:`~x_likes_mcp.index.TweetIndex` search seam, propagating the
  date filters from :class:`~x_likes_mcp.synthesis.types.ReportOptions`
  unchanged.
- ``run_round_two`` skips entirely when ``options.hops < 2`` (Req 2.1).
  Otherwise it picks the top-K boosted entities from the round-1
  knowledge graph, decodes each entity's ID back into a search-friendly
  query string, and runs K parallel ``index.search`` calls under a
  :class:`concurrent.futures.ThreadPoolExecutor` (Req 2.2). The same
  ``year`` / ``month_start`` / ``month_end`` filters from round-1 flow
  into every round-2 call (Req 2.3).
- ``fuse_results`` dedupes hits by ``tweet_id``: round-1 ordering wins
  for shared ids and the round-1 hit object is preserved verbatim;
  round-2-only hits are appended at the end in their original order
  (Req 1.4).
- ``validate_hops`` is the single ``hops`` gate the CLI / MCP boundary
  consults before any search call. It rejects values outside
  ``1..DEFAULT_MAX_HOPS`` (Req 2.4) by raising :class:`HopsOutOfRange`,
  a :class:`ValueError` subclass so callers can ``except ValueError``
  without importing this module.

The boundary is intentionally narrow: this module imports the
:class:`~x_likes_mcp.synthesis.kg.KG` graph for entity ranking, the
:class:`~x_likes_mcp.synthesis.types.ReportOptions` dataclass for
inputs, and only the type-checking surface of ``TweetIndex`` /
``ScoredHit``. Composition into the orchestrator (task 5.x) wires
these primitives together; nothing in this file builds a KG or
constructs an index.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from .kg import KG, Node, NodeKind

if TYPE_CHECKING:  # pragma: no cover - import-time-only types
    from x_likes_mcp.index import TweetIndex
    from x_likes_mcp.ranker import ScoredHit

    from .types import ReportOptions


__all__ = [
    "DEFAULT_MAX_HOPS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_ROUND_TWO_K",
    "ENTITY_BOOSTS",
    "HopsOutOfRange",
    "fuse_results",
    "run_round_one",
    "run_round_two",
    "validate_hops",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_ROUND_TWO_K: int = 5
"""Default number of entities the round-2 fan-out picks (operator override:
``SYNTHESIS_ROUND_TWO_K``; the override is wired in the orchestrator boundary,
not here)."""


DEFAULT_MAX_WORKERS: int = 4
"""Default ``ThreadPoolExecutor`` cap for parallel round-2 search calls.
Mirrors the existing pattern in :class:`~x_likes_mcp.index.TweetIndex`."""


DEFAULT_MAX_HOPS: int = 2
"""Maximum supported ``hops`` value. v1 only fans out one extra hop."""


ENTITY_BOOSTS: dict[NodeKind, float] = {
    NodeKind.HANDLE: 1.0,
    NodeKind.HASHTAG: 0.7,
    NodeKind.DOMAIN: 0.5,
    NodeKind.CONCEPT: 0.3,
}
"""Multiplicative weight applied to each entity kind when ranking the
round-2 candidates. ``HANDLE`` wins ties so author-centric expansion
takes precedence over softer signals."""


# Node kinds eligible for round-2 fan-out (queries / tweets are intrinsic
# to the graph, not search seeds).
_ENTITY_KINDS: tuple[NodeKind, ...] = (
    NodeKind.HANDLE,
    NodeKind.HASHTAG,
    NodeKind.DOMAIN,
    NodeKind.CONCEPT,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HopsOutOfRange(ValueError):
    """Raised when ``options.hops`` is outside the supported range.

    Subclassing :class:`ValueError` lets boundary code catch the stdlib
    type without importing the synthesis package; the message names the
    offending value and the maximum so the CLI / MCP error path can
    surface it verbatim.
    """


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def validate_hops(hops: int, *, max_hops: int = DEFAULT_MAX_HOPS) -> None:
    """Reject hop counts outside ``1..max_hops`` inclusive.

    The CLI / MCP boundary calls this *before* any search seam runs, so
    a bad ``hops`` value never costs a network round-trip. The error
    message names the offending value and the supported range so the
    caller can echo it back to the operator.
    """

    if hops < 1 or hops > max_hops:
        raise HopsOutOfRange(
            f"hops={hops} is out of range; must be between 1 and {max_hops}",
        )


def run_round_one(
    index: TweetIndex,
    options: ReportOptions,
) -> list[ScoredHit]:
    """Run the round-1 query against the existing search seam.

    The same date filters that gate the round-1 corpus also gate every
    round-2 call (Req 2.3); they are read straight off ``options`` and
    forwarded to :meth:`TweetIndex.search` unchanged. ``options.limit``
    is propagated as ``top_n`` so a small report cap does not pull a
    full default page.
    """

    return list(
        index.search(
            options.query,
            year=options.year,
            month_start=options.month_start,
            month_end=options.month_end,
            top_n=options.limit,
        )
    )


def run_round_two(
    index: TweetIndex,
    options: ReportOptions,
    round_one_kg: KG,
    *,
    k: int = DEFAULT_ROUND_TWO_K,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[ScoredHit]:
    """Fan out to K parallel round-2 searches keyed on the top-K entities.

    Behavior:

    - Returns ``[]`` immediately when ``options.hops < 2``: callers do
      not need to special-case the hops==1 branch (Req 2.1).
    - Picks at most ``k`` entities from ``round_one_kg`` across the four
      entity kinds, ranking by ``node.weight * ENTITY_BOOSTS[kind]``;
      ties are broken by ``id`` ascending so the same round-1 KG always
      yields the same round-2 fan-out.
    - Submits one ``index.search`` job per entity through a stdlib
      :class:`ThreadPoolExecutor` (``max_workers``), forwarding the
      identical ``year`` / ``month_start`` / ``month_end`` filters from
      ``options`` (Req 2.3).
    - Concatenates per-entity hit lists in the same order the entities
      were ranked. Cross-entity dedupe happens later in
      :func:`fuse_results`.
    """

    if options.hops < 2:
        return []

    entities = _pick_top_entities(round_one_kg, k)
    if not entities:
        return []

    queries = [_query_for_entity(node) for node in entities]
    filter_kwargs: dict[str, object] = {
        "year": options.year,
        "month_start": options.month_start,
        "month_end": options.month_end,
        "top_n": options.limit,
    }

    results: list[list[ScoredHit]] = []
    # ThreadPoolExecutor.map preserves submission order, which keeps the
    # concatenated output deterministic for the test suite (the entity
    # list itself is sorted with a stable tiebreaker above).
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for hits in pool.map(
            lambda query: list(index.search(query, **filter_kwargs)),  # type: ignore[arg-type]
            queries,
        ):
            results.append(hits)

    # Flatten while preserving per-entity order.
    flattened: list[ScoredHit] = []
    for hits in results:
        flattened.extend(hits)
    return flattened


def fuse_results(
    round_one: list[ScoredHit],
    round_two: list[ScoredHit],
) -> list[ScoredHit]:
    """Merge round-1 and round-2 hits, deduping by ``tweet_id`` (Req 1.4).

    Round-1 ordering is preserved verbatim; for shared ids the round-1
    hit object wins (its score, ``why``, and feature breakdown all come
    through). Round-2-only ids are appended in their original round-2
    order, and internal duplicates inside ``round_two`` collapse onto
    the first occurrence.
    """

    fused: dict[str, ScoredHit] = {}
    # ``dict`` insertion order is the iteration order; round-1 first
    # gives round-1 priority for shared ids automatically.
    for hit in round_one:
        if hit.tweet_id not in fused:
            fused[hit.tweet_id] = hit
    for hit in round_two:
        if hit.tweet_id not in fused:
            fused[hit.tweet_id] = hit
    return list(fused.values())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pick_top_entities(kg: KG, k: int) -> list[Node]:
    """Return up to ``k`` entities sorted by boosted weight desc, id asc.

    Pulls each entity kind's own top-``k`` slice (cheap because
    :meth:`KG.top_entities` already sorts), boosts each candidate by
    its kind weight, then merges and re-sorts. The id tiebreaker is
    inherited from ``KG.top_entities`` for intra-kind ordering and
    explicit here for cross-kind ordering.
    """

    if k <= 0:
        return []

    candidates: list[tuple[float, str, Node]] = []
    for kind in _ENTITY_KINDS:
        boost = ENTITY_BOOSTS.get(kind, 0.0)
        if boost == 0.0:
            continue
        for node in kg.top_entities(kind, k):
            boosted = node.weight * boost
            candidates.append((boosted, node.id, node))

    # ``-boosted`` so higher weights sort first; ``id`` ascending breaks
    # ties for cross-kind nodes that score the same boosted weight.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [node for _boost, _id, node in candidates[:k]]


def _query_for_entity(node: Node) -> str:
    """Decode a namespaced entity id back to a search-friendly query.

    The KG keys nodes by ``"{kind}:{value}"`` (e.g. ``handle:tom_doerr``,
    ``concept:ai_pentesting``); the search seam wants the bare value.
    Concept slugs round-trip through underscore -> space so the BM25
    tokenizer sees the original phrase, not a single compound token.
    Handles, hashtags, and domains keep their literal value.
    """

    prefix = f"{node.kind.value}:"
    raw = node.id[len(prefix) :] if node.id.startswith(prefix) else node.id
    if node.kind is NodeKind.CONCEPT:
        return raw.replace("_", " ")
    return raw
