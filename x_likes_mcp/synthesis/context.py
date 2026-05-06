"""Fenced synthesis-context assembly for the synthesis-report feature.

This module builds the fenced blob the synthesizer's ``fenced_context``
input field receives. Every untrusted text source (tweet body, fetched
URL link, fetched URL body, entity string, KG node label, KG edge
caption) is sanitized, truncated to a per-source byte cap, and wrapped
in its matching marker family so a crafted source cannot rewrite the
synthesis instructions (Req 7.1, 7.2, 7.4).

A total-byte budget is enforced as a defense-in-depth ceiling. When the
assembled context would exceed the budget, the lowest-rank URL bodies
are dropped first (in reverse rank order), then the lowest-rank tweets,
until the blob fits. The user query and the report-shape directive
never reach this module's blob — the synthesizer carries them in
separate signature fields, outside any fence (Req 6.5).

The function returns ``(fenced_blob, known_source_ids)`` where
``known_source_ids`` is exactly the set of namespaced IDs (``tweet:<id>``
and ``url:<final_url>``) the LM can legitimately cite. Sources dropped
under the budget are removed from this set so a downstream
claim-source validator (Req 6.5) never accepts a citation pointing at
content the LM never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from x_likes_mcp.sanitize import (
    fence_entity_for_llm,
    fence_kg_edge_for_llm,
    fence_kg_node_for_llm,
    fence_url_body_for_llm,
    fence_url_for_llm,
    sanitize_text,
)
from x_likes_mcp.sanitize import (
    fence_for_llm as fence_tweet_for_llm,
)

from .kg import NodeKind

if TYPE_CHECKING:  # pragma: no cover
    from x_likes_mcp.ranker import ScoredHit

    from .kg import KG
    from .shapes import ReportShape
    from .types import FetchedUrl

__all__ = [
    "MAX_KG_EDGES",
    "MAX_KG_NODES",
    "FencingBudget",
    "build_fenced_context",
]


# Defensive caps on KG node / edge emission so a huge graph cannot fill
# the budget by itself. The total-budget enforcer is the final
# guarantee, but trimming the KG up front keeps the dropped tweet and
# URL counts correct in the common case.
MAX_KG_NODES: int = 32
MAX_KG_EDGES: int = 32


@dataclass(frozen=True)
class FencingBudget:
    """Per-source and total byte budgets for the fenced synthesis context.

    All fields are byte counts measured against the post-sanitize UTF-8
    encoding of the body, before the fence markers are added. Defaults
    match design.md: ``per_tweet_bytes=280`` (a tweet is at most 280
    chars), ``per_url_body_bytes=4096`` (a few kilobytes of fetched
    page), ``per_entity_bytes=64`` and ``per_kg_label_bytes=64`` (single
    short identifiers), and ``total_bytes=32768`` (32 KiB hard ceiling).
    """

    per_tweet_bytes: int = 280
    per_url_body_bytes: int = 4096
    per_entity_bytes: int = 64
    per_kg_label_bytes: int = 64
    total_bytes: int = 32768


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate_utf8(text: str, cap: int) -> str:
    """Truncate ``text`` to at most ``cap`` UTF-8 bytes.

    Decodes with ``errors="ignore"`` so a multi-byte codepoint that
    straddles the cap drops cleanly instead of leaving a half-character
    that would later confuse the LM tokenizer.
    """

    if cap <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= cap:
        return text
    return encoded[:cap].decode("utf-8", errors="ignore")


def _prepare_body(raw: str, cap: int) -> str:
    """Sanitize ``raw`` then truncate to ``cap`` UTF-8 bytes.

    The fence helpers re-sanitize internally; that is idempotent
    (``sanitize_text(sanitize_text(x)) == sanitize_text(x)``). We
    sanitize here so the byte cap is measured against the cleaned
    representation rather than against escape-laden raw input.
    """

    return _truncate_utf8(sanitize_text(raw), cap)


def _byte_len(text: str) -> int:
    """Return the UTF-8 byte length of ``text``."""

    return len(text.encode("utf-8"))


def _join_section(header: str, blocks: list[str]) -> str:
    """Render ``header`` followed by ``blocks`` separated by newlines.

    Returns an empty string when ``blocks`` is empty so the caller can
    omit empty sections from the blob without explicit guards.
    """

    if not blocks:
        return ""
    return f"{header}\n" + "\n".join(blocks)


def _assemble_blob(
    shape_value: str,
    tweet_blocks: list[str],
    url_blocks: list[str],
    url_body_blocks: list[str],
    entity_blocks: list[str],
    kg_node_blocks: list[str],
    kg_edge_blocks: list[str],
) -> str:
    """Render the full fenced blob from the per-section block lists.

    The plain-text section headers (``SHAPE:``, ``TWEETS:`` etc.) sit
    outside any fence — they're the structural signposts the LM reads
    to understand what kind of payload follows. Every body inside a
    fence is treated as data per the synthesizer's system prompt rule.
    """

    parts: list[str] = [f"SHAPE: {shape_value}"]
    sections = [
        _join_section("TWEETS:", tweet_blocks),
        _join_section("URLS:", url_blocks),
        _join_section("URL_BODIES:", url_body_blocks),
        _join_section("ENTITIES:", entity_blocks),
        _join_section("KG_NODES:", kg_node_blocks),
        _join_section("KG_EDGES:", kg_edge_blocks),
    ]
    parts.extend(section for section in sections if section)
    return "\n\n".join(parts)


def _fence_tweet_section(
    hits: list[ScoredHit], budget: FencingBudget
) -> tuple[list[str], list[str]]:
    """Render fenced tweet blocks and their parallel ``tweet:<id>`` list."""

    tweet_blocks: list[str] = []
    tweet_ids: list[str] = []
    for hit in hits:
        # The orchestrator may inject the snippet onto the hit (or pass a
        # ScoredHit subclass that carries it). We fall back to "" so a
        # ScoredHit without a snippet still emits a structural slot the
        # claim-source validator can count against.
        body = getattr(hit, "snippet", "") or ""
        prepared = _prepare_body(body, budget.per_tweet_bytes)
        tweet_blocks.append(fence_tweet_for_llm(prepared))
        tweet_ids.append(f"tweet:{hit.tweet_id}")
    return tweet_blocks, tweet_ids


def _fence_url_section(
    fetched_urls: list[FetchedUrl], budget: FencingBudget
) -> tuple[list[str], list[str], list[str]]:
    """Render fenced URL link + body blocks and their ``url:<final>`` ids."""

    url_blocks: list[str] = []
    url_body_blocks: list[str] = []
    url_ids: list[str] = []
    for fetched in fetched_urls:
        # The URL link itself is short and is not subject to the per-URL
        # body cap (it carries one normalized HTTP(S) URL). The fence
        # helper returns ``None`` when the URL fails the safe-HTTP
        # check; in that case both the URL block and its body block
        # are dropped together so the blob never references a URL the
        # LM cannot cite back.
        url_block = fence_url_for_llm(fetched.final_url)
        if url_block is None:
            continue
        body_prepared = _prepare_body(fetched.sanitized_markdown, budget.per_url_body_bytes)
        url_blocks.append(url_block)
        url_body_blocks.append(fence_url_body_for_llm(body_prepared))
        url_ids.append(f"url:{fetched.final_url}")
    return url_blocks, url_body_blocks, url_ids


def _fence_entity_section(kg: KG, budget: FencingBudget) -> list[str]:
    """Render fenced top-K entity blocks across handle/hashtag/domain/concept."""

    entity_blocks: list[str] = []
    for kind in (
        NodeKind.HANDLE,
        NodeKind.HASHTAG,
        NodeKind.DOMAIN,
        NodeKind.CONCEPT,
    ):
        for node in kg.top_entities(kind, 8):
            label = _prepare_body(node.label, budget.per_entity_bytes)
            fenced = fence_entity_for_llm(label)
            if fenced is not None:
                entity_blocks.append(fenced)
    return entity_blocks


def _fence_kg_section(kg: KG, budget: FencingBudget) -> tuple[list[str], list[str]]:
    """Render fenced KG node + edge blocks (capped by MAX_KG_NODES/EDGES)."""

    # KG nodes / edges access the KG's internal node + edge stores
    # directly: there is no public iterator on :class:`KG` and the
    # boundary for this task forbids edits outside ``context.py``. The
    # KG's ``__slots__`` surface keeps the access well-defined.
    kg_node_blocks: list[str] = []
    for idx, node in enumerate(kg._nodes.values()):
        if idx >= MAX_KG_NODES:
            break
        label = _prepare_body(node.label, budget.per_kg_label_bytes)
        kg_node_blocks.append(fence_kg_node_for_llm(label))

    kg_edge_blocks: list[str] = []
    for idx, edge in enumerate(kg._edges):
        if idx >= MAX_KG_EDGES:
            break
        caption = f"{edge.src} {edge.kind.value} {edge.dst}"
        prepared = _prepare_body(caption, budget.per_kg_label_bytes)
        kg_edge_blocks.append(fence_kg_edge_for_llm(prepared))
    return kg_node_blocks, kg_edge_blocks


def _enforce_total_budget(
    shape_value: str,
    tweet_blocks: list[str],
    tweet_ids: list[str],
    url_blocks: list[str],
    url_body_blocks: list[str],
    url_ids: list[str],
    entity_blocks: list[str],
    kg_node_blocks: list[str],
    kg_edge_blocks: list[str],
    budget: FencingBudget,
) -> str:
    """Drop lowest-rank URL bodies, then tweets, until the blob fits.

    Mutates the passed-in block / id lists in place so the caller's
    parallel ``tweet_ids`` / ``url_ids`` lists track the surviving
    sources after enforcement.
    """

    def _current_blob() -> str:
        return _assemble_blob(
            shape_value,
            tweet_blocks,
            url_blocks,
            url_body_blocks,
            entity_blocks,
            kg_node_blocks,
            kg_edge_blocks,
        )

    blob = _current_blob()
    while _byte_len(blob) > budget.total_bytes and url_body_blocks:
        url_body_blocks.pop()
        url_blocks.pop()
        url_ids.pop()
        blob = _current_blob()
    while _byte_len(blob) > budget.total_bytes and tweet_blocks:
        tweet_blocks.pop()
        tweet_ids.pop()
        blob = _current_blob()
    return blob


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_fenced_context(
    query: str,
    shape: ReportShape,
    hits: list[ScoredHit],
    fetched_urls: list[FetchedUrl],
    kg: KG,
    budget: FencingBudget = FencingBudget(),
) -> tuple[str, set[str]]:
    """Assemble the fenced synthesis context and the known-source-ID set.

    Parameters
    ----------
    query:
        The user query. NOT included in the blob (the synthesizer
        carries it in a separate signature field outside any fence per
        Req 6.5 / 7.3). Accepted as a parameter so callers pass the
        full set of inputs in one call site, keeping the boundary
        symmetric with the synthesizer's own signature.
    shape:
        The report shape. Surfaces in the blob as the single
        ``SHAPE: <value>`` header — the LM uses it to pick its output
        contract (brief vs. synthesis vs. trend).
    hits:
        Round-1 (and optionally fused round-2) hits. Order is treated
        as the rank order: ``hits[0]`` is the highest-rank tweet,
        ``hits[-1]`` the lowest. The lowest-rank tweets are the first
        to drop under budget pressure.
    fetched_urls:
        Already-fetched and already-sanitized URL bodies. Order is
        treated as the rank order analogously to ``hits``.
    kg:
        The shared knowledge graph. Top entities (handle / hashtag /
        domain / concept), node labels, and edge captions are mined
        from this graph and contribute fenced context but never
        contribute to ``known_source_ids`` (claims must cite tweets or
        URLs, not entities).
    budget:
        Per-source and total byte budgets. Defaults match design.md.

    Returns
    -------
    tuple[str, set[str]]
        ``(fenced_blob, known_source_ids)``. ``known_source_ids`` is
        the set of namespaced IDs (``tweet:<id>``, ``url:<final_url>``)
        the LM can legitimately cite. Sources dropped under the
        total-budget enforcer are absent from both the blob and the
        set.
    """

    # ---- per-source rendering ------------------------------------------
    # Tweets / URLs keep parallel id lists so the budget enforcer can
    # synchronize drops between the rendered block lists and the
    # known-source set.
    tweet_blocks, tweet_ids = _fence_tweet_section(hits, budget)
    url_blocks, url_body_blocks, url_ids = _fence_url_section(fetched_urls, budget)
    entity_blocks = _fence_entity_section(kg, budget)
    kg_node_blocks, kg_edge_blocks = _fence_kg_section(kg, budget)

    # ---- total-budget enforcement --------------------------------------
    # Drop URL bodies in reverse rank order first (lowest-rank — last in
    # the list — goes first). Each drop also removes the paired URL link
    # block and the URL's known-source ID. After URL bodies are
    # exhausted, drop tweets in reverse rank order.
    blob = _enforce_total_budget(
        shape.value,
        tweet_blocks,
        tweet_ids,
        url_blocks,
        url_body_blocks,
        url_ids,
        entity_blocks,
        kg_node_blocks,
        kg_edge_blocks,
        budget,
    )

    known_source_ids: set[str] = set(tweet_ids) | set(url_ids)
    return blob, known_source_ids
