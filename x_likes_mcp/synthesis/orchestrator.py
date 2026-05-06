"""Synthesis-report orchestrator for the synthesis-report feature.

The orchestrator is the single ``run_report`` entry point that drives
the pipeline:

    round-1 search
        → optional round-2 fan-out (when ``hops == 2``)
        → optional URL fetch (only when ``options.fetch_urls=True``,
          including the crawl4ai startup probe)
        → KG build
        → fenced-context assembly
        → DSPy synthesis
        → markdown render

Boundary discipline: this module never writes to disk on its own. The
URL cache writes its own files behind the scenes, but those are gated
by ``fetch_urls=True``. The CLI / MCP layer is responsible for
persisting the rendered markdown.

Every downstream exception is translated into :class:`OrchestratorError`
with a structured ``category`` so the boundary can map errors to the
right surface (HTTP code, CLI exit code, MCP error envelope) without
sniffing exception types from the synthesis package.

See ``.kiro/specs/synthesis-report/design.md`` (``orchestrator``
component) and requirements 1.1, 1.3, 1.4, 5.2, 6.3, 9.4, 12.4.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..config import Config, ConfigError
from .compiled import load_compiled
from .context import FencingBudget, build_fenced_context
from .dspy_modules import (
    SynthesisError,
    SynthesisResult,
    SynthesisValidationError,
    configure_lm,
    extract_entities,
    filter_entities_by_relevance,
    synthesize,
)
from .entities import extract_regex
from .fetcher import ContainerUnreachable, fetch_all, probe_container
from .kg import (
    KG,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    concept_id,
    domain_id,
    handle_id,
    hashtag_id,
    query_id,
    tweet_id,
)
from .multihop import (
    HopsOutOfRange,
    fuse_results,
    run_round_one,
    run_round_two,
    validate_hops,
)
from .report_render import render_empty_report, render_report
from .shapes import ReportShape, parse_report_shape
from .types import Entity, EntityKind, FetchedUrl, ReportOptions, ReportResult
from .url_cache import UrlCache

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from x_likes_mcp.index import TweetIndex
    from x_likes_mcp.ranker import ScoredHit


__all__ = [
    "OrchestratorError",
    "build_kg",
    "run_report",
]


def build_kg(
    query: str,
    hits: list[ScoredHit],
    index: TweetIndex,
    *,
    edge_kind: EdgeKind = EdgeKind.RECALL_FOR,
    existing: KG | None = None,
    dspy_fallback: bool = True,
) -> KG:
    """Construct or extend a KG seeded with ``query`` plus a tweet node per hit.

    Public surface for callers that want the KG without running the full
    synthesis pipeline (CLI ``--kg`` mode, downstream tools). Reuses the
    same regex-first / DSPy-fallback entity extraction the orchestrator
    runs internally so the graph is identical to what a synthesis pass
    would build given the same hits.

    When ``existing`` is supplied, the new hits are merged into that KG
    in place (and the same instance is returned). The query root node
    is added only when ``existing`` is ``None`` so a round-2 enrichment
    call does not duplicate it.

    ``dspy_fallback=False`` skips the DSPy fallback for hits where the
    regex pass returned no entities. Hits with empty regex output simply
    contribute no entity nodes. Used by the CLI ``--kg`` mode, which
    must not call the LM.
    """

    kg = existing if existing is not None else KG()
    if existing is None:
        kg.add_node(Node(id=query_id(), kind=NodeKind.QUERY, label=query, weight=1.0))
    _populate_kg_from_hits(
        kg=kg, hits=hits, index=index, edge_kind=edge_kind, dspy_fallback=dspy_fallback
    )
    return kg


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class OrchestratorError(Exception):
    """Top-level orchestrator failure with a structured ``category``.

    The boundary layer (CLI / MCP) maps ``category`` values to surface
    representations:

    * ``"invalid_input"`` — the caller supplied an unknown shape, an
      out-of-range ``hops``, or a malformed filter shape. Translates to
      a 400-style response.
    * ``"config"`` — the LM endpoint env vars are missing. Translates
      to a configuration-error response.
    * ``"upstream"`` — a downstream dependency (LM, crawl4ai container)
      is unreachable or returned an unrecoverable error. Translates to
      a 502-style response.
    * ``"validation"`` — the synthesizer kept citing unknown source IDs
      after the corrective retry. Translates to a synthesis-validation
      response.
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_report(
    index: TweetIndex,
    options: ReportOptions,
    *,
    config: Config,
) -> ReportResult:
    """Drive the full synthesis-report pipeline.

    See module docstring for the pipeline order. Every downstream
    exception is translated into :class:`OrchestratorError` so the
    boundary layer never has to import the leaf-module exception
    families.
    """

    # 1) Validation (no side effects yet) --------------------------------
    shape = _validate_options(options, config)

    # 2) Round-1 search --------------------------------------------------
    hits_round_one = run_round_one(index, options)

    # 3) Empty corpus shortcut (Req 9.4) -- must NOT touch the LM endpoint.
    if not hits_round_one:
        markdown = render_empty_report(options)
        return ReportResult(
            markdown=markdown,
            shape=shape,
            used_hops=1,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    # 4) Configure LM (only when we will actually synthesize) -----------
    try:
        configure_lm(config)
    except ConfigError as exc:
        raise OrchestratorError("config", str(exc)) from exc

    # 5) Build round-1 KG -----------------------------------------------
    kg = _build_round_one_kg(options.query, hits_round_one, index)

    # 5a) Optional LM-backed entity-relevance filter --------------------
    # Keeps off-topic handles / hashtags / domains / concepts from
    # seeding round-2 fan-out and from polluting the synthesis context.
    # Gated by options.filter_entities so existing callers and the
    # offline test fixtures that do not stub filter_entities_by_relevance
    # keep their current behavior.
    if options.filter_entities:
        kg = _filter_kg_by_relevance(options.query, kg)

    # 6) Round-2 fan-out (if hops==2) -----------------------------------
    fused_hits = _run_round_two_phase(index, options, kg, hits_round_one, config)

    # 7) Optional URL fetch (Req 3.1, 3.2, 12.4) ------------------------
    fetched_urls = _fetch_phase(options, fused_hits, config, index)

    # 8-10) Build fenced context, load compiled program, run synthesis. -
    synthesis_result = _synthesize_phase(
        options=options,
        shape=shape,
        fused_hits=fused_hits,
        fetched_urls=fetched_urls,
        kg=kg,
        config=config,
    )

    # 11) Render --------------------------------------------------------
    # Inject snippet + handle onto every hit so the renderer can build
    # canonical ``https://x.com/{handle}/status/{id}`` links instead of
    # falling back to the bare ``i/status/{id}`` shape. ScoredHit is a
    # frozen dataclass; ``object.__setattr__`` is the documented escape
    # hatch the search seam uses for the same purpose.
    _inject_hit_metadata(fused_hits, index)

    markdown = render_report(
        shape,
        options,
        list(fused_hits),
        list(fetched_urls),
        kg,
        synthesis_result,
    )

    # 12) Return --------------------------------------------------------
    return ReportResult(
        markdown=markdown,
        shape=shape,
        used_hops=options.hops,
        fetched_url_count=len(fetched_urls),
        synthesis_token_count=0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_options(options: ReportOptions, config: Config) -> ReportShape:
    """Parse the report shape, validate hops + year, raise OrchestratorError on bad input."""

    try:
        shape = parse_report_shape(options.shape)
    except ValueError as exc:
        raise OrchestratorError("invalid_input", str(exc)) from exc

    try:
        validate_hops(options.hops, max_hops=config.synthesis_max_hops)
    except HopsOutOfRange as exc:
        raise OrchestratorError("invalid_input", str(exc)) from exc

    # ``year`` is a positive int or ``None``; the index handles the rest
    # of the filter validation but a negative year here would later
    # surface as a confusing search miss, so reject it early.
    if options.year is not None and options.year <= 0:
        raise OrchestratorError(
            "invalid_input",
            f"year={options.year} is not a positive integer",
        )
    return shape


_FILTERABLE_NODE_KINDS: frozenset[NodeKind] = frozenset(
    {NodeKind.HANDLE, NodeKind.HASHTAG, NodeKind.DOMAIN, NodeKind.CONCEPT}
)


def _filter_kg_by_relevance(query: str, kg: KG) -> KG:
    """Return a new KG that keeps only entity nodes the LM marks relevant.

    Calls :func:`filter_entities_by_relevance` once over the union of
    HANDLE / HASHTAG / DOMAIN / CONCEPT labels in ``kg`` and rebuilds a
    fresh KG containing the QUERY root, every TWEET node, and only the
    entity nodes whose labels survived the LM pass. Edges that touch a
    dropped node are dropped with it.

    Hallucinated labels (anything the LM returns that was not in the
    candidate set) are ignored — :func:`filter_entities_by_relevance`
    already filters them, but the post-filter ``relevant_set`` check
    here is the second line of defense. Empty / single-candidate KGs
    short-circuit through the no-op fast path inside
    :func:`filter_entities_by_relevance`.
    """

    candidates_with_id: list[tuple[str, str]] = []
    for node in kg.nodes():
        if node.kind in _FILTERABLE_NODE_KINDS:
            candidates_with_id.append((node.label, node.id))

    if not candidates_with_id:
        return kg

    candidate_labels = [label for label, _ in candidates_with_id]
    relevant_labels = filter_entities_by_relevance(query, candidate_labels)
    relevant_set = set(relevant_labels)

    keep_ids: set[str] = {node_id for label, node_id in candidates_with_id if label in relevant_set}

    new_kg = KG()
    for node in kg.nodes():
        if node.kind in (NodeKind.QUERY, NodeKind.TWEET) or node.id in keep_ids:
            new_kg.add_node(node)

    surviving_ids = {n.id for n in new_kg.nodes()}
    for edge in kg.edges():
        if edge.src in surviving_ids and edge.dst in surviving_ids:
            new_kg.add_edge(edge)
    return new_kg


def _build_round_one_kg(
    query: str,
    hits_round_one: list[ScoredHit],
    index: TweetIndex,
) -> KG:
    """Build a fresh KG seeded with the query node and round-1 hit entities."""

    kg = KG()
    kg.add_node(Node(id=query_id(), kind=NodeKind.QUERY, label=query, weight=1.0))
    _populate_kg_from_hits(
        kg=kg,
        hits=hits_round_one,
        index=index,
        edge_kind=EdgeKind.RECALL_FOR,
    )
    return kg


def _run_round_two_phase(
    index: TweetIndex,
    options: ReportOptions,
    kg: KG,
    hits_round_one: list[ScoredHit],
    config: Config,
) -> list[ScoredHit]:
    """Run the round-2 fan-out when ``hops>=2`` and enrich the KG with new hits."""

    if options.hops < 2:
        return hits_round_one

    hits_round_two = run_round_two(
        index,
        options,
        kg,
        k=config.synthesis_round_two_k,
    )
    fused_hits = fuse_results(hits_round_one, hits_round_two)
    # Extend the KG with round-2 entities so the synthesizer benefits
    # from a richer graph; round-2 hits not already in round-1 still
    # contribute their entities.
    round_two_only = [
        hit for hit in hits_round_two if hit.tweet_id not in {h.tweet_id for h in hits_round_one}
    ]
    if round_two_only:
        _populate_kg_from_hits(
            kg=kg,
            hits=round_two_only,
            index=index,
            edge_kind=EdgeKind.RECALL_FOR,
        )
    return fused_hits


def _fetch_phase(
    options: ReportOptions,
    fused_hits: list[ScoredHit],
    config: Config,
    index: TweetIndex,
) -> list[FetchedUrl]:
    """Probe crawl4ai and fetch the per-hit URL set when ``fetch_urls=True``."""

    if not options.fetch_urls:
        return []

    try:
        probe_container(config.crawl4ai_base_url)
    except ContainerUnreachable as exc:
        raise OrchestratorError(
            "upstream",
            f"crawl4ai container unreachable: {exc}",
        ) from exc

    cache = UrlCache(config.url_cache_dir, ttl_days=config.url_cache_ttl_days)
    urls = _collect_urls(fused_hits, index)
    if not urls:
        return []
    return fetch_all(
        urls,
        crawl4ai_base_url=config.crawl4ai_base_url,
        cache=cache,
        max_body_bytes=config.synthesis_per_source_bytes,
        private_allowlist=config.url_fetch_allowed_private_cidrs,
    )


def _synthesize_phase(
    *,
    options: ReportOptions,
    shape: ReportShape,
    fused_hits: list[ScoredHit],
    fetched_urls: list[FetchedUrl],
    kg: KG,
    config: Config,
) -> SynthesisResult:
    """Build fenced context, load the compiled program, and call synthesize.

    Translates :class:`SynthesisValidationError` and :class:`SynthesisError`
    into :class:`OrchestratorError` ``"validation"`` and ``"upstream"``
    categories respectively.
    """

    budget = FencingBudget(
        per_url_body_bytes=config.synthesis_per_source_bytes,
        total_bytes=config.synthesis_total_context_bytes,
    )
    fenced_blob, known_source_ids = build_fenced_context(
        options.query,
        shape,
        list(fused_hits),
        list(fetched_urls),
        kg,
        budget,
    )

    compiled_root = config.output_dir / "synthesis_compiled"
    program = load_compiled(shape, compiled_root)

    month_buckets: list[str] | None = None
    if shape is ReportShape.TREND:
        month_buckets = sorted(
            {ym for hit in fused_hits if (ym := getattr(hit, "year_month", None))}
        )

    try:
        return synthesize(
            shape,
            options.query,
            fenced_blob,
            known_source_ids=known_source_ids,
            month_buckets=month_buckets,
            program=program,
        )
    except SynthesisValidationError as exc:
        raise OrchestratorError(
            "validation",
            f"synthesis validation failed: {exc}",
        ) from exc
    except SynthesisError as exc:
        raise OrchestratorError(
            "upstream",
            f"synthesis call failed: {exc}",
        ) from exc


_TCO_RE = re.compile(r"https?://t\.co/\S+")


def _hit_text(hit: ScoredHit, index: TweetIndex) -> str:
    """Return the best-effort text for ``hit``.

    Prefers an orchestrator-injected ``snippet`` (the search seam adds
    one for some shapes), falls back to the underlying tweet body via
    ``index.tweets_by_id``, and returns ``""`` when neither is present.
    """

    snippet = getattr(hit, "snippet", None)
    if snippet:
        return str(snippet)
    tweet = index.tweets_by_id.get(hit.tweet_id)
    if tweet is None:
        return ""
    return str(getattr(tweet, "text", "") or "")


def _hit_extraction_text(hit: ScoredHit, index: TweetIndex) -> str:
    """Return the text used for entity extraction.

    Strips raw ``t.co`` shortlinks (so the domain regex does not learn
    "t.co" as an entity) and appends the resolved URLs the exporter
    captured in ``Tweet.urls``. The result is what the dense / BM25
    indexers also see (mirrors :func:`corpus_text.tweet_index_text`),
    which keeps the KG entity surface aligned with what the search
    pipeline already retrieves on.
    """

    text = _hit_text(hit, index)
    cleaned = _TCO_RE.sub("", text).strip()

    tweet = index.tweets_by_id.get(hit.tweet_id)
    raw_urls: list[str] = []
    if tweet is not None:
        raw_urls = list(getattr(tweet, "urls", []) or [])

    parts: list[str] = []
    if cleaned:
        parts.append(cleaned)
    parts.extend(url for url in raw_urls if url)
    return " ".join(parts)


def _hit_handle(hit: ScoredHit, index: TweetIndex) -> str:
    """Return the screen-name for ``hit`` or ``""`` if unknown."""

    handle = getattr(hit, "handle", None)
    if handle:
        return str(handle)
    tweet = index.tweets_by_id.get(hit.tweet_id)
    if tweet is None or tweet.user is None:
        return ""
    return str(tweet.user.screen_name or "")


def _inject_hit_metadata(hits: list[ScoredHit], index: TweetIndex) -> None:
    """Best-effort attach ``handle`` and ``snippet`` to each hit in place.

    The orchestrator's leaf modules (entities, KG seed, fenced context)
    already cope with hits that may or may not carry these injected
    fields; the renderer is what relies on them to build canonical
    x.com URLs and per-anchor snippets. Using ``object.__setattr__``
    keeps :class:`ScoredHit`'s frozen-dataclass invariant honoured for
    every other touch point.
    """

    for hit in hits:
        if not getattr(hit, "handle", None):
            object.__setattr__(hit, "handle", _hit_handle(hit, index))
        if not getattr(hit, "snippet", None):
            object.__setattr__(hit, "snippet", _hit_text(hit, index))
        if not getattr(hit, "urls", None):
            tweet = index.tweets_by_id.get(hit.tweet_id)
            urls = list(getattr(tweet, "urls", []) or []) if tweet is not None else []
            object.__setattr__(hit, "urls", urls)


def _populate_kg_from_hits(
    *,
    kg: KG,
    hits: list[ScoredHit],
    index: TweetIndex,
    edge_kind: EdgeKind,
    dspy_fallback: bool = True,
) -> None:
    """Populate ``kg`` with one tweet node per hit plus its entities.

    Per design + Req 5.2, the regex extractor runs first; only the hits
    where the regex pass returned nothing trigger the DSPy fallback.
    ``dspy_fallback=False`` short-circuits that fallback so the caller
    can build the KG without an LM endpoint configured (CLI ``--kg``).
    """

    for hit in hits:
        text = _hit_extraction_text(hit, index)
        handle = _hit_handle(hit, index)

        tnode_id = tweet_id(hit.tweet_id)
        kg.add_node(Node(id=tnode_id, kind=NodeKind.TWEET, label=hit.tweet_id, weight=1.0))
        kg.add_edge(Edge(src=query_id(), dst=tnode_id, kind=edge_kind))

        # Author edge (handle:<screen_name>).
        if handle:
            hnode_id = handle_id(handle)
            kg.add_node(Node(id=hnode_id, kind=NodeKind.HANDLE, label=handle, weight=1.0))
            kg.add_edge(Edge(src=tnode_id, dst=hnode_id, kind=EdgeKind.AUTHORED_BY))

        # Wire entity extraction explicitly so the orchestrator's test
        # seam (monkeypatching ``orchestrator.extract_regex`` and
        # ``orchestrator.extract_entities``) takes effect. Per Req 5.2
        # the DSPy fallback fires only when the regex pass returned
        # nothing for that hit.
        entities = extract_regex(text, [])
        if not entities and dspy_fallback:
            entities = extract_entities(text)

        for entity in entities:
            _add_entity_to_kg(kg, entity, tnode_id)


def _add_entity_to_kg(kg: KG, entity: Entity, source_tweet_id: str) -> None:
    """Add a single entity node and the corresponding edge.

    Maps each :class:`EntityKind` to its KG namespace + edge kind.
    Concept slugs are passed through ``concept_id`` so the KG and the
    regex extractor share the same canonicalization.
    """

    if entity.kind is EntityKind.HANDLE:
        node_id = handle_id(entity.value)
        node_kind = NodeKind.HANDLE
        edge_kind = EdgeKind.MENTIONS
    elif entity.kind is EntityKind.HASHTAG:
        node_id = hashtag_id(entity.value)
        node_kind = NodeKind.HASHTAG
        edge_kind = EdgeKind.MENTIONS
    elif entity.kind is EntityKind.DOMAIN:
        node_id = domain_id(entity.value)
        node_kind = NodeKind.DOMAIN
        edge_kind = EdgeKind.CITES
    elif entity.kind is EntityKind.CONCEPT:
        node_id = concept_id(entity.value)
        node_kind = NodeKind.CONCEPT
        edge_kind = EdgeKind.MENTIONS
    else:  # pragma: no cover - EntityKind is a closed StrEnum.
        return

    kg.add_node(Node(id=node_id, kind=node_kind, label=entity.value, weight=entity.weight))
    kg.add_edge(Edge(src=source_tweet_id, dst=node_id, kind=edge_kind))


def _collect_urls(hits: list[ScoredHit], index: TweetIndex) -> list[str]:
    """Collect the URL list from each hit's underlying tweet, preserving order."""

    urls: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        tweet = index.tweets_by_id.get(hit.tweet_id)
        if tweet is None:
            continue
        for url in getattr(tweet, "urls", []) or []:
            if not isinstance(url, str):
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls
