"""Markdown report renderer for the synthesis-report feature (task 4.4).

The renderer consumes the orchestrator's structured inputs (round-1 +
fused round-2 hits, fetched URL bodies, in-memory KG, and the
:class:`~x_likes_mcp.synthesis.dspy_modules.SynthesisResult` produced by
the LM) and assembles a markdown report for one of three shapes:

* ``brief`` — a ~300-word concept brief, top entities, 5-10 anchor
  tweets. No mermaid mindmap (per design.md).
* ``synthesis`` — longer narrative grouped into headed sections, plus a
  mermaid mindmap and a per-cluster anchor tweet list.
* ``trend`` — month-bucketed timeline grouped by each tweet's
  ``year_month`` field, plus a mermaid mindmap.

The renderer NEVER calls the LM; it only formats. Every anchor tweet
link is the canonical ``https://x.com/{handle}/status/{id}`` URL via
:func:`x_likes_mcp.tools._build_status_url`. The final markdown body
runs through :func:`x_likes_mcp.sanitize.sanitize_text` once before
return so a malicious URL excerpt cannot smuggle ANSI escapes into the
written report (Req 7.5).

If ``hits`` is empty, the renderer delegates to
:func:`render_empty_report` and never consults ``synthesis`` (Req 9.4).

See ``.kiro/specs/synthesis-report/design.md`` (``report_render``
component) and requirements 1.2, 7.5, 8.1, 8.3, 8.4, 9.4.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from x_likes_mcp.sanitize import safe_http_url, sanitize_text
from x_likes_mcp.synthesis.mindmap import render_mindmap
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.tools import _build_status_url

# t.co shortlink pattern. The exporter scrapes ``Tweet.text`` with the
# raw t.co tokens but resolves them out into ``Tweet.urls``; the
# renderer rewrites the snippet by stripping the shortlinks and
# appending the resolved URLs so the markdown body contains
# navigable links instead of opaque t.co/abc redirects. Mirrors the
# same pattern used by ``__main__._expand_snippet``.
_TCO_RE = re.compile(r"https?://t\.co/\S+")
_WHITESPACE_RUN_RE = re.compile(r"\s+")

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from x_likes_mcp.synthesis.dspy_modules import SynthesisResult
    from x_likes_mcp.synthesis.kg import KG
    from x_likes_mcp.synthesis.types import (
        Claim,
        FetchedUrl,
        ReportOptions,
        Section,
    )

__all__ = [
    "render_empty_report",
    "render_report",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Anchor tweets in the brief shape are capped at 10 (5-10 per design).
_BRIEF_ANCHOR_LIMIT: int = 10

# Snippet truncation length on anchor lines. Long snippets crowd the
# bullet list; 120 chars matches the search-tool ``_SNIPPET_MAX_CHARS``
# convention.
_SNIPPET_MAX_CHARS: int = 120

# Soft trim threshold for the brief: stop appending claim sentences if
# the running word count would push past this number.
_BRIEF_WORD_BUDGET: int = 350


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` truncated to ``limit`` characters with an ellipsis suffix."""

    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _hit_field(hit: Any, name: str, default: str = "") -> str:
    """Read an attribute from ``hit`` defensively.

    The renderer accepts either the production ``ScoredHit`` or a
    duck-typed test double; both may or may not carry the
    orchestrator-injected ``handle`` / ``snippet`` / ``year_month`` /
    ``created_at`` fields. Missing fields collapse to ``default`` so the
    renderer never raises on partial inputs.
    """

    value = getattr(hit, name, default)
    if value is None:
        return default
    return str(value)


def _expand_tco_in_snippet(snippet: str, urls: list[str]) -> str:
    """Strip ``t.co`` shortlinks from ``snippet`` and append resolved URLs.

    The exporter writes raw t.co tokens into ``Tweet.text`` while the
    resolved URLs live in ``Tweet.urls``. Substituting positionally is
    fragile, so we follow the same approach the existing CLI search
    output uses: drop every t.co occurrence and append the resolved
    URLs at the end of the snippet, deduped and filtered through
    :func:`safe_http_url` so non-HTTP(S) garbage cannot reach the
    rendered markdown.
    """

    cleaned = _TCO_RE.sub("", sanitize_text(snippet))
    cleaned = _WHITESPACE_RUN_RE.sub(" ", cleaned).strip()
    if not urls:
        return cleaned
    real_urls: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        safe = safe_http_url(raw)
        if safe and safe not in seen:
            seen.add(safe)
            real_urls.append(safe)
    if not real_urls:
        return cleaned
    if cleaned:
        return cleaned + "  " + " ".join(real_urls)
    return " ".join(real_urls)


def _anchor_tweet_line(hit: Any, *, snippet_chars: int = _SNIPPET_MAX_CHARS) -> str:
    """Render a single anchor-tweet bullet for ``hit``.

    Format::

        - [<id>](<canonical x.com URL>): <truncated snippet with resolved URLs>

    The snippet is sanitized so any control / BiDi codepoints carried on
    a freshly indexed hit do not leak into the markdown body, then any
    raw t.co shortlinks are stripped and replaced with the resolved
    URLs the exporter captured in ``Tweet.urls``. The status URL is
    built via :func:`_build_status_url`, which falls back to
    ``https://x.com/i/status/{id}`` when the handle is empty or malformed.
    """

    tweet_id = _hit_field(hit, "tweet_id")
    handle = _hit_field(hit, "handle")
    snippet_raw = _hit_field(hit, "snippet")
    raw_urls = getattr(hit, "urls", None) or []
    expanded = _expand_tco_in_snippet(snippet_raw, list(raw_urls))
    snippet = _truncate(expanded, snippet_chars)
    url = _build_status_url(handle, tweet_id)
    if not url:
        # Fall back to the bare tweet ID label when no link is available.
        return f"- {tweet_id}: {snippet}"
    return f"- [{tweet_id}]({url}): {snippet}"


def _resolve_hit(tweet_id: str, hits: list[Any]) -> Any | None:
    """Return the hit whose ``tweet_id`` matches ``tweet_id`` or ``None``."""

    for hit in hits:
        if _hit_field(hit, "tweet_id") == tweet_id:
            return hit
    return None


def _link_for_source(source: str, hits: list[Any]) -> str | None:
    """Return a clickable URL for a ``tweet:<id>`` or ``url:<final_url>`` cite.

    For ``tweet:`` IDs we look the hit up so we can prefer the canonical
    ``https://x.com/{handle}/status/{id}`` URL when the handle is known;
    when it is not, ``_build_status_url`` falls back to the
    ``i/status/`` shape so the link is still navigable. For ``url:``
    cites we surface the literal URL stored in the cite — that is the
    final URL the fetcher resolved or, when fetching was off, the
    pre-resolved value the exporter wrote into ``Tweet.urls``.
    """

    if source.startswith("tweet:"):
        tweet_id = source[len("tweet:") :]
        hit = _resolve_hit(tweet_id, hits)
        handle = _hit_field(hit, "handle") if hit is not None else ""
        return _build_status_url(handle, tweet_id) or None
    if source.startswith("url:"):
        url = source[len("url:") :]
        return url or None
    return None


def _format_sources(sources: list[str], hits: list[Any]) -> str:
    """Format a claim's source list as ``(see: [id1](url1), [id2](url2))``.

    Each cite the synthesizer emitted is rewritten as a markdown link
    when we can resolve a destination URL for it; cites without a
    resolvable URL are passed through as bare tokens so the reader can
    still see what the model claimed to use. An empty list produces an
    empty string so the rendered sentence does not gain a stray
    parenthetical.
    """

    if not sources:
        return ""
    rendered: list[str] = []
    for raw in sources:
        url = _link_for_source(raw, hits)
        if url:
            rendered.append(f"[{raw}]({url})")
        else:
            rendered.append(raw)
    return f" (see: {', '.join(rendered)})"


def _claim_to_sentence(claim: Claim, hits: list[Any]) -> str:
    """Turn a single :class:`Claim` into a sentence with inline source cites."""

    text = sanitize_text(claim.text).strip()
    return f"{text}{_format_sources(list(claim.sources), hits)}"


def _render_claims_inline(
    claims: list[Claim],
    hits: list[Any],
    *,
    word_budget: int = _BRIEF_WORD_BUDGET,
) -> str:
    """Render claims as inline prose, trimming when the running word count would exceed ``word_budget``.

    Each claim becomes one sentence followed by its ``(see: ...)``
    reference. When a sentence would push the running total above
    ``word_budget``, we stop appending so the brief stays close to its
    soft target.
    """

    parts: list[str] = []
    running = 0
    for claim in claims:
        sentence = _claim_to_sentence(claim, hits)
        sentence_words = len(sentence.split())
        if running + sentence_words > word_budget:
            break
        parts.append(sentence)
        running += sentence_words
    return " ".join(parts)


def _render_top_entities(top_entities: list[str]) -> list[str]:
    """Return markdown lines for the **Top entities** bullet list."""

    lines: list[str] = ["## Top entities"]
    if not top_entities:
        lines.append("- (none)")
        return lines
    for entity in top_entities:
        lines.append(f"- {sanitize_text(entity).strip()}")
    return lines


def _render_anchor_tweet_list(
    hits: list[Any],
    *,
    limit: int | None = None,
    heading: str = "## Anchor tweets",
) -> list[str]:
    """Return markdown lines for an anchor-tweet bullet list."""

    lines: list[str] = [heading]
    selection = hits if limit is None else hits[:limit]
    if not selection:
        lines.append("- (no anchor tweets)")
        return lines
    for hit in selection:
        lines.append(_anchor_tweet_line(hit))
    return lines


def _filter_summary(options: ReportOptions) -> str:
    """Return a short suffix describing any active date filters."""

    parts: list[str] = []
    if options.year is not None:
        parts.append(f"year={options.year}")
    if options.month_start is not None:
        parts.append(f"month_start={options.month_start}")
    if options.month_end is not None:
        parts.append(f"month_end={options.month_end}")
    if not parts:
        return ""
    return " (filters: " + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Empty report (Req 9.4)
# ---------------------------------------------------------------------------


def render_empty_report(options: ReportOptions) -> str:
    """Render the "no matching tweets" report.

    The synthesizer is NOT called — this is the orchestrator's
    short-circuit when the corpus has zero matches. The returned body
    runs through :func:`sanitize_text` once at the end so any control
    codepoints in ``options.query`` cannot leak through the title.
    """

    query = sanitize_text(options.query).strip() or "(empty query)"
    suffix = _filter_summary(options)
    lines = [
        f"# Synthesis report — {query}",
        "",
        (f"No matching tweets were found in the corpus for the query " f"{query!r}{suffix}."),
    ]
    body = "\n".join(lines).rstrip() + "\n"
    return sanitize_text(body)


# ---------------------------------------------------------------------------
# Per-shape renderers
# ---------------------------------------------------------------------------


def _title(options: ReportOptions) -> str:
    """Render the report's first heading."""

    query = sanitize_text(options.query).strip() or "(empty query)"
    return f"# Synthesis report — {query}"


def _render_brief(
    options: ReportOptions,
    hits: list[Any],
    synthesis: SynthesisResult | None,
) -> str:
    """Render the BRIEF shape body."""

    lines: list[str] = [_title(options), ""]

    # ---- Brief paragraph ------------------------------------------------
    lines.append("## Brief")
    if synthesis is not None and synthesis.claims:
        prose = _render_claims_inline(synthesis.claims, hits)
        lines.append(prose if prose else "(no synthesized claims)")
    else:
        # Synthesis-less fallback (e.g. orchestrator skipped the LM
        # call). The brief still gives the reader something to read.
        lines.append(
            f"This brief lists the top hits for {options.query!r}. "
            "The synthesizer did not produce a structured summary."
        )
    lines.append("")

    # ---- Top entities ---------------------------------------------------
    top_entities = list(synthesis.top_entities) if synthesis is not None else []
    lines.extend(_render_top_entities(top_entities))
    lines.append("")

    # ---- Anchor tweets --------------------------------------------------
    lines.extend(_render_anchor_tweet_list(hits, limit=_BRIEF_ANCHOR_LIMIT))

    return "\n".join(lines)


def _render_section(section: Section, hits: list[Any]) -> list[str]:
    """Render a single :class:`Section` as ``## heading`` plus claim paragraphs."""

    heading = sanitize_text(section.heading).strip() or "(untitled section)"
    lines: list[str] = [f"## {heading}", ""]
    if not section.claims:
        lines.append("(no claims)")
        return lines
    for claim in section.claims:
        lines.append(_claim_to_sentence(claim, hits))
        lines.append("")
    # Drop the trailing blank line so consecutive sections do not stack
    # double-blank spacers.
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_synthesis(
    options: ReportOptions,
    hits: list[Any],
    kg: KG,
    synthesis: SynthesisResult | None,
) -> str:
    """Render the SYNTHESIS shape body."""

    lines: list[str] = [_title(options), ""]

    # ---- Mindmap --------------------------------------------------------
    lines.append("## Mindmap")
    lines.append("")
    lines.append(render_mindmap(options.query, kg))
    lines.append("")

    # ---- Sections -------------------------------------------------------
    sections = list(synthesis.sections) if synthesis is not None and synthesis.sections else []
    for section in sections:
        lines.extend(_render_section(section, hits))
        lines.append("")

    # ---- Top entities ---------------------------------------------------
    top_entities = list(synthesis.top_entities) if synthesis is not None else []
    lines.extend(_render_top_entities(top_entities))
    lines.append("")

    # ---- Per-cluster anchor list ---------------------------------------
    cluster_assignments: dict[str, list[str]] = (
        dict(synthesis.cluster_assignments)
        if synthesis is not None and synthesis.cluster_assignments
        else {}
    )
    if cluster_assignments:
        lines.append("## Clusters")
        lines.append("")
        for entity, tweet_ids in cluster_assignments.items():
            entity_label = sanitize_text(entity).strip() or "(unnamed)"
            lines.append(f"### Cluster: {entity_label}")
            for tid in tweet_ids:
                hit = _resolve_hit(tid, hits)
                if hit is None:
                    lines.append(f"- {tid} (no matching hit)")
                    continue
                lines.append(_anchor_tweet_line(hit))
            lines.append("")

    return "\n".join(lines)


def _bucket_hits_by_month(hits: list[Any]) -> dict[str, list[Any]]:
    """Group hits by their ``year_month`` field."""

    buckets: dict[str, list[Any]] = defaultdict(list)
    for hit in hits:
        ym = _hit_field(hit, "year_month")
        if not ym:
            ym = "unknown"
        buckets[ym].append(hit)
    return buckets


def _render_trend(
    options: ReportOptions,
    hits: list[Any],
    kg: KG,
    synthesis: SynthesisResult | None,
) -> str:
    """Render the TREND shape body."""

    lines: list[str] = [_title(options), ""]

    # ---- Mindmap --------------------------------------------------------
    lines.append("## Mindmap")
    lines.append("")
    lines.append(render_mindmap(options.query, kg))
    lines.append("")

    # ---- Per-month sections --------------------------------------------
    per_month = list(synthesis.per_month) if synthesis is not None and synthesis.per_month else []

    if per_month:
        # Sort by ``year_month`` lexicographically — YYYY-MM strings sort
        # chronologically (Req 8.3).
        for month in sorted(per_month, key=lambda m: m.year_month):
            ym = sanitize_text(month.year_month).strip() or "unknown"
            lines.append(f"## {ym}")
            lines.append("")
            summary = sanitize_text(month.summary).strip()
            if summary:
                lines.append(summary)
                lines.append("")
            for tid in month.anchor_tweets:
                hit = _resolve_hit(tid, hits)
                if hit is None:
                    lines.append(f"- {tid} (no matching hit)")
                    continue
                lines.append(_anchor_tweet_line(hit))
            lines.append("")
    else:
        # Defensive fallback: bucket by hit ``year_month`` when the LM
        # returned no ``per_month`` entries.
        buckets = _bucket_hits_by_month(hits)
        for ym in sorted(buckets):
            lines.append(f"## {ym}")
            lines.append("")
            for hit in buckets[ym]:
                lines.append(_anchor_tweet_line(hit))
            lines.append("")

    # ---- Top entities ---------------------------------------------------
    top_entities = list(synthesis.top_entities) if synthesis is not None else []
    lines.extend(_render_top_entities(top_entities))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_report(
    shape: ReportShape,
    options: ReportOptions,
    hits: list[Any],
    fetched_urls: list[FetchedUrl],
    kg: KG,
    synthesis: SynthesisResult | None,
) -> str:
    """Render a markdown report for ``shape``.

    Empty hit lists short-circuit to :func:`render_empty_report` and
    never consult ``synthesis``. Otherwise the per-shape renderer
    assembles the body and the result runs through
    :func:`sanitize_text` once before return (Req 7.5).

    Parameters
    ----------
    shape:
        The validated :class:`ReportShape` enum member. An unknown value
        raises :class:`ValueError` (Req 1.3 / dispatch exhaustiveness).
    options:
        The orchestrator's :class:`ReportOptions`. Carries the user
        query, the active date filters, and the chosen shape.
    hits:
        Round-1 + fused round-2 hits, deduped by ``tweet_id`` upstream.
        Empty list → empty report.
    fetched_urls:
        Already-fetched and already-sanitized URL bodies. Currently
        unused by the renderer — held for forward compatibility with
        per-section "sources" listings.
    kg:
        The shared in-memory knowledge graph; only the synthesis and
        trend shapes consume it (for the mindmap).
    synthesis:
        The :class:`SynthesisResult` produced by the LM. ``None`` when
        the orchestrator skipped the LM call (e.g. validation failure
        upstream); the renderer falls back to anchor-tweets-only with a
        placeholder summary.
    """

    # ---- Empty corpus short-circuit ------------------------------------
    if not hits:
        return render_empty_report(options)

    # ---- Per-shape dispatch --------------------------------------------
    if shape is ReportShape.BRIEF:
        body = _render_brief(options, hits, synthesis)
    elif shape is ReportShape.SYNTHESIS:
        body = _render_synthesis(options, hits, kg, synthesis)
    elif shape is ReportShape.TREND:
        body = _render_trend(options, hits, kg, synthesis)
    else:
        # Defensive: ``ReportShape`` is a closed enum so this branch
        # only fires when a caller bypasses the type system.
        raise ValueError(f"Unknown report shape: {shape!r}")

    # ---- Final sanitize (Req 7.5) --------------------------------------
    return sanitize_text(body)
