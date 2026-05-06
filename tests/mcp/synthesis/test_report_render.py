"""Tests for the markdown report renderer (synthesis-report task 4.4).

The renderer assembles the final markdown for the three report shapes
(``brief``, ``synthesis``, ``trend``) from the orchestrator's structured
inputs. It never calls the LM, so these tests stay fully offline.

Behavioral contract pinned here:

* Empty corpus produces a short "no matching tweets" report and never
  consults the synthesizer (Req 9.4).
* ``brief`` emits a ~300-word concept brief, top entities, and 5-10
  anchor tweets — no mermaid mindmap (per design).
* ``synthesis`` emits a longer narrative with a mermaid mindmap block
  and per-cluster tweet list (Req 8.1).
* ``trend`` groups anchor tweets into chronologically-ordered month
  buckets and includes the mindmap (Req 8.3).
* Anchor tweet links use the canonical ``https://x.com/{handle}/status/{id}``
  URL via ``tools._build_status_url`` (Req 8.4).
* The final markdown body passes through ``sanitize_text`` once (Req 7.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from x_likes_mcp.synthesis.dspy_modules import SynthesisResult
from x_likes_mcp.synthesis.kg import KG, Node, NodeKind
from x_likes_mcp.synthesis.report_render import (
    render_empty_report,
    render_report,
)
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import Claim, MonthSummary, ReportOptions, Section

# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeHit:
    """Duck-typed stand-in for ``ranker.ScoredHit`` plus orchestrator-injected fields.

    The orchestrator decorates ``ScoredHit`` with the auxiliary fields
    the renderer needs (``handle``, ``snippet``, ``year_month``,
    ``created_at``). We mirror that contract here so the test inputs do
    not depend on the production dataclass shape.
    """

    tweet_id: str
    handle: str = ""
    snippet: str = ""
    year_month: str = ""
    created_at: str = ""
    score: float = 0.0
    walker_relevance: float = 0.0
    why: str = ""
    feature_breakdown: dict[str, float] = field(default_factory=dict)
    urls: list[str] = field(default_factory=list)


def _make_hit(
    tweet_id: str,
    handle: str = "alice",
    year_month: str = "2024-06",
    snippet: str = "tweet body",
    created_at: str = "Sat Jun 01 12:00:00 +0000 2024",
    urls: list[str] | None = None,
) -> FakeHit:
    return FakeHit(
        tweet_id=tweet_id,
        handle=handle,
        snippet=snippet,
        year_month=year_month,
        created_at=created_at,
        urls=list(urls or []),
    )


def _make_options(
    *,
    query: str = "ai pentesting",
    shape: ReportShape = ReportShape.BRIEF,
    year: int | None = None,
    month_start: str | None = None,
    month_end: str | None = None,
) -> ReportOptions:
    return ReportOptions(
        query=query,
        shape=shape,
        year=year,
        month_start=month_start,
        month_end=month_end,
    )


def _empty_kg() -> KG:
    return KG()


def _kg_with_one_handle() -> KG:
    """Minimal KG with a single HANDLE node so the mindmap has at least one category."""

    kg = KG()
    kg.add_node(Node(id="handle:alice", kind=NodeKind.HANDLE, label="alice", weight=1.0))
    return kg


# ---------------------------------------------------------------------------
# 1. Empty corpus / render_empty_report
# ---------------------------------------------------------------------------


def test_render_empty_report_states_no_tweets() -> None:
    """Output mentions the query and explains that no matching tweets were found."""

    options = _make_options(query="ai pentesting")
    body = render_empty_report(options)

    assert "ai pentesting" in body
    assert "no matching tweets" in body.lower()
    # Title is the first heading so the report is identifiable in any
    # browser preview.
    assert body.lstrip().startswith("# Synthesis report")


def test_render_empty_report_does_not_call_synthesizer() -> None:
    """``render_report`` with an empty hit list never touches ``synthesis``.

    We pass ``synthesis=None`` to make accidental access trip a
    :class:`AttributeError`. Calling the function must not raise.
    """

    options = _make_options(query="anything", shape=ReportShape.BRIEF)
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=[],
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=None,  # type: ignore[arg-type]
    )

    assert "no matching tweets" in body.lower()
    assert "anything" in body


def test_render_empty_report_mentions_active_filters() -> None:
    """A year filter surfaces in the empty-report body so the user knows why."""

    options = _make_options(query="x", year=2024, month_start="06", month_end="08")
    body = render_empty_report(options)

    # Filter mention is best-effort — we just look for the year value.
    assert "2024" in body


# ---------------------------------------------------------------------------
# 2. BRIEF shape
# ---------------------------------------------------------------------------


def _brief_synthesis() -> SynthesisResult:
    """Return a brief-shaped synthesis with several claims and a small entity list.

    Multiple claims keep the rendered brief inside the 50-350 word
    envelope without being too verbose.
    """

    claims = [
        Claim(
            text=(
                "The corpus highlights agentic security tooling that pairs language "
                "models with code execution sandboxes for offensive testing."
            ),
            sources=["tweet:t1"],
        ),
        Claim(
            text=(
                "Several authors compare DSPy and LangChain prompt patterns when "
                "building autonomous pentest agents."
            ),
            sources=["tweet:t2"],
        ),
        Claim(
            text=(
                "Discussion threads stress sandboxed Docker containers and strict "
                "egress allowlists when running automated browsing for recon work."
            ),
            sources=["tweet:t3"],
        ),
    ]
    return SynthesisResult(
        claims=claims,
        top_entities=["handle:alice", "concept:ai_pentesting"],
    )


def test_brief_target_word_count() -> None:
    """The brief stays in the loose envelope of 50-350 words."""

    options = _make_options(query="ai pentesting", shape=ReportShape.BRIEF)
    hits = [_make_hit("t1", snippet="hello world")]
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    word_count = len(body.split())
    assert 50 <= word_count <= 350, f"brief word count out of envelope: {word_count}"


def test_brief_no_mindmap_block() -> None:
    """Brief reports do NOT carry a mermaid mindmap fence (design rule)."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [_make_hit("t1")]
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_kg_with_one_handle(),
        synthesis=_brief_synthesis(),
    )

    assert "```mermaid" not in body
    assert "mindmap" not in body.lower()


def test_brief_includes_top_entities() -> None:
    """Top entities appear as a bulleted list."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [_make_hit("t1")]
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    assert "handle:alice" in body
    assert "concept:ai_pentesting" in body


def test_brief_anchor_tweets_present() -> None:
    """Anchor tweets render with a clickable canonical x.com URL."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [_make_hit(f"t{i}", handle="alice", snippet=f"tweet {i}") for i in range(3)]
    # Tweet IDs need to be all-digits to land on the per-handle status URL.
    hits = [_make_hit(str(1000 + i), handle="alice", snippet=f"tweet {i}") for i in range(3)]
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    for hit in hits:
        assert f"https://x.com/alice/status/{hit.tweet_id}" in body


# ---------------------------------------------------------------------------
# 3. SYNTHESIS shape
# ---------------------------------------------------------------------------


def _synthesis_synthesis() -> SynthesisResult:
    """Return a synthesis-shaped result with one section and a cluster mapping."""

    section = Section(
        heading="Tooling",
        claims=[
            Claim(
                text="Agentic agents pair web search with code execution.",
                sources=["tweet:t1"],
            )
        ],
    )
    return SynthesisResult(
        sections=[section],
        top_entities=["handle:alice"],
        cluster_assignments={"handle:alice": ["t1"]},
    )


def test_synthesis_includes_mindmap_block() -> None:
    """Synthesis output carries a fenced mermaid block with a mindmap directive."""

    options = _make_options(query="agents", shape=ReportShape.SYNTHESIS)
    hits = [_make_hit("t1")]
    body = render_report(
        shape=ReportShape.SYNTHESIS,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_kg_with_one_handle(),
        synthesis=_synthesis_synthesis(),
    )

    assert "```mermaid" in body
    assert "mindmap" in body


def test_synthesis_includes_per_cluster_section() -> None:
    """Each ``cluster_assignments`` entry produces its own ``### Cluster:`` block."""

    options = _make_options(query="agents", shape=ReportShape.SYNTHESIS)
    hits = [_make_hit("1", handle="foo", snippet="hi")]
    synth = SynthesisResult(
        sections=[],
        top_entities=[],
        cluster_assignments={"handle:foo": ["1"]},
    )
    body = render_report(
        shape=ReportShape.SYNTHESIS,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=synth,
    )

    assert "### Cluster: handle:foo" in body
    # The cluster's tweet ID resolves to a clickable link in the bullet list.
    assert "https://x.com/foo/status/1" in body


def test_synthesis_emits_section_headings() -> None:
    """Each ``Section`` from the synthesizer becomes its own ``## {heading}``."""

    options = _make_options(query="agents", shape=ReportShape.SYNTHESIS)
    hits = [_make_hit("1")]
    body = render_report(
        shape=ReportShape.SYNTHESIS,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_kg_with_one_handle(),
        synthesis=_synthesis_synthesis(),
    )

    assert "## Tooling" in body


# ---------------------------------------------------------------------------
# 4. TREND shape
# ---------------------------------------------------------------------------


def _trend_synthesis(year_months: list[str]) -> SynthesisResult:
    """Return a trend-shaped synthesis with one MonthSummary per ``year_month``."""

    per_month = [
        MonthSummary(
            year_month=ym,
            summary=f"Activity in {ym}.",
            anchor_tweets=[f"t-{ym}"],
        )
        for ym in year_months
    ]
    return SynthesisResult(
        per_month=per_month,
        top_entities=[],
    )


def test_trend_orders_months_chronologically() -> None:
    """Months emitted by the synthesizer in arbitrary order render in YYYY-MM order."""

    options = _make_options(query="agents", shape=ReportShape.TREND)
    hits = [
        _make_hit("t-2024-12", year_month="2024-12"),
        _make_hit("t-2025-01", year_month="2025-01"),
        _make_hit("t-2024-06", year_month="2024-06"),
    ]
    synth = _trend_synthesis(["2024-12", "2025-01", "2024-06"])

    body = render_report(
        shape=ReportShape.TREND,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_kg_with_one_handle(),
        synthesis=synth,
    )

    # Locate each month heading and confirm the order.
    pos_06 = body.find("## 2024-06")
    pos_12 = body.find("## 2024-12")
    pos_01 = body.find("## 2025-01")
    assert pos_06 != -1
    assert pos_12 != -1
    assert pos_01 != -1
    assert pos_06 < pos_12 < pos_01


def test_trend_includes_mindmap() -> None:
    """Trend reports carry the mermaid mindmap block."""

    options = _make_options(query="agents", shape=ReportShape.TREND)
    hits = [_make_hit("t-2024-06", year_month="2024-06")]
    synth = _trend_synthesis(["2024-06"])

    body = render_report(
        shape=ReportShape.TREND,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_kg_with_one_handle(),
        synthesis=synth,
    )

    assert "```mermaid" in body
    assert "mindmap" in body


def test_trend_falls_back_to_hit_buckets_when_per_month_empty() -> None:
    """When the synthesizer fails to bucket, the renderer groups hits by ``year_month``."""

    options = _make_options(query="agents", shape=ReportShape.TREND)
    hits = [
        _make_hit("a", year_month="2024-12"),
        _make_hit("b", year_month="2024-06"),
    ]
    synth = SynthesisResult(per_month=[], top_entities=[])

    body = render_report(
        shape=ReportShape.TREND,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=synth,
    )

    pos_06 = body.find("## 2024-06")
    pos_12 = body.find("## 2024-12")
    assert pos_06 != -1
    assert pos_12 != -1
    assert pos_06 < pos_12


# ---------------------------------------------------------------------------
# 5. Anchor URLs / sanitize / shape dispatch
# ---------------------------------------------------------------------------


def test_anchor_tweet_links_use_canonical_x_com_url() -> None:
    """Every anchor list item carries the canonical ``https://x.com/...`` URL."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [
        _make_hit("100", handle="alice", snippet="first"),
        _make_hit("200", handle="bob", snippet="second"),
    ]
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    assert "https://x.com/alice/status/100" in body
    assert "https://x.com/bob/status/200" in body


def test_output_passes_through_sanitize_text() -> None:
    """ANSI / BiDi codepoints in the synthesizer output are stripped from the body."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    # Built via ``chr`` so the source file itself stays free of literal
    # Trojan-Source-style hidden direction overrides (mirrors the
    # technique in ``x_likes_mcp.sanitize``).
    bidi = chr(0x202E)  # RIGHT-TO-LEFT OVERRIDE
    ansi = "\x1b[31m"
    payload = f"hidden {bidi}{ansi} content"
    claim = Claim(text=payload, sources=["tweet:1"])
    synth = SynthesisResult(claims=[claim], top_entities=[])
    hits = [_make_hit("1", snippet="hi")]

    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=synth,
    )

    assert bidi not in body
    assert "\x1b" not in body
    # The visible text survives.
    assert "hidden" in body


def test_unknown_shape_raises_value_error() -> None:
    """Dispatch is exhaustive — a non-ReportShape value trips a ``ValueError``."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [_make_hit("1")]

    class _NotAShape:
        value = "bogus"

    with pytest.raises(ValueError):
        render_report(
            shape=_NotAShape(),  # type: ignore[arg-type]
            options=options,
            hits=hits,
            fetched_urls=[],
            kg=_empty_kg(),
            synthesis=_brief_synthesis(),
        )


def test_render_report_with_no_synthesis_uses_hits_only() -> None:
    """``synthesis=None`` with a non-empty hit list still produces a brief.

    The renderer falls back to anchor-tweets-only with a placeholder
    summary so a synthesis-validation error upstream still produces a
    readable report rather than a ``NoneType`` crash.
    """

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [_make_hit("100", handle="alice", snippet="first")]

    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=None,
    )

    assert "https://x.com/alice/status/100" in body
    assert body.lstrip().startswith("# Synthesis report")


def test_render_report_handles_missing_handle_gracefully() -> None:
    """A hit with an empty handle still emits a usable status URL."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    # Numeric ID without a handle falls through to the ``i/status`` path.
    hits = [_make_hit("12345", handle="", snippet="anonymous")]

    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    # The fallback path uses the ``i/status`` form when the handle is missing.
    assert "https://x.com/i/status/12345" in body


def test_brief_caps_anchor_tweets_at_ten() -> None:
    """Brief never lists more than 10 anchor tweets even when more hits are passed."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hits = [_make_hit(str(i), handle="alice", snippet=f"snippet {i}") for i in range(20)]
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    # Count how many distinct status URLs appear.
    found = sum(1 for hit in hits if f"/status/{hit.tweet_id}" in body)
    assert found <= 10


# ---------------------------------------------------------------------------
# 6. Defensive: hits-only path keeps the title + filter info
# ---------------------------------------------------------------------------


def test_render_report_returns_str() -> None:
    """The renderer always returns a non-empty ``str`` for any populated input."""

    options = _make_options(query="x", shape=ReportShape.SYNTHESIS)
    hits = [_make_hit("1")]
    body: Any = render_report(
        shape=ReportShape.SYNTHESIS,
        options=options,
        hits=hits,
        fetched_urls=[],
        kg=_kg_with_one_handle(),
        synthesis=_synthesis_synthesis(),
    )
    assert isinstance(body, str)
    assert body.strip() != ""


# ---------------------------------------------------------------------------
# t.co rewrite in anchor snippets
# ---------------------------------------------------------------------------


def test_anchor_snippet_strips_tco_and_appends_resolved_urls() -> None:
    """Raw t.co tokens are removed from the snippet and the resolved
    URLs from ``Tweet.urls`` are appended at the end."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hit = _make_hit(
        "100",
        handle="alice",
        snippet="check this https://t.co/abcdef out",
        urls=["https://example.com/article"],
    )
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=[hit],
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    assert "https://t.co/abcdef" not in body
    assert "https://example.com/article" in body


def test_anchor_snippet_drops_unsafe_url_schemes() -> None:
    """Non-HTTP(S) entries in ``Tweet.urls`` are filtered out."""

    options = _make_options(query="x", shape=ReportShape.BRIEF)
    hit = _make_hit(
        "100",
        handle="alice",
        snippet="see https://t.co/x",
        urls=["javascript:alert(1)", "https://safe.example/"],
    )
    body = render_report(
        shape=ReportShape.BRIEF,
        options=options,
        hits=[hit],
        fetched_urls=[],
        kg=_empty_kg(),
        synthesis=_brief_synthesis(),
    )

    assert "javascript:" not in body
    assert "https://safe.example/" in body
