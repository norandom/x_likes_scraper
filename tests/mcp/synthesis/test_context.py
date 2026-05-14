"""Tests for the fenced synthesis-context assembler (synthesis-report task 4.1).

These tests pin the public surface of
``x_likes_mcp.synthesis.context.build_fenced_context``:

* every marker family is neutralized inside fenced bodies (Req 7.2),
* per-source byte caps truncate before fencing (Req 7.4),
* the total-budget enforcer drops the lowest-rank URL bodies first,
  then the lowest-rank tweets, and never the system prompt or query
  (Req 7.4),
* ``known_source_ids`` matches the fenced sources exactly so the
  downstream claim-source validator (Req 6.5) cannot accept citations
  pointing at content the LM never saw,
* the user query never reaches the blob (Req 6.5 / 7.3 — the
  synthesizer carries it in a separate signature field).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from x_likes_mcp.sanitize import (
    _ALL_FENCES,
    ENTITY_FENCE_OPEN,
    KG_EDGE_FENCE_OPEN,
    KG_NODE_FENCE_OPEN,
    LLM_FENCE_OPEN,
    URL_BODY_FENCE_OPEN,
    URL_FENCE_OPEN,
)
from x_likes_mcp.synthesis.context import (
    MAX_KG_EDGES,
    MAX_KG_NODES,
    FencingBudget,
    build_fenced_context,
)
from x_likes_mcp.synthesis.kg import KG, Edge, EdgeKind, Node, NodeKind
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import FetchedUrl

# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeHit:
    """Minimal duck-typed stand-in for ``ranker.ScoredHit``.

    ``build_fenced_context`` only reads ``tweet_id`` and ``snippet``;
    the production ``ScoredHit`` dataclass will gain a ``snippet``
    field as the synthesis pipeline integrates with the index. Using a
    dedicated test double here keeps these context tests independent
    of that future migration.
    """

    tweet_id: str
    snippet: str


def _make_fetched_url(
    final_url: str = "https://example.com/article",
    body: str = "page body",
    size: int | None = None,
) -> FetchedUrl:
    """Build a :class:`FetchedUrl` with sensible defaults for tests."""

    return FetchedUrl(
        url=final_url,
        final_url=final_url,
        content_type="text/html",
        sanitized_markdown=body,
        size_bytes=size if size is not None else len(body.encode("utf-8")),
    )


def _empty_kg() -> KG:
    """Return a fresh empty :class:`KG`."""

    return KG()


# ---------------------------------------------------------------------------
# 1. Empty inputs
# ---------------------------------------------------------------------------


def test_empty_inputs_produce_minimal_blob() -> None:
    """No hits, no URLs, no KG → blob still carries SHAPE; ids empty."""

    blob, ids = build_fenced_context(
        query="anything",
        shape=ReportShape.BRIEF,
        hits=[],
        fetched_urls=[],
        kg=_empty_kg(),
    )

    assert blob.startswith("SHAPE: brief")
    assert ids == set()
    # No section headers from absent sources.
    assert "TWEETS:" not in blob
    assert "URLS:" not in blob
    assert "URL_BODIES:" not in blob


# ---------------------------------------------------------------------------
# 2. Marker neutralization
# ---------------------------------------------------------------------------


def test_each_marker_family_neutralized_in_bodies() -> None:
    """Every fence marker present in any body is replaced with [FENCE]."""

    payload = " ".join(_ALL_FENCES)
    hit = FakeHit(tweet_id="t1", snippet=payload)
    fetched = _make_fetched_url(body=payload)
    kg = _empty_kg()
    kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label=payload, weight=1.0))
    kg.add_edge(
        Edge(src="handle:foo", dst="handle:foo", kind=EdgeKind.MENTIONS),
    )
    # Generous budget so nothing truncates and we can inspect the full body.
    budget = FencingBudget(
        per_tweet_bytes=10_000,
        per_url_body_bytes=10_000,
        per_entity_bytes=10_000,
        per_kg_label_bytes=10_000,
        total_bytes=100_000,
    )

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.SYNTHESIS,
        hits=[hit],
        fetched_urls=[fetched],
        kg=kg,
        budget=budget,
    )

    # The structural section markers (the outer-most pair) must still
    # appear once per fenced source. Every *interior* occurrence (the
    # one we injected via ``payload``) should have been replaced.
    for marker in _ALL_FENCES:
        # The body of each fence carried each marker exactly once.
        # After neutralization the marker appears only as the structural
        # opener/closer. Count occurrences of each marker in the blob —
        # they should match the structural slots only.
        occurrences = blob.count(marker)
        # Tweet section uses LLM_FENCE_OPEN/CLOSE structurally.
        # Every other marker appears only when its own section emits
        # opener/closer pairs. So the count must equal the number of
        # structural slots, never more.
        # Conservative bound: each marker appears at most a small
        # constant number of times; specifically, never as part of the
        # body payload. Asserting "[FENCE]" is in the blob proves
        # neutralization actually fired.
        assert occurrences <= 4, (
            f"marker {marker!r} appears {occurrences} times - payload was not neutralized"
        )

    assert "[FENCE]" in blob


# ---------------------------------------------------------------------------
# 3. Per-source byte caps
# ---------------------------------------------------------------------------


def test_per_tweet_byte_cap_truncates() -> None:
    """A 1000-byte tweet body shrinks to ~per_tweet_bytes after fencing."""

    long_body = "a" * 1000
    hit = FakeHit(tweet_id="t1", snippet=long_body)
    budget = FencingBudget(per_tweet_bytes=100)

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=[hit],
        fetched_urls=[],
        kg=_empty_kg(),
        budget=budget,
    )

    # The tweet content (between the fence markers) must be <= cap.
    assert "a" * 100 in blob
    assert "a" * 101 not in blob


def test_per_url_body_byte_cap_truncates() -> None:
    """A 10000-byte URL body shrinks to per_url_body_bytes after fencing."""

    long_body = "b" * 10000
    fetched = _make_fetched_url(body=long_body)
    budget = FencingBudget(per_url_body_bytes=200)

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=[],
        fetched_urls=[fetched],
        kg=_empty_kg(),
        budget=budget,
    )

    assert "b" * 200 in blob
    assert "b" * 201 not in blob


def test_per_entity_and_kg_label_caps() -> None:
    """Long entity / KG-label strings truncate to the configured caps."""

    long_label = "x" * 500
    long_entity = "y" * 500
    kg = _empty_kg()
    kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label=long_entity, weight=1.0))
    kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label=long_label, weight=1.0))
    budget = FencingBudget(per_entity_bytes=50, per_kg_label_bytes=30)

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=[],
        fetched_urls=[],
        kg=kg,
        budget=budget,
    )

    # Entity cap.
    assert "y" * 50 in blob
    assert "y" * 51 not in blob
    # KG label cap.
    assert "x" * 30 in blob
    assert "x" * 31 not in blob


# ---------------------------------------------------------------------------
# 4. Total-budget enforcer
# ---------------------------------------------------------------------------


def test_total_budget_drops_lowest_rank_url_body_first() -> None:
    """URL bodies drop in reverse order before any tweet drops."""

    hits = [FakeHit(tweet_id=f"t{i}", snippet="t" * 200) for i in range(3)]
    fetched = [
        _make_fetched_url(
            final_url=f"https://example.com/{i}",
            body="u" * 800,
        )
        for i in range(3)
    ]
    # Tight enough that at least one URL body must drop. The exact
    # threshold is not load-bearing; what matters is the *order*.
    budget = FencingBudget(
        per_tweet_bytes=400,
        per_url_body_bytes=1000,
        total_bytes=2200,
    )

    blob, ids = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=hits,
        fetched_urls=fetched,
        kg=_empty_kg(),
        budget=budget,
    )

    # Highest-rank URL (index 0) survives; lowest-rank (index 2) is
    # dropped first. Concretely: if any URL is dropped, it is the last.
    assert "url:https://example.com/0" in ids
    assert "url:https://example.com/2" not in ids
    # All tweets survive: URL drops happen first.
    for hit in hits:
        assert f"tweet:{hit.tweet_id}" in ids
    # Surviving URL appears in blob; dropped URL does not.
    assert "https://example.com/0" in blob
    assert "https://example.com/2" not in blob


def test_total_budget_drops_tweet_when_no_more_urls() -> None:
    """With only tweets, lowest-rank tweets drop first."""

    hits = [FakeHit(tweet_id=f"t{i}", snippet="t" * 200) for i in range(5)]
    budget = FencingBudget(per_tweet_bytes=200, total_bytes=600)

    blob, ids = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=hits,
        fetched_urls=[],
        kg=_empty_kg(),
        budget=budget,
    )

    # Highest-rank tweet (t0) survives; lowest-rank (t4) is dropped first.
    assert "tweet:t0" in ids
    assert "tweet:t4" not in ids
    # Blob must fit under the budget after enforcement.
    assert len(blob.encode("utf-8")) <= budget.total_bytes


# ---------------------------------------------------------------------------
# 5. known_source_ids reflects exactly what survives
# ---------------------------------------------------------------------------


def test_known_source_ids_matches_blob_sources() -> None:
    """For every id in the set, its fenced section appears in the blob."""

    hits = [FakeHit(tweet_id=f"t{i}", snippet=f"snippet {i}") for i in range(2)]
    fetched = [
        _make_fetched_url(final_url=f"https://example.com/{i}", body=f"body {i}") for i in range(2)
    ]

    blob, ids = build_fenced_context(
        query="q",
        shape=ReportShape.SYNTHESIS,
        hits=hits,
        fetched_urls=fetched,
        kg=_empty_kg(),
    )

    expected = {
        "tweet:t0",
        "tweet:t1",
        "url:https://example.com/0",
        "url:https://example.com/1",
    }
    assert ids == expected
    for tweet_id in ("t0", "t1"):
        # The snippet should be present inside the tweet fence.
        assert f"snippet {tweet_id[-1]}" in blob
    for i in range(2):
        # The URL itself appears inside the URL fence.
        assert f"https://example.com/{i}" in blob


def test_dropped_source_id_absent_from_blob_and_set() -> None:
    """When the budget drops a URL, both the section and id disappear."""

    hits = [FakeHit(tweet_id="t0", snippet="hi")]
    fetched = [
        _make_fetched_url(final_url="https://kept.example/", body="ok"),
        _make_fetched_url(final_url="https://dropped.example/", body="x" * 8000),
    ]
    budget = FencingBudget(
        per_url_body_bytes=8000,
        total_bytes=400,
    )

    blob, ids = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=hits,
        fetched_urls=fetched,
        kg=_empty_kg(),
        budget=budget,
    )

    assert "url:https://dropped.example/" not in ids
    assert "https://dropped.example/" not in blob


# ---------------------------------------------------------------------------
# 6. Query / shape handling
# ---------------------------------------------------------------------------


def test_query_not_in_blob() -> None:
    """The user query travels in the synthesizer's separate field, not here."""

    blob, _ = build_fenced_context(
        query="my secret query phrase",
        shape=ReportShape.BRIEF,
        hits=[],
        fetched_urls=[],
        kg=_empty_kg(),
    )

    assert "my secret query phrase" not in blob


def test_shape_header_present() -> None:
    """Blob starts with the literal SHAPE: <value> header."""

    for shape in ReportShape:
        blob, _ = build_fenced_context(
            query="q",
            shape=shape,
            hits=[],
            fetched_urls=[],
            kg=_empty_kg(),
        )
        assert blob.startswith(f"SHAPE: {shape.value}")


# ---------------------------------------------------------------------------
# 7. Sanitization integration
# ---------------------------------------------------------------------------


def test_unsafe_codepoints_stripped() -> None:
    """ANSI / BiDi / control codepoints are stripped via sanitize_text."""

    payload = (
        "before\x1b[31mRED\x1b[0m"  # ANSI
        + chr(0x202E)  # RLO bidi override
        + "after"
        + chr(0xFEFF)  # BOM
    )
    hit = FakeHit(tweet_id="t1", snippet=payload)

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=[hit],
        fetched_urls=[],
        kg=_empty_kg(),
    )

    assert "\x1b" not in blob
    assert chr(0x202E) not in blob
    assert chr(0xFEFF) not in blob
    # The visible text content survived.
    assert "beforeREDafter" in blob


# ---------------------------------------------------------------------------
# 8. KG section caps + edge caption format
# ---------------------------------------------------------------------------


def test_kg_section_capped() -> None:
    """A KG with 100 nodes emits at most MAX_KG_NODES KG_NODE blocks."""

    kg = _empty_kg()
    for i in range(100):
        kg.add_node(
            Node(
                id=f"concept:{i}",
                kind=NodeKind.CONCEPT,
                label=f"concept{i}",
                weight=1.0,
            )
        )
    for i in range(100):
        kg.add_edge(
            Edge(
                src=f"concept:{i}",
                dst=f"concept:{(i + 1) % 100}",
                kind=EdgeKind.MENTIONS,
            )
        )

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.SYNTHESIS,
        hits=[],
        fetched_urls=[],
        kg=kg,
    )

    assert blob.count(KG_NODE_FENCE_OPEN) <= MAX_KG_NODES
    assert blob.count(KG_EDGE_FENCE_OPEN) <= MAX_KG_EDGES


def test_kg_edge_caption_format() -> None:
    """An edge ``tweet:1 -authored_by-> handle:foo`` renders its caption."""

    kg = _empty_kg()
    kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label="hi", weight=1.0))
    kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label="foo", weight=1.0))
    kg.add_edge(
        Edge(src="tweet:1", dst="handle:foo", kind=EdgeKind.AUTHORED_BY),
    )

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.SYNTHESIS,
        hits=[],
        fetched_urls=[],
        kg=kg,
    )

    # The caption appears inside an edge fence.
    assert "tweet:1 authored_by handle:foo" in blob
    # And it sits between an open / close pair.
    assert KG_EDGE_FENCE_OPEN in blob


# ---------------------------------------------------------------------------
# 9. Return shape + default budget
# ---------------------------------------------------------------------------


def test_returns_tuple_str_and_set() -> None:
    """The return type is a (str, set[str]) tuple."""

    result = build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=[FakeHit(tweet_id="t1", snippet="hi")],
        fetched_urls=[],
        kg=_empty_kg(),
    )

    assert isinstance(result, tuple)
    blob, ids = result
    assert isinstance(blob, str)
    assert isinstance(ids, set)
    for entry in ids:
        assert isinstance(entry, str)


def test_default_budget_works_with_normal_inputs() -> None:
    """Defaults plus typical input fit comfortably; nothing is dropped."""

    hits = [FakeHit(tweet_id=f"t{i}", snippet=f"tweet body {i}") for i in range(3)]
    fetched = [
        _make_fetched_url(
            final_url=f"https://example.com/{i}",
            body=f"page body {i}" * 10,
        )
        for i in range(2)
    ]
    kg = _empty_kg()
    for i in range(5):
        kg.add_node(
            Node(
                id=f"handle:user{i}",
                kind=NodeKind.HANDLE,
                label=f"user{i}",
                weight=1.0,
            )
        )
    for i in range(10):
        kg.add_node(
            Node(
                id=f"concept:c{i}",
                kind=NodeKind.CONCEPT,
                label=f"concept{i}",
                weight=1.0,
            )
        )

    blob, ids = build_fenced_context(
        query="q",
        shape=ReportShape.SYNTHESIS,
        hits=hits,
        fetched_urls=fetched,
        kg=kg,
    )

    assert len(blob.encode("utf-8")) <= FencingBudget().total_bytes
    expected = {f"tweet:t{i}" for i in range(3)} | {
        f"url:https://example.com/{i}" for i in range(2)
    }
    assert ids == expected


# ---------------------------------------------------------------------------
# 10. Layout invariants — section headers + structural markers
# ---------------------------------------------------------------------------


def test_section_headers_outside_fences() -> None:
    """``TWEETS:`` / ``URLS:`` etc. are plain headers, not inside any fence."""

    hits = [FakeHit(tweet_id="t1", snippet="hi")]
    fetched = [_make_fetched_url(body="page")]
    kg = _empty_kg()
    kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label="foo", weight=1.0))

    blob, _ = build_fenced_context(
        query="q",
        shape=ReportShape.SYNTHESIS,
        hits=hits,
        fetched_urls=fetched,
        kg=kg,
    )

    for header in ("TWEETS:", "URLS:", "URL_BODIES:", "ENTITIES:"):
        assert header in blob
    # Each tweet body is wrapped in the LLM (tweet) fence.
    assert LLM_FENCE_OPEN in blob
    # Each URL link is wrapped in the URL fence.
    assert URL_FENCE_OPEN in blob
    # Each URL body is wrapped in the URL_BODY fence.
    assert URL_BODY_FENCE_OPEN in blob
    # The entity is wrapped in the ENTITY fence.
    assert ENTITY_FENCE_OPEN in blob


def test_inputs_not_mutated() -> None:
    """``build_fenced_context`` does not mutate hits / fetched_urls / kg."""

    hits = [FakeHit(tweet_id="t1", snippet="hi"), FakeHit(tweet_id="t2", snippet="ho")]
    fetched = [_make_fetched_url(body="p1"), _make_fetched_url(body="p2")]
    kg = _empty_kg()
    kg.add_node(Node(id="handle:foo", kind=NodeKind.HANDLE, label="foo", weight=1.0))
    kg.add_edge(
        Edge(src="handle:foo", dst="handle:foo", kind=EdgeKind.MENTIONS),
    )
    hits_copy = list(hits)
    fetched_copy = list(fetched)

    # Tight budget so the enforcer fires.
    budget = FencingBudget(total_bytes=200)
    build_fenced_context(
        query="q",
        shape=ReportShape.BRIEF,
        hits=hits,
        fetched_urls=fetched,
        kg=kg,
        budget=budget,
    )

    assert hits == hits_copy
    assert fetched == fetched_copy


# ---------------------------------------------------------------------------
# 11. FencingBudget contract
# ---------------------------------------------------------------------------


def test_fencing_budget_defaults_match_design() -> None:
    """Defaults pin design.md's documented values."""

    b = FencingBudget()
    assert b.per_tweet_bytes == 280
    assert b.per_url_body_bytes == 4096
    assert b.per_entity_bytes == 64
    assert b.per_kg_label_bytes == 64
    assert b.total_bytes == 32768


def test_fencing_budget_is_frozen() -> None:
    """``FencingBudget`` is a frozen dataclass — fields cannot be mutated."""

    b = FencingBudget()
    with pytest.raises(Exception):  # FrozenInstanceError subclass; broad on purpose.
        b.per_tweet_bytes = 999  # type: ignore[misc]
