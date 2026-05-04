"""In-process integration tests for :mod:`x_likes_mcp.server`.

These tests build the MCP :class:`~mcp.server.Server` via
:func:`x_likes_mcp.server.build_server` and exercise the registered tool
surface programmatically, without driving the real stdio transport. The
public entry points covered are:

1. :func:`x_likes_mcp.server._build_tool_definitions` — the four tool
   :class:`mcp.types.Tool` declarations and their JSON input schemas.
2. :func:`x_likes_mcp.server._dispatch` — the inner function the SDK's
   ``@server.call_tool()`` handler calls. Exercising ``_dispatch`` directly
   gives us the structured payload the SDK would return on success and the
   :class:`ToolError` it would translate into an MCP error response on
   failure, which is the behaviour the spec pins down.

The default ``search_likes`` path no longer touches the walker; it drives
hybrid recall (BM25 + dense) through ``index.search``. Tests that need to
exercise the explainer (``with_why=True``) stub
:func:`x_likes_mcp.tools._call_walker_explainer` directly. Tests that need
to simulate the both-retrievals-down failure mode patch
:meth:`x_likes_mcp.embeddings.Embedder.embed_query` and
:meth:`x_likes_mcp.bm25.BM25Index.top_k` to raise. The autouse
``_block_real_llm`` and ``_block_real_embeddings`` fixtures in
``tests/mcp/conftest.py`` keep all real network calls blocked by default.

Boundary: tests/mcp/test_server_integration.py only.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from x_likes_mcp import server as server_module
from x_likes_mcp import tools as tools_module
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.errors import ToolError
from x_likes_mcp.index import TweetIndex
from x_likes_mcp.walker import WalkerHit


# ---------------------------------------------------------------------------
# _build_tool_definitions: tool list + schema shape
# ---------------------------------------------------------------------------


def test_build_tool_definitions_lists_exactly_four_named_tools() -> None:
    """The server registers exactly the four documented tool names."""

    tools = server_module._build_tool_definitions()
    names = [t.name for t in tools]

    assert names == ["search_likes", "list_months", "get_month", "read_tweet"]


def test_build_tool_definitions_each_tool_has_non_empty_input_schema() -> None:
    """Every tool carries a non-empty ``inputSchema`` object."""

    tools = server_module._build_tool_definitions()
    for tool in tools:
        assert isinstance(tool.inputSchema, dict)
        # An object schema with at least the type key (some take no inputs
        # but the dict itself must not be empty/None).
        assert tool.inputSchema, f"{tool.name} has empty inputSchema"
        assert tool.inputSchema.get("type") == "object"


def test_search_likes_schema_year_and_month_fields_are_optional() -> None:
    """``search_likes`` schema lists ``query`` as required and the three
    structured-filter fields (``year``, ``month_start``, ``month_end``)
    plus the new ``with_why`` boolean as optional with the documented
    patterns/bounds.
    """

    tools = server_module._build_tool_definitions()
    by_name = {t.name: t for t in tools}
    search = by_name["search_likes"]
    schema = search.inputSchema

    # query is required; year/month_start/month_end/with_why are NOT.
    assert schema.get("required") == ["query"]

    props = schema["properties"]
    assert "year" in props
    assert "month_start" in props
    assert "month_end" in props
    assert "with_why" in props

    # year: integer, lower-bound 2006.
    year_prop = props["year"]
    assert year_prop["type"] == "integer"
    assert year_prop["minimum"] == 2006

    # month_start / month_end: string with the documented zero-padded pattern.
    month_pattern = "^(0[1-9]|1[0-2])$"
    for field in ("month_start", "month_end"):
        prop = props[field]
        assert prop["type"] == "string"
        assert prop["pattern"] == month_pattern
        # The pattern compiles and validates representative values.
        compiled = re.compile(prop["pattern"])
        assert compiled.match("01")
        assert compiled.match("12")
        assert not compiled.match("00")
        assert not compiled.match("13")
        assert not compiled.match("1")

    # with_why: boolean, default False, documented description.
    with_why_prop = props["with_why"]
    assert with_why_prop["type"] == "boolean"
    assert with_why_prop["default"] is False
    assert "description" in with_why_prop
    assert with_why_prop["description"]


# ---------------------------------------------------------------------------
# _dispatch: end-to-end via build_server(index)
# ---------------------------------------------------------------------------


def _default_weights() -> RankerWeights:
    """Shared ranker weights for the integration tests."""

    return RankerWeights()


def _build_index(fake_export: Config) -> TweetIndex:
    """Construct a real :class:`TweetIndex` from the ``fake_export`` fixture.

    The walker is *not* invoked at construction time; index build is
    parser/IO only.
    """

    return TweetIndex.open_or_build(fake_export, _default_weights())


def test_dispatch_search_likes_returns_documented_payload(
    fake_export: Config,
) -> None:
    """The default ``search_likes`` path drives hybrid recall via the
    autouse-stubbed embedder seam plus the real BM25 index, then runs
    the ranker. The structured payload is ``{"results": [hit, ...]}``
    and each hit carries the eight documented keys. The walker is never
    invoked on the default path (``with_why=False``).
    """

    index = _build_index(fake_export)
    # build_server has the side effect of registering the handlers; we
    # call it for parity with the real startup path even though we drive
    # _dispatch directly.
    server_module.build_server(index)

    payload = server_module._dispatch(index, "search_likes", {"query": "test"})

    assert isinstance(payload, dict)
    assert "results" in payload
    results = payload["results"]
    assert isinstance(results, list)
    # The fake_export fixture has a small but non-empty corpus; hybrid
    # recall over it produces at least one ranked hit.
    assert len(results) >= 1

    hit = results[0]
    assert set(hit.keys()) == {
        "tweet_id",
        "year_month",
        "handle",
        "snippet",
        "urls",
        "score",
        "walker_relevance",
        "why",
        "feature_breakdown",
    }
    # Default path: walker did not run, so `why` is empty (req 8.1 / 7.8).
    assert hit["why"] == ""
    # walker_relevance is the cosine score, clamped to [0, 1].
    assert 0.0 <= hit["walker_relevance"] <= 1.0


def test_dispatch_search_likes_with_why_true_integrates(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``with_why=True``, ``_dispatch`` runs hybrid recall + ranker
    and then invokes the walker explainer. The walker's ``why`` and
    ``relevance`` for matching ids reach the response unchanged (req 8.2,
    8.3).
    """

    # Patch the explainer helper rather than walker.walk so the test does
    # not have to construct a synthetic TweetTree. Returning a hit for a
    # real fixture tweet id lets the merge step land on that result.
    captured: dict[str, Any] = {"calls": 0, "query": None, "top_ids": None}

    def fake_explainer(top_results, query, index):
        captured["calls"] += 1
        captured["query"] = query
        captured["top_ids"] = [hit.tweet_id for hit in top_results]
        return {
            "1001": WalkerHit(
                tweet_id="1001", relevance=0.87, why="explainer rationale"
            )
        }

    monkeypatch.setattr(tools_module, "_call_walker_explainer", fake_explainer)

    index = _build_index(fake_export)
    server_module.build_server(index)

    payload = server_module._dispatch(
        index, "search_likes", {"query": "test", "with_why": True}
    )

    assert captured["calls"] == 1
    assert captured["query"] == "test"
    # Top-20 cap is enforced inside tools.search_likes; with the small
    # fixture corpus we get fewer than that.
    assert captured["top_ids"], "explainer received an empty top-results list"

    results = payload["results"]
    # Find the merged hit (if "1001" is among the ranker's top results).
    merged = [h for h in results if h["tweet_id"] == "1001"]
    if merged:
        hit = merged[0]
        assert hit["why"] == "explainer rationale"
        assert hit["walker_relevance"] == pytest.approx(0.87)
    # Either way, every result still carries the documented keys.
    for hit in results:
        assert set(hit.keys()) == {
            "tweet_id",
            "year_month",
            "handle",
            "snippet",
            "urls",
            "score",
            "walker_relevance",
            "why",
            "feature_breakdown",
        }


def test_dispatch_search_likes_blank_query_raises_invalid_input_tool_error(
    fake_export: Config,
) -> None:
    """A whitespace-only query raises a ``ToolError`` with category
    ``invalid_input`` out of ``_dispatch``.

    ``_dispatch`` deliberately leaves the ToolError uncaught — the SDK's
    ``@call_tool`` handler in :func:`server.build_server` is what
    translates it into the structured MCP error response. Asserting on
    the raised error here is the most direct way to verify the boundary
    contract without driving the SDK's request loop.
    """

    index = _build_index(fake_export)
    server_module.build_server(index)

    with pytest.raises(ToolError) as excinfo:
        server_module._dispatch(index, "search_likes", {"query": "   "})

    assert excinfo.value.category == "invalid_input"
    assert "query" in excinfo.value.message


def test_dispatch_search_likes_blank_query_translates_via_call_tool_wrapper(
    fake_export: Config,
) -> None:
    """The same blank-query path, observed through the SDK wrapper that
    :func:`server.build_server` registers via ``@server.call_tool()``.

    The ``call_tool`` decorator stores the wrapped handler on the
    server's ``request_handlers`` map keyed by
    :class:`mcp.types.CallToolRequest`. The wrapper catches the
    :class:`ToolError` and returns an MCP error result. We assert the
    structured payload carries the documented ``"category":
    "invalid_input"`` shape.
    """

    import asyncio

    import mcp.types as mcp_types

    index = _build_index(fake_export)
    server = server_module.build_server(index)

    # The SDK exposes registered request handlers via Server.request_handlers.
    # CallToolRequest is the key.
    handler = server.request_handlers[mcp_types.CallToolRequest]

    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name="search_likes", arguments={"query": "   "}
        ),
    )

    # The SDK wraps results in ServerResult; unwrap to inspect.
    server_result = asyncio.run(handler(request))
    inner = server_result.root

    assert isinstance(inner, mcp_types.CallToolResult)
    assert inner.isError is True
    assert inner.structuredContent == {
        "error": {
            "category": "invalid_input",
            "message": inner.structuredContent["error"]["message"],
        }
    }
    assert "query" in inner.structuredContent["error"]["message"]


def test_dispatch_search_likes_both_retrievals_down_becomes_upstream_failure(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both retrieval paths in ``index.search`` fail (dense via the
    embedder seam, BM25 via :meth:`BM25Index.top_k`), the resulting
    :class:`EmbeddingError` propagates out of ``_dispatch`` as a
    :class:`ToolError` with category ``upstream_failure`` — the
    translation :func:`tools.search_likes` performs (req 7.6).
    """

    def dense_boom(self, query):
        raise RuntimeError("embeddings API down")

    def bm25_boom(self, query, k=200, restrict_to_ids=None):
        raise RuntimeError("bm25 down")

    monkeypatch.setattr(
        "x_likes_mcp.embeddings.Embedder.embed_query", dense_boom
    )
    monkeypatch.setattr(
        "x_likes_mcp.bm25.BM25Index.top_k", bm25_boom
    )

    index = _build_index(fake_export)
    server_module.build_server(index)

    with pytest.raises(ToolError) as excinfo:
        server_module._dispatch(index, "search_likes", {"query": "anything"})

    assert excinfo.value.category == "upstream_failure"
    assert "both retrieval paths failed" in excinfo.value.message


def test_dispatch_list_months_succeeds_after_search_failure(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a simulated dual-retrieval failure, a fresh ``list_months``
    dispatch against the same in-process server still succeeds. The
    server stays alive across tool errors.
    """

    def dense_boom(self, query):
        raise RuntimeError("embeddings API down")

    def bm25_boom(self, query, k=200, restrict_to_ids=None):
        raise RuntimeError("bm25 down")

    monkeypatch.setattr(
        "x_likes_mcp.embeddings.Embedder.embed_query", dense_boom
    )
    monkeypatch.setattr(
        "x_likes_mcp.bm25.BM25Index.top_k", bm25_boom
    )

    index = _build_index(fake_export)
    server_module.build_server(index)

    # First call raises (both retrievals are down).
    with pytest.raises(ToolError):
        server_module._dispatch(index, "search_likes", {"query": "any"})

    # Second call to a non-search tool still works.
    payload = server_module._dispatch(index, "list_months", {})

    assert isinstance(payload, dict)
    assert "months" in payload
    months = payload["months"]
    assert isinstance(months, list)
    # Three months in the fake_export fixture.
    assert len(months) == 3
    year_months = {m["year_month"] for m in months}
    assert year_months == {"2025-01", "2025-02", "2025-03"}
    # Each month entry carries the documented keys.
    for entry in months:
        assert set(entry.keys()) == {"year_month", "path", "tweet_count"}


def test_dispatch_unknown_tool_name_raises_invalid_input(
    fake_export: Config,
) -> None:
    """``_dispatch`` defends against unknown tool names with an
    ``invalid_input`` :class:`ToolError`. The SDK's schema layer would
    normally filter these out, but the boundary check protects against
    direct in-process callers.
    """

    index = _build_index(fake_export)
    server_module.build_server(index)

    with pytest.raises(ToolError) as excinfo:
        server_module._dispatch(index, "no_such_tool", {})

    assert excinfo.value.category == "invalid_input"
    assert "no_such_tool" in excinfo.value.message
