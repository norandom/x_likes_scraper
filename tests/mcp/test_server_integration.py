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

The walker is the only LLM call site in the package; tests that need to
drive ``search_likes`` end to end stub :func:`x_likes_mcp.walker.walk`
with :func:`pytest.MonkeyPatch.setattr`. The autouse ``_block_real_llm``
fixture in ``tests/mcp/conftest.py`` is an additional safety net at the
chat-completions seam.

Boundary: tests/mcp/test_server_integration.py only.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from x_likes_mcp import server as server_module
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.errors import ToolError
from x_likes_mcp.index import TweetIndex
from x_likes_mcp.walker import WalkerError, WalkerHit


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
    as optional with the documented patterns/bounds.
    """

    tools = server_module._build_tool_definitions()
    by_name = {t.name: t for t in tools}
    search = by_name["search_likes"]
    schema = search.inputSchema

    # query is required; year/month_start/month_end are NOT.
    assert schema.get("required") == ["query"]

    props = schema["properties"]
    assert "year" in props
    assert "month_start" in props
    assert "month_end" in props

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
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stubbed walker run flows through ``_dispatch`` and produces the
    documented ``{"results": [hit, ...]}`` structured payload, with each
    hit carrying the eight documented keys.
    """

    def fake_walk(tree, query, months_in_scope, config, chunk_size=30):
        # Return one hit pointing at a real fixture tweet so the tools
        # layer has metadata to shape into the response.
        return [WalkerHit(tweet_id="1001", relevance=0.9, why="thematic match")]

    monkeypatch.setattr("x_likes_mcp.walker.walk", fake_walk)

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
    assert len(results) == 1

    hit = results[0]
    assert set(hit.keys()) == {
        "tweet_id",
        "year_month",
        "handle",
        "snippet",
        "score",
        "walker_relevance",
        "why",
        "feature_breakdown",
    }
    assert hit["tweet_id"] == "1001"
    assert hit["walker_relevance"] == pytest.approx(0.9)
    # The walker's `why` is preserved (truncated to 240 chars upstream).
    assert hit["why"] == "thematic match"
    # year_month is resolved from the real Tweet's created_at.
    assert hit["year_month"] == "2025-01"


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


def test_dispatch_search_likes_walker_failure_becomes_upstream_failure_tool_error(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A :class:`WalkerError` raised by the stubbed walker propagates out
    of ``_dispatch`` as a :class:`ToolError` with category
    ``upstream_failure`` — this is the translation the
    :func:`tools.search_likes` layer performs.
    """

    def boom(tree, query, months_in_scope, config, chunk_size=30):
        raise WalkerError("LLM down")

    monkeypatch.setattr("x_likes_mcp.walker.walk", boom)

    index = _build_index(fake_export)
    server_module.build_server(index)

    with pytest.raises(ToolError) as excinfo:
        server_module._dispatch(index, "search_likes", {"query": "anything"})

    assert excinfo.value.category == "upstream_failure"
    assert "LLM down" in excinfo.value.message


def test_dispatch_list_months_succeeds_after_walker_failure(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a simulated walker failure, a fresh ``list_months`` dispatch
    against the same in-process server still succeeds. The server does
    not crash on the prior failure.
    """

    def boom(tree, query, months_in_scope, config, chunk_size=30):
        raise WalkerError("transient")

    monkeypatch.setattr("x_likes_mcp.walker.walk", boom)

    index = _build_index(fake_export)
    server_module.build_server(index)

    # First call raises (walker is down).
    with pytest.raises(ToolError):
        server_module._dispatch(index, "search_likes", {"query": "any"})

    # Second call to a non-walker tool still works.
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
