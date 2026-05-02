"""MCP server wiring: build the SDK :class:`Server`, register the four
tools with their JSON schemas, and run the stdio transport loop.

Boundary: imports the ``mcp`` SDK, the tool handlers from :mod:`tools`,
the :class:`ToolError` shape from :mod:`errors`, and the package version
from :mod:`__init__`. No direct OpenAI SDK use; no filesystem I/O beyond
what the tool handlers do via :class:`TweetIndex`.

The boundary error wrapper is per-tool: :class:`ToolError` raised by a
handler becomes a structured-content MCP error response carrying the
stable ``category`` and ``message`` strings; any other exception is
logged to ``stderr`` (full traceback) and reshaped as a generic
``upstream_failure`` so the process stays alive across bad tool calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import __version__, tools
from .errors import ToolError

if TYPE_CHECKING:  # pragma: no cover
    from .index import TweetIndex


# ---------------------------------------------------------------------------
# JSON schemas

# Matches ScoredHit shape produced by tools.search_likes.
_SEARCH_HIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tweet_id": {"type": "string"},
        "year_month": {"type": "string"},
        "handle": {"type": "string"},
        "snippet": {"type": "string"},
        "score": {"type": "number"},
        "walker_relevance": {"type": "number"},
        "why": {"type": "string"},
        "feature_breakdown": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "required": ["tweet_id", "score", "walker_relevance"],
    "additionalProperties": True,
}

_MONTH_INFO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "year_month": {"type": "string"},
        "path": {"type": "string"},
        "tweet_count": {"type": ["integer", "null"]},
    },
    "required": ["year_month", "path"],
    "additionalProperties": False,
}

_TWEET_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tweet_id": {"type": "string"},
        "handle": {"type": "string"},
        "display_name": {"type": "string"},
        "text": {"type": "string"},
        "created_at": {"type": "string"},
        "view_count": {"type": "integer"},
        "like_count": {"type": "integer"},
        "retweet_count": {"type": "integer"},
        "url": {"type": "string"},
    },
    "required": ["tweet_id"],
    "additionalProperties": True,
}


def _current_year() -> int:
    """Year used as the upper bound for the ``year`` filter at startup."""
    return datetime.now(timezone.utc).year


def _build_tool_definitions() -> list[mcp_types.Tool]:
    """Return the four tool definitions with input/output JSON schemas.

    The ``year`` upper bound is the current year at server-startup time
    (Requirement 6.8). The patterns match the spec exactly.
    """
    year_max = _current_year()

    search_input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Natural-language search prompt.",
            },
            "year": {
                "type": "integer",
                "minimum": 2006,
                "maximum": year_max,
                "description": "Optional year filter (X launched in 2006).",
            },
            "month_start": {
                "type": "string",
                "pattern": "^(0[1-9]|1[0-2])$",
                "description": "Optional zero-padded month (01..12). Requires year.",
            },
            "month_end": {
                "type": "string",
                "pattern": "^(0[1-9]|1[0-2])$",
                "description": "Optional zero-padded month (01..12). Requires month_start.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    search_output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": _SEARCH_HIT_SCHEMA,
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    }

    list_months_input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    list_months_output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "months": {
                "type": "array",
                "items": _MONTH_INFO_SCHEMA,
            },
        },
        "required": ["months"],
        "additionalProperties": False,
    }

    get_month_input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "year_month": {
                "type": "string",
                "pattern": r"^\d{4}-\d{2}$",
                "description": "Month identifier in YYYY-MM form.",
            },
        },
        "required": ["year_month"],
        "additionalProperties": False,
    }

    get_month_output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "markdown": {"type": "string"},
        },
        "required": ["markdown"],
        "additionalProperties": False,
    }

    read_tweet_input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "tweet_id": {
                "type": "string",
                "pattern": r"^\d+$",
                "description": "Numeric tweet id as a string.",
            },
        },
        "required": ["tweet_id"],
        "additionalProperties": False,
    }

    return [
        mcp_types.Tool(
            name="search_likes",
            description=(
                "Search the user's liked tweets by natural-language query. "
                "Optional year/month_start/month_end pre-filter narrows the "
                "set of months the walker LLM looks at."
            ),
            inputSchema=search_input_schema,
            outputSchema=search_output_schema,
        ),
        mcp_types.Tool(
            name="list_months",
            description=(
                "List the months for which per-month Markdown exists, "
                "reverse-chronologically, with tweet counts when available."
            ),
            inputSchema=list_months_input_schema,
            outputSchema=list_months_output_schema,
        ),
        mcp_types.Tool(
            name="get_month",
            description="Return the raw per-month Markdown for one YYYY-MM.",
            inputSchema=get_month_input_schema,
            outputSchema=get_month_output_schema,
        ),
        mcp_types.Tool(
            name="read_tweet",
            description="Return metadata for one tweet by numeric id.",
            inputSchema=read_tweet_input_schema,
            outputSchema=_TWEET_METADATA_SCHEMA,
        ),
    ]


# ---------------------------------------------------------------------------
# Boundary error wrapper

def _error_payload(category: str, message: str) -> dict[str, Any]:
    """Shape an MCP-error payload carrying our stable category/message."""
    return {"error": {"category": category, "message": message}}


def _error_result(category: str, message: str) -> mcp_types.CallToolResult:
    """Build a ``CallToolResult`` with ``isError=True`` carrying the
    structured error payload plus a JSON text content for clients that
    only read ``content``."""
    payload = _error_payload(category, message)
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=True,
    )


def _success_result(structured: dict[str, Any]) -> mcp_types.CallToolResult:
    """Build a successful ``CallToolResult`` with structured + text content."""
    return mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(
                type="text", text=json.dumps(structured, indent=2)
            )
        ],
        structuredContent=structured,
        isError=False,
    )


# ---------------------------------------------------------------------------
# Tool dispatch

def _dispatch(
    index: TweetIndex, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Call into the matching tools handler and shape the structured output.

    Raises ``ToolError`` for invalid-input / not-found / upstream-failure
    cases. Other exceptions propagate to the caller which logs and reshapes.
    """
    if name == "search_likes":
        results = tools.search_likes(
            index,
            query=arguments.get("query", ""),
            year=arguments.get("year"),
            month_start=arguments.get("month_start"),
            month_end=arguments.get("month_end"),
        )
        return {"results": results}

    if name == "list_months":
        return {"months": tools.list_months(index)}

    if name == "get_month":
        markdown = tools.get_month(index, arguments.get("year_month", ""))
        return {"markdown": markdown}

    if name == "read_tweet":
        return tools.read_tweet(index, arguments.get("tweet_id", ""))

    # Unknown tool name. Should be filtered by the SDK at the schema layer,
    # but defend anyway.
    raise ToolError("invalid_input", f"unknown tool: {name}")


# ---------------------------------------------------------------------------
# Public API

def build_server(index: TweetIndex) -> Server:
    """Construct the MCP :class:`Server`, register the four tools, and
    return the server instance.

    Tools:
        - ``search_likes``: ranked search over liked tweets.
        - ``list_months``: enumerate per-month markdown files.
        - ``get_month``: read a single month's markdown.
        - ``read_tweet``: read one tweet's metadata by id.
    """
    server: Server = Server(name="x-likes-mcp", version=__version__)
    tool_defs = _build_tool_definitions()

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:  # pragma: no cover (driven by SDK)
        return tool_defs

    @server.call_tool()
    async def _call_tool(  # pragma: no cover (driven by SDK)
        name: str, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        try:
            structured = _dispatch(index, name, arguments)
        except ToolError as exc:
            return _error_result(exc.category, exc.message)
        except Exception:  # noqa: BLE001 — boundary translation
            traceback.print_exc(file=sys.stderr)
            return _error_result(
                "upstream_failure",
                "internal error; see server logs",
            )
        return _success_result(structured)

    return server


async def _run_async(index: TweetIndex) -> None:
    """Run the SDK stdio loop until the client disconnects."""
    server = build_server(index)
    init_options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def run(index: TweetIndex) -> None:
    """Drive the stdio loop synchronously (entry from ``__main__``).

    Wraps :func:`_run_async` with :func:`asyncio.run`. Returns when the
    client disconnects (``stdio_server`` exits its context).
    """
    asyncio.run(_run_async(index))
