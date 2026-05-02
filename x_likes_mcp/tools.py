"""Four MCP tool handlers: ``search_likes``, ``list_months``, ``get_month``,
``read_tweet``.

Each handler is thin: validate input, call into :class:`TweetIndex`, shape
the response. All input-shape errors raise :class:`ToolError` instances built
through the :mod:`x_likes_mcp.errors` factories. Failures from the walker (or
any other non-``ToolError`` exception bubbling out of ``index.search``) are
re-shaped as ``upstream_failure`` so the server boundary can return a clean
error response without crashing.

Boundary: this module imports only from ``index``, ``errors``, and stdlib /
``re``. It does not import the OpenAI SDK or perform I/O directly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from . import errors

if TYPE_CHECKING:  # pragma: no cover
    from .index import TweetIndex


# ---------------------------------------------------------------------------
# Constants

# Snippets and `why` strings are truncated to the same length the walker uses.
_SNIPPET_MAX_CHARS = 240

# Maximum number of results returned to the MCP client. Mirrors
# TweetIndex.search's default top_n.
_DEFAULT_TOP_N = 50

_MONTH_RE = re.compile(r"^(0[1-9]|1[0-2])$")
_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_TWEET_ID_RE = re.compile(r"^\d+$")

# Year bounds: Twitter launched in 2006; upper bound is loose (the server
# schema layer in server.py pins the upper bound to the current year).
_YEAR_MIN = 2006
_YEAR_MAX = 9999


# ---------------------------------------------------------------------------
# Internal helpers

def _truncate(text: str, max_chars: int) -> str:
    """Return ``text`` truncated to ``max_chars`` characters."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _resolve_year_month(index: TweetIndex, tweet_id: str) -> str:
    """Best-effort ``YYYY-MM`` for a tweet.

    Prefers ``Tweet.get_created_datetime().strftime('%Y-%m')``. Falls back to
    the matching ``TreeNode.year_month`` when ``created_at`` is unparseable.
    Returns ``"unknown"`` when neither source resolves.
    """
    tweet = index.tweets_by_id.get(tweet_id)
    if tweet is not None:
        try:
            return tweet.get_created_datetime().strftime("%Y-%m")
        except (ValueError, TypeError):
            pass

    node = index.tree.nodes_by_id.get(tweet_id)
    if node is not None:
        return node.year_month

    return "unknown"


def _tweet_url(tweet: Any) -> str:
    """Canonical URL for a tweet; falls back to the i/status path when the
    handle is empty (older anonymized rows)."""
    handle = tweet.user.screen_name if tweet.user is not None else ""
    if handle:
        return f"https://x.com/{handle}/status/{tweet.id}"
    return f"https://x.com/i/status/{tweet.id}"


# ---------------------------------------------------------------------------
# search_likes

def _validate_query(query: Any) -> str:
    """Strip and validate the query string. Returns the cleaned string."""
    if not isinstance(query, str):
        raise errors.invalid_input("query", "must be a string")
    stripped = query.strip()
    if not stripped:
        raise errors.invalid_input("query", "must be non-empty")
    return stripped


def _validate_filter(
    year: Any,
    month_start: Any,
    month_end: Any,
) -> None:
    """Validate the structured-filter shape. Raises ``invalid_input`` on any
    rule violation; returns nothing on success.

    The resolver enforces these rules too. We check here so the error message
    keeps "filter" as the field name even when the caller hands us a mock
    that bypasses the resolver.
    """
    if year is not None:
        if isinstance(year, bool) or not isinstance(year, int):
            raise errors.invalid_input("filter", "year must be an integer")
        if year < _YEAR_MIN or year > _YEAR_MAX:
            raise errors.invalid_input(
                "filter", f"year must be in {_YEAR_MIN}..{_YEAR_MAX}"
            )

    if month_start is not None and (
        not isinstance(month_start, str) or not _MONTH_RE.match(month_start)
    ):
        raise errors.invalid_input(
            "filter", "month_start must match ^(0[1-9]|1[0-2])$"
        )

    if month_end is not None and (
        not isinstance(month_end, str) or not _MONTH_RE.match(month_end)
    ):
        raise errors.invalid_input(
            "filter", "month_end must match ^(0[1-9]|1[0-2])$"
        )

    if month_start is not None and year is None:
        raise errors.invalid_input("filter", "month_start requires year")
    if month_end is not None and month_start is None:
        raise errors.invalid_input("filter", "month_end requires month_start")


def _shape_hit(index: TweetIndex, hit: Any) -> dict[str, Any]:
    """Convert a ``ScoredHit`` plus the loaded tweet/tree to the dict shape
    the MCP client receives."""
    tweet = index.tweets_by_id.get(hit.tweet_id)
    node = index.tree.nodes_by_id.get(hit.tweet_id)

    if tweet is not None:
        handle = tweet.user.screen_name if tweet.user is not None else ""
        snippet = _truncate(tweet.text or "", _SNIPPET_MAX_CHARS)
    elif node is not None:
        handle = node.handle
        snippet = _truncate(node.text or "", _SNIPPET_MAX_CHARS)
    else:
        handle = ""
        snippet = ""

    return {
        "tweet_id": hit.tweet_id,
        "year_month": _resolve_year_month(index, hit.tweet_id),
        "handle": handle,
        "snippet": snippet,
        "score": hit.score,
        "walker_relevance": hit.walker_relevance,
        "why": hit.why,
        "feature_breakdown": dict(hit.feature_breakdown),
    }


def search_likes(
    index: TweetIndex,
    query: str,
    year: int | None = None,
    month_start: str | None = None,
    month_end: str | None = None,
) -> list[dict[str, Any]]:
    """Return up to N ranked search hits for ``query``.

    Validates ``query`` and the structured filter; translates resolver
    ``ValueError``\\ s to ``invalid_input`` and any non-``ToolError``
    exception (notably :class:`x_likes_mcp.walker.WalkerError`) to
    ``upstream_failure``.

    Returns:
        List of dicts shaped
        ``{"tweet_id", "year_month", "handle", "snippet", "score",
        "walker_relevance", "why", "feature_breakdown"}``.
    """
    stripped = _validate_query(query)
    _validate_filter(year, month_start, month_end)

    try:
        scored = index.search(stripped, year, month_start, month_end)
    except errors.ToolError:
        raise
    except ValueError as exc:
        raise errors.invalid_input("filter", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — boundary translation is the point
        raise errors.upstream_failure(str(exc)) from exc

    return [_shape_hit(index, hit) for hit in scored[:_DEFAULT_TOP_N]]


# ---------------------------------------------------------------------------
# list_months

def list_months(index: TweetIndex) -> list[dict[str, Any]]:
    """Return per-month metadata in whatever order ``TweetIndex.list_months``
    produced (reverse-chronological by current implementation).

    Each dict is ``{"year_month", "path", "tweet_count"}``. ``tweet_count``
    may be ``None`` for months whose tweet ``created_at`` did not parse.
    """
    return [
        {
            "year_month": info.year_month,
            "path": str(info.path),
            "tweet_count": info.tweet_count,
        }
        for info in index.list_months()
    ]


# ---------------------------------------------------------------------------
# get_month

def get_month(index: TweetIndex, year_month: str) -> str:
    """Return the raw Markdown for one month.

    Raises ``invalid_input`` when ``year_month`` does not match
    ``^\\d{4}-\\d{2}$``; raises ``not_found`` when the file is missing.
    """
    if not isinstance(year_month, str) or not _YEAR_MONTH_RE.match(year_month):
        raise errors.invalid_input(
            "year_month", "must match ^\\d{4}-\\d{2}$"
        )

    markdown = index.get_month_markdown(year_month)
    if markdown is None:
        raise errors.not_found("month", year_month)
    return markdown


# ---------------------------------------------------------------------------
# read_tweet

def read_tweet(index: TweetIndex, tweet_id: str) -> dict[str, Any]:
    """Return the metadata for one tweet by id.

    Raises ``invalid_input`` when ``tweet_id`` is empty or non-numeric;
    raises ``not_found`` when the id is unknown to the loaded export.

    Output dict keys: ``tweet_id``, ``handle``, ``display_name``, ``text``,
    ``created_at``, ``view_count``, ``like_count``, ``retweet_count``,
    ``url``. Fields with falsy values (e.g. ``view_count == 0``) are omitted.
    """
    if not isinstance(tweet_id, str) or not tweet_id:
        raise errors.invalid_input("tweet_id", "must be a non-empty string")
    if not _TWEET_ID_RE.match(tweet_id):
        raise errors.invalid_input("tweet_id", "must match ^\\d+$")

    tweet = index.lookup_tweet(tweet_id)
    if tweet is None:
        raise errors.not_found("tweet", tweet_id)

    user = tweet.user
    handle = user.screen_name if user is not None else ""
    display_name = user.name if user is not None else ""

    candidates: list[tuple[str, Any]] = [
        ("tweet_id", tweet.id),
        ("handle", handle),
        ("display_name", display_name),
        ("text", tweet.text),
        ("created_at", tweet.created_at),
        ("view_count", tweet.view_count),
        ("like_count", tweet.favorite_count),
        ("retweet_count", tweet.retweet_count),
        ("url", _tweet_url(tweet)),
    ]
    return {key: value for key, value in candidates if value}
