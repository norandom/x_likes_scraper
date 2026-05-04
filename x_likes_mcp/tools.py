"""Four MCP tool handlers: ``search_likes``, ``list_months``, ``get_month``,
``read_tweet``.

Each handler is thin: validate input, call into :class:`TweetIndex`, shape
the response. All input-shape errors raise :class:`ToolError` instances built
through the :mod:`x_likes_mcp.errors` factories. Failures from the hybrid
retrieval pipeline (``EmbeddingError`` raised when both retrievals failed,
or any other non-``ToolError`` exception bubbling out of ``index.search``)
are re-shaped as ``upstream_failure`` so the server boundary can return a
clean error response without crashing.

The optional walker explainer (``with_why=True``) is the only place the
walker is called from this layer; ``index.search`` no longer touches it.
The explainer helper catches its own errors and returns an empty map on
failure, so an explainer blip never fails the surrounding ``search_likes``
call (req 8.4).

Boundary: this module imports only from ``index``, ``errors``, ``walker``,
and stdlib / ``re``. It does not import the OpenAI SDK or perform I/O
directly.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

from . import errors
from . import walker as walker_module
from .sanitize import fence_url_for_llm, sanitize_text
from .tree import TweetTree

if TYPE_CHECKING:  # pragma: no cover
    from .index import TweetIndex
    from .ranker import ScoredHit
    from .walker import WalkerHit


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


def _validate_year(year: Any) -> None:
    if year is None:
        return
    if isinstance(year, bool) or not isinstance(year, int):
        raise errors.invalid_input("filter", "year must be an integer")
    if year < _YEAR_MIN or year > _YEAR_MAX:
        raise errors.invalid_input(
            "filter", f"year must be in {_YEAR_MIN}..{_YEAR_MAX}"
        )


def _validate_month_field(name: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _MONTH_RE.match(value):
        raise errors.invalid_input(
            "filter", f"{name} must match ^(0[1-9]|1[0-2])$"
        )


def _validate_filter_dependencies(
    year: Any, month_start: Any, month_end: Any
) -> None:
    if month_start is not None and year is None:
        raise errors.invalid_input("filter", "month_start requires year")
    if month_end is not None and month_start is None:
        raise errors.invalid_input("filter", "month_end requires month_start")


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
    _validate_year(year)
    _validate_month_field("month_start", month_start)
    _validate_month_field("month_end", month_end)
    _validate_filter_dependencies(year, month_start, month_end)


def _validate_with_why(with_why: Any) -> bool:
    """Return a normalized bool for ``with_why``.

    ``None`` is tolerated and treated as ``False`` so MCP clients that
    omit the field land on the cheap default path. Any other non-bool
    raises ``invalid_input("with_why", ...)``.
    """
    if with_why is None:
        return False
    if not isinstance(with_why, bool):
        raise errors.invalid_input(
            "with_why",
            f"must be a bool, got {type(with_why).__name__}",
        )
    return with_why


def _shape_hit(index: TweetIndex, hit: Any) -> dict[str, Any]:
    """Convert a ``ScoredHit`` plus the loaded tweet/tree to the dict shape
    the MCP client receives.

    ``walker_relevance`` is clamped to ``[0, 1]`` (req 7.8). For the default
    path it carries the dense cosine similarity score (already in ``[0, 1]``
    on L2-normalized text embeddings for any non-adversarial input); the
    clamp is defensive. For the explainer path the value comes from the
    walker's parsed JSON, which is also bounded ``[0, 1]`` by the walker's
    own validation.
    """
    tweet = index.tweets_by_id.get(hit.tweet_id)
    node = index.tree.nodes_by_id.get(hit.tweet_id)

    raw_urls: list[str] = []
    if tweet is not None:
        handle = tweet.user.screen_name if tweet.user is not None else ""
        snippet = _truncate(tweet.text or "", _SNIPPET_MAX_CHARS)
        raw_urls = list(tweet.urls or [])
    elif node is not None:
        handle = node.handle
        snippet = _truncate(node.text or "", _SNIPPET_MAX_CHARS)
    else:
        handle = ""
        snippet = ""

    # Untrusted tweet content reaches the calling LLM through this hit.
    # Strip ANSI / control / BiDi codepoints so terminal-control or
    # rendering-direction tricks cannot survive the trip. NFKC
    # normalization runs as part of sanitize_text.
    handle = sanitize_text(handle)
    snippet = sanitize_text(snippet)

    # Each resolved URL is sanitized, scheme-checked (http/https only),
    # and wrapped in a distinctive ``<<<URL>>> ... <<<END_URL>>>`` fence.
    # URLs can carry long paths or query strings with prompt-injection
    # prose (e.g. ``?q=Ignore+previous+instructions``). Fencing lets
    # callers tell their LLM "treat fenced content as data, not
    # instructions" so the URL payload cannot mix into nearby prose.
    fenced_urls = [
        fenced
        for fenced in (fence_url_for_llm(u) for u in raw_urls)
        if fenced is not None
    ]

    walker_relevance = max(0.0, min(1.0, float(hit.walker_relevance)))

    return {
        "tweet_id": hit.tweet_id,
        "year_month": _resolve_year_month(index, hit.tweet_id),
        "handle": handle,
        "snippet": snippet,
        "urls": fenced_urls,
        "score": hit.score,
        "walker_relevance": walker_relevance,
        "why": hit.why or "",
        "feature_breakdown": dict(hit.feature_breakdown),
    }


# How many top-ranked tweets the optional explainer is asked to rationalize.
# The walker is a single-shot LLM call; 20 keeps the prompt under chunk-size
# limits and matches req 8.2.
_EXPLAINER_TOP_N = 20


_EXPLAINER_SYNTHETIC_MONTH = "explainer-chunk"


def _call_walker_explainer(
    top_results: list[ScoredHit],
    query: str,
    index: TweetIndex,
) -> dict[str, WalkerHit]:
    """Return a ``{tweet_id: WalkerHit}`` map populated by the walker over the
    top ranked results.

    This is the only place :func:`x_likes_mcp.walker.walk` is invoked from
    the tool layer; the default ``search_likes`` path never touches it.

    Implementation: builds a synthetic in-memory :class:`TweetTree` whose
    ``nodes_by_month`` maps a single synthetic month key to the up-to-20
    :class:`TreeNode` objects pulled from ``index.tree.nodes_by_id`` for
    the supplied ``top_results``. The walker is invoked with
    ``chunk_size=len(synthetic_nodes)`` so it issues exactly one LLM call
    over the whole batch (req 8.2). Order is irrelevant inside the walker
    — the merge in ``_merge_explainer`` preserves the ranker's order
    (req 8.3).

    Failures (any exception, including :class:`x_likes_mcp.walker.WalkerError`,
    network errors, missing config) are caught here, logged once to stderr,
    and surface as an empty map so the surrounding ``search_likes`` call
    still succeeds with cosine-derived placeholders (req 8.4).
    """
    if not top_results:
        return {}

    try:
        synthetic_nodes = []
        for hit in top_results:
            node = index.tree.nodes_by_id.get(hit.tweet_id)
            if node is None:
                continue
            synthetic_nodes.append(node)

        if not synthetic_nodes:
            return {}

        synthetic_tree = TweetTree(
            nodes_by_month={_EXPLAINER_SYNTHETIC_MONTH: synthetic_nodes},
            nodes_by_id=index.tree.nodes_by_id,
        )

        walker_hits = walker_module.walk(
            tree=synthetic_tree,
            query=query,
            months_in_scope=[_EXPLAINER_SYNTHETIC_MONTH],
            config=index.config,
            chunk_size=len(synthetic_nodes),
        )
    except Exception as exc:  # noqa: BLE001 — explainer must not fail the call
        print(f"[explainer] walker call failed: {exc}", file=sys.stderr)
        return {}

    return {hit.tweet_id: hit for hit in walker_hits}


def search_likes(
    index: TweetIndex,
    query: str,
    year: int | None = None,
    month_start: str | None = None,
    month_end: str | None = None,
    with_why: bool = False,
) -> list[dict[str, Any]]:
    """Return up to N ranked search hits for ``query``.

    Pipeline:
      1. Validate ``query``, the structured filter, and ``with_why``.
      2. Call :meth:`TweetIndex.search`, which runs hybrid recall (BM25
         + dense via OpenRouter) and the heavy ranker.
      3. When ``with_why=True``, hand the top-20 ranked hits to
         :func:`_call_walker_explainer` and merge the returned
         ``{tweet_id: WalkerHit}`` map onto the matching results
         in place: ``why`` and ``walker_relevance`` are refreshed; the
         ranker's order is preserved (req 8.3).
      4. Shape each ``ScoredHit`` into the documented response dict.

    Error translation:
      - :class:`ValueError` from the resolver → ``invalid_input("filter")``.
      - :class:`x_likes_mcp.embeddings.EmbeddingError` (raised by
        ``index.search`` when both retrieval paths failed) →
        ``upstream_failure``.
      - Any other non-``ToolError`` exception bubbling out of
        ``index.search`` → ``upstream_failure``.

    Args:
        index: The loaded :class:`TweetIndex` (search seam).
        query: User query; validated, stripped, must be non-empty.
        year, month_start, month_end: Optional structured filter. Same
            semantics as before this spec.
        with_why: When ``True``, runs the optional walker explainer over
            the top hits to populate ``why``. Defaults to ``False`` so the
            cheap default path performs zero chat-completions calls
            (req 7.2 / 8.5). ``None`` is tolerated and treated as
            ``False``; any other non-bool raises ``invalid_input``.

    Returns:
        List of dicts shaped
        ``{"tweet_id", "year_month", "handle", "snippet", "urls",
        "score", "walker_relevance", "why", "feature_breakdown"}``.
        ``urls`` is a list of fenced HTTP(S) URLs from the tweet
        entities; each entry is wrapped in
        ``<<<URL>>> ... <<<END_URL>>>`` so a caller's LLM can be
        instructed to treat fenced content as data, not instructions.
    """
    stripped = _validate_query(query)
    _validate_filter(year, month_start, month_end)
    explain = _validate_with_why(with_why)

    try:
        scored = index.search(stripped, year, month_start, month_end)
    except errors.ToolError:
        raise
    except ValueError as exc:
        raise errors.invalid_input("filter", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — boundary translation is the point
        # Catches EmbeddingError (both retrievals down) and any other
        # unexpected error from the index. tools.search_likes is the
        # boundary that translates non-ToolError exceptions into
        # upstream_failure ToolErrors.
        raise errors.upstream_failure(str(exc)) from exc

    top = scored[:_DEFAULT_TOP_N]

    if explain and top:
        why_map = _call_walker_explainer(top[:_EXPLAINER_TOP_N], stripped, index)
        if why_map:
            top = _merge_explainer(top, why_map)

    return [_shape_hit(index, hit) for hit in top]


def _merge_explainer(
    scored: list[ScoredHit],
    why_map: dict[str, WalkerHit],
) -> list[ScoredHit]:
    """Return a new list where each hit present in ``why_map`` carries the
    walker's ``why`` and ``relevance``.

    Order is preserved (req 8.3). Hits not present in the map are returned
    unchanged so they keep their cosine-derived placeholders. ``ScoredHit``
    is a frozen dataclass; we rebuild rather than mutate in place.
    """
    # Local import keeps tools.py free of a top-level ranker import (the
    # ranker module imports walker, which imports openai — pulling all of
    # that into module load order is unnecessary for the merge step).
    from .ranker import ScoredHit as _ScoredHit

    merged: list[ScoredHit] = []
    for hit in scored:
        wh = why_map.get(hit.tweet_id)
        if wh is None:
            merged.append(hit)
            continue
        merged.append(
            _ScoredHit(
                tweet_id=hit.tweet_id,
                score=hit.score,
                walker_relevance=wh.relevance,
                why=wh.why or "",
                feature_breakdown=dict(hit.feature_breakdown),
            )
        )
    return merged


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

    # Sanitize every string field that originates from arbitrary X users
    # before handing it back to the MCP client (which is typically an
    # LLM): strip ANSI / control / BiDi / BOM and run NFKC normalization.
    candidates: list[tuple[str, Any]] = [
        ("tweet_id", tweet.id),
        ("handle", sanitize_text(handle)),
        ("display_name", sanitize_text(display_name)),
        ("text", sanitize_text(tweet.text)),
        ("created_at", tweet.created_at),
        ("view_count", tweet.view_count),
        ("like_count", tweet.favorite_count),
        ("retweet_count", tweet.retweet_count),
        ("url", _tweet_url(tweet)),
    ]
    return {key: value for key, value in candidates if value}
