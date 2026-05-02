"""Tests for :mod:`x_likes_mcp.tools`.

Covers the four MCP tool handlers (:func:`search_likes`, :func:`list_months`,
:func:`get_month`, :func:`read_tweet`) end to end at the boundary level. The
:class:`TweetIndex` collaborator is replaced with a :class:`MagicMock` for
nearly every test so the tests stay focused on input validation, error
translation, and response shaping. Two of the ``search_likes`` happy-path
tests use real :class:`~x_likes_exporter.models.Tweet` objects loaded from
``tests/mcp/fixtures/likes.json`` so the handle / snippet / year_month
shaping path is exercised against real data.

Boundary: tests/mcp/test_tools.py only. We do not import the OpenAI SDK and
we never reach into the index's internal collaborators (walker, ranker).
The autouse ``_block_real_llm`` guard from ``conftest.py`` covers the case
where a misconfigured test path would accidentally drive the walker; none
of these tests touch the walker.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from x_likes_exporter import load_export
from x_likes_mcp import tools
from x_likes_mcp.errors import ToolError
from x_likes_mcp.index import MonthInfo
from x_likes_mcp.ranker import ScoredHit
from x_likes_mcp.tree import TreeNode, TweetTree
from x_likes_mcp.walker import WalkerError


FIXTURES_DIR = Path(__file__).parent / "fixtures"
LIKES_JSON = FIXTURES_DIR / "likes.json"


# ---------------------------------------------------------------------------
# Helpers


def _make_index_mock(
    *,
    tweets_by_id: dict[str, object] | None = None,
    nodes_by_id: dict[str, TreeNode] | None = None,
) -> MagicMock:
    """Build a :class:`MagicMock` standing in for a :class:`TweetIndex`.

    ``tweets_by_id`` and ``nodes_by_id`` are wired onto the documented
    public attributes so :mod:`x_likes_mcp.tools` can read them when
    shaping responses.
    """

    idx = MagicMock(name="TweetIndex")
    idx.tweets_by_id = tweets_by_id or {}
    # ``tools._resolve_year_month`` reaches through ``index.tree.nodes_by_id``.
    idx.tree = MagicMock(name="TweetTree")
    idx.tree.nodes_by_id = nodes_by_id or {}
    return idx


def _make_user_mock(
    *,
    screen_name: str = "alice",
    name: str = "Alice Author",
    verified: bool = True,
) -> MagicMock:
    user = MagicMock(name="User")
    user.screen_name = screen_name
    user.name = name
    user.verified = verified
    return user


def _make_tweet_mock(
    *,
    tweet_id: str = "1001",
    text: str = "hello world",
    created_at: str = "Wed Jan 15 09:30:00 +0000 2025",
    favorite_count: int = 10,
    retweet_count: int = 2,
    view_count: int = 100,
    user: MagicMock | None = None,
    parses_datetime: bool = True,
    year_month: str = "2025-01",
) -> MagicMock:
    """Build a :class:`MagicMock` standing in for a :class:`Tweet`.

    When ``parses_datetime`` is true, ``get_created_datetime().strftime``
    returns ``year_month``. When false, it raises ``ValueError`` so the
    fallback path through ``TreeNode.year_month`` is exercised.
    """
    tweet = MagicMock(name="Tweet")
    tweet.id = tweet_id
    tweet.text = text
    tweet.created_at = created_at
    tweet.favorite_count = favorite_count
    tweet.retweet_count = retweet_count
    tweet.view_count = view_count
    tweet.user = user if user is not None else _make_user_mock()

    if parses_datetime:
        dt = MagicMock(name="datetime")
        dt.strftime.return_value = year_month
        tweet.get_created_datetime.return_value = dt
    else:
        tweet.get_created_datetime.side_effect = ValueError("unparseable")

    return tweet


# ---------------------------------------------------------------------------
# search_likes — input validation


def test_search_likes_whitespace_query_raises_invalid_input() -> None:
    """A blank/whitespace query is rejected before any downstream call."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "   ")
    assert excinfo.value.category == "invalid_input"
    assert "query" in excinfo.value.message
    idx.search.assert_not_called()


def test_search_likes_month_start_without_year_raises_invalid_input() -> None:
    """``month_start`` set without ``year`` is invalid (filter shape)."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x", year=None, month_start="01")
    assert excinfo.value.category == "invalid_input"
    assert "filter" in excinfo.value.message
    idx.search.assert_not_called()


def test_search_likes_month_start_out_of_range_raises_invalid_input() -> None:
    """A month_start value outside ``01..12`` fails the regex check."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x", year=2025, month_start="13")
    assert excinfo.value.category == "invalid_input"
    idx.search.assert_not_called()


# ---------------------------------------------------------------------------
# search_likes — index.search forwarding


def test_search_likes_forwards_filter_to_index_search() -> None:
    """``search_likes`` passes the four arguments through to ``index.search``."""
    idx = _make_index_mock()
    idx.search.return_value = []  # empty result list -> empty output
    result = tools.search_likes(
        idx, "x", year=2025, month_start="01", month_end="02"
    )
    idx.search.assert_called_once_with("x", 2025, "01", "02")
    assert result == []


# ---------------------------------------------------------------------------
# search_likes — error translation from index.search


def test_search_likes_translates_value_error_to_invalid_input() -> None:
    """``ValueError`` from the resolver becomes an ``invalid_input`` tool error."""
    idx = _make_index_mock()
    idx.search.side_effect = ValueError("filter: bad shape")
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x", year=2025, month_start="01")
    assert excinfo.value.category == "invalid_input"
    assert "filter" in excinfo.value.message


def test_search_likes_translates_walker_error_to_upstream_failure() -> None:
    """``WalkerError`` from the walker becomes an ``upstream_failure`` tool error."""
    idx = _make_index_mock()
    idx.search.side_effect = WalkerError("LLM down")
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x")
    assert excinfo.value.category == "upstream_failure"
    assert "LLM down" in excinfo.value.message


def test_search_likes_translates_generic_exception_to_upstream_failure() -> None:
    """Any non-``ToolError`` exception becomes ``upstream_failure``."""
    idx = _make_index_mock()
    idx.search.side_effect = RuntimeError("disk gone")
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x")
    assert excinfo.value.category == "upstream_failure"
    assert "disk gone" in excinfo.value.message


# ---------------------------------------------------------------------------
# search_likes — response shaping


def test_search_likes_shapes_results_with_documented_keys() -> None:
    """Each ``ScoredHit`` becomes a dict with the eight documented keys."""
    tweet = _make_tweet_mock(
        tweet_id="1001",
        text="hello there, this is the snippet body",
        year_month="2025-01",
    )
    idx = _make_index_mock(tweets_by_id={"1001": tweet})
    breakdown = {
        "relevance": 7.5,
        "favorite": 1.2,
        "retweet": 0.8,
        "reply": 0.4,
        "view": 0.3,
        "affinity": 1.0,
        "recency": 0.9,
        "verified": 0.5,
        "media": 0.0,
    }
    idx.search.return_value = [
        ScoredHit(
            tweet_id="1001",
            score=12.6,
            walker_relevance=0.75,
            why="thematically related",
            feature_breakdown=breakdown,
        )
    ]

    result = tools.search_likes(idx, "retrieval")
    assert len(result) == 1
    hit = result[0]
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
    assert hit["year_month"] == "2025-01"
    assert hit["handle"] == "alice"
    assert hit["snippet"] == "hello there, this is the snippet body"
    assert hit["score"] == 12.6
    assert hit["walker_relevance"] == 0.75
    assert hit["why"] == "thematically related"
    assert hit["feature_breakdown"] == breakdown
    # The feature_breakdown should be a *copy* so the caller cannot mutate
    # the original ScoredHit's dict by accident.
    assert hit["feature_breakdown"] is not breakdown


def test_search_likes_works_against_real_loaded_export() -> None:
    """Smoke-test against a real ``Tweet`` loaded from the fixture export.

    Confirms the ``handle`` / ``snippet`` / ``year_month`` shaping path
    doesn't choke on a real :class:`Tweet` (which has a real
    :meth:`Tweet.get_created_datetime`).
    """
    tweets = load_export(LIKES_JSON)
    tweets_by_id = {t.id: t for t in tweets}
    idx = _make_index_mock(tweets_by_id=tweets_by_id)
    idx.search.return_value = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.9,
            why="real tweet path",
            feature_breakdown={"relevance": 9.0, "favorite": 1.0},
        )
    ]
    result = tools.search_likes(idx, "anything")
    assert len(result) == 1
    assert result[0]["tweet_id"] == "1001"
    assert result[0]["handle"] == "alice"
    assert result[0]["year_month"] == "2025-01"
    assert "retrieval-augmented" in result[0]["snippet"]


# ---------------------------------------------------------------------------
# list_months


def test_list_months_returns_dict_list_per_month_info() -> None:
    """``list_months`` returns the dict-shaped per-MonthInfo list."""
    idx = _make_index_mock()
    idx.list_months.return_value = [
        MonthInfo(
            year_month="2025-03",
            path=Path("/tmp/by_month/likes_2025-03.md"),
            tweet_count=1,
        ),
        MonthInfo(
            year_month="2025-02",
            path=Path("/tmp/by_month/likes_2025-02.md"),
            tweet_count=1,
        ),
        MonthInfo(
            year_month="2025-01",
            path=Path("/tmp/by_month/likes_2025-01.md"),
            tweet_count=2,
        ),
    ]
    result = tools.list_months(idx)
    assert len(result) == 3
    assert result[0] == {
        "year_month": "2025-03",
        "path": "/tmp/by_month/likes_2025-03.md",
        "tweet_count": 1,
    }
    assert result[2] == {
        "year_month": "2025-01",
        "path": "/tmp/by_month/likes_2025-01.md",
        "tweet_count": 2,
    }
    # All entries must carry the three documented keys.
    for entry in result:
        assert set(entry.keys()) == {"year_month", "path", "tweet_count"}


# ---------------------------------------------------------------------------
# get_month


def test_get_month_invalid_pattern_raises_invalid_input() -> None:
    """``YYYY/MM`` or any other slash form fails the regex check."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.get_month(idx, "2025/01")
    assert excinfo.value.category == "invalid_input"
    assert "year_month" in excinfo.value.message
    idx.get_month_markdown.assert_not_called()


def test_get_month_missing_month_raises_not_found() -> None:
    """An unknown month surfaces ``not_found`` rather than ``None``."""
    idx = _make_index_mock()
    idx.get_month_markdown.return_value = None
    with pytest.raises(ToolError) as excinfo:
        tools.get_month(idx, "2099-12")
    assert excinfo.value.category == "not_found"
    assert "2099-12" in excinfo.value.message
    idx.get_month_markdown.assert_called_once_with("2099-12")


def test_get_month_returns_markdown_string() -> None:
    """A present month returns the raw Markdown content."""
    idx = _make_index_mock()
    markdown = "## 2025-01\n\n### [@alice](https://x.com/alice)\n\nhello\n"
    idx.get_month_markdown.return_value = markdown
    assert tools.get_month(idx, "2025-01") == markdown
    idx.get_month_markdown.assert_called_once_with("2025-01")


# ---------------------------------------------------------------------------
# read_tweet


def test_read_tweet_empty_id_raises_invalid_input() -> None:
    """An empty ``tweet_id`` is rejected before any lookup."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.read_tweet(idx, "")
    assert excinfo.value.category == "invalid_input"
    assert "tweet_id" in excinfo.value.message
    idx.lookup_tweet.assert_not_called()


def test_read_tweet_non_numeric_id_raises_invalid_input() -> None:
    """A non-numeric id fails the ``^\\d+$`` regex."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.read_tweet(idx, "abc")
    assert excinfo.value.category == "invalid_input"
    assert "tweet_id" in excinfo.value.message
    idx.lookup_tweet.assert_not_called()


def test_read_tweet_unknown_id_raises_not_found() -> None:
    """``lookup_tweet`` returning ``None`` becomes ``not_found``."""
    idx = _make_index_mock()
    idx.lookup_tweet.return_value = None
    with pytest.raises(ToolError) as excinfo:
        tools.read_tweet(idx, "999")
    assert excinfo.value.category == "not_found"
    assert "999" in excinfo.value.message
    idx.lookup_tweet.assert_called_once_with("999")


def test_read_tweet_returns_metadata_dict_with_documented_keys() -> None:
    """A present tweet returns a dict with the documented keys populated."""
    user = _make_user_mock(screen_name="alice", name="Alice Author")
    tweet = _make_tweet_mock(
        tweet_id="999",
        text="hello there",
        created_at="Wed Jan 15 09:30:00 +0000 2025",
        favorite_count=42,
        retweet_count=7,
        view_count=1234,
        user=user,
    )
    idx = _make_index_mock()
    idx.lookup_tweet.return_value = tweet

    result = tools.read_tweet(idx, "999")
    assert result["tweet_id"] == "999"
    assert result["handle"] == "alice"
    assert result["display_name"] == "Alice Author"
    assert result["text"] == "hello there"
    assert result["created_at"] == "Wed Jan 15 09:30:00 +0000 2025"
    assert result["view_count"] == 1234
    assert result["like_count"] == 42
    assert result["retweet_count"] == 7
    assert result["url"] == "https://x.com/alice/status/999"
    # All documented keys are present (none of these values are falsy).
    assert set(result.keys()) == {
        "tweet_id",
        "handle",
        "display_name",
        "text",
        "created_at",
        "view_count",
        "like_count",
        "retweet_count",
        "url",
    }


def test_read_tweet_url_falls_back_to_i_status_when_handle_empty() -> None:
    """An empty ``screen_name`` falls back to the ``/i/status/{id}`` form."""
    user = _make_user_mock(screen_name="", name="Anonymous")
    tweet = _make_tweet_mock(
        tweet_id="42",
        text="anon post",
        favorite_count=1,
        retweet_count=1,
        view_count=1,
        user=user,
    )
    idx = _make_index_mock()
    idx.lookup_tweet.return_value = tweet

    result = tools.read_tweet(idx, "42")
    assert result["url"] == "https://x.com/i/status/42"
    # An empty screen_name is falsy so ``handle`` is omitted entirely.
    assert "handle" not in result


def test_read_tweet_omits_falsy_fields() -> None:
    """Fields with falsy values (``view_count == 0``, etc.) are omitted."""
    user = _make_user_mock(screen_name="bob", name="Bob")
    tweet = _make_tweet_mock(
        tweet_id="500",
        text="some text",
        created_at="Wed Jan 15 09:30:00 +0000 2025",
        favorite_count=0,
        retweet_count=0,
        view_count=0,
        user=user,
    )
    idx = _make_index_mock()
    idx.lookup_tweet.return_value = tweet

    result = tools.read_tweet(idx, "500")
    # Truthy fields stay.
    assert result["tweet_id"] == "500"
    assert result["handle"] == "bob"
    assert result["display_name"] == "Bob"
    assert result["text"] == "some text"
    assert result["created_at"] == "Wed Jan 15 09:30:00 +0000 2025"
    assert result["url"] == "https://x.com/bob/status/500"
    # Falsy numeric fields are dropped.
    assert "view_count" not in result
    assert "like_count" not in result
    assert "retweet_count" not in result
