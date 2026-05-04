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
from x_likes_mcp.embeddings import EmbeddingError
from x_likes_mcp.errors import ToolError
from x_likes_mcp.index import MonthInfo
from x_likes_mcp.ranker import ScoredHit
from x_likes_mcp.tree import TreeNode, TweetTree
from x_likes_mcp.walker import WalkerError, WalkerHit


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
    """``WalkerError`` bubbling out of ``index.search`` becomes ``upstream_failure``.

    Post-task-3.3 ``index.search`` no longer calls the walker, so this is now
    a pure boundary translation test: any non-``ToolError`` exception from the
    seam (including a hypothetical re-raised ``WalkerError``) maps to
    ``upstream_failure``.
    """
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


def test_search_likes_embedding_error_becomes_upstream_failure() -> None:
    """``EmbeddingError`` (raised when both retrievals fail) maps to ``upstream_failure``."""
    idx = _make_index_mock()
    idx.search.side_effect = EmbeddingError("both retrieval paths failed")
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x")
    assert excinfo.value.category == "upstream_failure"
    assert "both retrieval paths failed" in excinfo.value.message


# ---------------------------------------------------------------------------
# search_likes — with_why semantics


def test_search_likes_default_with_why_false_returns_shaped_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default call returns shaped dicts and never invokes the explainer.

    Patches ``_call_walker_explainer`` to raise so the test asserts the
    default path does not touch it.
    """

    def _explode(*_args: object, **_kwargs: object) -> dict[str, WalkerHit]:
        raise AssertionError("explainer must not be called when with_why=False")

    monkeypatch.setattr(tools, "_call_walker_explainer", _explode)

    tweet = _make_tweet_mock(tweet_id="1001", text="hello", year_month="2025-01")
    idx = _make_index_mock(tweets_by_id={"1001": tweet})
    idx.search.return_value = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.42,
            why="",
            feature_breakdown={"relevance": 5.0},
        )
    ]

    result = tools.search_likes(idx, "x")
    assert len(result) == 1
    assert result[0]["tweet_id"] == "1001"
    assert result[0]["why"] == ""
    assert result[0]["walker_relevance"] == 0.42


def test_search_likes_with_why_true_calls_walker_explainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``with_why=True`` invokes the explainer with the top results and
    merges the returned ``WalkerHit`` map onto the response."""
    captured: dict[str, object] = {}

    def _fake_explainer(
        top_results: list[ScoredHit],
        query: str,
        index: object,
    ) -> dict[str, WalkerHit]:
        captured["top_results"] = top_results
        captured["query"] = query
        captured["index"] = index
        return {
            "1001": WalkerHit(
                tweet_id="1001", relevance=0.91, why="explainer reason"
            )
        }

    monkeypatch.setattr(tools, "_call_walker_explainer", _fake_explainer)

    tweet = _make_tweet_mock(tweet_id="1001", text="hello", year_month="2025-01")
    idx = _make_index_mock(tweets_by_id={"1001": tweet})
    scored = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.42,
            why="",
            feature_breakdown={"relevance": 5.0},
        )
    ]
    idx.search.return_value = scored

    result = tools.search_likes(idx, "deep query", with_why=True)
    assert len(result) == 1
    assert result[0]["why"] == "explainer reason"
    assert result[0]["walker_relevance"] == 0.91
    # The explainer received the ScoredHit slice (top-20-or-fewer), the query,
    # and the index reference.
    assert captured["query"] == "deep query"
    assert captured["index"] is idx
    assert captured["top_results"] == scored


def test_search_likes_with_why_true_explainer_passed_at_most_top_20(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explainer receives at most 20 ScoredHits even when search returned more."""
    captured: dict[str, object] = {}

    def _fake_explainer(
        top_results: list[ScoredHit],
        query: str,
        index: object,
    ) -> dict[str, WalkerHit]:
        captured["count"] = len(top_results)
        return {}

    monkeypatch.setattr(tools, "_call_walker_explainer", _fake_explainer)

    # 30 scored hits; tools should pass only top-20 to the explainer.
    tweets_by_id: dict[str, object] = {}
    scored: list[ScoredHit] = []
    for n in range(30):
        tid = str(1000 + n)
        tweets_by_id[tid] = _make_tweet_mock(
            tweet_id=tid, text=f"t{n}", year_month="2025-01"
        )
        scored.append(
            ScoredHit(
                tweet_id=tid,
                score=10.0 - n * 0.1,
                walker_relevance=0.5,
                why="",
                feature_breakdown={"relevance": 1.0},
            )
        )
    idx = _make_index_mock(tweets_by_id=tweets_by_id)
    idx.search.return_value = scored

    tools.search_likes(idx, "x", with_why=True)
    assert captured["count"] == 20


def test_search_likes_with_why_true_empty_map_uses_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the explainer returns ``{}``, results keep cosine-derived placeholders.

    Order is preserved; ``why`` and ``walker_relevance`` come from the
    ``ScoredHit`` (set by ``index.search``) untouched.
    """
    monkeypatch.setattr(
        tools,
        "_call_walker_explainer",
        lambda *_a, **_kw: {},
    )

    tweet = _make_tweet_mock(tweet_id="1001", text="hello", year_month="2025-01")
    idx = _make_index_mock(tweets_by_id={"1001": tweet})
    idx.search.return_value = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.42,
            why="",
            feature_breakdown={"relevance": 5.0},
        )
    ]

    result = tools.search_likes(idx, "x", with_why=True)
    assert len(result) == 1
    assert result[0]["why"] == ""
    assert result[0]["walker_relevance"] == 0.42


def test_search_likes_with_why_true_partial_merge_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hits not present in the explainer map keep their cosine placeholders;
    order is preserved across the merge."""

    def _fake_explainer(
        _top_results: list[ScoredHit],
        _query: str,
        _index: object,
    ) -> dict[str, WalkerHit]:
        return {
            "1002": WalkerHit(
                tweet_id="1002", relevance=0.7, why="explained 1002"
            )
        }

    monkeypatch.setattr(tools, "_call_walker_explainer", _fake_explainer)

    tweets_by_id = {
        "1001": _make_tweet_mock(tweet_id="1001", text="a", year_month="2025-01"),
        "1002": _make_tweet_mock(tweet_id="1002", text="b", year_month="2025-01"),
        "1003": _make_tweet_mock(tweet_id="1003", text="c", year_month="2025-01"),
    }
    idx = _make_index_mock(tweets_by_id=tweets_by_id)
    idx.search.return_value = [
        ScoredHit(
            tweet_id="1001",
            score=12.0,
            walker_relevance=0.4,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
        ScoredHit(
            tweet_id="1002",
            score=11.0,
            walker_relevance=0.3,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
        ScoredHit(
            tweet_id="1003",
            score=10.0,
            walker_relevance=0.2,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
    ]

    result = tools.search_likes(idx, "x", with_why=True)
    assert [r["tweet_id"] for r in result] == ["1001", "1002", "1003"]
    # 1001 untouched
    assert result[0]["why"] == ""
    assert result[0]["walker_relevance"] == 0.4
    # 1002 merged
    assert result[1]["why"] == "explained 1002"
    assert result[1]["walker_relevance"] == 0.7
    # 1003 untouched
    assert result[2]["why"] == ""
    assert result[2]["walker_relevance"] == 0.2


# ---------------------------------------------------------------------------
# _call_walker_explainer — synthetic single-chunk tree


def _make_tree_node(tweet_id: str, *, year_month: str = "2025-01") -> TreeNode:
    """Build a synthetic :class:`TreeNode` for explainer tests."""
    return TreeNode(
        year_month=year_month,
        tweet_id=tweet_id,
        handle="alice",
        text=f"text for {tweet_id}",
        raw_section=f"### [@alice](https://x.com/alice)\ntext for {tweet_id}",
    )


def _make_explainer_index(
    *,
    nodes_by_id: dict[str, TreeNode] | None = None,
    config: object | None = None,
) -> MagicMock:
    """Build a mock index with a real :class:`TweetTree` and a config attr.

    The explainer reads ``index.tree.nodes_by_id`` and ``index.config``;
    the synthetic tree it constructs is passed to ``walker.walk`` so the
    field shapes must match the real :class:`TweetTree` dataclass.
    """
    nodes_by_id = nodes_by_id or {}
    idx = MagicMock(name="TweetIndex")
    idx.tree = TweetTree(
        nodes_by_month={"2025-01": list(nodes_by_id.values())},
        nodes_by_id=dict(nodes_by_id),
    )
    idx.config = config if config is not None else MagicMock(name="Config")
    return idx


def test_call_walker_explainer_calls_walker_walk_with_top_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper invokes ``walker.walk`` once with a synthetic single-chunk tree.

    The synthetic tree carries the looked-up ``TreeNode`` objects under
    one synthetic month key, ``months_in_scope`` lists that month, and
    ``chunk_size`` matches the node count so the walker issues a single
    LLM call (req 8.2).
    """
    captured: dict[str, object] = {}

    def _fake_walk(**kwargs: object) -> list[WalkerHit]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("x_likes_mcp.tools.walker_module.walk", _fake_walk)

    nodes = {
        "1001": _make_tree_node("1001"),
        "1002": _make_tree_node("1002"),
    }
    config_sentinel = object()
    idx = _make_explainer_index(nodes_by_id=nodes, config=config_sentinel)
    scored = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
        ScoredHit(
            tweet_id="1002",
            score=9.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
    ]

    result = tools._call_walker_explainer(scored, "deep query", idx)

    assert result == {}
    # Walker called exactly once with the documented kwargs.
    assert captured["query"] == "deep query"
    assert captured["chunk_size"] == 2
    assert captured["config"] is config_sentinel
    months_in_scope = captured["months_in_scope"]
    synthetic_tree = captured["tree"]
    assert isinstance(synthetic_tree, TweetTree)
    # Synthetic tree has exactly one month entry, named after the synthetic key.
    assert list(synthetic_tree.nodes_by_month.keys()) == months_in_scope
    [synth_month] = synthetic_tree.nodes_by_month.keys()
    # months_in_scope contains the same single synthetic month.
    assert months_in_scope == [synth_month]
    # Synthetic chunk holds the looked-up TreeNodes for the supplied hits.
    chunk = synthetic_tree.nodes_by_month[synth_month]
    assert [n.tweet_id for n in chunk] == ["1001", "1002"]


def test_call_walker_explainer_returns_walker_hit_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``walker.walk`` results become a ``{tweet_id: WalkerHit}`` map."""
    canned = [
        WalkerHit(tweet_id="1001", relevance=0.9, why="match"),
        WalkerHit(tweet_id="1002", relevance=0.4, why="weak"),
    ]
    monkeypatch.setattr(
        "x_likes_mcp.tools.walker_module.walk",
        lambda **_kw: canned,
    )

    nodes = {
        "1001": _make_tree_node("1001"),
        "1002": _make_tree_node("1002"),
    }
    idx = _make_explainer_index(nodes_by_id=nodes)
    scored = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
        ScoredHit(
            tweet_id="1002",
            score=9.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
    ]

    result = tools._call_walker_explainer(scored, "q", idx)
    assert result == {
        "1001": canned[0],
        "1002": canned[1],
    }


def test_call_walker_explainer_walker_failure_returns_empty_map(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Any walker failure becomes an empty map plus a single stderr line (req 8.4)."""

    def _boom(**_kwargs: object) -> list[WalkerHit]:
        raise RuntimeError("upstream gone")

    monkeypatch.setattr("x_likes_mcp.tools.walker_module.walk", _boom)

    nodes = {"1001": _make_tree_node("1001")}
    idx = _make_explainer_index(nodes_by_id=nodes)
    scored = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        )
    ]

    result = tools._call_walker_explainer(scored, "q", idx)
    assert result == {}
    captured = capsys.readouterr()
    # One line on stderr; stdout untouched.
    assert "walker call failed" in captured.err
    assert "upstream gone" in captured.err
    assert captured.out == ""


def test_call_walker_explainer_walker_error_returns_empty_map(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A :class:`WalkerError` (LLM-side failure) is also swallowed."""

    def _boom(**_kwargs: object) -> list[WalkerHit]:
        raise WalkerError("LLM said no")

    monkeypatch.setattr("x_likes_mcp.tools.walker_module.walk", _boom)

    nodes = {"1001": _make_tree_node("1001")}
    idx = _make_explainer_index(nodes_by_id=nodes)
    scored = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        )
    ]

    assert tools._call_walker_explainer(scored, "q", idx) == {}
    assert "LLM said no" in capsys.readouterr().err


def test_call_walker_explainer_empty_top_results_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``top_results`` short-circuits without calling ``walker.walk``."""

    def _explode(**_kwargs: object) -> list[WalkerHit]:
        raise AssertionError("walker.walk must not be called for empty top_results")

    monkeypatch.setattr("x_likes_mcp.tools.walker_module.walk", _explode)

    idx = _make_explainer_index(nodes_by_id={})
    assert tools._call_walker_explainer([], "q", idx) == {}


def test_call_walker_explainer_missing_tree_nodes_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When none of the top_results' ids are in the tree, return ``{}`` without calling walker."""

    def _explode(**_kwargs: object) -> list[WalkerHit]:
        raise AssertionError("walker.walk must not be called when no nodes resolve")

    monkeypatch.setattr("x_likes_mcp.tools.walker_module.walk", _explode)

    # nodes_by_id is empty but we pass a ScoredHit that won't resolve.
    idx = _make_explainer_index(nodes_by_id={})
    scored = [
        ScoredHit(
            tweet_id="9999",
            score=10.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        )
    ]
    assert tools._call_walker_explainer(scored, "q", idx) == {}


def test_call_walker_explainer_partial_missing_nodes_skips_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hits whose ids aren't in the tree are dropped from the synthetic chunk;
    resolved hits still flow through to the walker."""
    captured: dict[str, object] = {}

    def _fake_walk(**kwargs: object) -> list[WalkerHit]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("x_likes_mcp.tools.walker_module.walk", _fake_walk)

    # Only 1001 has a TreeNode; 9999 will be silently skipped.
    nodes = {"1001": _make_tree_node("1001")}
    idx = _make_explainer_index(nodes_by_id=nodes)
    scored = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
        ScoredHit(
            tweet_id="9999",
            score=9.0,
            walker_relevance=0.5,
            why="",
            feature_breakdown={"relevance": 1.0},
        ),
    ]
    tools._call_walker_explainer(scored, "q", idx)
    chunk = captured["tree"].nodes_by_month[captured["months_in_scope"][0]]
    assert [n.tweet_id for n in chunk] == ["1001"]
    # chunk_size sized to actual chunk, not requested length.
    assert captured["chunk_size"] == 1


# ---------------------------------------------------------------------------
# search_likes — with_why semantics (continued)


def test_search_likes_non_bool_with_why_raises_invalid_input() -> None:
    """A non-bool, non-None ``with_why`` raises ``invalid_input``."""
    idx = _make_index_mock()
    with pytest.raises(ToolError) as excinfo:
        tools.search_likes(idx, "x", with_why="yes")  # type: ignore[arg-type]
    assert excinfo.value.category == "invalid_input"
    assert "with_why" in excinfo.value.message
    idx.search.assert_not_called()


def test_search_likes_with_why_none_treated_as_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``with_why=None`` is tolerated and treated as ``False`` (no explainer call)."""

    def _explode(*_args: object, **_kwargs: object) -> dict[str, WalkerHit]:
        raise AssertionError("explainer must not run when with_why is None")

    monkeypatch.setattr(tools, "_call_walker_explainer", _explode)

    tweet = _make_tweet_mock(tweet_id="1001", text="hello", year_month="2025-01")
    idx = _make_index_mock(tweets_by_id={"1001": tweet})
    idx.search.return_value = [
        ScoredHit(
            tweet_id="1001",
            score=10.0,
            walker_relevance=0.42,
            why="",
            feature_breakdown={"relevance": 5.0},
        )
    ]

    result = tools.search_likes(idx, "x", with_why=None)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0]["why"] == ""


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
        "urls",
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
