"""Tests for :mod:`x_likes_mcp.walker`.

Every test in this file replaces the ``_call_chat_completions`` seam with its
own stub. The autouse ``_block_real_llm`` fixture in
``tests/mcp/conftest.py`` installs a guard that raises
``RealLLMCallAttempted``; each test below overrides that guard via
``monkeypatch.setattr('x_likes_mcp.walker._call_chat_completions', stub)`` so
the walker exercises real parsing logic without making a real HTTP call.

These tests cover the public ``walk`` entry point, the ``WalkerHit``
dataclass shape, the ``WalkerError`` failure mode, and the response-parsing
guard rails (markdown fences, top-level arrays, id/relevance/why
validation).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.tree import TreeNode, TweetTree
from x_likes_mcp.walker import WalkerError, WalkerHit, walk


# ---------------------------------------------------------------------------
# Helpers


def _make_config() -> Config:
    """Build a minimal :class:`Config` for walker tests.

    The walker only reads ``config.openai_model`` and ``config.openai_api_key``
    (to construct an OpenAI client), so the directory paths can point at
    ``/tmp`` placeholders.
    """

    return Config(
        output_dir=Path("/tmp/walker-test"),
        by_month_dir=Path("/tmp/walker-test/by_month"),
        likes_json=Path("/tmp/walker-test/likes.json"),
        cache_path=Path("/tmp/walker-test/tweet_tree_cache.pkl"),
        openai_base_url="http://fake/v1",
        openai_api_key="",
        openai_model="fake-model",
        ranker_weights=RankerWeights(),
    )


def _node(year_month: str, tweet_id: str, handle: str = "alice", text: str = "hi") -> TreeNode:
    """Build a synthetic :class:`TreeNode`."""

    return TreeNode(
        year_month=year_month,
        tweet_id=tweet_id,
        handle=handle,
        text=text,
        raw_section=f"### [@{handle}](https://x.com/{handle})\n{text}",
    )


def _make_tree(months: dict[str, list[TreeNode]]) -> TweetTree:
    """Build a :class:`TweetTree` from a ``{month: [TreeNode, ...]}`` map."""

    nodes_by_id = {n.tweet_id: n for nodes in months.values() for n in nodes}
    return TweetTree(nodes_by_month=dict(months), nodes_by_id=nodes_by_id)


def _install_stub(monkeypatch: pytest.MonkeyPatch, stub) -> None:
    """Replace the autouse guard with the test's own ``_call_chat_completions`` stub."""

    monkeypatch.setattr(
        "x_likes_mcp.walker._call_chat_completions",
        stub,
        raising=True,
    )


# ---------------------------------------------------------------------------
# Tests


def test_walk_returns_hit_for_valid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stubbed JSON with one valid hit yields a single ``WalkerHit``."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    def stub(_client, _model, _messages):  # noqa: ANN001 - test stub
        return '{"hits": [{"id": "1001", "relevance": 0.9, "why": "match"}]}'

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "query", ["2025-01"], _make_config())

    assert len(hits) == 1
    assert hits[0] == WalkerHit(tweet_id="1001", relevance=0.9, why="match")


def test_walk_parses_markdown_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON wrapped in ```json ... ``` parses correctly."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    def stub(_client, _model, _messages):  # noqa: ANN001
        return (
            "```json\n"
            '{"hits": [{"id": "1001", "relevance": 0.5, "why": "fenced"}]}\n'
            "```"
        )

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2025-01"], _make_config())

    assert len(hits) == 1
    assert hits[0].tweet_id == "1001"
    assert hits[0].relevance == 0.5
    assert hits[0].why == "fenced"


def test_walk_parses_top_level_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """A top-level JSON array (no ``hits`` wrapper) parses correctly."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    def stub(_client, _model, _messages):  # noqa: ANN001
        return '[{"id": "1001", "relevance": 0.7, "why": "array"}]'

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2025-01"], _make_config())

    assert len(hits) == 1
    assert hits[0].tweet_id == "1001"
    assert hits[0].relevance == 0.7


def test_walk_drops_unknown_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries whose id is not in the chunk are dropped."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    def stub(_client, _model, _messages):  # noqa: ANN001
        return (
            '{"hits": ['
            '{"id": "9999", "relevance": 0.9, "why": "ghost"},'
            '{"id": "1001", "relevance": 0.4, "why": "real"}'
            ']}'
        )

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2025-01"], _make_config())

    assert len(hits) == 1
    assert hits[0].tweet_id == "1001"


def test_walk_drops_relevance_outside_unit_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entries with relevance > 1.0 or < 0 are dropped."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001"), _node("2025-01", "1002")]})

    def stub(_client, _model, _messages):  # noqa: ANN001
        return (
            '{"hits": ['
            '{"id": "1001", "relevance": 1.5, "why": "too high"},'
            '{"id": "1002", "relevance": -0.1, "why": "negative"}'
            ']}'
        )

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2025-01"], _make_config())

    assert hits == []


def test_walk_drops_non_finite_relevance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries with NaN or inf relevance are dropped."""

    tree = _make_tree(
        {
            "2025-01": [
                _node("2025-01", "1001"),
                _node("2025-01", "1002"),
                _node("2025-01", "1003"),
            ]
        }
    )

    # Use Python's float literals for NaN and Infinity in JSON via the
    # standard ``json`` module's tolerant parsing — strict JSON forbids them
    # but ``json.loads`` accepts them by default. Use an array literal so
    # the values are clear.
    def stub(_client, _model, _messages):  # noqa: ANN001
        return (
            '{"hits": ['
            '{"id": "1001", "relevance": NaN, "why": "nan"},'
            '{"id": "1002", "relevance": Infinity, "why": "inf"},'
            '{"id": "1003", "relevance": -Infinity, "why": "neg inf"}'
            ']}'
        )

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2025-01"], _make_config())

    assert hits == []


def test_walk_truncates_why_to_240_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """``why`` longer than 240 chars is truncated to 240."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    long_why = "x" * 500

    def stub(_client, _model, _messages):  # noqa: ANN001
        import json

        return json.dumps(
            {"hits": [{"id": "1001", "relevance": 0.5, "why": long_why}]}
        )

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2025-01"], _make_config())

    assert len(hits) == 1
    assert len(hits[0].why) == 240
    assert hits[0].why == "x" * 240


def test_walk_wraps_runtime_error_in_walker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A helper raising ``RuntimeError`` becomes ``WalkerError`` with chunk index."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    def stub(_client, _model, _messages):  # noqa: ANN001
        raise RuntimeError("LLM down")

    _install_stub(monkeypatch, stub)

    with pytest.raises(WalkerError) as excinfo:
        walk(tree, "q", ["2025-01"], _make_config())

    # The walker's error message includes the chunk index ("chunk 0" for the
    # first chunk).
    assert "chunk 0" in str(excinfo.value)
    assert "LLM down" in str(excinfo.value)


def test_walk_raises_walker_error_for_unsalvageable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response with no parseable JSON raises ``WalkerError``."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    def stub(_client, _model, _messages):  # noqa: ANN001
        return "not json at all"

    _install_stub(monkeypatch, stub)

    with pytest.raises(WalkerError):
        walk(tree, "q", ["2025-01"], _make_config())


def test_walk_none_scope_iterates_all_months_ascending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``months_in_scope=None`` walks every month in the tree, sorted ascending."""

    tree = _make_tree(
        {
            # Insert out of order to verify the walker re-sorts ascending.
            "2025-03": [_node("2025-03", "3001")],
            "2025-01": [_node("2025-01", "1001")],
            "2025-02": [_node("2025-02", "2001")],
        }
    )

    months_seen: list[str] = []

    def stub(_client, _model, messages):  # noqa: ANN001
        # The user message lists tweets by id; we extract the ids to record
        # which month was processed in this call. Each chunk in this test has
        # exactly one tweet, so the id maps uniquely to a month.
        user_content = messages[1]["content"]
        for line in user_content.splitlines():
            if line.startswith("[id="):
                tweet_id = line.split("=", 1)[1].split("]", 1)[0]
                months_seen.append(tweet_id)
        return '{"hits": []}'

    _install_stub(monkeypatch, stub)

    result = walk(tree, "q", None, _make_config())

    assert result == []
    # The recorded tweet ids reflect the order of processing. With
    # chunk_size=30 (default) and one tweet per month, each call covers one
    # month. The walker sorts months ascending, so we should see 1001 (Jan),
    # 2001 (Feb), 3001 (Mar) in that order.
    assert months_seen == ["1001", "2001", "3001"]


def test_walk_unknown_month_in_scope_makes_no_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scope listing only months not in the tree triggers zero LLM calls."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})

    call_count = 0

    def stub(_client, _model, _messages):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return '{"hits": []}'

    _install_stub(monkeypatch, stub)

    hits = walk(tree, "q", ["2099-12"], _make_config())

    assert hits == []
    assert call_count == 0


def test_walk_chunk_size_one_calls_helper_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``chunk_size=1`` against a 2-node month invokes the helper exactly twice."""

    tree = _make_tree(
        {
            "2025-01": [
                _node("2025-01", "1001"),
                _node("2025-01", "1002"),
            ]
        }
    )

    call_count = 0

    def stub(_client, _model, _messages):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        return '{"hits": []}'

    _install_stub(monkeypatch, stub)

    walk(tree, "q", ["2025-01"], _make_config(), chunk_size=1)

    assert call_count == 2


def test_walk_raises_walker_error_when_openai_config_missing() -> None:
    """The walker is opt-in; if it's invoked without OPENAI_BASE_URL or
    OPENAI_MODEL set, surface a clear WalkerError naming the cause."""

    tree = _make_tree({"2025-01": [_node("2025-01", "1001")]})
    config = Config(
        output_dir=Path("/tmp/walker-test"),
        by_month_dir=Path("/tmp/walker-test/by_month"),
        likes_json=Path("/tmp/walker-test/likes.json"),
        cache_path=Path("/tmp/walker-test/tweet_tree_cache.pkl"),
        openai_base_url=None,
        openai_api_key="",
        openai_model=None,
        ranker_weights=RankerWeights(),
    )

    with pytest.raises(WalkerError) as excinfo:
        walk(tree, "q", ["2025-01"], config)
    msg = str(excinfo.value)
    assert "OPENAI_BASE_URL" in msg
    assert "OPENAI_MODEL" in msg
    assert "with_why" in msg


# ---------------------------------------------------------------------------
# Regression: walker preservation (task 5.7, requirements 11.1-11.3)
#
# These tests are passes-on-first-run regression anchors, not a TDD RED→GREEN
# cycle. They guard the documented mock seam and the "walker is the only
# chat.completions call site" invariant from drifting silently as the
# fast-search work lands alongside the existing walker.


class TestWalkerPreservation:
    """Regression anchors for walker module preservation (req 11.1-11.3)."""

    def test_walker_call_chat_completions_seam_preserved(self) -> None:
        """``walker._call_chat_completions`` is the documented LLM mock seam.

        The walker test suite and the explainer path patch this exact
        attribute name; renaming or removing it would silently break every
        walker test (they would fall back to the autouse ``_block_real_llm``
        guard or, worse, hit the real network).
        """

        from x_likes_mcp import walker

        assert hasattr(walker, "_call_chat_completions"), (
            "walker._call_chat_completions is the documented LLM mock seam; "
            "walker tests and the explainer path patch this name."
        )
        assert callable(walker._call_chat_completions)

    def test_walker_is_only_chat_completions_call_site(self) -> None:
        """Only ``walker.py`` may invoke ``client.chat.completions.create``.

        ``embeddings.py`` talks to ``/v1/embeddings`` via
        ``openai.embeddings.create`` (a different endpoint); no other module
        in the package should call the chat-completions endpoint. This is a
        grep-style invariant guard for requirement 11.4 / design boundary
        commitments.
        """

        package_dir = Path(__file__).parent.parent.parent / "x_likes_mcp"
        assert package_dir.is_dir(), f"Expected package at {package_dir}"

        chat_pattern = re.compile(r"\.chat\.completions\.create")
        offenders: list[Path] = []
        for py_file in package_dir.rglob("*.py"):
            if py_file.name == "walker.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            text = py_file.read_text(encoding="utf-8")
            if chat_pattern.search(text):
                offenders.append(py_file.relative_to(package_dir))
        assert offenders == [], (
            "Modules other than walker.py invoke openai chat.completions.create: "
            f"{offenders}. By design, walker is the only chat-completions call "
            "site (per req 11.4)."
        )
