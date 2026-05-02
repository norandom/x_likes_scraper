"""Tests for :mod:`x_likes_mcp.tree`.

Covers the pure-Python parser ``build_tree`` and its dataclasses
(:class:`TreeNode`, :class:`TweetTree`). These tests are entirely offline:
``tree.py`` performs no network or LLM access, so the autouse
``_block_real_llm`` guard from ``conftest.py`` does not need to fire here.

The fixtures at ``tests/mcp/fixtures/by_month/likes_2025-{01,02,03}.md`` were
created in task 6.1 and exercise:

* three months, four tweet IDs (1001, 1002, 2001, 3001),
* the canonical ``### [@handle](https://x.com/handle)`` heading shape,
* the canonical ``🔗 [View on X](https://x.com/{handle}/status/{id})`` link
  line.
"""

from __future__ import annotations

from pathlib import Path

from x_likes_mcp import tree
from x_likes_mcp.tree import TreeNode, TweetTree, build_tree


FIXTURES_BY_MONTH = Path(__file__).parent / "fixtures" / "by_month"


# Expected handle for each fixture tweet id, derived from the fixtures.
_EXPECTED_HANDLE_BY_ID: dict[str, str] = {
    "1001": "alice",
    "1002": "bob",
    "2001": "alice",
    "3001": "carol",
}

# Expected month for each fixture tweet id.
_EXPECTED_MONTH_BY_ID: dict[str, str] = {
    "1001": "2025-01",
    "1002": "2025-01",
    "2001": "2025-02",
    "3001": "2025-03",
}


# ---------------------------------------------------------------------------
# Smoke tests against the on-disk fixtures.

def test_build_tree_returns_tweet_tree_with_expected_months() -> None:
    """build_tree's nodes_by_month keys must match the three fixture months."""
    result = build_tree(FIXTURES_BY_MONTH)

    assert isinstance(result, TweetTree)
    assert set(result.nodes_by_month.keys()) == {"2025-01", "2025-02", "2025-03"}


def test_build_tree_nodes_by_id_contains_all_fixture_ids() -> None:
    """nodes_by_id must contain every fixture tweet ID."""
    result = build_tree(FIXTURES_BY_MONTH)

    assert set(result.nodes_by_id.keys()) == set(_EXPECTED_HANDLE_BY_ID.keys())


def test_build_tree_each_node_has_expected_handle_and_id() -> None:
    """Each TreeNode must expose the handle and tweet_id from the markdown."""
    result = build_tree(FIXTURES_BY_MONTH)

    for tweet_id, expected_handle in _EXPECTED_HANDLE_BY_ID.items():
        node = result.nodes_by_id[tweet_id]
        assert isinstance(node, TreeNode)
        assert node.tweet_id == tweet_id
        assert node.handle == expected_handle
        assert node.year_month == _EXPECTED_MONTH_BY_ID[tweet_id]


def test_build_tree_text_excludes_heading_and_view_on_x_lines() -> None:
    """Node.text must omit the ``### `` heading line and the View-on-X link line."""
    result = build_tree(FIXTURES_BY_MONTH)

    for node in result.nodes_by_id.values():
        # The heading line ("### [@handle](...)") must not appear in text.
        assert "### " not in node.text, (
            f"text for tweet {node.tweet_id} unexpectedly contains a '### ' "
            f"heading line: {node.text!r}"
        )
        # The "View on X" link line must not appear in text.
        assert "View on X" not in node.text, (
            f"text for tweet {node.tweet_id} unexpectedly contains the "
            f"View-on-X line: {node.text!r}"
        )
        # Sanity: text should still carry the tweet body (non-empty, no leading
        # blank line because _strip_heading_and_link strips those).
        assert node.text.strip() != ""


def test_build_tree_raw_section_starts_with_h3_heading() -> None:
    """raw_section must preserve the ``### `` heading at the top of each section."""
    result = build_tree(FIXTURES_BY_MONTH)

    for node in result.nodes_by_id.values():
        assert node.raw_section.startswith("### "), (
            f"raw_section for tweet {node.tweet_id} should start with '### ' "
            f"but starts with {node.raw_section[:10]!r}"
        )


def test_build_tree_is_deterministic() -> None:
    """Two consecutive build_tree calls must produce equal TweetTree objects."""
    first = build_tree(FIXTURES_BY_MONTH)
    second = build_tree(FIXTURES_BY_MONTH)

    # TweetTree and TreeNode are frozen dataclasses, so __eq__ compares fields
    # (including the dict / list contents) structurally.
    assert first == second


# ---------------------------------------------------------------------------
# Edge cases driven through tmp_path.

def test_build_tree_empty_directory_returns_empty_tree(tmp_path: Path) -> None:
    """A directory with no ``likes_*.md`` files yields an empty TweetTree."""
    # tmp_path is a fresh, empty directory; build_tree should walk it
    # without raising and return a TweetTree with empty containers.
    result = build_tree(tmp_path)

    assert isinstance(result, TweetTree)
    assert result.nodes_by_month == {}
    assert result.nodes_by_id == {}


def test_build_tree_skips_section_without_view_on_x_link(tmp_path: Path) -> None:
    """A section that lacks the ``🔗 [View on X]`` link is skipped silently."""
    # Build a single-month file with one *valid* section and one *malformed*
    # section (missing the View-on-X link line, hence no parseable tweet ID).
    # build_tree must keep the valid one and skip the malformed one without
    # raising.
    md = (
        "## 2025-04 (2 tweets)\n"
        "\n"
        "\n"
        "### [@dave](https://x.com/dave)\n"
        "**Dave Developer** \n"
        "*2025-04-02 10:00:00*\n"
        "\n"
        "Valid section with a proper link line.\n"
        "\n"
        "*\U0001f504 1 • ❤️ 1 • \U0001f4ac 0 • \U0001f441️ 10*\n"
        "\n"
        "\U0001f517 [View on X](https://x.com/dave/status/4001)\n"
        "\n"
        "---\n"
        "\n"
        "### [@eve](https://x.com/eve)\n"
        "**Eve Engineer** \n"
        "*2025-04-03 11:00:00*\n"
        "\n"
        "Malformed section: no View-on-X link, so no tweet id can be parsed.\n"
        "\n"
        "*\U0001f504 0 • ❤️ 0 • \U0001f4ac 0 • \U0001f441️ 0*\n"
        "\n"
        "---\n"
    )
    (tmp_path / "likes_2025-04.md").write_text(md, encoding="utf-8")

    result = build_tree(tmp_path)

    # The valid section made it in.
    assert "4001" in result.nodes_by_id
    assert result.nodes_by_id["4001"].handle == "dave"
    # The malformed section was skipped: no nodes for @eve, and the only
    # entry for 2025-04 is the valid one.
    assert len(result.nodes_by_id) == 1
    assert "2025-04" in result.nodes_by_month
    assert len(result.nodes_by_month["2025-04"]) == 1
    assert result.nodes_by_month["2025-04"][0].tweet_id == "4001"


# ---------------------------------------------------------------------------
# Module hygiene: tree.py must not import the OpenAI SDK.

def test_tree_module_does_not_import_openai() -> None:
    """``tree.py`` is the pure parser; the LLM lives in ``walker.py``.

    The module docstring legitimately mentions "OpenAI SDK import" while
    documenting that *no* such import happens, so we can't simply scan for
    the substring ``openai``. Instead, scan each non-comment, non-docstring
    code line for an actual ``import`` statement that pulls in the SDK.
    """
    source = Path(tree.__file__).read_text(encoding="utf-8")

    forbidden = ("import openai", "from openai")
    offenders: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        # Skip blank lines and ordinary ``#`` comments. Docstring text is
        # left in place but the patterns we look for are import statements,
        # which would not appear inside a docstring as live code.
        if not line or line.startswith("#"):
            continue
        if any(pat in line for pat in forbidden):
            offenders.append(raw_line)

    assert offenders == [], (
        "x_likes_mcp/tree.py must not import the OpenAI SDK; the walker is "
        f"the single LLM call site in this package. Offending lines: {offenders!r}"
    )
