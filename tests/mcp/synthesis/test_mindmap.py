"""Tests for the depth-capped mermaid mindmap renderer (synthesis-report task 2.5).

These tests pin the contract for ``render_mindmap``:

* The output is always a fenced ``mermaid`` block with a ``mindmap`` directive.
* The root label is the sanitized user query.
* Level 1 emits an entity category header (Authors / Sources / Themes /
  Hashtags) only when the corresponding ``NodeKind`` has at least one node.
* Level 2 lists the top-K nodes per category by weight (default K=8,
  override-able), with deterministic tie-breaking already enforced by
  ``KG.top_entities``.
* Level 3 lists the tweet neighbors of each top-K node.
* The depth cap honors ``MAX_MINDMAP_DEPTH`` from ``shapes.py`` — no
  level-4 children are emitted even when the KG carries deeper neighbors.
* Labels are sanitized (``sanitize.sanitize_text`` plus mermaid-unsafe
  ASCII filtering) and empty labels fall back to a stable ``unnamed``
  placeholder.

The renderer itself never calls the LM and never reads the network; the
tests therefore stay fully offline.
"""

from __future__ import annotations

from x_likes_mcp.synthesis.kg import (
    KG,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    handle_id,
    tweet_id,
)
from x_likes_mcp.synthesis.mindmap import render_mindmap
from x_likes_mcp.synthesis.shapes import MAX_MINDMAP_DEPTH

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_lines(block: str) -> list[str]:
    return block.splitlines()


def _inside_fence(block: str) -> list[str]:
    """Return the lines between the opening ```mermaid fence and the closing ``` fence."""

    lines = _split_lines(block)
    assert lines[0] == "```mermaid", f"expected leading mermaid fence, got {lines[0]!r}"
    assert lines[-1] == "```", f"expected trailing fence, got {lines[-1]!r}"
    return lines[1:-1]


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestEmpty:
    def test_empty_kg_renders_root_only(self) -> None:
        kg = KG()
        out = render_mindmap("hello world", kg)
        inside = _inside_fence(out)
        # First inside-line is the ``mindmap`` directive; second is the root.
        assert inside[0] == "mindmap"
        assert inside[1].lstrip(" ").startswith("root((")
        assert "hello world" in inside[1]
        # No category headers — the empty KG yields exactly two lines inside
        # the fence (the directive and the root).
        assert len(inside) == 2


class TestRootLabel:
    def test_root_label_sanitized(self) -> None:
        # ANSI escape, BOM, plus a mermaid-unsafe pair of brackets.
        query = "\x1b[31mfoo (bar)﻿"
        out = render_mindmap(query, KG())
        inside = _inside_fence(out)
        root_line = inside[1]
        # The ANSI escape must be gone.
        assert "\x1b" not in root_line
        # The BOM must be gone.
        assert "﻿" not in root_line
        # The mermaid-unsafe parens around "bar" must be stripped from the
        # *label*. The wrapping ``root((...))`` brackets remain, so we only
        # check the inner label by extracting it.
        assert root_line.lstrip(" ").startswith("root((")
        assert root_line.rstrip().endswith("))")
        inner = root_line.lstrip(" ")[len("root((") : -2]
        assert "(" not in inner
        assert ")" not in inner
        assert "foo" in inner
        assert "bar" in inner


class TestLevel1Categories:
    def test_level1_category_emitted_only_when_non_empty(self) -> None:
        kg = KG()
        kg.add_node(Node(id=handle_id("alice"), kind=NodeKind.HANDLE, label="alice", weight=1.0))
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        inside = _inside_fence(out)
        text = "\n".join(inside)
        assert "Authors" in text
        assert "Sources" not in text
        assert "Themes" not in text
        assert "Hashtags" not in text


class TestLevel2:
    def test_level2_top_k_per_category(self) -> None:
        kg = KG()
        for i in range(10):
            kg.add_node(
                Node(
                    id=handle_id(f"user{i:02d}"),
                    kind=NodeKind.HANDLE,
                    label=f"user{i:02d}",
                    weight=float(10 - i),
                ),
            )
        out = render_mindmap("q", kg)
        inside = _inside_fence(out)
        # Find the ``Authors`` category line, then count its level-2 children
        # (the lines that follow at indent + 2 spaces, until indent drops back).
        author_idx = next(i for i, line in enumerate(inside) if line.lstrip(" ") == "Authors")
        author_indent = _indent_of(inside[author_idx])
        children_indent = author_indent + 2
        children = []
        for line in inside[author_idx + 1 :]:
            ind = _indent_of(line)
            if ind <= author_indent:
                break
            if ind == children_indent:
                children.append(line)
        assert len(children) == 8

    def test_level2_ordered_by_weight_desc(self) -> None:
        kg = KG()
        kg.add_node(Node(id=handle_id("low"), kind=NodeKind.HANDLE, label="low", weight=1.0))
        kg.add_node(Node(id=handle_id("mid"), kind=NodeKind.HANDLE, label="mid", weight=2.0))
        kg.add_node(Node(id=handle_id("hi"), kind=NodeKind.HANDLE, label="hi", weight=3.0))
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        inside = _inside_fence(out)
        order: list[str] = []
        for line in inside:
            stripped = line.lstrip(" ")
            for label in ("hi", "mid", "low"):
                if stripped == label:
                    order.append(label)
                    break
        assert order == ["hi", "mid", "low"]


class TestLevel3:
    def test_level3_tweet_neighbors_under_handle(self) -> None:
        kg = KG()
        kg.add_node(Node(id=handle_id("foo"), kind=NodeKind.HANDLE, label="foo", weight=1.0))
        kg.add_node(Node(id=tweet_id("1"), kind=NodeKind.TWEET, label="tweet one", weight=1.0))
        kg.add_node(Node(id=tweet_id("2"), kind=NodeKind.TWEET, label="tweet two", weight=1.0))
        kg.add_edge(Edge(src=handle_id("foo"), dst=tweet_id("1"), kind=EdgeKind.AUTHORED_BY))
        kg.add_edge(Edge(src=handle_id("foo"), dst=tweet_id("2"), kind=EdgeKind.AUTHORED_BY))
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        inside = _inside_fence(out)
        # Find the ``foo`` line (level 2 under Authors); collect its level-3 children.
        foo_idx = next(i for i, line in enumerate(inside) if line.lstrip(" ") == "foo")
        foo_indent = _indent_of(inside[foo_idx])
        children_indent = foo_indent + 2
        children: list[str] = []
        for line in inside[foo_idx + 1 :]:
            ind = _indent_of(line)
            if ind <= foo_indent:
                break
            if ind == children_indent:
                children.append(line.lstrip(" "))
        assert children == ["tweet one", "tweet two"]


class TestDepthCap:
    def test_depth_cap_honored(self) -> None:
        # Build a chain handle -> tweet -> url -> concept where each link is
        # a different node kind. The CONCEPT only exists as a deep-walk leaf
        # (it is not added to the KG as a top-level node), so it must NOT
        # appear in the rendering — both because the depth cap stops at
        # level 4 and because the level-4 emitter only takes TWEET-kind
        # neighbors of the level-3 entries.
        kg = KG()
        kg.add_node(Node(id=handle_id("foo"), kind=NodeKind.HANDLE, label="foo", weight=1.0))
        kg.add_node(Node(id=tweet_id("1"), kind=NodeKind.TWEET, label="t1", weight=1.0))
        # A second tweet hangs off the first one; if the renderer ever
        # accidentally recurses, we'd see ``t2`` at level 5 (indent 10),
        # which the indent assertion catches.
        kg.add_node(Node(id=tweet_id("2"), kind=NodeKind.TWEET, label="t2", weight=1.0))
        kg.add_edge(Edge(src=handle_id("foo"), dst=tweet_id("1"), kind=EdgeKind.AUTHORED_BY))
        kg.add_edge(Edge(src=tweet_id("1"), dst=tweet_id("2"), kind=EdgeKind.MENTIONS))
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        inside = _inside_fence(out)
        # MAX_MINDMAP_DEPTH=4 means root + 3 child levels. After stripping
        # the ``mindmap`` directive line, every remaining line must indent
        # by at most 4 * 2 = 8 spaces (root sits at indent 2; the deepest
        # legal line is level 4 at indent 8).
        assert MAX_MINDMAP_DEPTH == 4
        max_indent = MAX_MINDMAP_DEPTH * 2
        for line in inside[1:]:
            assert _indent_of(line) <= max_indent, line
        # ``t1`` should appear at level 4 (the legal leaf); ``t2`` should
        # NOT appear because that would require recursing past the cap.
        text = "\n".join(inside)
        assert "t1" in text
        assert "t2" not in text


class TestUnsafeChars:
    def test_unsafe_characters_dropped_from_labels(self) -> None:
        kg = KG()
        kg.add_node(
            Node(
                id=handle_id("badname"),
                kind=NodeKind.HANDLE,
                label="@bad/name",
                weight=1.0,
            ),
        )
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        inside = _inside_fence(out)
        # Find the level-2 label line. The category header "Authors" is at
        # level 1, so its child sits at indent 6 (root=2, Authors=4, child=6).
        child_lines = [line for line in inside if _indent_of(line) == 6]
        assert child_lines, inside
        label = child_lines[0].lstrip(" ")
        assert "@" not in label
        assert "/" not in label
        assert "bad" in label
        assert "name" in label

    def test_empty_label_falls_back_to_unnamed(self) -> None:
        kg = KG()
        # A label that sanitizes to nothing: only mermaid-unsafe ASCII.
        kg.add_node(
            Node(
                id=handle_id("bad"),
                kind=NodeKind.HANDLE,
                label="@@//::",
                weight=1.0,
            ),
        )
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        inside = _inside_fence(out)
        text = "\n".join(inside)
        assert "unnamed" in text


class TestMermaidBlockShape:
    def test_mermaid_block_is_well_formed(self) -> None:
        out = render_mindmap("q", KG())
        lines = _split_lines(out)
        # ```mermaid opener.
        assert lines[0] == "```mermaid"
        # ``` closer.
        assert lines[-1] == "```"
        # Directive on the first inside-line.
        assert lines[1] == "mindmap"
        # All inside lines indented in 2-space steps.
        for line in lines[2:-1]:
            assert _indent_of(line) % 2 == 0, line


class TestMinLevel3Weight:
    def test_default_min_weight_drops_singletons(self) -> None:
        """Default ``min_level3_weight=2.0`` filters out one-off entities."""

        kg = KG()
        kg.add_node(Node(id=handle_id("once"), kind=NodeKind.HANDLE, label="once", weight=1.0))
        kg.add_node(
            Node(id=handle_id("recurring"), kind=NodeKind.HANDLE, label="recurring", weight=3.0)
        )
        out = render_mindmap("q", kg)
        text = "\n".join(_inside_fence(out))
        assert "recurring" in text
        assert "once" not in text

    def test_min_weight_zero_keeps_everything(self) -> None:
        kg = KG()
        kg.add_node(Node(id=handle_id("once"), kind=NodeKind.HANDLE, label="once", weight=1.0))
        out = render_mindmap("q", kg, min_level3_weight=0.0)
        text = "\n".join(_inside_fence(out))
        assert "once" in text

    def test_min_weight_drops_entire_category_when_no_survivors(self) -> None:
        """A category whose top entities are all below the threshold is
        omitted entirely from the rendered mindmap (no empty header)."""

        kg = KG()
        kg.add_node(Node(id=handle_id("a"), kind=NodeKind.HANDLE, label="a", weight=1.0))
        kg.add_node(Node(id=handle_id("b"), kind=NodeKind.HANDLE, label="b", weight=1.0))
        out = render_mindmap("q", kg, min_level3_weight=2.0)
        text = "\n".join(_inside_fence(out))
        assert "Authors" not in text


class TestTopPerCategoryArg:
    def test_top_per_category_arg_overrides_default(self) -> None:
        kg = KG()
        for i in range(5):
            kg.add_node(
                Node(
                    id=handle_id(f"user{i}"),
                    kind=NodeKind.HANDLE,
                    label=f"user{i}",
                    weight=float(5 - i),
                ),
            )
        out = render_mindmap("q", kg, top_per_category=2)
        inside = _inside_fence(out)
        author_idx = next(i for i, line in enumerate(inside) if line.lstrip(" ") == "Authors")
        author_indent = _indent_of(inside[author_idx])
        children_indent = author_indent + 2
        children: list[str] = []
        for line in inside[author_idx + 1 :]:
            ind = _indent_of(line)
            if ind <= author_indent:
                break
            if ind == children_indent:
                children.append(line)
        assert len(children) == 2
