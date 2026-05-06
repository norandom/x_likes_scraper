"""Mermaid mindmap renderer for the synthesis-report feature (task 2.5).

Generates a depth-capped mermaid ``mindmap`` block whose root is the user
query and whose level-1 children are the entity categories present in
the KG (Authors, Sources, Themes, Hashtags). The renderer is pure (no
LM, no network, no disk) and delegates label safety to
:mod:`x_likes_mcp.sanitize` plus an additional pass that strips the
ASCII subset that mermaid's parser rejects (``()[]{}``, ``"``, ``/``,
``@``, ``:``).

Tree layout (root counts as level 1; depth cap is :data:`MAX_MINDMAP_DEPTH`
from :mod:`x_likes_mcp.synthesis.shapes`):

* Level 1 — the root, the sanitized user query.
* Level 2 — entity-category headers (Authors / Sources / Themes /
  Hashtags). Categories whose corresponding :class:`NodeKind` has no
  members are omitted entirely.
* Level 3 — the top-K nodes per category by weight.
* Level 4 — tweet neighbors of each top-K node (via
  :meth:`KG.neighbors`), capped at ``MAX_MINDMAP_DEPTH`` levels total.

An empty KG still emits a valid mermaid block: the fenced opener, the
``mindmap`` directive, the root line, and the fenced closer.

See ``.kiro/specs/synthesis-report/design.md`` (``mindmap`` component)
and requirements 8.1, 8.2.
"""

from __future__ import annotations

import re

from x_likes_mcp.sanitize import sanitize_text
from x_likes_mcp.synthesis.kg import KG, Node, NodeKind
from x_likes_mcp.synthesis.shapes import MAX_MINDMAP_DEPTH

__all__ = [
    "DEFAULT_MIN_LEVEL3_WEIGHT",
    "DEFAULT_TOP_PER_CATEGORY",
    "render_mindmap",
]


# Default per-category fan-out. Eight is small enough to keep the
# rendered diagram legible in GitHub / Obsidian / VS Code preview but
# large enough to expose the interesting tail beyond the top three.
DEFAULT_TOP_PER_CATEGORY: int = 8


# Default minimum cumulative weight required for a level-3 (top-K)
# node to surface. Entities mentioned only once across the whole hit
# set are likely tangential — they accumulate weight via ``add_node``
# only when they recur, so the threshold filters one-off proper nouns
# without removing genuinely topical ones. Set to ``0.0`` to disable.
DEFAULT_MIN_LEVEL3_WEIGHT: float = 2.0


# Order is fixed so the output is deterministic across runs and across
# minor KG mutations. The label is the on-screen header; the kind is the
# :class:`NodeKind` whose nodes feed into level 3 of the tree.
_CATEGORIES: tuple[tuple[str, NodeKind], ...] = (
    ("Authors", NodeKind.HANDLE),
    ("Sources", NodeKind.DOMAIN),
    ("Themes", NodeKind.CONCEPT),
    ("Hashtags", NodeKind.HASHTAG),
)


# Mermaid's mindmap parser rejects these ASCII glyphs inside node
# labels: parentheses, square brackets, and curly braces double as node
# shape delimiters; a bare double quote opens a string literal; ``/``,
# ``@``, and ``:`` are interpreted as mindmap modifiers / icon prefixes
# in newer mermaid versions. We strip every occurrence rather than try
# to escape — escaping varies by renderer (GitHub vs VS Code vs Obsidian
# vs mermaid-cli), and the labels are display-only.
_MERMAID_UNSAFE_RE = re.compile(r"[()\[\]{}\"/@:]")
_WHITESPACE_RUN_RE = re.compile(r"\s+")


# Stable placeholder for labels that sanitize to an empty string. The
# placeholder itself is ASCII-only and contains no mermaid-unsafe
# characters, so it survives ``_safe_label`` unchanged.
_EMPTY_LABEL_PLACEHOLDER: str = "unnamed"


def _safe_label(raw: str) -> str:
    """Return a mermaid-safe single-line label for ``raw``.

    Pipeline:

    1. :func:`sanitize_text` — strips ANSI escape sequences, BiDi /
       formatting codepoints, the BOM, and C0 / C1 controls.
    2. Drop the mermaid-unsafe ASCII subset (parens, brackets, braces,
       double quote, slash, ``@``, colon).
    3. Collapse runs of any whitespace into a single space and strip.
    4. If the resulting label is empty, fall back to
       :data:`_EMPTY_LABEL_PLACEHOLDER` so the structural slot is still
       visible to readers.
    """

    cleaned = sanitize_text(raw)
    cleaned = _MERMAID_UNSAFE_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RUN_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return _EMPTY_LABEL_PLACEHOLDER
    return cleaned


def _indent(level: int) -> str:
    """Return the 2-space indentation prefix for ``level`` (1-based)."""

    # Level 1 (the root) sits at one indent step under the ``mindmap``
    # directive. Level 2 is two steps, and so on.
    return "  " * level


def render_mindmap(
    query: str,
    kg: KG,
    max_depth: int = MAX_MINDMAP_DEPTH,
    top_per_category: int = DEFAULT_TOP_PER_CATEGORY,
    min_level3_weight: float = DEFAULT_MIN_LEVEL3_WEIGHT,
) -> str:
    r"""Render a depth-capped mermaid ``mindmap`` block for ``query`` + ``kg``.

    The output is always a fenced ``mermaid`` block (``\`\`\`mermaid``
    opener, ``\`\`\``  closer) wrapping a ``mindmap`` directive plus the
    tree. Empty KGs still produce a valid block — only the root is
    emitted.

    Parameters
    ----------
    query:
        The user query, used as the root label. Sanitized and
        mermaid-safe-filtered before emission.
    kg:
        The in-memory knowledge graph. Read-only; the renderer only
        calls :meth:`KG.top_entities` and :meth:`KG.neighbors`.
    max_depth:
        Maximum number of tree levels to emit (root counts as level 1).
        Defaults to :data:`MAX_MINDMAP_DEPTH`. Anything beyond is
        truncated silently — readers see a well-formed but smaller tree.
    top_per_category:
        Number of top-K nodes to emit under each level-2 category
        header. Defaults to :data:`DEFAULT_TOP_PER_CATEGORY`. Tie-breaks
        on weight are deferred to :meth:`KG.top_entities`, which sorts
        by ``(-weight, id)`` for determinism.
    """

    lines: list[str] = ["```mermaid", "mindmap"]

    # --- level 1 ---------------------------------------------------------
    root_label = _safe_label(query)
    lines.append(f"{_indent(1)}root(({root_label}))")

    # ``max_depth`` of 1 means "root only". Honor the cap up front so the
    # category-loop never runs in the trivial degenerate case.
    if max_depth <= 1:
        lines.append("```")
        return "\n".join(lines)

    # --- level 2: category headers --------------------------------------
    for category_label, kind in _CATEGORIES:
        top_nodes: list[Node] = kg.top_entities(kind, top_per_category)
        # Apply the min-weight floor before deciding whether the category
        # has any survivors. Below-threshold one-off mentions ("Yeah",
        # "The") would otherwise leak through the regex extractor and
        # eat a category slot — filter them out here so the visual still
        # honors the empty-category-header skip.
        if min_level3_weight > 0:
            top_nodes = [n for n in top_nodes if n.weight >= min_level3_weight]
        if not top_nodes:
            # Skip categories with no nodes so empty headers don't clutter
            # the rendered diagram.
            continue
        lines.append(f"{_indent(2)}{category_label}")

        if max_depth <= 2:
            continue

        # --- level 3: top-K nodes per category --------------------------
        for node in top_nodes:
            lines.append(f"{_indent(3)}{_safe_label(node.label)}")

            if max_depth <= 3:
                continue

            # --- level 4: tweet neighbors -------------------------------
            for neighbor in kg.neighbors(node.id):
                if neighbor.kind is not NodeKind.TWEET:
                    # Only tweets are interesting at the leaf level for
                    # the mindmap; other neighbor kinds (concepts,
                    # domains) would just duplicate the level-2 headers.
                    continue
                lines.append(f"{_indent(4)}{_safe_label(neighbor.label)}")

    lines.append("```")
    return "\n".join(lines)
