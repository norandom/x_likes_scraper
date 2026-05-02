"""Pure-Python parser for the per-month likes Markdown.

Parses files of the shape produced by
``x_likes_exporter.formatters.MarkdownFormatter.export(..., omit_global_header=True)``:

    ## YYYY-MM (N tweets)


    ### [@handle](https://x.com/handle)
    **Display Name** ...
    *YYYY-MM-DD HH:MM:SS*

    Tweet text body...

    *stats line*

    🔗 [View on X](https://x.com/handle/status/{id})

    ---

Each ``### ...`` block becomes a :class:`TreeNode`. The file's own ``## YYYY-MM``
heading is *not* a tweet section; only ``### ``-prefixed sections are.

This module is deliberately stdlib-only: no LLM call, no network access, no
OpenAI SDK import. The walker is the single LLM call site in this package.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Public dataclasses

@dataclass(frozen=True)
class TreeNode:
    """One liked tweet parsed out of a per-month Markdown file."""

    year_month: str       # "2026-04"
    tweet_id: str         # extracted from the canonical "View on X" link
    handle: str           # screen_name without the leading "@"
    text: str             # section body minus the heading and the link line
    raw_section: str      # full section text including the heading


@dataclass(frozen=True)
class TweetTree:
    """In-memory tree of all parsed liked tweets."""

    nodes_by_month: dict[str, list[TreeNode]]
    nodes_by_id:    dict[str, TreeNode]


# ---------------------------------------------------------------------------
# Internal regexes

_FILENAME_RE = re.compile(r"^likes_(\d{4}-\d{2})\.md$")

# The canonical link line. We accept both x.com and twitter.com just in case,
# and tolerate an empty handle path component (the exporter occasionally
# emits ``https://x.com//status/{id}`` for tweets whose handle was stripped).
_TWEET_URL_RE = re.compile(
    r"https?://(?:x|twitter)\.com/([^/\s)]*)/status/(\d+)"
)

# Heading forms emitted by MarkdownFormatter._format_tweet:
#   ### [@handle](https://x.com/handle)
# We also accept the bare fallback ``### @handle`` per the design.
_HEADING_LINK_RE = re.compile(r"^###\s+\[@([^\]]+)\]\(")
_HEADING_BARE_RE = re.compile(r"^###\s+@([A-Za-z0-9_]+)")

# A "View on X" line. Kept loose so a stray emoji or whitespace doesn't break it.
_VIEW_ON_X_LINE_RE = re.compile(r"^\s*🔗\s*\[View on X\]\(.*\)\s*$")


# ---------------------------------------------------------------------------
# Section splitter

def _split_sections(content: str) -> list[str]:
    """Split file content into per-tweet ``### ...`` sections.

    The file's ``## YYYY-MM`` heading and any preamble before the first
    ``### `` line are discarded. Each returned string starts with ``### ``.
    """
    # Anchor on lines that *begin* with "### " (multiline mode). Using a
    # lookahead split keeps the heading attached to its body.
    parts = re.split(r"(?m)^(?=### )", content)
    return [p for p in parts if p.startswith("### ")]


def _extract_handle(section: str, link_handle: str | None) -> str | None:
    """Return the handle from the section heading; fall back to ``link_handle``."""
    first_line = section.splitlines()[0] if section else ""
    m = _HEADING_LINK_RE.match(first_line)
    if m:
        return m.group(1)
    m = _HEADING_BARE_RE.match(first_line)
    if m:
        return m.group(1)
    return link_handle


def _strip_heading_and_link(section: str) -> str:
    """Return the section body with the heading line and the View-on-X line removed.

    Other content (display name, date, tweet text, stats, media block) is
    preserved verbatim. Leading/trailing blank lines are stripped so callers
    get clean text.
    """
    lines = section.splitlines()
    # Drop the heading line (always the first line of a section).
    if lines and lines[0].startswith("### "):
        lines = lines[1:]
    # Drop the "View on X" line wherever it appears.
    lines = [ln for ln in lines if not _VIEW_ON_X_LINE_RE.match(ln)]
    return "\n".join(lines).strip("\n")


# ---------------------------------------------------------------------------
# Public entry point

def build_tree(by_month_dir: Path) -> TweetTree:
    """Parse every ``likes_YYYY-MM.md`` under ``by_month_dir`` into a :class:`TweetTree`.

    Pure function: same input directory yields the same ``TweetTree`` (up to
    dataclass equality). No LLM call, no network access.

    Files whose names don't match ``likes_YYYY-MM.md`` are silently ignored.
    Sections with no recognizable tweet ID are logged to stderr and skipped.
    """
    by_month_dir = Path(by_month_dir)

    nodes_by_month: dict[str, list[TreeNode]] = {}
    nodes_by_id: dict[str, TreeNode] = {}

    if not by_month_dir.is_dir():
        # Empty tree; the caller (TweetIndex) is responsible for surfacing
        # the missing-directory error per Requirement 3.4.
        return TweetTree(nodes_by_month={}, nodes_by_id={})

    # Sort by filename for deterministic month ordering.
    for path in sorted(by_month_dir.iterdir()):
        if not path.is_file():
            continue
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        year_month = m.group(1)

        content = path.read_text(encoding="utf-8")
        sections = _split_sections(content)

        month_nodes: list[TreeNode] = []
        for section in sections:
            url_match = _TWEET_URL_RE.search(section)
            if not url_match:
                print(
                    f"tree: skipping section without tweet id in {path}",
                    file=sys.stderr,
                )
                continue

            link_handle = url_match.group(1)
            tweet_id = url_match.group(2)
            handle = _extract_handle(section, link_handle) or link_handle

            text = _strip_heading_and_link(section)
            raw_section = section.rstrip("\n")

            node = TreeNode(
                year_month=year_month,
                tweet_id=tweet_id,
                handle=handle,
                text=text,
                raw_section=raw_section,
            )
            month_nodes.append(node)
            # Last write wins on duplicate IDs across files. In practice the
            # exporter shouldn't produce duplicates, but we don't crash on them.
            nodes_by_id[tweet_id] = node

        if month_nodes:
            nodes_by_month[year_month] = month_nodes

    return TweetTree(nodes_by_month=nodes_by_month, nodes_by_id=nodes_by_id)
