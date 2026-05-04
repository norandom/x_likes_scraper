"""Tests for :mod:`x_likes_mcp.__main__`.

Covers the small CLI helpers that don't fit anywhere else: the local
media-file resolver and the snippet-expansion / metadata formatting
helpers used by the ``--search`` printer. The full ``main()`` wiring is
covered by the integration smoke tests; the unit tests here pin the
defensive validation that protects ``Path.glob`` from a malformed
``tweet_id`` (defense-in-depth recommended by the codeql review).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from x_likes_mcp.__main__ import (
    _expand_snippet,
    _format_meta_line,
    _local_media_files,
)


def test_local_media_files_returns_matching_files(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "12345_0.jpg").write_bytes(b"")
    (media / "12345_1.png").write_bytes(b"")
    (media / "99999_0.jpg").write_bytes(b"")

    result = _local_media_files("12345", media)

    assert [p.name for p in result] == ["12345_0.jpg", "12345_1.png"]


def test_local_media_files_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    assert _local_media_files("12345", tmp_path / "absent") == []


def test_local_media_files_returns_empty_when_no_match(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "11111_0.jpg").write_bytes(b"")

    assert _local_media_files("99999", media) == []


@pytest.mark.parametrize(
    "bad_id",
    [
        "../etc/passwd",
        "/absolute/path",
        "12345/..",
        "12*",
        "../../secret",
        "abc123",  # contains letters
        "",
        "12 34",
    ],
)
def test_local_media_files_rejects_non_numeric_tweet_id(tmp_path: Path, bad_id: str) -> None:
    """``tweet_id`` must match ``^[0-9]+$`` to reach the glob.

    Twitter IDs are numeric; anything else is either programmer error or
    an attempt to escape ``media_dir`` via ``Path.glob``'s literal
    handling of ``..`` / ``/`` / wildcard characters.
    """

    media = tmp_path / "media"
    media.mkdir()

    assert _local_media_files(bad_id, media) == []


def test_expand_snippet_strips_tco_appends_resolved() -> None:
    out = _expand_snippet(
        "check this https://t.co/abc123",
        ["https://github.com/owner/repo"],
    )
    # Two-space separator between snippet and resolved URLs is intentional.
    assert out == "check this  https://github.com/owner/repo"


def test_expand_snippet_collapses_whitespace() -> None:
    out = _expand_snippet(
        "before https://t.co/x https://t.co/y after",
        [],
    )
    assert out == "before after"


def test_expand_snippet_no_urls_returns_clean_text() -> None:
    out = _expand_snippet("plain text", [])
    assert out == "plain text"


def test_format_meta_line_plain() -> None:
    hit = {
        "score": 12.345,
        "walker_relevance": 0.6,
        "year_month": "2026-04",
        "handle": "alice",
        "tweet_id": "999",
    }
    line = _format_meta_line(1, hit, color=False)

    assert "score=12.35" in line
    assert "wr=0.60" in line
    assert "2026-04" in line
    assert "@alice" in line
    assert "id=999" in line
    assert "\x1b[" not in line  # no ANSI when color=False


def test_format_meta_line_color_emits_ansi() -> None:
    hit = {
        "score": 1.0,
        "walker_relevance": 0.0,
        "year_month": "",
        "handle": "",
        "tweet_id": "1",
    }
    line = _format_meta_line(1, hit, color=True)

    assert "\x1b[1m" in line  # bold prefix
    assert "\x1b[2m" in line  # dim body
    assert "\x1b[0m" in line  # reset
