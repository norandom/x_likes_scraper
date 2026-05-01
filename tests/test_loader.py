"""Tests for ``x_likes_exporter.loader``.

Covers Requirements 7.1 through 7.6 of the codebase-foundation spec:

- 7.1/7.2: ``load_export`` round-trips a ``likes.json`` file produced by the
  exporter into a ``list[Tweet]`` whose ``to_dict()`` payloads structurally
  match the source JSON entries.
- 7.3: ``load_export`` raises ``FileNotFoundError`` (with the offending path
  in the message) when the input file does not exist.
- 7.4: ``load_export`` raises ``ValueError`` when the file exists but does
  not contain a JSON list of tweet dicts (e.g. a top-level JSON string).
- 7.5: ``iter_monthly_markdown`` yields ``likes_YYYY-MM.md`` files in
  reverse-chronological order and silently skips non-matching files
  (e.g. ``notes.md``); raises ``FileNotFoundError`` for a missing directory.
- 7.6: The public read API (``load_export``, ``iter_monthly_markdown``) is
  importable directly from the top-level ``x_likes_exporter`` package.

These tests are pure: no network I/O, no cookie reads. The autouse guards in
``conftest.py`` will fail the suite if either is accidentally triggered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from x_likes_exporter.loader import iter_monthly_markdown, load_export


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "likes_export.json"


def test_load_export_round_trip() -> None:
    """``load_export`` reconstructs Tweets whose ``to_dict()`` matches source JSON."""
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tweets = load_export(FIXTURE_PATH)

    assert isinstance(tweets, list)
    assert len(tweets) == len(raw)
    assert [t.to_dict() for t in tweets] == raw


def test_load_export_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent path raises ``FileNotFoundError`` mentioning the path."""
    missing = tmp_path / "no.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_export(missing)

    assert str(missing) in str(exc_info.value)


def test_load_export_wrong_shape_raises(tmp_path: Path) -> None:
    """A JSON file whose top-level value is not a list raises ``ValueError``."""
    bad = tmp_path / "bad.json"
    # Top-level JSON string ("not a list") - syntactically valid JSON but
    # not the list-of-dicts shape ``load_export`` expects.
    bad.write_text('"not a list"', encoding="utf-8")

    with pytest.raises(ValueError):
        load_export(bad)


def test_iter_monthly_markdown_sorts_reverse_chrono(tmp_path: Path) -> None:
    """Yields matching files newest-first; non-matching files are skipped."""
    (tmp_path / "likes_2024-03.md").write_text("march 2024", encoding="utf-8")
    (tmp_path / "likes_2024-01.md").write_text("january 2024", encoding="utf-8")
    (tmp_path / "likes_2025-02.md").write_text("february 2025", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not a monthly file", encoding="utf-8")

    paths = list(iter_monthly_markdown(tmp_path))

    assert [p.name for p in paths] == [
        "likes_2025-02.md",
        "likes_2024-03.md",
        "likes_2024-01.md",
    ]
    # Sanity: notes.md is not anywhere in the result.
    assert all(p.name != "notes.md" for p in paths)


def test_iter_monthly_markdown_missing_dir_raises(tmp_path: Path) -> None:
    """A non-existent directory raises ``FileNotFoundError``."""
    missing_dir = tmp_path / "no_such_dir"

    with pytest.raises(FileNotFoundError):
        # ``iter_monthly_markdown`` is a generator; force evaluation.
        list(iter_monthly_markdown(missing_dir))


def test_top_level_imports() -> None:
    """The public read API is exposed at the package top level."""
    from x_likes_exporter import iter_monthly_markdown as top_iter
    from x_likes_exporter import load_export as top_load

    # Identity check: top-level names should be the same callables defined in
    # the loader module (not re-implementations).
    assert top_load is load_export
    assert top_iter is iter_monthly_markdown
