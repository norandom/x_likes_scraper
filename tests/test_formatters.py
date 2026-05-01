"""Tests for ``x_likes_exporter.formatters``.

Covers Requirements 8.3 and 9.1-9.5 of the codebase-foundation spec by
exercising each formatter (JSON, Markdown, HTML, Pandas) against the same
hand-built three-tweet fixture:

* ``tweet1`` - a plain tweet with one ``photo`` media item and a parseable
  ``created_at`` in January 2025.
* ``tweet2`` - a retweet (``is_retweet=True``) with a parseable ``created_at``
  in March 2025.
* ``tweet3`` - a tweet with an unparseable ``created_at`` ("garbage"), which
  the Markdown grouping logic must route under the ``unknown`` month bucket.

The tests assert observable file-level behavior (round-trip equality for
JSON, section ordering and group routing for Markdown, ``<div class='tweet'>``
multiplicity for HTML, schema columns for Pandas) without reaching into
private formatter state. All file writes are scoped to ``tmp_path``.
"""

from __future__ import annotations

import json
from typing import List

import pandas as pd
import pytest

from x_likes_exporter.formatters import (
    HTMLFormatter,
    JSONFormatter,
    MarkdownFormatter,
    PandasFormatter,
)
from x_likes_exporter.models import Media, Tweet, User


@pytest.fixture
def sample_tweets() -> List[Tweet]:
    """Return three hand-built tweets covering the documented edge cases.

    The trio is intentionally small so each assertion below can name the
    specific tweet it is targeting (plain-with-media, retweet, unparseable-
    date). Distinct ``screen_name`` values let the HTML/Markdown tests
    confirm per-tweet rendering by handle.
    """
    user1 = User(
        id="100",
        screen_name="alice",
        name="Alice Example",
        verified=True,
    )
    user2 = User(
        id="200",
        screen_name="bob",
        name="Bob Example",
        verified=False,
    )
    user3 = User(
        id="300",
        screen_name="carol",
        name="Carol Example",
        verified=False,
    )

    tweet1 = Tweet(
        id="1001",
        text="Hello world from Alice",
        created_at="Sun Jan 12 10:00:00 +0000 2025",
        user=user1,
        retweet_count=2,
        favorite_count=5,
        reply_count=1,
        view_count=100,
        media=[
            Media(
                type="photo",
                url="http://example.com/photo/1",
                media_url="http://example.com/1.jpg",
            )
        ],
        hashtags=["greeting"],
    )

    tweet2 = Tweet(
        id="2002",
        text="RT something interesting",
        created_at="Wed Mar 05 14:00:00 +0000 2025",
        user=user2,
        retweet_count=10,
        favorite_count=20,
        is_retweet=True,
    )

    # ``created_at`` is the empty string here rather than a freeform
    # token like "garbage". Both shapes are "unparseable" per the
    # ``parse_x_datetime`` contract (Requirement 8.2), but the empty
    # string also round-trips through ``pd.to_datetime`` as ``NaT``
    # without raising, so the same fixture exercises both the Markdown
    # ``unknown`` routing path (Req 9.3) and the Pandas
    # one-row-per-tweet path (Req 9.5).
    tweet3 = Tweet(
        id="3003",
        text="Tweet with a busted timestamp",
        created_at="",
        user=user3,
    )

    return [tweet1, tweet2, tweet3]


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------


def test_json_formatter_round_trip(sample_tweets, tmp_path) -> None:
    """JSON export round-trips to ``[t.to_dict() for t in tweets]``.

    Requirement 9.1: the JSON formatter writes a faithful list-of-dicts
    serialization of the model. Re-loading the file and comparing to
    ``to_dict()`` for each tweet pins the contract end-to-end.
    """
    output_file = tmp_path / "tweets.json"

    JSONFormatter.export(sample_tweets, str(output_file))

    with output_file.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)

    expected = [t.to_dict() for t in sample_tweets]
    assert loaded == expected


# ---------------------------------------------------------------------------
# MarkdownFormatter
# ---------------------------------------------------------------------------


def test_markdown_formatter_basic(sample_tweets, tmp_path) -> None:
    """Markdown export groups by month, sorts reverse-chrono, and renders blocks.

    Requirement 9.2 / 9.3: tweets group by ``YYYY-MM`` (parseable) or the
    ``Unknown Date`` bucket (unparseable), sorted in reverse chronological
    order. Each tweet block surfaces the handle, name, date, text, and
    stats so the human-readable export carries the same data the JSON
    export does.
    """
    output_file = tmp_path / "tweets.md"

    MarkdownFormatter().export(sample_tweets, str(output_file), include_media=False)

    content = output_file.read_text(encoding="utf-8")

    # The unparseable tweet routes under the ``Unknown Date`` heading the
    # formatter emits for the ``unknown`` month_key.
    assert "## Unknown Date" in content

    # Both parseable months appear as their own ``## YYYY-MM (N tweets)``
    # sections.
    assert "## 2025-03" in content
    assert "## 2025-01" in content

    # Reverse chronological order: March precedes January in the file.
    march_idx = content.index("## 2025-03")
    january_idx = content.index("## 2025-01")
    assert march_idx < january_idx

    # Every tweet block carries the rendering essentials. Handle is rendered
    # via the ``[@<screen_name>]`` link; name appears bold; the formatted
    # date string for the parseable cases appears verbatim; the unparseable
    # case falls through to the original ``created_at`` string.
    assert "[@alice]" in content
    assert "[@bob]" in content
    assert "[@carol]" in content

    assert "Alice Example" in content
    assert "Bob Example" in content
    assert "Carol Example" in content

    # Parseable dates render via ``%Y-%m-%d %H:%M:%S``.
    assert "2025-01-12 10:00:00" in content
    assert "2025-03-05 14:00:00" in content

    # Tweet text bodies appear in the rendered output.
    assert "Hello world from Alice" in content
    assert "RT something interesting" in content
    assert "Tweet with a busted timestamp" in content

    # Stats line uses the documented emoji glyphs; tweet1 has retweet,
    # favorite, reply, and view counts > 0 so all four show up.
    assert "🔄 2" in content
    assert "❤️ 5" in content
    assert "💬 1" in content
    assert "👁️ 100" in content


def test_markdown_formatter_unknown_routing(sample_tweets, tmp_path) -> None:
    """Tweet with unparseable ``created_at`` lands in the ``Unknown Date`` group.

    Requirement 9.3: explicit assertion that the ``unknown`` month_key
    bucket actually owns the unparseable-date tweet's body, not just that
    the heading exists somewhere in the file.
    """
    output_file = tmp_path / "tweets.md"

    MarkdownFormatter().export(sample_tweets, str(output_file), include_media=False)

    content = output_file.read_text(encoding="utf-8")

    unknown_header = "## Unknown Date"
    assert unknown_header in content

    # Slice from the ``Unknown Date`` heading to the next ``## `` heading
    # (or end-of-file) - the unparseable tweet's body must live in that
    # window. Other tweets (alice, bob) must NOT.
    unknown_start = content.index(unknown_header)
    after_unknown = content[unknown_start + len(unknown_header) :]
    next_heading = after_unknown.find("\n## ")
    if next_heading == -1:
        unknown_section = after_unknown
    else:
        unknown_section = after_unknown[:next_heading]

    assert "Tweet with a busted timestamp" in unknown_section
    assert "[@carol]" in unknown_section
    # Sanity: the other handles must not have leaked into this section.
    assert "[@alice]" not in unknown_section
    assert "[@bob]" not in unknown_section


# ---------------------------------------------------------------------------
# HTMLFormatter
# ---------------------------------------------------------------------------


def test_html_formatter_three_divs(sample_tweets, tmp_path) -> None:
    """HTML export emits one document with three ``<div class='tweet'>`` blocks.

    Requirement 9.4: the HTML formatter produces a single self-contained
    page with one tweet ``div`` per input tweet, and each tweet's
    ``screen_name`` is rendered into the page so the reader can identify
    the author.
    """
    output_file = tmp_path / "tweets.html"

    HTMLFormatter().export(sample_tweets, str(output_file))

    html = output_file.read_text(encoding="utf-8")

    # Single-document structure: one html/body pair, no concatenation
    # artifacts.
    assert html.count("<!DOCTYPE html>") == 1
    assert html.count("<html") == 1
    assert html.count("</html>") == 1

    # Three tweet blocks, exactly. The formatter quotes the class with
    # single quotes ("<div class='tweet'>") so the assertion mirrors that.
    assert html.count("<div class='tweet'>") == 3

    # Every tweet's screen_name appears in the rendered HTML (prefixed
    # with ``@`` per the formatter's user block).
    assert "@alice" in html
    assert "@bob" in html
    assert "@carol" in html


# ---------------------------------------------------------------------------
# PandasFormatter
# ---------------------------------------------------------------------------


def test_pandas_formatter_dataframe(sample_tweets) -> None:
    """``to_dataframe`` returns three rows with the documented column set.

    Requirement 9.5: the Pandas DataFrame export is the structured
    tabular form of the tweet collection. Three input tweets yield three
    rows, and the column schema includes the documented identifier,
    text, timestamp, and user handle columns at minimum.
    """
    df = PandasFormatter.to_dataframe(sample_tweets)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    # Documented column subset must be present. The formatter emits more
    # columns than listed here; this check asserts the contract floor
    # without locking in incidental columns.
    required_columns = {
        "tweet_id",
        "text",
        "created_at",
        "user_screen_name",
        "user_name",
        "is_retweet",
        "has_media",
        "media_count",
    }
    assert required_columns.issubset(set(df.columns))

    # The retweet row carries ``is_retweet=True``; the others do not.
    retweet_row = df[df["tweet_id"] == "2002"].iloc[0]
    assert bool(retweet_row["is_retweet"]) is True

    # The plain-with-media row reports media_count == 1.
    media_row = df[df["tweet_id"] == "1001"].iloc[0]
    assert int(media_row["media_count"]) == 1
    assert bool(media_row["has_media"]) is True

    # All three screen_names made it into the DataFrame.
    assert set(df["user_screen_name"]) == {"alice", "bob", "carol"}
