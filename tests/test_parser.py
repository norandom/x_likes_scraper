"""
Tests for the pure response-parsing functions in :mod:`x_likes_exporter.parser`.

These tests cover the contract documented in the parser module: a malformed
response yields ``[]`` from :func:`extract_tweets` and ``None`` from
:func:`extract_cursor`, and a per-entry failure inside :func:`extract_tweets`
is silently skipped. Hand-built dicts exercise specific edge cases (missing
``legacy``, missing ``core``, non-numeric ``views.count``, retweet variant,
quote variant) that the on-disk fixtures do not all hit.

Requirements covered: 4.1, 4.2, 4.3, 4.4, 4.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from x_likes_exporter.parser import extract_cursor, extract_tweets, parse_tweet

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """Load and parse a JSON fixture from ``tests/fixtures/<name>``."""
    path = FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# Helper to build the minimum response envelope that extract_tweets walks.
def _build_response(entries: list) -> dict[str, Any]:
    return {
        "data": {
            "user": {
                "result": {
                    "timeline": {
                        "timeline": {
                            "instructions": [
                                {
                                    "type": "TimelineAddEntries",
                                    "entries": entries,
                                }
                            ]
                        }
                    }
                }
            }
        }
    }


def _wrap_tweet_entry(tweet_data: dict[str, Any], entry_id: str = "tweet-1") -> dict[str, Any]:
    return {
        "entryId": entry_id,
        "content": {
            "entryType": "TimelineTimelineItem",
            "itemContent": {
                "tweet_results": {
                    "result": tweet_data,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# extract_tweets — fixture-based cases
# ---------------------------------------------------------------------------


def test_extract_tweets_from_success_fixture():
    """Success fixture yields three tweets with expected ids and screen_names."""
    response = load_fixture("likes_page_success.json")
    tweets = extract_tweets(response)

    assert len(tweets) == 3, f"expected 3 tweets, got {len(tweets)}"

    ids = [t.id for t in tweets]
    screen_names = [t.user.screen_name for t in tweets]

    assert ids == [
        "1000000000000000001",
        "1000000000000000002",
        "1000000000000000003",
    ]
    assert screen_names == [
        "test_user",
        "test_user_two",
        "test_user_three",
    ]

    # First tweet sanity check
    first = tweets[0]
    assert first.text == "This is a plain test tweet for the success fixture."
    assert first.favorite_count == 12
    assert first.view_count == 456


def test_extract_tweets_empty_fixture():
    """Empty fixture (cursors only, no tweet entries) returns []."""
    response = load_fixture("likes_page_empty.json")
    tweets = extract_tweets(response)

    assert tweets == []


def test_extract_tweets_malformed_fixture():
    """Malformed fixture (UserUnavailable error envelope) returns [] and does not raise."""
    response = load_fixture("likes_page_malformed.json")
    # The fixture is missing the timeline.timeline.instructions chain entirely.
    tweets = extract_tweets(response)

    assert tweets == []


# ---------------------------------------------------------------------------
# extract_cursor
# ---------------------------------------------------------------------------


def test_extract_cursor_present():
    """Bottom cursor entry from the success fixture is returned verbatim."""
    response = load_fixture("likes_page_success.json")
    cursor = extract_cursor(response)

    assert cursor == "DAABCgABREDACTEDBOTTOMCURSOR"


def test_extract_cursor_absent():
    """An empty/structurally absent response yields None instead of raising."""
    assert extract_cursor({}) is None
    assert extract_cursor(load_fixture("likes_page_malformed.json")) is None


# ---------------------------------------------------------------------------
# Per-entry edge cases (hand-built)
# ---------------------------------------------------------------------------


def test_extract_tweets_skips_entry_missing_legacy():
    """An entry whose ``tweet_results.result`` lacks ``legacy`` is skipped silently."""
    tweet_data_no_legacy = {
        "rest_id": "9999999999999999999",
        "core": {
            "user_results": {
                "result": {
                    "rest_id": "1",
                    "legacy": {"screen_name": "x", "name": "X"},
                }
            }
        },
        # no "legacy" key on purpose
    }
    response = _build_response([_wrap_tweet_entry(tweet_data_no_legacy)])

    assert extract_tweets(response) == []


def test_parse_tweet_missing_core_returns_default_user():
    """
    parse_tweet on a dict missing ``core`` does not raise; user fields fall back
    to empty strings / defaults. This documents the per-tweet fault tolerance
    that backs extract_tweets's skip behavior.
    """
    tweet_data = {
        "rest_id": "1234",
        "legacy": {
            "id_str": "1234",
            "full_text": "no core key",
            "created_at": "Sun Nov 09 11:05:17 +0000 2025",
        },
        # no "core" key
    }

    tweet = parse_tweet(tweet_data)

    assert tweet is not None
    assert tweet.id == "1234"
    assert tweet.text == "no core key"
    assert tweet.user.screen_name == ""
    assert tweet.user.id == ""


def test_parse_tweet_non_numeric_views_count():
    """
    A non-numeric ``views.count`` (e.g. "abc") leaves ``view_count`` at the
    default 0 instead of raising.
    """
    tweet_data = {
        "rest_id": "5555",
        "core": {
            "user_results": {
                "result": {
                    "rest_id": "42",
                    "legacy": {
                        "screen_name": "junk_views",
                        "name": "Junk Views",
                    },
                }
            }
        },
        "legacy": {
            "id_str": "5555",
            "full_text": "tweet with bogus views.count",
            "created_at": "Sun Nov 09 11:05:17 +0000 2025",
            "lang": "en",
        },
        "views": {"count": "abc"},
    }

    tweet = parse_tweet(tweet_data)

    assert tweet is not None
    assert tweet.view_count == 0


def test_parse_tweet_retweet_flag():
    """
    A ``retweeted_status_result`` inside ``legacy`` flips ``is_retweet`` to True.
    """
    tweet_data = {
        "rest_id": "7777",
        "core": {
            "user_results": {
                "result": {
                    "rest_id": "42",
                    "legacy": {"screen_name": "rt_user", "name": "RT User"},
                }
            }
        },
        "legacy": {
            "id_str": "7777",
            "full_text": "RT @someone: original",
            "created_at": "Sun Nov 09 11:05:17 +0000 2025",
            "lang": "en",
            "retweeted_status_result": {
                "result": {
                    "rest_id": "111",
                    "legacy": {"id_str": "111", "full_text": "original"},
                }
            },
        },
    }

    tweet = parse_tweet(tweet_data)

    assert tweet is not None
    assert tweet.is_retweet is True
    assert tweet.is_quote is False


def test_parse_tweet_quote_flag():
    """
    A top-level ``quoted_status_result`` on the tweet_data flips ``is_quote``
    to True (the parser checks the outer dict, not legacy).
    """
    tweet_data = {
        "rest_id": "8888",
        "quoted_status_result": {
            "result": {
                "rest_id": "222",
                "legacy": {"id_str": "222", "full_text": "quoted source"},
            }
        },
        "core": {
            "user_results": {
                "result": {
                    "rest_id": "42",
                    "legacy": {"screen_name": "qt_user", "name": "QT User"},
                }
            }
        },
        "legacy": {
            "id_str": "8888",
            "full_text": "Quote tweeting an interesting take.",
            "created_at": "Sun Nov 09 11:05:17 +0000 2025",
            "lang": "en",
        },
    }

    tweet = parse_tweet(tweet_data)

    assert tweet is not None
    assert tweet.is_quote is True
    assert tweet.is_retweet is False


def test_parse_tweet_reads_screen_name_from_core_block() -> None:
    """X moved screen_name and name from legacy to a new 'core' sub-block.

    The parser must read from user_results.core first and fall back to
    user_results.legacy when core is absent. This regression check covers
    the May 2026 API drift that surfaced empty handles on a fresh scrape
    even though the request and response otherwise looked healthy.
    """
    tweet_data = {
        "rest_id": "9999",
        "core": {
            "user_results": {
                "result": {
                    "rest_id": "u9",
                    # Brand-new shape: screen_name and name only in core.
                    "core": {"screen_name": "new_shape_user", "name": "New Shape"},
                    # legacy keeps everything else but no longer carries
                    # the handle or display name.
                    "legacy": {
                        "followers_count": 12345,
                        "friends_count": 678,
                        "verified": False,
                    },
                }
            }
        },
        "legacy": {
            "id_str": "9999",
            "full_text": "core-block tweet",
            "created_at": "Sun Nov 09 11:05:17 +0000 2025",
            "lang": "en",
        },
    }

    tweet = parse_tweet(tweet_data)

    assert tweet is not None
    assert tweet.user.screen_name == "new_shape_user"
    assert tweet.user.name == "New Shape"
    # followers_count is still in legacy in the new shape, which the parser
    # also has to keep handling correctly.
    assert tweet.user.followers_count == 12345
