"""Tests for :mod:`x_likes_mcp.corpus_text`.

The helper is the single source of truth for the text BM25 and the
embedder both index. It must include the resolved URLs from
``Tweet.urls`` (the export already replaces ``t.co`` shortlinks with
their resolved targets) so that lexical and semantic search can match on
domains and slugs the original ``t.co/abc`` text hides.
"""

from __future__ import annotations

from x_likes_exporter.models import Tweet, User
from x_likes_mcp.corpus_text import tweet_index_text


def _make_tweet(
    text: str = "",
    urls: list[str] | None = None,
) -> Tweet:
    user = User(id="1", screen_name="alice", name="Alice")
    return Tweet(id="1", text=text, created_at="", user=user, urls=urls or [])


def test_text_only_returns_text() -> None:
    tweet = _make_tweet(text="hello world")
    assert tweet_index_text(tweet) == "hello world"


def test_appends_resolved_urls_after_text() -> None:
    tweet = _make_tweet(
        text="check this https://t.co/abc",
        urls=["https://github.com/owner/repo"],
    )
    assert tweet_index_text(tweet) == ("check this https://t.co/abc https://github.com/owner/repo")


def test_appends_multiple_urls() -> None:
    tweet = _make_tweet(
        text="dual link",
        urls=["https://example.com/a", "https://example.org/b"],
    )
    assert tweet_index_text(tweet) == ("dual link https://example.com/a https://example.org/b")


def test_empty_text_with_urls() -> None:
    tweet = _make_tweet(text="", urls=["https://example.com"])
    assert tweet_index_text(tweet) == "https://example.com"


def test_empty_tweet_returns_empty_string() -> None:
    tweet = _make_tweet(text="", urls=[])
    assert tweet_index_text(tweet) == ""


def test_skips_falsy_urls() -> None:
    tweet = _make_tweet(text="t", urls=["", "https://example.com", ""])
    assert tweet_index_text(tweet) == "t https://example.com"
