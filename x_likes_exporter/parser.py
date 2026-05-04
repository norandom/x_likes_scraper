"""
Pure response-parsing functions for the X Likes timeline.

These functions take a parsed JSON response (a dict) and produce Tweet objects
plus the next pagination cursor. They have no I/O, no state, and no dependency
on the API client. They never raise on missing or malformed keys: a malformed
response yields [] from extract_tweets and None from extract_cursor, and a
per-entry failure inside extract_tweets is skipped.

The bodies are lifted verbatim from XAPIClient._extract_tweets, _parse_tweet,
and _extract_cursor with `self` removed. parse_response is a convenience wrapper
that returns both extract_tweets and extract_cursor results in one call.
"""

import contextlib
from typing import Any

from .models import Media, Tweet, User


def extract_tweets(response: dict[str, Any]) -> list[Tweet]:
    """Extract tweets from an X Likes API response.

    Returns an empty list if the response is missing the expected
    `data.user.result.timeline` chain or any other structural key. Per-entry
    parse failures are skipped.
    """
    tweets: list[Tweet] = []

    try:
        instructions = (
            response.get("data", {})
            .get("user", {})
            .get("result", {})
            .get("timeline", {})
            .get("timeline", {})
            .get("instructions", [])
        )

        for instruction in instructions:
            if instruction.get("type") == "TimelineAddEntries":
                entries = instruction.get("entries", [])

                for entry in entries:
                    if entry.get("content", {}).get("entryType") == "TimelineTimelineItem":
                        tweet_data = (
                            entry.get("content", {})
                            .get("itemContent", {})
                            .get("tweet_results", {})
                            .get("result", {})
                        )

                        if tweet_data and "legacy" in tweet_data:
                            tweet = parse_tweet(tweet_data)
                            if tweet:
                                tweets.append(tweet)

    except Exception as e:
        print(f"Error extracting tweets: {e}")

    return tweets


def parse_tweet(tweet_data: dict[str, Any]) -> Tweet | None:
    """Parse a single tweet entry's `result` dict into a Tweet model.

    Returns None on any failure (missing legacy/core, type errors). Per-tweet
    failures here are how extract_tweets skips bad entries without raising.
    """
    try:
        legacy = tweet_data.get("legacy", {})
        core = tweet_data.get("core", {})
        user_results = core.get("user_results", {}).get("result", {})
        user_legacy = user_results.get("legacy", {})
        # X moved screen_name and name from legacy to a new "core" sub-block.
        # Read core first, fall back to legacy so old fixtures still parse.
        user_core = user_results.get("core", {})

        # Parse user
        user = User(
            id=user_results.get("rest_id", ""),
            screen_name=user_core.get("screen_name") or user_legacy.get("screen_name", ""),
            name=user_core.get("name") or user_legacy.get("name", ""),
            profile_image_url=user_legacy.get("profile_image_url_https", ""),
            verified=user_legacy.get("verified", user_results.get("is_blue_verified", False)),
            followers_count=user_legacy.get("followers_count", 0),
            following_count=user_legacy.get("friends_count", 0),
        )

        # Parse media
        media_list = []
        extended_entities = legacy.get("extended_entities", {})
        for media_item in extended_entities.get("media", []):
            media = Media(
                type=media_item.get("type", "photo"),
                url=media_item.get("url", ""),
                media_url=media_item.get("media_url_https", ""),
                width=media_item.get("original_info", {}).get("width"),
                height=media_item.get("original_info", {}).get("height"),
            )
            media_list.append(media)

        # Parse entities
        entities = legacy.get("entities", {})
        urls = [url["expanded_url"] for url in entities.get("urls", [])]
        hashtags = [tag["text"] for tag in entities.get("hashtags", [])]
        mentions = [mention["screen_name"] for mention in entities.get("user_mentions", [])]

        # Create tweet
        tweet = Tweet(
            id=tweet_data.get("rest_id", legacy.get("id_str", "")),
            text=legacy.get("full_text", ""),
            created_at=legacy.get("created_at", ""),
            user=user,
            retweet_count=legacy.get("retweet_count", 0),
            favorite_count=legacy.get("favorite_count", 0),
            reply_count=legacy.get("reply_count", 0),
            quote_count=legacy.get("quote_count", 0),
            lang=legacy.get("lang", "en"),
            is_retweet="retweeted_status_result" in legacy,
            is_quote="quoted_status_result" in tweet_data,
            media=media_list,
            urls=urls,
            hashtags=hashtags,
            mentions=mentions,
            conversation_id=legacy.get("conversation_id_str"),
            in_reply_to_user_id=legacy.get("in_reply_to_user_id_str"),
            raw_data=tweet_data,
        )

        views = tweet_data.get("views", {}).get("count")
        if views:
            # view counts are best-effort; leave default 0 if X returns garbage
            with contextlib.suppress(ValueError, TypeError):
                tweet.view_count = int(views)

        return tweet

    except Exception as e:
        print(f"Error parsing tweet: {e}")
        return None


def extract_cursor(response: dict[str, Any]) -> str | None:
    """Extract the next-page cursor from an X Likes API response.

    Returns None if the response is malformed or no Bottom cursor entry is
    present.
    """
    try:
        instructions = (
            response.get("data", {})
            .get("user", {})
            .get("result", {})
            .get("timeline", {})
            .get("timeline", {})
            .get("instructions", [])
        )

        for instruction in instructions:
            if instruction.get("type") == "TimelineAddEntries":
                entries = instruction.get("entries", [])

                for entry in entries:
                    content = entry.get("content", {})
                    if (
                        content.get("entryType") == "TimelineTimelineCursor"
                        and content.get("cursorType") == "Bottom"
                    ):
                        return content.get("value")

    except Exception as e:
        print(f"Error extracting cursor: {e}")

    return None


def parse_response(response: dict[str, Any]) -> tuple[list[Tweet], str | None]:
    """Convenience: extract_tweets and extract_cursor in one call.

    Returns (tweets, next_cursor). Both extract_tweets and extract_cursor are
    individually safe against malformed input, so this wrapper is also safe.
    """
    return extract_tweets(response), extract_cursor(response)
