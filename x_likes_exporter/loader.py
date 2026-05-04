"""
Public read API for an existing X Likes export.

This module loads a previously exported `likes.json` (produced by this
library's exporter / formatters) back into `Tweet` objects, and walks the
per-month Markdown directory in deterministic order. It does not require a
`cookies.json` file, does not perform any network I/O, and does not import
from `cookies.py`, `auth.py`, `client.py`, or `exporter.py`. Its only
internal dependency is `models.py`; everything else is the standard library.

The reconstruction in `_dict_to_tweet` is the inverse of `Tweet.to_dict()`.
Round-trip property: for any Tweet `t`, the result of
`_dict_to_tweet(t.to_dict()).to_dict()` is structurally equal to `t.to_dict()`
(modulo dict insertion ordering).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .models import Media, Tweet, User

_PathLike = str | Path

# Filenames of the form `likes_YYYY-MM.md` exactly. Anything else is skipped
# by `iter_monthly_markdown` (no warning, just skipped).
_MONTHLY_RE = re.compile(r"^likes_(\d{4})-(\d{2})\.md$")


def load_export(path: _PathLike) -> list[Tweet]:
    """Load an existing `likes.json` into `Tweet` objects.

    Args:
        path: Path to a `likes.json` file produced by this library.

    Returns:
        A list of `Tweet` objects. The list may be empty if the export was
        empty.

    Raises:
        FileNotFoundError: The given path does not exist. The path is in the
            error message.
        ValueError: The file exists but does not contain a JSON list of tweet
            dicts in the expected shape. The failing field is in the message.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"likes.json not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"failed to parse JSON at {p}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"expected a list of tweet dicts, got {type(data).__name__} at {p}")

    tweets: list[Tweet] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"tweet entry at index {index} is not a dict, got {type(entry).__name__}"
            )
        tweets.append(_dict_to_tweet(entry, index=index))
    return tweets


def iter_monthly_markdown(path: _PathLike) -> Iterator[Path]:
    """Yield `likes_YYYY-MM.md` files under `path` in reverse-chronological order.

    Files that do not match the `likes_YYYY-MM.md` pattern are silently
    skipped.

    Args:
        path: Directory containing per-month Markdown files (typically
            `output/by_month/`).

    Yields:
        `Path` objects for each matching file, sorted by `(year, month)` in
        descending order (e.g. 2025-04 before 2025-03 before 2024-12).

    Raises:
        FileNotFoundError: The given path does not exist or is not a
            directory. The path is in the error message.
    """
    d = Path(path)
    if not d.exists():
        raise FileNotFoundError(f"directory not found: {d}")
    if not d.is_dir():
        raise FileNotFoundError(f"not a directory: {d}")

    matches: list[tuple[int, int, Path]] = []
    for child in d.iterdir():
        if not child.is_file():
            continue
        m = _MONTHLY_RE.match(child.name)
        if not m:
            continue
        year = int(m.group(1))
        month = int(m.group(2))
        matches.append((year, month, child))

    # Reverse-chronological: newest (year, month) first.
    matches.sort(key=lambda t: (t[0], t[1]), reverse=True)

    for _year, _month, file_path in matches:
        yield file_path


def _dict_to_tweet(entry: dict[str, Any], *, index: int | None = None) -> Tweet:
    """Reconstruct a Tweet from the dict shape Tweet.to_dict() produces.

    Recurses into `quoted_tweet` and `retweeted_tweet` if present.

    Raises:
        ValueError: Required fields (`id`, `text`, `created_at`, `user`) are
            missing or wrong-typed. The failing field is in the message.
    """
    where = f" (index {index})" if index is not None else ""

    # --- required scalar fields ---
    for required in ("id", "text", "created_at"):
        if required not in entry:
            raise ValueError(f"tweet entry missing field: {required!r}{where}")

    if "user" not in entry:
        raise ValueError(f"tweet entry missing field: 'user'{where}")
    user_entry = entry["user"]
    if not isinstance(user_entry, dict):
        raise ValueError(
            f"tweet field 'user' is not a dict, got {type(user_entry).__name__}{where}"
        )

    user = _dict_to_user(user_entry, where=where)

    # --- media list ---
    media_list: list[Media] = []
    raw_media = entry.get("media", []) or []
    if not isinstance(raw_media, list):
        raise ValueError(
            f"tweet field 'media' is not a list, got {type(raw_media).__name__}{where}"
        )
    for m_index, m in enumerate(raw_media):
        if not isinstance(m, dict):
            raise ValueError(f"tweet media[{m_index}] is not a dict, got {type(m).__name__}{where}")
        media_list.append(_dict_to_media(m, where=f"{where} media[{m_index}]"))

    # --- nested tweets (recursive) ---
    quoted = entry.get("quoted_tweet")
    quoted_tweet: Tweet | None = None
    if quoted is not None:
        if not isinstance(quoted, dict):
            raise ValueError(
                f"tweet field 'quoted_tweet' is not a dict, got {type(quoted).__name__}{where}"
            )
        quoted_tweet = _dict_to_tweet(quoted)

    retweeted = entry.get("retweeted_tweet")
    retweeted_tweet: Tweet | None = None
    if retweeted is not None:
        if not isinstance(retweeted, dict):
            raise ValueError(
                f"tweet field 'retweeted_tweet' is not a dict, got "
                f"{type(retweeted).__name__}{where}"
            )
        retweeted_tweet = _dict_to_tweet(retweeted)

    # --- optional list fields ---
    urls = entry.get("urls", []) or []
    hashtags = entry.get("hashtags", []) or []
    mentions = entry.get("mentions", []) or []
    if not isinstance(urls, list):
        raise ValueError(f"tweet field 'urls' is not a list{where}")
    if not isinstance(hashtags, list):
        raise ValueError(f"tweet field 'hashtags' is not a list{where}")
    if not isinstance(mentions, list):
        raise ValueError(f"tweet field 'mentions' is not a list{where}")

    return Tweet(
        id=entry["id"],
        text=entry["text"],
        created_at=entry["created_at"],
        user=user,
        retweet_count=entry.get("retweet_count", 0),
        favorite_count=entry.get("favorite_count", 0),
        reply_count=entry.get("reply_count", 0),
        quote_count=entry.get("quote_count", 0),
        view_count=entry.get("view_count", 0),
        lang=entry.get("lang", "en"),
        is_retweet=entry.get("is_retweet", False),
        is_quote=entry.get("is_quote", False),
        quoted_tweet=quoted_tweet,
        retweeted_tweet=retweeted_tweet,
        media=media_list,
        urls=list(urls),
        hashtags=list(hashtags),
        mentions=list(mentions),
        conversation_id=entry.get("conversation_id"),
        in_reply_to_user_id=entry.get("in_reply_to_user_id"),
        raw_data=entry.get("raw_data"),
    )


def _dict_to_user(entry: dict[str, Any], *, where: str = "") -> User:
    """Reconstruct a User from a dict shape produced by User.to_dict()."""
    for required in ("id", "screen_name", "name"):
        if required not in entry:
            raise ValueError(f"user entry missing field: {required!r}{where}")

    return User(
        id=entry["id"],
        screen_name=entry["screen_name"],
        name=entry["name"],
        profile_image_url=entry.get("profile_image_url"),
        verified=entry.get("verified", False),
        followers_count=entry.get("followers_count", 0),
        following_count=entry.get("following_count", 0),
    )


def _dict_to_media(entry: dict[str, Any], *, where: str = "") -> Media:
    """Reconstruct a Media from a dict shape produced by Media.to_dict()."""
    for required in ("type", "url"):
        if required not in entry:
            raise ValueError(f"media entry missing field: {required!r}{where}")

    return Media(
        type=entry["type"],
        url=entry["url"],
        media_url=entry.get("media_url"),
        preview_image_url=entry.get("preview_image_url"),
        width=entry.get("width"),
        height=entry.get("height"),
        local_path=entry.get("local_path"),
    )
