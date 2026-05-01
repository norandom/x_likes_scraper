"""Tests for ``x_likes_exporter.checkpoint.Checkpoint``.

Covers Requirements 6.1, 6.2, and 6.3 of the codebase-foundation spec:

- 6.1: Saving a list of tweets, a cursor, and a user id and then loading
  through a fresh ``Checkpoint`` against the same directory shall round-trip
  the original tweets, cursor, and user id.
- 6.2: ``clear()`` shall remove the checkpoint files, after which ``exists()``
  shall report no checkpoint.
- 6.3: ``is_valid(user_id)`` shall return true only when the saved
  checkpoint's user id matches the supplied user id.

All file I/O is performed inside ``tmp_path`` so the suite never writes
outside the temporary directory pytest hands us.
"""

from __future__ import annotations

from pathlib import Path

from x_likes_exporter.checkpoint import Checkpoint
from x_likes_exporter.models import Tweet, User


def _make_tweet(tweet_id: str, text: str, user_id: str = "u1") -> Tweet:
    """Build a minimal ``Tweet`` for round-trip testing."""
    user = User(id=user_id, screen_name="someuser", name="Some User")
    return Tweet(
        id=tweet_id,
        text=text,
        created_at="Sun Nov 09 11:05:17 +0000 2025",
        user=user,
    )


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    """Save tweets/cursor/user, then load with a fresh manager and compare."""
    saver = Checkpoint(str(tmp_path))
    tweets = [_make_tweet("1", "first"), _make_tweet("2", "second")]

    saver.save(
        user_id="user_a",
        tweets=tweets,
        cursor="abc",
        total_fetched=len(tweets),
    )

    # Construct a fresh manager against the same directory: the load path
    # must not depend on the in-memory state of the saver.
    loader = Checkpoint(str(tmp_path))
    loaded = loader.load()

    assert loaded is not None
    assert loaded["user_id"] == "user_a"
    assert loaded["cursor"] == "abc"

    loaded_tweets = loaded["tweets"]
    assert len(loaded_tweets) == 2
    assert [t.id for t in loaded_tweets] == ["1", "2"]
    assert loaded_tweets[0].text == "first"
    assert loaded_tweets[1].text == "second"


def test_checkpoint_clear_removes_files(tmp_path: Path) -> None:
    """``clear()`` deletes both files and ``exists()`` reports false."""
    cp = Checkpoint(str(tmp_path))
    cp.save(
        user_id="user_a",
        tweets=[_make_tweet("1", "hi")],
        cursor="abc",
        total_fetched=1,
    )

    # Sanity: both files are present before clear.
    assert cp.exists() is True
    assert cp.checkpoint_file.exists()
    assert cp.tweets_file.exists()

    cp.clear()

    assert cp.exists() is False
    assert not cp.checkpoint_file.exists()
    assert not cp.tweets_file.exists()


def test_checkpoint_is_valid_matching_user(tmp_path: Path) -> None:
    """``is_valid`` returns ``True`` for the user id that was saved."""
    cp = Checkpoint(str(tmp_path))
    cp.save(
        user_id="user_a",
        tweets=[_make_tweet("1", "hi")],
        cursor="abc",
        total_fetched=1,
    )

    assert cp.is_valid("user_a") is True


def test_checkpoint_is_valid_different_user(tmp_path: Path) -> None:
    """``is_valid`` returns ``False`` for a user id that does not match."""
    cp = Checkpoint(str(tmp_path))
    cp.save(
        user_id="user_a",
        tweets=[_make_tweet("1", "hi")],
        cursor="abc",
        total_fetched=1,
    )

    assert cp.is_valid("user_b") is False
