"""Tests for ``XLikesExporter`` resume behaviour.

Covers Requirements 6.4 and 6.5 of the codebase-foundation spec:

- 6.4: When ``fetch_likes`` is called with ``resume=True`` and a checkpoint
  exists for the same ``user_id``, the exporter shall load the checkpoint's
  tweets and cursor, pass that cursor to ``XAPIClient.fetch_all_likes`` as
  ``start_cursor``, and merge newly-fetched tweets with the checkpointed ones
  while deduplicating by tweet id.
- 6.5: When ``fetch_likes`` is called with ``resume=True`` and a checkpoint
  exists but for a *different* ``user_id``, the exporter shall clear the
  stale checkpoint and start the fetch fresh (no merge, no inherited cursor).

Both cases run entirely under ``tmp_path`` and stub
``XAPIClient.fetch_all_likes`` via ``monkeypatch.setattr`` so no network or
real pagination occurs. Cookie validation is satisfied by the autouse
``_no_real_cookies`` fixture in ``conftest.py``, so the cookies path argument
to ``XLikesExporter`` does not need to point at a real file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from x_likes_exporter.checkpoint import Checkpoint
from x_likes_exporter.client import XAPIClient
from x_likes_exporter.exporter import XLikesExporter
from x_likes_exporter.models import Tweet, User


def _make_tweet(tweet_id: str, text: str = "") -> Tweet:
    """Build a minimal ``Tweet`` for resume testing."""
    user = User(id="u1", screen_name="someuser", name="Some User")
    return Tweet(
        id=tweet_id,
        text=text or f"tweet {tweet_id}",
        created_at="Sun Nov 09 11:05:17 +0000 2025",
        user=user,
    )


def test_resume_same_user_dedupes_and_uses_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume with matching user_id loads cursor and dedupes by tweet id.

    Pre-populate a checkpoint under ``tmp_path`` for ``user_id="A"`` with
    two tweets (ids ``"1"`` and ``"2"``) and a saved cursor. Stub
    ``XAPIClient.fetch_all_likes`` to return a list containing a duplicate
    of tweet ``"2"`` plus a new tweet ``"3"``, and capture the
    ``start_cursor`` it was called with. The merged result must contain
    ids ``{"1", "2", "3"}`` (no duplicate ``"2"``) and the stub must have
    been invoked with the saved cursor.
    """
    # 1. Pre-populate checkpoint for user "A".
    checkpoint = Checkpoint(str(tmp_path))
    t1 = _make_tweet("1", "first")
    t2 = _make_tweet("2", "second")
    checkpoint.save(
        user_id="A",
        tweets=[t1, t2],
        cursor="saved_cursor",
        total_fetched=2,
    )

    # 2. Stub ``XAPIClient.fetch_all_likes`` to return one duplicate id and
    #    one new id, capturing the ``start_cursor`` argument.
    captured: dict = {}
    t2_dup = _make_tweet("2", "second-dup")
    t3 = _make_tweet("3", "third")

    def fake_fetch_all_likes(
        self: XAPIClient,
        user_id: str,
        progress_callback=None,
        stop_callback=None,
        start_cursor: str | None = None,
        checkpoint_callback=None,
        checkpoint_interval: int = 10,
    ) -> list[Tweet]:
        captured["user_id"] = user_id
        captured["start_cursor"] = start_cursor
        return [t2_dup, t3]

    monkeypatch.setattr(XAPIClient, "fetch_all_likes", fake_fetch_all_likes)

    # 3. Build exporter and call fetch_likes for user "A" with resume=True.
    exporter = XLikesExporter(
        cookies_file="any/path/cookies.json",
        output_dir=str(tmp_path),
    )
    result = exporter.fetch_likes(
        user_id="A",
        resume=True,
        download_media=False,
    )

    # 4. Assertions: cursor was passed in, ids are deduplicated.
    assert captured["user_id"] == "A"
    assert captured["start_cursor"] == "saved_cursor"

    result_ids = [t.id for t in result]
    # Explicit duplicate-id assertion: no id appears more than once.
    assert len(result_ids) == len(set(result_ids)), f"Result contains duplicate ids: {result_ids}"
    assert set(result_ids) == {"1", "2", "3"}


def test_resume_different_user_clears_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume with mismatched user_id clears checkpoint and starts fresh.

    Pre-populate a checkpoint for ``user_id="A"``, then call
    ``fetch_likes(user_id="B", resume=True)``. The exporter must not merge
    A's tweets into B's result, must invoke ``fetch_all_likes`` with
    ``start_cursor=None`` (fresh fetch), and must clear the stale
    checkpoint from disk.
    """
    # 1. Pre-populate checkpoint for user "A".
    checkpoint = Checkpoint(str(tmp_path))
    t_a1 = _make_tweet("a1", "from-A-checkpoint")
    checkpoint.save(
        user_id="A",
        tweets=[t_a1],
        cursor="cursor_for_A",
        total_fetched=1,
    )
    assert checkpoint.exists() is True

    # 2. Stub ``XAPIClient.fetch_all_likes`` to return B's tweets and
    #    capture the ``start_cursor`` argument.
    captured: dict = {}
    t_new1 = _make_tweet("b1", "fresh-1")
    t_new2 = _make_tweet("b2", "fresh-2")

    def fake_fetch_all_likes(
        self: XAPIClient,
        user_id: str,
        progress_callback=None,
        stop_callback=None,
        start_cursor: str | None = None,
        checkpoint_callback=None,
        checkpoint_interval: int = 10,
    ) -> list[Tweet]:
        captured["user_id"] = user_id
        captured["start_cursor"] = start_cursor
        return [t_new1, t_new2]

    monkeypatch.setattr(XAPIClient, "fetch_all_likes", fake_fetch_all_likes)

    # 3. Build exporter and call fetch_likes for user "B" with resume=True.
    exporter = XLikesExporter(
        cookies_file="any/path/cookies.json",
        output_dir=str(tmp_path),
    )
    result = exporter.fetch_likes(
        user_id="B",
        resume=True,
        download_media=False,
    )

    # 4. Assertions: fresh fetch, no merge with A's data, checkpoint cleared.
    assert captured["user_id"] == "B"
    assert (
        captured["start_cursor"] is None
    ), "Mismatched checkpoint user must not contribute its cursor"

    result_ids = [t.id for t in result]
    assert result_ids == ["b1", "b2"]
    assert "a1" not in result_ids, "Tweets from a different user's checkpoint must not be merged in"

    # The stale checkpoint for "A" must be gone after the call.
    fresh_checkpoint = Checkpoint(str(tmp_path))
    assert fresh_checkpoint.exists() is False
