"""
Checkpoint system for resuming interrupted exports
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Tweet


class Checkpoint:
    """Manages export checkpoints for resume functionality"""

    def __init__(self, checkpoint_dir: str = "output"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / ".export_checkpoint.json"
        self.tweets_file = self.checkpoint_dir / ".export_tweets.pkl"

    def save(
        self,
        user_id: str,
        tweets: list[Tweet],
        cursor: str | None,
        total_fetched: int,
        download_media: bool = True,
    ):
        """
        Save current export state

        Args:
            user_id: User ID being exported
            tweets: List of fetched tweets
            cursor: Current pagination cursor
            total_fetched: Total number of tweets fetched
            download_media: Whether media download is enabled
        """
        # Save checkpoint metadata
        checkpoint_data = {
            "user_id": user_id,
            "cursor": cursor,
            "total_fetched": total_fetched,
            "download_media": download_media,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        # pickle is faster than re-serializing Tweet objects to JSON each save.
        # Trust boundary: this file is the exporter's own checkpoint, written
        # under output/ on the same machine that loads it. Single-user, no
        # network exposure, no untrusted producer; deserialization risk is
        # bounded to whatever the user puts in their own output directory.
        with open(self.tweets_file, "wb") as f:
            pickle.dump(
                tweets, f
            )  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle

        print(f"✓ Checkpoint saved: {total_fetched} tweets")

    def load(self) -> dict[str, Any] | None:
        """
        Load checkpoint if exists

        Returns:
            Dictionary with checkpoint data or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            # Load checkpoint metadata
            with open(self.checkpoint_file, encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            # Load tweets. See save() for the trust-boundary rationale that
            # makes pickle here acceptable (single-user, local-only file).
            if self.tweets_file.exists():
                with open(self.tweets_file, "rb") as f:
                    tweets = pickle.load(
                        f
                    )  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
                checkpoint_data["tweets"] = tweets
            else:
                checkpoint_data["tweets"] = []

            return checkpoint_data

        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None

    def exists(self) -> bool:
        """Check if a checkpoint exists"""
        return self.checkpoint_file.exists() and self.tweets_file.exists()

    def clear(self):
        """Remove checkpoint files"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.tweets_file.exists():
            self.tweets_file.unlink()
        print("✓ Checkpoint cleared")

    def get_info(self) -> dict[str, Any] | None:
        """
        Get checkpoint information without loading tweets

        Returns:
            Checkpoint metadata or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}")
            return None

    def is_valid(self, user_id: str) -> bool:
        """
        Check if checkpoint is valid for given user ID

        Args:
            user_id: User ID to validate against

        Returns:
            True if checkpoint is valid for this user
        """
        info = self.get_info()
        if not info:
            return False

        return info.get("user_id") == user_id

    def get_progress(self) -> str:
        """
        Get human-readable progress string

        Returns:
            Progress description
        """
        info = self.get_info()
        if not info:
            return "No checkpoint found"

        timestamp = info.get("timestamp", "unknown")
        total = info.get("total_fetched", 0)
        user_id = info.get("user_id", "unknown")

        return f"User {user_id}: {total} tweets (saved at {timestamp})"
