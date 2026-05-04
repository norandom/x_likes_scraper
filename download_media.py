#!/usr/bin/env python3
"""
Download media from existing likes.json export
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from x_likes_exporter.downloader import MediaDownloader
from x_likes_exporter.models import Media


def main():
    # Load existing JSON export
    json_file = Path("output/likes.json")
    if not json_file.exists():
        print("Error: output/likes.json not found")
        sys.exit(1)

    print("Loading tweets from likes.json...")
    with open(json_file) as f:
        tweets_data = json.load(f)

    print(f"Found {len(tweets_data)} tweets")

    # Initialize media downloader
    downloader = MediaDownloader("output/media")

    # Collect all media items
    media_items = []
    for tweet in tweets_data:
        if tweet.get("media"):
            tweet_id = tweet["id"]
            for idx, media_data in enumerate(tweet["media"]):
                if media_data.get("media_url"):
                    media = Media(
                        type=media_data["type"],
                        url=media_data["url"],
                        media_url=media_data["media_url"],
                        preview_image_url=media_data.get("preview_image_url"),
                        width=media_data.get("width"),
                        height=media_data.get("height"),
                    )
                    media_items.append((tweet_id, idx, media))

    print(f"Found {len(media_items)} media items to download")
    print("Downloading media...")

    # Skip files already on disk; filenames are tweet_id_index.*
    media_dir = Path("output/media")
    existing_prefixes = {
        f.stem.rsplit("_", 1)[0] + "_" + f.stem.rsplit("_", 1)[1]
        for f in media_dir.glob("*")
        if "_" in f.stem
    }

    successful = 0
    failed = 0
    skipped = 0

    for tweet_id, idx, media in tqdm(media_items, desc="Downloading"):
        if f"{tweet_id}_{idx}" in existing_prefixes:
            skipped += 1
            continue
        try:
            local_path = downloader.download_media(media, tweet_id, idx)
            if local_path:
                successful += 1
        except Exception as e:
            failed += 1
            if failed <= 10:  # Only show first 10 errors
                print(f"\nError downloading {media.media_url}: {e}")

    print("\n✓ Download complete!")
    print(f"  Skipped (already on disk): {skipped}")
    print(f"  Newly downloaded: {successful}")
    print(f"  Failed: {failed}")
    print(
        f"  Total size: {sum(f.stat().st_size for f in Path('output/media').glob('*')) / 1024 / 1024:.1f} MB"
    )


if __name__ == "__main__":
    main()
