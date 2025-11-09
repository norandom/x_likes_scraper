#!/usr/bin/env python3
"""
Example: Using resume functionality with checkpoints
"""

from x_likes_exporter import XLikesExporter
from x_likes_exporter.checkpoint import Checkpoint
import time


def example_basic_resume():
    """Basic resume example"""
    print("=== Basic Resume ===\n")

    exporter = XLikesExporter("cookies.json", "output")

    # First run - may be interrupted
    print("Starting export (you can press Ctrl+C to interrupt)...")
    try:
        tweets = exporter.fetch_likes(
            user_id="123456789",
            download_media=True
        )
        print(f"✓ Complete! Fetched {len(tweets)} tweets")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! But checkpoint was saved.")
        print("Run again with resume=True to continue")

    # Second run - resume from checkpoint
    print("\n\nResuming from checkpoint...")
    tweets = exporter.fetch_likes(
        user_id="123456789",
        download_media=True,
        resume=True  # ← Resume from checkpoint
    )
    print(f"✓ Complete! Total: {len(tweets)} tweets")


def example_check_checkpoint():
    """Check checkpoint before resuming"""
    print("\n=== Check Checkpoint ===\n")

    checkpoint = Checkpoint("output")

    if checkpoint.exists():
        info = checkpoint.get_info()
        print("Found checkpoint:")
        print(f"  User: {info['user_id']}")
        print(f"  Tweets: {info['total_fetched']}")
        print(f"  Date: {info['timestamp']}")

        # Validate before resuming
        user_id = "123456789"
        if checkpoint.is_valid(user_id):
            print(f"\n✓ Checkpoint is valid for user {user_id}")
            print("You can resume safely")
        else:
            print(f"\n⚠ Checkpoint is for different user")
            print("Will start fresh if you try to resume")
    else:
        print("No checkpoint found - will start from beginning")


def example_manual_control():
    """Manually control checkpoints"""
    print("\n=== Manual Control ===\n")

    exporter = XLikesExporter("cookies.json", "output")

    # Check for existing checkpoint
    if exporter.checkpoint and exporter.checkpoint.exists():
        choice = input("Found checkpoint. Resume? (y/n): ")

        if choice.lower() == 'y':
            # Resume
            tweets = exporter.fetch_likes(
                user_id="123456789",
                resume=True
            )
        else:
            # Clear and start fresh
            exporter.checkpoint.clear()
            tweets = exporter.fetch_likes(
                user_id="123456789"
            )
    else:
        # No checkpoint, start fresh
        tweets = exporter.fetch_likes(
            user_id="123456789"
        )


def example_with_progress():
    """Monitor progress with resume"""
    print("\n=== Progress Monitoring ===\n")

    exporter = XLikesExporter("cookies.json", "output")

    # Track progress
    start_time = time.time()
    last_count = 0

    def on_progress(current, total):
        nonlocal last_count
        new_tweets = current - last_count
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0

        print(f"Progress: {current} tweets (+{new_tweets}) | "
              f"Rate: {rate:.1f} tweets/sec | "
              f"Elapsed: {elapsed:.0f}s")

        last_count = current

    # Resume with progress tracking
    tweets = exporter.fetch_likes(
        user_id="123456789",
        progress_callback=on_progress,
        resume=True
    )

    total_time = time.time() - start_time
    print(f"\n✓ Complete in {total_time:.0f}s")


def example_stop_and_resume():
    """Example: Stop callback with checkpoint"""
    print("\n=== Stop and Resume ===\n")

    exporter = XLikesExporter("cookies.json", "output")

    # Fetch with stop condition
    max_tweets = 1000

    def should_stop():
        return len(exporter.tweets) >= max_tweets

    print(f"Fetching up to {max_tweets} tweets...")
    tweets = exporter.fetch_likes(
        user_id="123456789",
        stop_callback=should_stop,
        resume=True
    )

    print(f"✓ Stopped at {len(tweets)} tweets")
    print("Checkpoint saved - can resume later")


def example_recover_from_error():
    """Recover from network errors"""
    print("\n=== Error Recovery ===\n")

    exporter = XLikesExporter("cookies.json", "output")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            tweets = exporter.fetch_likes(
                user_id="123456789",
                download_media=True,
                resume=True  # Always try to resume
            )

            print(f"✓ Success! Fetched {len(tweets)} tweets")
            break

        except Exception as e:
            retry_count += 1
            print(f"\n⚠ Error: {e}")

            if retry_count < max_retries:
                wait_time = 60 * retry_count  # Exponential backoff
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts")
                print("Checkpoint saved - try again later")


def example_cleanup():
    """Clean up checkpoints"""
    print("\n=== Cleanup ===\n")

    checkpoint = Checkpoint("output")

    if checkpoint.exists():
        info = checkpoint.get_info()
        print(f"Found checkpoint: {info['total_fetched']} tweets")

        # Check if it's old
        from datetime import datetime
        timestamp = datetime.fromisoformat(info['timestamp'])
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600

        if age_hours > 24:
            print(f"Checkpoint is {age_hours:.1f} hours old")
            checkpoint.clear()
            print("✓ Cleared old checkpoint")
        else:
            print(f"Checkpoint is recent ({age_hours:.1f} hours old)")
    else:
        print("No checkpoint to clean up")


if __name__ == "__main__":
    # Run examples
    # Uncomment the example you want to try

    # example_basic_resume()
    # example_check_checkpoint()
    # example_manual_control()
    # example_with_progress()
    # example_stop_and_resume()
    # example_recover_from_error()
    # example_cleanup()

    print("\nUncomment an example function to run it!")
