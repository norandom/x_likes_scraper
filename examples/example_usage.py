#!/usr/bin/env python3
"""
Example usage of X Likes Exporter library
"""

from x_likes_exporter import XLikesExporter


def example_basic():
    """Basic usage example"""
    print("=== Basic Usage ===\n")

    # Initialize exporter with cookies
    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    # Fetch all likes
    tweets = exporter.fetch_likes(
        user_id="123456789",
        download_media=True
    )

    print(f"Fetched {len(tweets)} liked tweets")

    # Export to all formats
    exporter.export_all()


def example_selective_export():
    """Export to specific formats only"""
    print("\n=== Selective Export ===\n")

    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    tweets = exporter.fetch_likes(user_id="123456789")

    # Export only to JSON and Markdown
    exporter.export_json(filename="my_likes.json")
    exporter.export_markdown(filename="my_likes.md")


def example_with_pandas():
    """Work with Pandas DataFrame"""
    print("\n=== Pandas DataFrame ===\n")

    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    tweets = exporter.fetch_likes(user_id="123456789", download_media=False)

    # Get as DataFrame
    df = exporter.get_dataframe()

    # Analyze data
    print(f"Total tweets: {len(df)}")
    print(f"Most liked tweet: {df['favorite_count'].max()} likes")
    print(f"Most retweeted: {df['retweet_count'].max()} retweets")
    print(f"\nTop 5 most liked:")
    print(df.nlargest(5, 'favorite_count')[['user_screen_name', 'text', 'favorite_count']])

    # Export to CSV
    df.to_csv("output/likes_analysis.csv", index=False)


def example_with_progress():
    """Monitor progress during fetch"""
    print("\n=== With Progress Monitoring ===\n")

    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    # Define progress callback
    def on_progress(current, total):
        if total:
            print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
        else:
            print(f"Fetched: {current} tweets")

    tweets = exporter.fetch_likes(
        user_id="123456789",
        download_media=True,
        progress_callback=on_progress
    )


def example_no_media():
    """Fetch without downloading media"""
    print("\n=== Without Media Download ===\n")

    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    # Don't download media (faster)
    tweets = exporter.fetch_likes(
        user_id="123456789",
        download_media=False
    )

    # Export without media embeds
    exporter.export_markdown(include_media=False)


def example_statistics():
    """Get statistics about likes"""
    print("\n=== Statistics ===\n")

    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    tweets = exporter.fetch_likes(user_id="123456789")

    # Get statistics
    stats = exporter.get_stats()

    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")


def example_filtered_export():
    """Filter and export specific tweets"""
    print("\n=== Filtered Export ===\n")

    exporter = XLikesExporter(
        cookies_file="cookies.json",
        output_dir="output"
    )

    all_tweets = exporter.fetch_likes(user_id="123456789", download_media=False)

    # Filter tweets (e.g., only tweets with media)
    tweets_with_media = [t for t in all_tweets if t.media]
    print(f"Tweets with media: {len(tweets_with_media)}")

    # Filter by date
    from datetime import datetime, timedelta
    recent_date = datetime.now() - timedelta(days=30)
    recent_tweets = [
        t for t in all_tweets
        if t.get_created_datetime() > recent_date
    ]
    print(f"Tweets from last 30 days: {len(recent_tweets)}")

    # Filter by engagement
    popular_tweets = [
        t for t in all_tweets
        if t.favorite_count > 1000 or t.retweet_count > 100
    ]
    print(f"Popular tweets: {len(popular_tweets)}")

    # Export filtered tweets
    exporter.tweets = popular_tweets
    exporter.export_json("popular_tweets.json")


if __name__ == "__main__":
    # Run examples
    # Uncomment the example you want to run

    # example_basic()
    # example_selective_export()
    # example_with_pandas()
    # example_with_progress()
    # example_no_media()
    # example_statistics()
    # example_filtered_export()

    print("\nUncomment an example function to run it!")
