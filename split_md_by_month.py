#!/usr/bin/env python3
"""
Split likes.md into separate files by year/month
"""

from x_likes_exporter import XLikesExporter
from x_likes_exporter.models import Tweet, User, Media
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

# Load tweets
with open('output/likes.json') as f:
    tweets_data = json.load(f)

# Group by year/month
tweets_by_month = defaultdict(list)

for t in tweets_data:
    user = User(**t['user'])
    media = [Media(**m) for m in t.get('media', [])]
    tweet = Tweet(
        id=t['id'],
        text=t['text'],
        created_at=t['created_at'],
        user=user,
        retweet_count=t['retweet_count'],
        favorite_count=t['favorite_count'],
        reply_count=t['reply_count'],
        quote_count=t['quote_count'],
        view_count=t.get('view_count', 0),
        lang=t['lang'],
        is_retweet=t['is_retweet'],
        is_quote=t['is_quote'],
        media=media,
        urls=t['urls'],
        hashtags=t['hashtags'],
        mentions=t['mentions'],
        conversation_id=t.get('conversation_id'),
        in_reply_to_user_id=t.get('in_reply_to_user_id'),
        raw_data=t.get('raw_data')
    )

    # Parse date
    try:
        # Format: "Sun Nov 09 11:05:17 +0000 2025"
        dt = datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S %z %Y")
        year_month = dt.strftime("%Y-%m")
        tweets_by_month[year_month].append(tweet)
    except:
        # Fallback
        tweets_by_month["unknown"].append(tweet)

# Create output directory
output_dir = Path("output/by_month")
output_dir.mkdir(exist_ok=True)

# Export each month
exporter = XLikesExporter('cookies.json', 'output')

for year_month in sorted(tweets_by_month.keys(), reverse=True):
    tweets = tweets_by_month[year_month]
    exporter.tweets = tweets

    # Generate filename
    filename = output_dir / f"likes_{year_month}.md"

    # Export
    from x_likes_exporter.formatters import MarkdownFormatter
    formatter = MarkdownFormatter(exporter.media_downloader)
    formatter.export(tweets, str(filename), include_media=True)

    print(f"✓ {year_month}: {len(tweets)} tweets -> {filename}")

print(f"\n✓ Created {len(tweets_by_month)} monthly files in output/by_month/")
