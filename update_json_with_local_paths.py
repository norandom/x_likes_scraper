#!/usr/bin/env python3
"""
Update likes.json with local media paths
"""

import json
from pathlib import Path

# Load JSON
with open('output/likes.json', 'r') as f:
    tweets = json.load(f)

# Get list of downloaded files
media_dir = Path('output/media')
downloaded_files = {f.name for f in media_dir.glob('*')}

# Update media local_path
updated = 0
for tweet in tweets:
    if tweet.get('media'):
        tweet_id = tweet['id']
        for idx, media_item in enumerate(tweet['media']):
            # Construct expected filename
            filename = f"{tweet_id}_{idx}.jpg"
            if filename in downloaded_files:
                media_item['local_path'] = f"media/{filename}"
                updated += 1

# Save updated JSON
with open('output/likes.json', 'w') as f:
    json.dump(tweets, f, indent=2)

print(f"✓ Updated {updated} media items with local paths")
