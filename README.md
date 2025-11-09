# X Likes Exporter (Python)

A comprehensive Python library to export your liked tweets from X (formerly Twitter) to multiple formats including JSON, CSV, Pandas DataFrame, Markdown with images, and HTML.

## Features

- 🚀 **Multiple Export Formats**: JSON, CSV, Excel, Pandas DataFrame, Markdown, HTML
- 📷 **Media Download**: Automatically downloads images and videos from tweets
- 📊 **Data Analysis**: Export to Pandas for easy data analysis
- 📝 **Markdown Export**: Beautiful Markdown files with locally embedded images
- ⏱️ **Rate Limit Handling**: Automatic rate limit detection and waiting
- 📄 **Cursor Pagination**: Efficiently fetches all likes using cursor-based pagination
- 💾 **Progress Tracking**: Real-time progress callbacks during export
- 🔒 **Privacy First**: All processing happens locally, no data sent to third parties

## Installation

```bash
# Clone or download this repository
cd x_likes_exporter_py

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Requirements

- Python 3.8+
- requests
- pandas
- beautifulsoup4
- Pillow
- tqdm
- python-dateutil

## Quick Start

### 1. Export Your Cookies

You need to export your X (Twitter) cookies to authenticate:

#### Using a Browser Extension:

1. Install a cookie export extension:
   - **Chrome/Edge**: [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm)
   - **Firefox**: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. Go to [x.com](https://x.com) and log in

3. Open the extension and export cookies as JSON

4. Save as `cookies.json`

#### Manual Method:

1. Open Developer Tools (F12) on [x.com](https://x.com)
2. Go to Application → Cookies → https://x.com
3. Find and copy these important cookies:
   - `ct0` (CSRF token)
   - `auth_token`
4. Create a `cookies.json` file (see format below)

<details>
<summary>cookies.json Format</summary>

```json
[
  {
    "domain": ".x.com",
    "name": "ct0",
    "value": "your_ct0_token_here",
    "path": "/",
    "secure": true,
    "httpOnly": false
  },
  {
    "domain": ".x.com",
    "name": "auth_token",
    "value": "your_auth_token_here",
    "path": "/",
    "secure": true,
    "httpOnly": true
  }
]
```

</details>

### 2. Find Your User ID

Your User ID is a numeric ID, not your username.

**Method 1: From HTML**
1. Go to your profile on x.com
2. Right-click → Inspect Element
3. Search for `data-user-id` in the HTML
4. Copy the numeric ID

**Method 2: Using an API**
```bash
# Using this tool
curl "https://tweeterid.com/ajax.php?username=YOUR_USERNAME"
```

### 3. Run the Exporter

#### Using CLI:

```bash
# Export to all formats
python cli.py cookies.json YOUR_USER_ID

# Export only to JSON and Markdown (Markdown will be split by month by default)
python cli.py cookies.json YOUR_USER_ID --format json --format markdown

# Export markdown as single file instead of splitting by month
python cli.py cookies.json YOUR_USER_ID --format markdown --single-file

# Skip media download (faster)
python cli.py cookies.json YOUR_USER_ID --no-media

# Custom output directory
python cli.py cookies.json YOUR_USER_ID --output my_export

# Show statistics
python cli.py cookies.json YOUR_USER_ID --stats
```

#### Using Python API:

```python
from x_likes_exporter import XLikesExporter

# Initialize
exporter = XLikesExporter(
    cookies_file="cookies.json",
    output_dir="output"
)

# Fetch all likes
tweets = exporter.fetch_likes(
    user_id="YOUR_USER_ID",
    download_media=True
)

# Export to all formats
exporter.export_all()

# Or export selectively
exporter.export_json()
exporter.export_markdown()
exporter.export_csv()
```

## Usage Examples

### Basic Export

```python
from x_likes_exporter import XLikesExporter

exporter = XLikesExporter("cookies.json", "output")
tweets = exporter.fetch_likes("123456789")
exporter.export_all()
```

### Export to Specific Formats

```python
# JSON only
exporter.export_json("my_likes.json")

# CSV for spreadsheet analysis
exporter.export_csv("my_likes.csv")

# Markdown with images
exporter.export_markdown("my_likes.md", include_media=True)

# HTML for viewing in browser
exporter.export_html("my_likes.html")
```

### Work with Pandas

```python
# Get as DataFrame
df = exporter.get_dataframe()

# Analyze
print(f"Total tweets: {len(df)}")
print(f"Most liked: {df['favorite_count'].max()}")
print(f"Average likes: {df['favorite_count'].mean()}")

# Filter
popular = df[df['favorite_count'] > 1000]
with_media = df[df['has_media'] == True]

# Export filtered data
popular.to_csv("popular_tweets.csv", index=False)
```

### Progress Monitoring

```python
def progress_callback(current, total):
    print(f"Fetched {current} tweets...")

tweets = exporter.fetch_likes(
    user_id="123456789",
    progress_callback=progress_callback
)
```

### Fetch Without Media

```python
# Faster if you don't need images
tweets = exporter.fetch_likes(
    user_id="123456789",
    download_media=False
)
```

### Get Statistics

```python
stats = exporter.get_stats()
print(f"Total tweets: {stats['total_tweets']}")
print(f"Total media: {stats['total_media']}")
print(f"Total likes: {stats['total_likes']}")
```

## Output Formats

### JSON

Complete tweet data including user info, media, engagement stats:

```json
[
  {
    "id": "1234567890",
    "text": "Tweet text here...",
    "created_at": "Wed Jan 01 12:00:00 +0000 2025",
    "user": {
      "id": "987654321",
      "screen_name": "username",
      "name": "Display Name"
    },
    "retweet_count": 10,
    "favorite_count": 50,
    "media": [
      {
        "type": "photo",
        "url": "https://...",
        "local_path": "media/1234567890_0.jpg"
      }
    ]
  }
]
```

### CSV / Excel

Tabular format perfect for spreadsheet analysis:

| tweet_id | text | user_screen_name | favorite_count | retweet_count | created_at |
|----------|------|------------------|----------------|---------------|------------|
| 123... | Tweet... | username | 50 | 10 | 2025-01-01 |

### Markdown

**By default, Markdown exports are split into separate files by year/month** in the `output/by_month/` directory (e.g., `likes_2025-01.md`, `likes_2025-02.md`). This makes large exports easier to browse.

Use `--single-file` flag to export to a single file instead.

Readable format with embedded images:

```markdown
## 2025-01 (15 tweets)

### @username
**Display Name** ✓
*2025-01-01 12:00:00*

Tweet text here...

![Image](media/1234567890_0.jpg)

*🔄 10 • ❤️ 50 • 💬 5*

🔗 [View on X](https://x.com/username/status/1234567890)

---
```

### HTML

Beautiful HTML page for viewing in browser with styled tweets and embedded media.

## Advanced Usage

### Filter Tweets Before Export

```python
# Filter by date
from datetime import datetime, timedelta
recent = datetime.now() - timedelta(days=30)
recent_tweets = [t for t in tweets if t.get_created_datetime() > recent]

# Filter by engagement
popular = [t for t in tweets if t.favorite_count > 1000]

# Filter by content
with_media = [t for t in tweets if t.media]
with_hashtag = [t for t in tweets if 'python' in t.hashtags]

# Export filtered
exporter.tweets = popular
exporter.export_json("popular.json")
```

### Custom Analysis

```python
import pandas as pd

df = exporter.get_dataframe()

# Top users you like
top_users = df['user_screen_name'].value_counts().head(10)
print("Top 10 users you like:")
print(top_users)

# Engagement over time
df['month'] = pd.to_datetime(df['created_at']).dt.to_period('M')
monthly_likes = df.groupby('month')['favorite_count'].sum()
print("\nMonthly engagement:")
print(monthly_likes)

# Most used hashtags
all_hashtags = []
for hashtags in df['hashtags']:
    all_hashtags.extend(hashtags.split(','))
hashtag_counts = pd.Series(all_hashtags).value_counts()
print("\nTop hashtags:")
print(hashtag_counts.head(20))
```

## Architecture

The library follows the same process as the Chrome extension:

```
┌──────────────────┐
│  CookieManager   │  ← Parse cookies.json
└────────┬─────────┘
         ↓
┌──────────────────┐
│  XAuthenticator  │  ← Extract Bearer token & Query ID
└────────┬─────────┘
         ↓
┌──────────────────┐
│   XAPIClient     │  ← Fetch likes with pagination
│                  │
│  • Rate limiting │
│  • Cursor paging │
│  • Data parsing  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ MediaDownloader  │  ← Download images/videos
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Formatters     │  ← Export to formats
│                  │
│  • JSON          │
│  • CSV/Excel     │
│  • Markdown      │
│  • HTML          │
└──────────────────┘
```

## Rate Limiting

X's API has rate limits (typically 500 requests per 15-minute window). The library:

1. **Monitors** rate limit headers from each response:
   ```
   x-rate-limit-limit: 500
   x-rate-limit-remaining: 499
   x-rate-limit-reset: 1704211200
   ```

2. **Waits automatically** when rate limit is reached (remaining ≤ 1)

3. **Adds polite delays** (1 second) between requests

4. **Shows progress** with rate limit info:
   ```
   Fetching page 25...
   Fetched 20 likes. Total: 500
   Rate limit: 475/500
   ```

For 10,000+ likes, expect 1-2 hours with rate limit waits.

## Troubleshooting

### "Invalid cookies"

- Make sure `ct0` and `auth_token` are in your cookies.json
- Try re-exporting cookies after logging out and back in

### "Authentication failed"

- Your session may have expired
- Log out of x.com, log back in, and export fresh cookies

### "Rate limit exceeded"

- The script should handle this automatically
- If it doesn't, wait 15 minutes and try again

### "No tweets found"

- Verify your User ID is correct
- Make sure you have liked tweets on your account
- Check that cookies are valid

### Images not downloading

- Check internet connection
- Some media URLs may have expired
- Try running with `download_media=False` first

## Project Structure

```
x_likes_exporter_py/
├── x_likes_exporter/
│   ├── __init__.py       # Package exports
│   ├── cookies.py        # Cookie parsing
│   ├── auth.py           # Token extraction
│   ├── client.py         # API client
│   ├── models.py         # Data models
│   ├── downloader.py     # Media downloader
│   ├── formatters.py     # Export formatters
│   └── exporter.py       # Main exporter class
├── cli.py                # Command-line interface
├── examples/
│   └── example_usage.py  # Usage examples
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## API Reference

### XLikesExporter

Main class for exporting likes.

```python
exporter = XLikesExporter(cookies_file: str, output_dir: str = "output")
```

**Methods:**

- `fetch_likes(user_id, download_media=True, progress_callback=None)` → List[Tweet]
- `export_json(filename, include_raw=False)` → None
- `export_csv(filename)` → None
- `export_excel(filename)` → None
- `export_markdown(filename, include_media=True)` → None
- `export_html(filename)` → None
- `export_all(base_name="likes", include_raw=False)` → None
- `get_dataframe()` → pandas.DataFrame
- `get_stats()` → dict

### Tweet Model

Represents a single tweet.

**Attributes:**
- `id`: str - Tweet ID
- `text`: str - Tweet text
- `created_at`: str - Creation timestamp
- `user`: User - User object
- `retweet_count`: int
- `favorite_count`: int
- `reply_count`: int
- `quote_count`: int
- `view_count`: int
- `media`: List[Media] - List of media items
- `urls`: List[str]
- `hashtags`: List[str]
- `mentions`: List[str]

**Methods:**
- `to_dict()` → dict
- `get_url()` → str
- `get_created_datetime()` → datetime

## Performance

### Speed

- **Fetching**: ~20 likes per second (rate limited)
- **Media download**: Depends on internet speed
- **Processing**: Very fast (< 1 second for 1000 tweets)

### Memory

- **~2-5 KB per tweet** in memory
- **10,000 likes**: ~20-50 MB
- **50,000 likes**: ~100-250 MB

### Time Estimates

| Likes | Time (approx) |
|-------|---------------|
| 100 | 5-10 seconds |
| 1,000 | 1-2 minutes |
| 10,000 | 15-30 minutes |
| 50,000 | 1-2 hours |

Time includes rate limit waits.

## Privacy & Security

- ✅ All processing happens locally
- ✅ No data sent to third-party servers
- ✅ Cookies stay on your machine
- ✅ Uses your existing X session
- ✅ Open source - audit the code yourself

## License

MIT License - Feel free to use, modify, and distribute.

## Credits

Inspired by the [Twitter Exporter](https://chrome.google.com/webstore/detail/twitter-exporter/lnklhjfbeicncichppfbhjijodjgaejm) Chrome extension.

## Disclaimer

This tool is not affiliated with X Corp or Twitter. Use at your own risk. Be respectful of API rate limits and terms of service.
