# X Likes Exporter (Python)

Export your liked tweets from X (formerly Twitter) to JSON, CSV, a Pandas DataFrame, Markdown with images, or HTML.

## What it does

- Exports to JSON, CSV, Excel, Pandas, Markdown, or HTML.
- Resumes interrupted exports from a checkpoint instead of restarting.
- Downloads images and videos and rewrites Markdown to point at the local files.
- Honors X's rate limit headers and waits when you run out of budget.
- Walks the cursor-based pagination so you get every like, not just the first page.
- Runs locally. Your cookies and tweets never leave the machine.

## Requirements

Python 3.8+, plus `requests`, `pandas`, `beautifulsoup4`, `Pillow`, `tqdm`, `python-dateutil` (all in `requirements.txt`). Install with `pip install -r requirements.txt`, or `pip install -e .` to install as a package.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Drop your exported cookies in the project root
#    (see "Exporting cookies" below for how)
cp /path/to/cookies.json .

# 3. Configure your account
cp .env.sample .env
$EDITOR .env       # fill in X_USER_ID (and X_USERNAME for your own reference)

# 4. Run it
./scrape.sh
```

That's it. `scrape.sh` reads `.env`, activates `venv/` if present, and runs the exporter with `--resume` so an interrupted run picks up where it left off. Anything you pass after `./scrape.sh` is forwarded to `cli.py`:

```bash
./scrape.sh --no-media         # skip media download
./scrape.sh --stats            # print stats at the end
./scrape.sh --format markdown  # only the per-month Markdown
```

### Exporting cookies

You need your X session cookies to authenticate. Two options.

**With a browser extension** — install one, log in to [x.com](https://x.com), export cookies as JSON, save as `cookies.json`:

- Chrome/Edge: [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm)
- Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

**Manually** — open DevTools (F12) on x.com → Application → Cookies → https://x.com, then copy `ct0` (CSRF token) and `auth_token` into a `cookies.json` file matching the format below.

<details>
<summary>cookies.json format</summary>

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

### Finding your user ID

`X_USER_ID` is the numeric ID, not the @handle. Look it up at [tweeterid.com](https://tweeterid.com/), or via:

```bash
curl "https://tweeterid.com/ajax.php?username=YOUR_USERNAME"
```

### Calling the CLI directly

`scrape.sh` is just a wrapper. If you'd rather invoke `cli.py` yourself:

```bash
python cli.py cookies.json YOUR_USER_ID --resume
python cli.py cookies.json YOUR_USER_ID --no-media
python cli.py cookies.json YOUR_USER_ID --format json --format markdown
python cli.py cookies.json YOUR_USER_ID --format markdown --single-file
```

### From Python

```python
from x_likes_exporter import XLikesExporter

exporter = XLikesExporter(cookies_file="cookies.json", output_dir="output")

tweets = exporter.fetch_likes(
    user_id="YOUR_USER_ID",
    download_media=True,
    resume=True,
)

exporter.export_all()
```

## Resume

Exporting tens of thousands of likes can take hours, and the run will eventually hit a network blip or a rate-limit wait. The exporter writes progress to `.export_checkpoint.json` and `.export_tweets.pkl` in the output directory as it goes. Pass `--resume` on the CLI (or `resume=True` in Python) to pick up where it stopped. The checkpoint holds the tweets fetched so far and the current pagination cursor; on resume the exporter merges new tweets with the saved set and deduplicates by ID. The checkpoint files are deleted automatically once an export finishes.

## Usage examples

### Basic export

```python
from x_likes_exporter import XLikesExporter

exporter = XLikesExporter("cookies.json", "output")
tweets = exporter.fetch_likes("123456789")
exporter.export_all()
```

### Export to specific formats

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

### Work with pandas

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

### Progress monitoring

```python
def progress_callback(current, total):
    print(f"Fetched {current} tweets...")

tweets = exporter.fetch_likes(
    user_id="123456789",
    progress_callback=progress_callback
)
```

### Fetch without media

```python
# Faster if you don't need images
tweets = exporter.fetch_likes(
    user_id="123456789",
    download_media=False
)
```

### Get statistics

```python
stats = exporter.get_stats()
print(f"Total tweets: {stats['total_tweets']}")
print(f"Total media: {stats['total_media']}")
print(f"Total likes: {stats['total_likes']}")
```

## Output formats

### JSON

Tweet data with user info, media, and engagement counts:

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

Flat table, one row per tweet:

| tweet_id | text | user_screen_name | favorite_count | retweet_count | created_at |
|----------|------|------------------|----------------|---------------|------------|
| 123... | Tweet... | username | 50 | 10 | 2025-01-01 |

### Markdown (split by month)

By default, Markdown exports are split per month. The exporter parses each tweet's `created_at`, groups by year-month, and writes one file per group to `output/by_month/` (e.g. `likes_2025-01.md`). A few years of liked tweets in a single Markdown file makes most editors crawl, which is the only reason this exists.

To force a single `likes.md` file instead, pass `--single-file` on the CLI or `split_by_month=False` in Python.

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

A single HTML file you can open in a browser. Tweets are styled and media is embedded inline.

## Advanced usage

### Filter tweets before export

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

### Custom analysis

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

Roughly the same flow as the Chrome extension that inspired it:

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

## Rate limiting

X's API gives you roughly 500 requests per 15-minute window. The client reads the `x-rate-limit-limit`, `x-rate-limit-remaining`, and `x-rate-limit-reset` headers from each response. When `remaining` drops to 1, it sleeps until the reset timestamp (plus a 5-second buffer) and continues. There's also a 1-second pause between requests so you're not slamming the endpoint.

Progress is printed as you go:

```
Fetching page 25...
Fetched 20 likes. Total: 500
Rate limit: 475/500
```

For 10,000+ likes, expect 1-2 hours including the rate-limit waits.

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

## Project structure

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

## API reference

### XLikesExporter

The main exporter.

```python
exporter = XLikesExporter(cookies_file: str, output_dir: str = "output")
```

**Methods:**

- `fetch_likes(user_id, download_media=True, progress_callback=None, resume=False)` → List[Tweet]
- `export_json(filename, include_raw=False)` → None
- `export_csv(filename)` → None
- `export_excel(filename)` → None
- `export_markdown(filename, include_media=True, split_by_month=True)` → None
- `export_html(filename)` → None
- `export_all(base_name="likes", include_raw=False)` → None
- `get_dataframe()` → pandas.DataFrame
- `get_stats()` → dict

### Tweet model

A single tweet.

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

Fetch throughput is roughly 20 likes per second, capped by X's rate limit. Media download speed depends on your connection. In-memory processing is negligible (under a second for 1,000 tweets).

Memory use is around 2-5 KB per tweet, so 10,000 likes is ~20-50 MB and 50,000 is ~100-250 MB.

Rough time-to-finish, including rate-limit waits:

| Likes  | Time            |
|--------|-----------------|
| 100    | 5-10 seconds    |
| 1,000  | 1-2 minutes     |
| 10,000 | 15-30 minutes   |
| 50,000 | 1-2 hours       |

## Privacy and security

Everything runs locally. The script reads your `cookies.json`, talks to X's API directly using your existing session, and writes files to disk. Nothing is sent to a third-party server. The source is here, so you can read it yourself before running it.

## License

MIT.

## Credits

Inspired by the [Twitter Exporter](https://chrome.google.com/webstore/detail/twitter-exporter/lnklhjfbeicncichppfbhjijodjgaejm) Chrome extension.

## Disclaimer

Not affiliated with X Corp or Twitter. Use at your own risk and don't hammer the API.
