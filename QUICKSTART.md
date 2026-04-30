# Quick start

The short path from "I have an X account" to "I have my likes on disk."

## Step 1: Install dependencies

```bash
cd x_likes_exporter_py
pip install -r requirements.txt
```

## Step 2: Export your cookies

### With a browser extension

1. Install [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm) for Chrome/Edge
2. Go to https://x.com and make sure you're logged in
3. Click the Cookie-Editor extension icon
4. Click "Export" → "JSON"
5. Save as `cookies.json` in the `x_likes_exporter_py` folder

### Manual method

1. Go to https://x.com while logged in.
2. Press F12 to open Developer Tools.
3. Go to Application → Cookies → https://x.com.
4. Find these two cookies and copy their values:
   - `ct0` (CSRF token)
   - `auth_token` (auth token)
5. Copy `examples/cookies.json.example` to `cookies.json`.
6. Replace `YOUR_CT0_TOKEN_HERE` and `YOUR_AUTH_TOKEN_HERE` with the real values.

## Step 3: Find your user ID

The exporter needs the numeric user ID, not the @handle.

From your profile page: open https://x.com/YOUR_USERNAME, hit F12, search the HTML for `data-user-id`, and copy the number.

Or use https://tweeterid.com/ and paste in your @handle.

## Step 4: Run it

### From the command line

```bash
# Export everything (JSON, CSV, Markdown, HTML)
python cli.py cookies.json YOUR_USER_ID

# See all options
python cli.py --help
```

### From Python

Create `export_my_likes.py`:

```python
from x_likes_exporter import XLikesExporter

# Your User ID (the number, not @username)
USER_ID = "123456789"  # Replace with your actual User ID

# Initialize exporter
exporter = XLikesExporter(
    cookies_file="cookies.json",
    output_dir="my_likes_export"
)

# Fetch all likes (this may take a while)
print("Fetching your likes...")
tweets = exporter.fetch_likes(
    user_id=USER_ID,
    download_media=True  # Set to False to skip images
)

print(f"✓ Found {len(tweets)} liked tweets!")

# Export to all formats
print("Exporting...")
exporter.export_all()

print("✓ Done! Check the 'my_likes_export' folder")
```

Run it:
```bash
python export_my_likes.py
```

## What you end up with

```
my_likes_export/
├── likes.json          # Complete data in JSON
├── likes.csv           # Spreadsheet-friendly format
├── likes.md            # Readable Markdown with images
├── likes.html          # Web page to view in browser
└── media/              # Downloaded images/videos
    ├── 123456_0.jpg
    ├── 123456_1.jpg
    └── ...
```

## Common commands

```bash
# Export only JSON and Markdown
python cli.py cookies.json YOUR_USER_ID --format json --format markdown

# Skip media download (faster)
python cli.py cookies.json YOUR_USER_ID --no-media

# Custom output folder
python cli.py cookies.json YOUR_USER_ID --output my_folder

# Show statistics
python cli.py cookies.json YOUR_USER_ID --stats
```

## Analyze with pandas

```python
from x_likes_exporter import XLikesExporter

exporter = XLikesExporter("cookies.json")
tweets = exporter.fetch_likes("YOUR_USER_ID", download_media=False)

# Get as DataFrame
df = exporter.get_dataframe()

# Top 10 most liked tweets
print(df.nlargest(10, 'favorite_count')[['user_screen_name', 'text', 'favorite_count']])

# Save analysis
df.to_csv("analysis.csv", index=False)
```

## Troubleshooting

`Invalid cookies`: re-export from the browser, making sure you're logged in.

`Authentication failed`: the session expired. Log out, log back in, export fresh cookies.

Takes a long time: X's rate limit is the bottleneck. For 10,000+ likes, plan on 1-2 hours.

No images downloading: pass `download_media=True` to `fetch_likes()` and check your connection.

## Next steps

- [README.md](README.md) for the full reference.
- [examples/example_usage.py](examples/example_usage.py) for more code samples.

If you hit something not covered above, open an issue on GitHub.
