# Quick Start Guide

Get up and running with X Likes Exporter in 5 minutes!

## Step 1: Install Dependencies

```bash
cd x_likes_exporter_py
pip install -r requirements.txt
```

## Step 2: Export Your Cookies

### Using a Browser Extension (Recommended)

1. Install [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm) for Chrome/Edge
2. Go to https://x.com and make sure you're logged in
3. Click the Cookie-Editor extension icon
4. Click "Export" → "JSON"
5. Save as `cookies.json` in the `x_likes_exporter_py` folder

### Manual Method

1. Go to https://x.com (logged in)
2. Press F12 to open Developer Tools
3. Go to Application → Cookies → https://x.com
4. Find these two cookies and copy their values:
   - `ct0` - your CSRF token
   - `auth_token` - your authentication token
5. Copy `examples/cookies.json.example` to `cookies.json`
6. Replace `YOUR_CT0_TOKEN_HERE` and `YOUR_AUTH_TOKEN_HERE` with your actual values

## Step 3: Find Your User ID

### Method 1: From Profile Page

1. Go to your profile: https://x.com/YOUR_USERNAME
2. Right-click anywhere → "Inspect" (or press F12)
3. Press Ctrl+F (or Cmd+F) to search
4. Search for: `data-user-id`
5. Copy the number (e.g., `123456789`)

### Method 2: Using a Tool

Visit: https://tweeterid.com/
Enter your @username and get your User ID

## Step 4: Run the Exporter

### Command Line (Easiest)

```bash
# Export everything (JSON, CSV, Markdown, HTML)
python cli.py cookies.json YOUR_USER_ID

# See all options
python cli.py --help
```

### Python Script

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

## What You'll Get

After export, you'll have:

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

## Common Commands

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

## Analyze with Pandas

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

### "Invalid cookies"
→ Re-export cookies from your browser. Make sure you're logged in to X.

### "Authentication failed"
→ Your session expired. Log out of X, log back in, and export fresh cookies.

### Takes too long
→ X has rate limits. For 10,000+ likes, it may take 1-2 hours. This is normal!

### No images downloading
→ Add `download_media=True` when calling `fetch_likes()`, or check your internet connection.

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [examples/example_usage.py](examples/example_usage.py) for more examples
- Analyze your data with Pandas!

## Need Help?

- Check the full README.md
- Look at example_usage.py
- Open an issue on GitHub
