# Quick start

The short path from "I have an X account" to "I have my likes on disk."

## Step 1: Install dependencies

```bash
cd x_likes_exporter_py
uv sync
```

If you don't have uv yet, see https://docs.astral.sh/uv/.

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

## Step 4: Configure and run

```bash
cp .env.sample .env && $EDITOR .env   # set X_USER_ID (and X_USERNAME for reference)
./scrape.sh
```

`scrape.sh` runs `cli.py` through `uv run` and resumes from a checkpoint if one is present.

## What you end up with

```
output/
├── likes.json
├── likes.csv
├── likes.html
├── by_month/             # per-month Markdown with embedded local images
│   ├── likes_2025-04.md
│   ├── likes_2025-03.md
│   └── ...
└── media/                # downloaded images and videos
    ├── 123456_0.jpg
    └── ...
```

## Common commands

```bash
./scrape.sh --no-media          # skip media download
./scrape.sh --stats             # print stats at the end
./scrape.sh --format markdown   # only the per-month Markdown
./scrape.sh --output my_export  # custom output directory
```

## Analyze with pandas

```python
from x_likes_exporter import XLikesExporter

exporter = XLikesExporter("cookies.json")
exporter.fetch_likes("YOUR_USER_ID", download_media=False)

df = exporter.get_dataframe()
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
