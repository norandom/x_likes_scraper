# X Likes Exporter (Python)

[![CI](https://github.com/norandom/x_likes_scraper/actions/workflows/build.yml/badge.svg)](https://github.com/norandom/x_likes_scraper/actions/workflows/build.yml)
[![Release](https://github.com/norandom/x_likes_scraper/actions/workflows/release.yml/badge.svg)](https://github.com/norandom/x_likes_scraper/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/norandom/x_likes_scraper)](https://github.com/norandom/x_likes_scraper/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/norandom/x_likes_scraper)

Three layers built on the same local archive:

1. **Scraper** — pulls liked tweets off X, writes JSON / CSV / Excel / Markdown / HTML, downloads media. Resumes from a checkpoint after a network blip or rate-limit wait.
2. **Search** — hybrid recall (BM25 + dense embeddings, fused with RRF and re-ranked) over the archive. Reachable as a CLI or as an MCP server for Claude Code, Claude Desktop, and other MCP clients.
3. **Social Network Intelligence** — synthesis reports (brief / synthesis / trend) with a mermaid mindmap of authors, sources, themes, and hashtags. DSPy-typed prompt with an entity-relevance filter; in-memory knowledge graph you can dump as mermaid or JSON.

Everything runs locally. Cookies, tweets, and synthesis prompts never cross a network boundary except for the embedding endpoint and the synthesis LM you point at.

## Requirements

Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                          # install dependencies
cp .env.sample .env && $EDITOR .env
```

`uv sync` creates `.venv/`, installs the deps from `pyproject.toml`, and pins them in `uv.lock`.

---

## Layer 1: Scraper

Pulls liked tweets off X, writes them in your chosen format.

### Quick start

```bash
cp /path/to/cookies.json .       # see "Exporting cookies" below
./scrape.sh
```

`scrape.sh` loads `.env`, calls `cli.py` through `uv run`, passes `--resume` so an interrupted run resumes from its checkpoint. Extra flags forward:

```bash
./scrape.sh --no-media         # skip media download
./scrape.sh --stats            # print stats at the end
./scrape.sh --format markdown  # only the per-month Markdown
```

### Exporting cookies

Two options.

**Browser extension** — install one, log into [x.com](https://x.com), export cookies as JSON, save as `cookies.json`:
- Chrome / Edge: [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm)
- Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

**Manually** — DevTools (F12) on x.com → Application → Cookies → https://x.com, copy `ct0` (CSRF token) and `auth_token` into a `cookies.json` file.

<details>
<summary>cookies.json format</summary>

```json
[
  {"domain": ".x.com", "name": "ct0", "value": "your_ct0", "path": "/", "secure": true, "httpOnly": false},
  {"domain": ".x.com", "name": "auth_token", "value": "your_auth_token", "path": "/", "secure": true, "httpOnly": true}
]
```
</details>

### Finding your user ID

`X_USER_ID` is the numeric ID, not the @handle. Look it up at [tweeterid.com](https://tweeterid.com/), or:

```bash
curl "https://tweeterid.com/ajax.php?username=YOUR_USERNAME"
```

### Calling the CLI directly

```bash
uv run python cli.py cookies.json YOUR_USER_ID --resume
uv run python cli.py cookies.json YOUR_USER_ID --no-media
uv run python cli.py cookies.json YOUR_USER_ID --format json --format markdown
uv run python cli.py cookies.json YOUR_USER_ID --format markdown --single-file
```

### From Python

```python
from x_likes_exporter import XLikesExporter

exporter = XLikesExporter(cookies_file="cookies.json", output_dir="output")
tweets = exporter.fetch_likes(user_id="YOUR_USER_ID", download_media=True, resume=True)
exporter.export_all()
```

### Resume

Tens of thousands of likes take hours and any run will hit a network blip or rate-limit wait. The exporter writes progress to `.export_checkpoint.json` and `.export_tweets.pkl` in the output directory as it goes. `--resume` (CLI) or `resume=True` (Python) picks up from the checkpoint. The exporter merges new tweets with the saved set and dedupes by ID. Checkpoints delete themselves once an export finishes.

### Output formats

- **JSON** — tweet records with user info, media, engagement counts.
- **CSV / Excel** — flat table, one row per tweet.
- **Markdown** — split per month under `output/by_month/likes_YYYY-MM.md` by default; `--single-file` produces one big file. The per-month split is what the search and synthesis layers index.
- **HTML** — single file you open in a browser, media inline.

### Rate limiting

X gives you roughly 500 requests per 15-minute window. The client reads `x-rate-limit-*` headers; when `remaining` drops to 1, it sleeps until reset (plus a 5-second buffer). A 1-second pause between requests rounds out the politeness budget.

For 10,000+ likes, expect 1–2 hours including rate-limit waits.

### Troubleshooting

- "Invalid cookies" — verify `ct0` and `auth_token` are present; re-export after a fresh login.
- "Authentication failed" — session probably expired; log out, log back in, re-export.
- "Rate limit exceeded" — the client handles this; if it does not, wait 15 minutes.
- "No tweets found" — verify the user ID, that the account has likes, and the cookies are valid.
- Images not downloading — check connectivity; some media URLs expire; try `--no-media` to isolate the problem.

---

## Layer 2: Search

Hybrid recall (BM25 lexical + dense embeddings via OpenRouter) fused with Reciprocal Rank Fusion, re-ranked by a heavy-ranker-style scorer with a per-query coverage penalty for missing high-IDF terms. Reachable two ways, sharing the same on-disk cache:

- **CLI** (`uv run x-likes-mcp --search "query"`) — standalone, no MCP client, good for one-off queries and cache warm-up.
- **MCP server** (`uv run x-likes-mcp` with no flags) — stdio MCP server for Claude Code, Claude Desktop, and other MCP clients.

### CLI search

```bash
# Build or load the index, print a summary, exit (warms the cache).
uv run x-likes-mcp --init

# Run a query.
uv run x-likes-mcp --search "AI pentesting" --limit 5

# Date filter.
uv run x-likes-mcp --search "rust async" --year 2025 --limit 10
uv run x-likes-mcp --search "kubernetes" --month-start 2025-01 --month-end 2025-06

# Walker explainer (one chat-completions call; opt-in).
uv run x-likes-mcp --search "system design" --with-why --limit 5

# Machine-readable output.
uv run x-likes-mcp --search "graph databases" --json | jq '.tweet_id'
```

Each hit prints as a small block: metadata (score, walker relevance, year-month, handle, tweet id), canonical `https://x.com/...` link, then the snippet. The snippet has `t.co` shortlinks stripped and resolved URLs appended. If `./scrape.sh` ran with media downloads enabled, the printer also lists each downloaded media file as a `file://` link. iTerm2, Kitty, Wezterm, and VS Code make those clickable.

```
 1. score=56.12 │ wr=0.54 │ 2026-05 │ @tom_doerr │ id=2050120307399147786
    https://x.com/tom_doerr/status/2050120307399147786
    AI vibe investing agent for financial markets  https://github.com/ginlix-ai/LangAlpha
```

First invocation embeds the whole corpus (30–90 seconds with the default model). Every later run hits the on-disk cache and starts in under a second.

### MCP tools

The MCP server exposes five tools:

- `search_likes(query, year, month_start, month_end, with_why)` — natural-language search. Default path runs hybrid recall over the corpus and re-ranks with the heavy ranker. No chat-completions call by default; one OpenRouter `/v1/embeddings` request per query. Each hit carries a `tweet_url` (canonical `https://x.com/{handle}/status/{id}`) and a `urls` list of resolved HTTP(S) destinations. Every entry in `urls` is wrapped in `<<<URL>>> ... <<<END_URL>>>` so a calling LLM can be told to read fenced content as a URL string.
- `synthesize_likes(query, report_shape, fetch_urls, hops, ...)` — the Layer 3 entry. See below.
- `list_months()` — months for which per-month Markdown exists, reverse-chronologically.
- `get_month(year_month)` — raw Markdown for one month.
- `read_tweet(tweet_id)` — one tweet's metadata.

The ranker design borrows the heavy-ranker shape from `twitter/the-algorithm` for the features the export already has. Not a port. Defaults are tuned for search: cosine relevance dominates, engagement is a soft prior. The coverage penalty subtracts `idf(token) * coverage_penalty` from a hit's score for each tokenized query term the hit's text does not contain, so a query like `AI factors for portfolio` stops surfacing unrelated AI tweets just because they share the word `AI`.

### Defenses against malicious tweet content

Tweets in `likes.json` carry text written by arbitrary X users, so the search path does not trust them.

Snippet, handle, and display name go through `sanitize_text` before leaving the index. That strips ANSI escape sequences, C0/C1 controls (newline and tab survive), Unicode bidirectional overrides in U+202A–U+202E and U+2066–U+2069, zero-width joiners, and the BOM. Text is NFKC-normalized first so visually identical confusables collapse to a canonical form. The result is round-tripped through UTF-8 with `errors="replace"` so a stray surrogate cannot crash whoever consumes the response.

The walker explainer (`with_why=true`) wraps each tweet body in `<<<TWEET_BODY>>> ... <<<END_TWEET_BODY>>>`, and the system prompt tells the model fenced content is data, not instructions. If a tweet body contains either marker, or any other fence marker, it is replaced with `[FENCE]` before fencing. A crafted tweet cannot close the fence early and resume control of the prompt.

Resolved URLs in `tweet.urls` get the same treatment. Each URL is filtered to HTTP(S), sanitized, and wrapped in `<<<URL>>> ... <<<END_URL>>>`. URLs are never re-resolved at query time. We trust whatever the export already pulled from Twitter's API, not anything reachable on the network.

### Why hosted dense embeddings (and not a local transformer)

The dense retrieval path is network-based rather than running a local transformer. The maintainer's primary platform (Intel macOS x86_64) has no recent PyTorch or ONNX Runtime wheels: `sentence-transformers` and `fastembed` both refuse to install. OpenRouter serves embedding models through the OpenAI-shape `/v1/embeddings` endpoint, which the existing `openai` SDK reaches by changing `base_url`. No new SDK dep, no transformer model in-process. The hybrid recall adds one new pure-python dep: `rank_bm25>=0.2`, ~50 KB, no native code.

### Configuration

OpenRouter (dense embeddings, required) in `.env`:

```ini
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Cost: roughly $0.01 to embed a 7,780-tweet corpus once (~400K tokens at $0.02/1M), and effectively free per query. The on-disk cache means you only pay the corpus cost when the model name changes or new likes are scraped.

Walker / chat-completions endpoint (opt-in via `with_why=true`):

```ini
OPENAI_BASE_URL=http://10.0.0.59:8317/v1
OPENAI_API_KEY=sk-dummy
OPENAI_MODEL=claude-opus-4-1-20250805
```

The `openai` Python SDK reads `OPENAI_BASE_URL` at client-construction time, so any OpenAI-compatible endpoint works. These three are unused on the default path and only consulted when a request sets `with_why=true`. **Single-key shortcut:** if only `OPENROUTER_API_KEY` is set, the loader points the LM at OpenRouter as well, with `OPENAI_MODEL` defaulting to `openai/gpt-4o-mini`. Explicit `OPENAI_*` always wins.

Optional ranker weights (override the in-code defaults). Shell environment variables win over `.env` file values, so a one-shot `RANKER_W_RELEVANCE=40 uv run x-likes-mcp --search ...` takes effect:

```ini
RANKER_W_RELEVANCE=80.0
RANKER_W_FAVORITE=0.5
RANKER_W_RETWEET=0.5
RANKER_W_REPLY=0.3
RANKER_W_VIEW=0.1
RANKER_W_AFFINITY=1.0
RANKER_W_RECENCY=1.5
RANKER_W_VERIFIED=0.5
RANKER_W_MEDIA=0.3
RANKER_RECENCY_HALFLIFE_DAYS=180
RANKER_W_COVERAGE_PENALTY=5.0
```

`author_affinity[handle]` is precomputed from your own like history as `log1p(count_of_likes_from_handle)`. To switch to feed-style recommendation (engagement leads, relevance is a soft prior), raise the engagement weights and lower `RANKER_W_RELEVANCE`. Set `RANKER_W_COVERAGE_PENALTY=0` to disable the missing-token penalty.

### Registering with Claude Code

`uv run x-likes-mcp` resolves `pyproject.toml`, `.venv/`, and `.env` from the current working directory. The MCP config either sets that directory or is invoked from the project root.

**Project-scoped** (`.mcp.json` at the project root):

```json
{"mcpServers": {"x-likes": {"command": "uv", "args": ["run", "x-likes-mcp"]}}}
```

**Globally available** (your user-scoped `~/.claude.json`, or `.mcp.json` here with an absolute path):

```json
{
  "mcpServers": {
    "x-likes": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/x_likes_exporter_py", "x-likes-mcp"]
    }
  }
}
```

Or use the CLI for user-scope registration:

```bash
claude mcp add x-likes --scope user -- \
  uv run --directory /absolute/path/to/x_likes_exporter_py x-likes-mcp
```

### Caches and first-run cost

Three on-disk caches under the configured output directory:

- `output/tweet_tree_cache.pkl` — per-month tweet tree (mtime-invalidated).
- `output/corpus_embeddings.npy` — float32 `(N, D)` matrix of L2-normalized tweet embeddings.
- `output/corpus_embeddings.meta.json` — schema version, model name, tweet count, ordered tweet ids.

Embedding-cache invalidation is structural: rebuild on `EMBEDDING_MODEL` change, on tweet-id-set change, or on schema-version bump.

First run embeds the whole corpus (~7,780 tweets) through OpenRouter, typically 30–90 seconds with the default paid model. Per-query cost on a warm cache is one OpenRouter request (~80–300 ms over LAN/WAN); typical queries return in well under 2 seconds end-to-end.

### Prerequisites

1. `./scrape.sh` has been run at least once so `output/likes.json` and `output/by_month/` exist.
2. An OpenRouter API key. Sign up at [openrouter.ai](https://openrouter.ai); the default model costs ~$0.01 for the one-time corpus embed and effectively nothing per query.
3. Optional: a local OpenAI-Chat-Completions-compatible LLM endpoint, only required when callers pass `with_why=true` (Layer 2) or use Layer 3 below.

---

## Layer 3: Social Network Intelligence

Turn a query into a structured markdown report or a knowledge-graph dump. Same hybrid retrieval as Layer 2, plus entity extraction, an in-memory knowledge graph, optional URL fetching through a sandboxed `crawl4ai` Docker container, an optional second-hop search seeded by the round-1 entities, and a DSPy-typed synthesis pass.

### Generating a report

```bash
# Brief (~300-word concept brief; cheapest).
uv run x-likes-mcp --report brief --query "AI factors for portfolio" --out brief.md

# Synthesis (long-form + mermaid mindmap + per-cluster tweet list).
uv run x-likes-mcp --report synthesis --query "AI factors for portfolio" --hops 2 --out report.md

# Trend (month-bucketed timeline + mindmap).
uv run x-likes-mcp --report trend --query "AI factors for portfolio" --hops 2 --out trend.md
```

See [`examples/synthesis_report_example.md`](examples/synthesis_report_example.md) for a full report.

Useful flags:
- `--hops 2` — seed a second hop of search from the top entities the first hop surfaced. Catches adjacent material the literal query misses.
- `--limit 50` — round-1 hit cap.
- `--year 2025 --month-start 06 --month-end 12` — same date filter as `--search`.
- `--fetch-urls` — opt in to the crawl4ai container for resolved URL bodies. Off by default.
- `--no-filter-entities` — skip the LM-backed entity-relevance filter (faster, noisier).
- omit `--out` — markdown to stdout.

Exit codes: `0` success, `2` failure (LM down, container unreachable while `--fetch-urls`, malformed shape, synthesis-validation error). Stderr names the category.

### Inspecting the knowledge graph

The `--kg` mode skips the LM and prints just the in-memory KG as a mermaid mindmap (default) or JSON. Useful for inspecting which entities the synthesizer would see for a query, and for piping the graph into external tools.

```bash
# Mermaid mindmap to stdout.
uv run x-likes-mcp --kg "AI factors for portfolio" --hops 2

# JSON dump to a file.
uv run x-likes-mcp --kg "AI factors for portfolio" --hops 2 --json --out kg.json

# Disable the singleton-entity filter to see everything.
uv run x-likes-mcp --kg "AI factors for portfolio" --min-weight 0
```

Entities: `Authors` (handles), `Sources` (URL domains), `Themes` (recurring concepts), `Hashtags`. The KG dedupes domains via `www.` collapse, normalizes concepts to `lower_snake_case`, and weights every node by occurrence count. Singleton-entity filtering happens at render time (`--min-weight`, default 2.0).

### How it works

```
query
  → round-1 search (BM25 + dense, fused, re-ranked)
  → KG seed (regex entities + DSPy fallback for empty hits)
  → optional LM-backed entity-relevance filter (drops off-topic entities)
  → optional round-2 fan-out (parallel search per top-K entity)
  → optional URL fetch via crawl4ai container (sandboxed, SSRF-checked)
  → fenced synthesis context (per-source byte caps, total budget, six fence families)
  → DSPy ChainOfThought synthesizer (claim-source validator, retry-once on hallucination)
  → markdown render (mermaid mindmap, t.co rewrite, clickable cite links)
```

The fenced context budget defaults to 32 KB total with per-source caps (280 B per tweet, 4 KB per URL body, 64 B per entity / KG label). Six fence families wrap every untrusted source: `<<<TWEET_BODY>>>`, `<<<URL>>>`, `<<<URL_BODY>>>`, `<<<ENTITY>>>`, `<<<KG_NODE>>>`, `<<<KG_EDGE>>>`. Cross-marker neutralization runs before fencing so a crafted tweet cannot prematurely close one fence and reopen another.

The synthesizer is `dspy.ChainOfThought` over a typed signature with Pydantic-typed structured outputs (`Claim`, `Section`, `MonthSummary`). Every claim must cite a `tweet:<id>` or `url:<final_url>` that was actually in the fenced context; a hallucinated citation triggers one corrective retry, and a second failure raises a structured validation error.

### URL fetching (opt-in)

`--fetch-urls` enables the crawl4ai HTTP client. The host code never imports `crawl4ai`; it talks to a sandboxed `unclecode/crawl4ai` Docker container over HTTP and trusts the markdown response only after host-side sanitization.

Run the container before the report:

```bash
docker run -d --name crawl4ai \
  -p 127.0.0.1:11235:11235 \
  --network bridge \
  unclecode/crawl4ai:latest
```

Override the endpoint via `CRAWL4AI_BASE_URL` (default `http://127.0.0.1:11235`). The fetcher applies an SSRF guard with two tiers:
- **Unconditional block:** loopback, cloud-metadata addresses (169.254.169.254, 192.0.0.192, fd00:ec2::254), broadcast, multicast, IANA-reserved.
- **Private block:** RFC1918, IPv4 link-local minus the metadata IP, IPv6 link-local, IPv6 ULA. The operator can punch holes through this tier via `URL_FETCH_ALLOWED_PRIVATE_CIDRS=10.100.0.0/16,...` for zero-trust deployments where internal services are legitimate fetch targets.

Per-URL: 5-second timeout, redirect re-validation (max 3 hops, each re-checked), content-type allowlist (`text/html`, `text/plain`, `application/json`, `application/pdf`), 4 KB body cap before fencing. Sanitized markdown gets cached at `output/url_cache/<sha256>.json` with a 30-day TTL; raw HTML / PDF bytes never touch disk.

### Entity-relevance filter

By default, `--report` runs an LM-backed entity filter once per report. The LM sees the union of regex-extracted handles / hashtags / domains / concepts and returns only the topical ones for the query. That keeps `handle:elonmusk` from polluting an `AI factors for portfolio` report just because elonmusk's tweets happen to mention AI. One LM call per report, not per hit. Disable with `--no-filter-entities`.

### Configuration

The synthesis layer reads:

```ini
# Required (or use the OpenRouter shortcut — see Layer 2).
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-...
OPENAI_MODEL=anthropic/claude-3.5-haiku  # any OpenAI-shape model

# Optional, with defaults.
CRAWL4AI_BASE_URL=http://127.0.0.1:11235
URL_CACHE_DIR=output/url_cache
URL_CACHE_TTL_DAYS=30
SYNTHESIS_MAX_HOPS=2
SYNTHESIS_PER_SOURCE_BYTES=4096
SYNTHESIS_TOTAL_CONTEXT_BYTES=32768
SYNTHESIS_ROUND_TWO_K=5
URL_FETCH_ALLOWED_PRIVATE_CIDRS=                   # comma-separated CIDRs
```

### Optimizing the synthesis prompt (DSPy BootstrapFewShot)

```bash
# Place 5–10 hand-labeled (query, expected output) pairs at:
#   output/synthesis_labeled/<shape>.json
# Then run the optimizer; the compiled program lands at
#   output/synthesis_compiled/<shape>.json
uv run x-likes-mcp --report-optimize brief
```

The orchestrator picks up the compiled program automatically. Without one it falls back to the un-optimized signature.

---

## Architecture

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
│   XAPIClient     │  ← Fetch likes with pagination, rate limiting, cursor paging
└────────┬─────────┘
         ↓
┌──────────────────┐
│ MediaDownloader  │  ← Download images / videos
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Formatters     │  ← JSON / CSV / Excel / Markdown / HTML
└────────┬─────────┘
         ↓
┌──────────────────┐
│   TweetIndex     │  ← Hybrid recall: BM25 + dense + RRF + heavy ranker
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Synthesis-report │  ← Round-2 fan-out + KG + DSPy synthesis + markdown render
└──────────────────┘
```

## Project structure

```
x_likes_exporter_py/
├── x_likes_exporter/         # Layer 1: scraper
├── x_likes_mcp/              # Layer 2 + 3: search + synthesis
│   └── synthesis/            # Layer 3: orchestrator, KG, DSPy, fetcher, render
├── cli.py                    # Layer 1 CLI
├── scrape.sh                 # .env-driven scraper entry point
├── examples/                 # usage examples + a sample synthesis report
├── tests/                    # 760+ tests, offline by default
├── pyproject.toml
└── README.md
```

## Performance

- **Scrape:** ~20 likes/second (rate-limit capped). 10,000 likes → 15–30 minutes; 50,000 → 1–2 hours.
- **Index build:** 30–90 seconds for ~7,780 tweets on first run; sub-second on cache hit.
- **Search query:** under 2 seconds end-to-end (one embedding round-trip + local fusion).
- **Synthesis report:** 5–30 seconds, dominated by the LM round-trip.

## Privacy and security

Everything runs locally. Cookies, tweets, and the corpus stay on disk. The only outbound calls are: (1) X's public API during scrape, (2) OpenRouter's `/v1/embeddings` endpoint during search, (3) your configured LM endpoint during synthesis (or `with_why=true`), (4) the optional `crawl4ai` container if you opt into URL fetching. No third-party telemetry, no analytics, no phone-home.

## Development

```bash
uv sync --group dev               # tests + lint + types tooling
uv run pytest                     # 760+ tests, offline by default
uv run ruff check .
uv run ruff format .
uv run mypy                       # scope is x_likes_mcp/ only
uv run pre-commit install         # one-time
uv run pre-commit run --all-files # full set on demand
```

`.kiro/`, `.claude/`, `output/`, `examples/`, and the `static_analysis_*` directories are excluded from every hook.

## License

MIT.

## Credits

Inspired by the [Twitter Exporter](https://chrome.google.com/webstore/detail/twitter-exporter/lnklhjfbeicncichppfbhjijodjgaejm) Chrome extension.

## Disclaimer

Not affiliated with X Corp or Twitter. Use at your own risk. Don't hammer the API.
