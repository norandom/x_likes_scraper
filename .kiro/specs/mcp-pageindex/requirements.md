# Requirements Document

## Project Description (Input)

I keep years of liked tweets as per-month Markdown files under `output/by_month/`. Today the only way to find anything is grep or scrolling. I want to ask "what was that thread about kernel scheduling I liked last spring?" and get an answer with tweet IDs and dates. The data is already heading-shaped (`## YYYY-MM`, `### @handle`), so I parse it into a tree, walk the tree per-month with a local LLM that returns JSON of plausibly-relevant tweet IDs with a `why` line, and rank the hits with a heavy-ranker-feature-shape combiner inspired by `twitter/the-algorithm` (the feature design, not the infrastructure). This spec adds a stdio MCP server that registers with Claude Code (or any MCP client) and exposes the like history through four tools backed by that pipeline. The LLM call goes to a local OpenAI-compatible endpoint configured in `.env`. The server consumes the read API defined in Spec 1 (`codebase-foundation`) and never re-fetches from X.

## Boundary Context

- **In scope**: A stdio MCP server packaged as `x_likes_mcp` with a `python -m x_likes_mcp` entry point and a `pyproject.toml` script entry. Three new modules in that package — `tree.py` (pure-Python markdown parser), `walker.py` (LLM-driven semantic walk per month), `ranker.py` (heavy-ranker-feature-shape weighted combiner) — plus the existing `config.py`, `errors.py`, `index.py`, `tools.py`, `server.py`, `__main__.py`. Four MCP tools: `search_likes(query, year=None, month_start=None, month_end=None)`, `list_months()`, `get_month(year_month)`, `read_tweet(tweet_id)`. A tree cache file alongside the export with mtime-based invalidation. Three OpenAI-compatible env vars (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`) plus `RANKER_W_*` weight overrides; `.env.sample` documents both. README section on registering the server with Claude Code. Tests that mock the walker (the LLM call); the tree parser and ranker are tested directly.
- **Out of scope**: Re-fetching from X. HTTP/SSE transport (stdio only). A web UI. Authentication. Tools that mutate the export. Hosted LLM services by default. The PageIndex PyPI package (replaced by our own implementation). Vector embeddings, MMR, BGE-style similarity ranking. Real-graph features, SimClusters, TwHIN, anything else from `twitter/the-algorithm` that needs Twitter-scale infrastructure. Anything Spec 1 owns beyond the targeted change to `MarkdownFormatter.export` already in this spec's foundation tasks.
- **Adjacent expectations**: Spec 1 (`codebase-foundation`) owns the read API. This spec consumes `load_export(path)` and `iter_monthly_markdown(path)` from `x_likes_exporter` and depends on the `Tweet` dataclass exposed there, including the engagement fields (`favorite_count`, `retweet_count`, `reply_count`, `view_count`), the author handle on `Tweet.user.screen_name`, the `verified` flag on the user, the `media` list, and the parseable `created_at`. If those signatures or the directory layout under `output/by_month/` change, this spec re-checks. The MCP server requires that `scrape.sh` has been run at least once; an empty or missing export is a documented startup error, not a feature.

## Requirements

### Requirement 1: Stdio MCP server runnable from a fresh checkout

**Objective:** As a single-user maintainer, I want to start the MCP server from the project root with one command after `uv sync`, so that any MCP client connected over stdio can talk to my like history without extra wiring.

#### Acceptance Criteria

1. When a user runs `python -m x_likes_mcp` from the repository root after `uv sync`, the X Likes MCP Server shall start a stdio MCP server that announces its tool surface and stays running until the client disconnects.
2. When a user runs the console script declared in `pyproject.toml` (for example `uv run x-likes-mcp`), the X Likes MCP Server shall start the same stdio server as `python -m x_likes_mcp`.
3. The X Likes MCP Server shall declare itself on startup with a stable server name and version that an MCP client can read.
4. If `uv sync` is run on a fresh checkout, the X Likes MCP Server shall install its runtime dependencies (the MCP Python SDK and the OpenAI SDK) without requiring any change to the existing `[project.dependencies]` set used by `scrape.sh`.
5. The X Likes MCP Server shall not require `cookies.json`, network access to X, or a real scrape at startup; it consumes only the existing files under `output/`.

### Requirement 2: Configuration through `.env`

**Objective:** As a single-user maintainer, I want the local LLM endpoint, the export directory, the cache location, and the ranker weights to come from `.env`, so that I can point the server at my own infrastructure and tune ranking without editing source.

#### Acceptance Criteria

1. When the X Likes MCP Server starts, the X Likes MCP Server shall read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` from the project's `.env` file (and from the process environment if `.env` is absent).
2. When the X Likes MCP Server starts, the X Likes MCP Server shall read `OUTPUT_DIR` from `.env` (defaulting to `output`) and locate `output/by_month/` and `output/likes.json` underneath it.
3. If `OPENAI_BASE_URL` or `OPENAI_MODEL` is unset or empty at startup, the X Likes MCP Server shall fail loudly with an error message that names the missing variable.
4. The X Likes MCP Server shall treat `OPENAI_BASE_URL` as an OpenAI-compatible base URL (for example `http://10.0.0.59:8317/v1`) and shall not call any hosted LLM service by default.
5. The X Likes MCP Server shall ensure `OPENAI_BASE_URL` (and `OPENAI_API_KEY` when set) is present in `os.environ` before the walker constructs the OpenAI SDK client, so the SDK picks the values up at client-construction time.
6. The X Likes MCP Server shall read optional ranker weight overrides from `.env` as `RANKER_W_RELEVANCE`, `RANKER_W_FAVORITE`, `RANKER_W_RETWEET`, `RANKER_W_REPLY`, `RANKER_W_VIEW`, `RANKER_W_AFFINITY`, `RANKER_W_RECENCY`, `RANKER_W_VERIFIED`, `RANKER_W_MEDIA`, plus an optional `RANKER_RECENCY_HALFLIFE_DAYS`. Missing variables fall back to the documented defaults; non-numeric values are treated as a configuration error and reported with the variable name.
7. The X Likes MCP Server shall extend `.env.sample` to document the three OpenAI variables and the ranker weight variables, including a note that the LLM endpoint is OpenAI-compatible and local by default.

### Requirement 3: Tree built from the per-month Markdown

**Objective:** As a single-user maintainer, I want the server to build a tree of tweets from the existing per-month Markdown on first run and reuse it across restarts, so that startup is fast after the first time.

#### Acceptance Criteria

1. When the X Likes MCP Server starts and no cache file exists, the X Likes MCP Server shall enumerate `output/by_month/likes_YYYY-MM.md` via the read API from Spec 1, build a `TweetTree` from those files using `tree.build_tree`, and persist the tree to a cache file under the output directory.
2. When the X Likes MCP Server starts and a cache file exists whose modification time is newer than every Markdown file under `output/by_month/`, the X Likes MCP Server shall load the cached tree without rebuilding.
3. If any Markdown file under `output/by_month/` has a modification time newer than the cache file, the X Likes MCP Server shall rebuild the tree and overwrite the cache.
4. If `output/by_month/` is missing or contains no `likes_YYYY-MM.md` files, the X Likes MCP Server shall fail loudly with an error message identifying the missing directory or empty content, rather than starting in a half-working state.
5. The X Likes MCP Server shall write the cache file alongside the export (under the configured output directory) and shall not write inside the package directory or the user's home.
6. The tree builder (`tree.build_tree`) shall be a pure-Python markdown parser with no LLM call and no network access. Each `TreeNode` shall carry `year_month`, `tweet_id`, `handle`, `text`, and `raw_section`. Tweet IDs shall be extracted from the canonical `🔗 [View on X](https://x.com/{handle}/status/{id})` link.

### Requirement 4: Walker — per-month LLM walk returning JSON hits

**Objective:** As an MCP client user, I want the server to surface tweets that are plausibly relevant to my question (including indirect or thematic relevance), not just keyword matches, so that "what was that thread about kernel scheduling" finds threads that don't literally say "scheduling."

#### Acceptance Criteria

1. When `walker.walk(tree, query, months_in_scope)` is called, the walker shall iterate the in-scope months in chronological order and, for each month, batch the month's tweets into chunks of at most N (default 30) and issue one LLM call per chunk via the OpenAI Python SDK at `OPENAI_BASE_URL` using `OPENAI_MODEL`.
2. The walker shall use a chat-completions prompt that asks the model to return JSON of the shape `[{"id": "...", "relevance": <0..1 float>, "why": "..."}, ...]` containing only tweets the model judges plausibly related; tweets the model judges irrelevant shall be omitted from the response.
3. The walker shall parse the JSON response, drop entries whose `id` is not present in the chunk it sent, drop entries whose `relevance` is not a finite number in `[0, 1]`, and accumulate the surviving entries as `WalkerHit(tweet_id, relevance, why)` instances across all chunks.
4. If the LLM call fails (network error, non-2xx response, malformed body) for one chunk, the walker shall surface that as an upstream failure for the search call as a whole rather than silently returning partial results; alternatively, the walker may emit zero hits for the failed chunk and proceed, but only if the implementation makes the choice explicit and consistent across chunks.
5. The walker module shall be the only place in the package that performs LLM calls; no other module (including `tree.py`, `ranker.py`, `index.py`, `tools.py`, `server.py`) shall import the OpenAI SDK or contact `OPENAI_BASE_URL`.
6. The walker shall be the test layer's mock seam: tests shall replace `walker.walk` (or its underlying chat-completions helper) and never make a real HTTP call.

### Requirement 5: Ranker — heavy-ranker-feature-shape weighted combiner

**Objective:** As an MCP client user, I want walker hits ordered by usefulness (engagement, my own affinity to the author, recency) so that the top results are actually the ones I would have wanted, not just the ones the model flagged first.

#### Acceptance Criteria

1. When `ranker.rank(walker_hits, tweets_by_id, author_affinity, weights, now)` is called, the ranker shall produce one `ScoredHit` per `WalkerHit` whose `tweet_id` exists in `tweets_by_id`, in descending `score` order; hits whose `tweet_id` is missing from the tweet map shall be skipped without raising.
2. The ranker score shall be the weighted sum: `walker_relevance * W_RELEVANCE + log1p(favorite_count) * W_FAVORITE + log1p(retweet_count) * W_RETWEET + log1p(reply_count) * W_REPLY + log1p(view_count) * W_VIEW + author_affinity[handle] * W_AFFINITY + recency_decay(created_at, anchor) * W_RECENCY + (1 if verified else 0) * W_VERIFIED + (1 if has_media else 0) * W_MEDIA`. The ranker shall record the per-feature contribution in `ScoredHit.feature_breakdown` for explainability.
3. The ranker shall use defaults `W_RELEVANCE=10.0`, `W_FAVORITE=2.0`, `W_RETWEET=2.5`, `W_REPLY=1.0`, `W_VIEW=0.5`, `W_AFFINITY=3.0`, `W_RECENCY=1.5`, `W_VERIFIED=0.5`, `W_MEDIA=0.3`, and recency half-life of 180 days, all overridable via `RANKER_W_*` and `RANKER_RECENCY_HALFLIFE_DAYS` env vars.
4. The recency decay shall be `exp(-days_apart / halflife_days)` where `days_apart = max(0, (anchor - tweet_created_at).total_days)`. The anchor shall be `now` when no year filter is set, the end of the year when only `year` is set, the end of `month_end` when a range is set, or the end of `month_start`'s month when only `month_start` is set.
5. The ranker shall be a pure function: same inputs produce the same outputs, no I/O, no LLM, no network.
6. The ranker shall not import the OpenAI SDK and shall not call the walker.
7. Author affinity shall be precomputed at index-build time as `{screen_name: log1p(count_of_user_likes_from_this_author)}` over the loaded `Tweet` list. Authors not present in the map shall contribute zero to the affinity term.

### Requirement 6: `search_likes` tool

**Objective:** As an MCP client user, I want to ask a natural-language question about my likes and get back the matching tweets with enough metadata to find them and a score that reflects how worth-my-time each result is.

#### Acceptance Criteria

1. When an MCP client calls `search_likes` with a non-empty `query` string, the X Likes MCP Server shall resolve the optional structured filter to a list of in-scope months, call `walker.walk` over those months, call `ranker.rank` over the walker hits, and return the top-N (bounded; default 50) `ScoredHit` results.
2. The X Likes MCP Server shall accept three optional structured-filter parameters (`year: int | None`, `month_start: str | None`, `month_end: str | None`) and shall pre-filter the months passed to the walker (year alone spans the whole year; year + month_start spans one month; year + month_start + month_end spans the inclusive range). When all three are None, the search covers every month.
3. The X Likes MCP Server shall validate the structured filter at the input layer: month values must match `^(0[1-9]|1[0-2])$`; if `month_end` is set without `month_start`, or `month_start` > `month_end`, or any month parameter is set without `year`, the server shall return an input-validation error naming the offending field.
4. Each `search_likes` result shall include `tweet_id`, `year_month`, `handle`, `snippet`, `score`, `walker_relevance`, `why`, and `feature_breakdown`. The snippet shall be drawn from the tweet's text in the loaded export (truncated to a reasonable length; e.g. 240 characters).
5. When an MCP client calls `search_likes` with an empty or whitespace-only query, the X Likes MCP Server shall return an input-validation error rather than calling the walker.
6. If the walker raises (LLM down, malformed response, etc.), the X Likes MCP Server shall surface a tool error with `category="upstream_failure"` and shall not crash the server process.
7. When the walker returns no hits for a valid query, the X Likes MCP Server shall return an empty result list rather than an error.
8. The X Likes MCP Server shall declare a JSON schema for `search_likes` input and output that an MCP client can introspect.

### Requirement 7: `list_months` tool

**Objective:** As an MCP client user, I want to see which months of likes are available without scanning the directory myself, so that I can scope follow-up calls.

#### Acceptance Criteria

1. When an MCP client calls `list_months`, the X Likes MCP Server shall return the set of `YYYY-MM` values for which `likes_YYYY-MM.md` exists under `output/by_month/`.
2. The X Likes MCP Server shall return the months in reverse chronological order so the most recent month is first.
3. The X Likes MCP Server shall include the file path and the tweet count for each month when that information is available; if tweet counts cannot be derived for a month, the server shall return that month with a null count rather than raising.
4. The X Likes MCP Server shall declare a JSON schema for `list_months` input (empty) and output that an MCP client can introspect.

### Requirement 8: `get_month` tool

**Objective:** As an MCP client user, I want the raw Markdown for a single month so that the calling LLM can read it directly when a follow-up question is scoped to that month.

#### Acceptance Criteria

1. When an MCP client calls `get_month` with a `year_month` value matching `^\d{4}-\d{2}$` and a corresponding file exists, the X Likes MCP Server shall return the full Markdown contents of `output/by_month/likes_{year_month}.md`.
2. If `year_month` does not match the `YYYY-MM` pattern, the X Likes MCP Server shall return an input-validation error naming the expected format.
3. If `year_month` matches the pattern but no corresponding file exists, the X Likes MCP Server shall return a not-found tool error identifying the missing month, rather than an empty string.
4. The X Likes MCP Server shall declare a JSON schema for `get_month` input and output that an MCP client can introspect.

### Requirement 9: `read_tweet` tool

**Objective:** As an MCP client user, I want to fetch one tweet by ID with full metadata so that I can drill into a specific result returned by `search_likes`.

#### Acceptance Criteria

1. When an MCP client calls `read_tweet` with a `tweet_id` that matches a tweet present in the loaded export, the X Likes MCP Server shall return that tweet's text, handle, display name, created-at timestamp, view/like/retweet counts where present, and the canonical URL.
2. The X Likes MCP Server shall source `read_tweet` results from the loaded `likes.json` via the Spec 1 read API rather than re-parsing the per-month Markdown.
3. If `tweet_id` does not match any tweet in the loaded export, the X Likes MCP Server shall return a not-found tool error identifying the missing ID.
4. If `tweet_id` is empty or non-numeric, the X Likes MCP Server shall return an input-validation error.
5. The X Likes MCP Server shall declare a JSON schema for `read_tweet` input and output that an MCP client can introspect.

### Requirement 10: Read-only over existing exports

**Objective:** As a single-user maintainer, I want strict assurance that the MCP server cannot modify the export, hit X, or contact a hosted LLM, so that running the server is risk-free relative to my data and credentials.

#### Acceptance Criteria

1. The X Likes MCP Server shall not write any file under `output/by_month/` or modify `output/likes.json`.
2. The X Likes MCP Server shall not import, instantiate, or call any code path in `x_likes_exporter` that performs a network request against `x.com`, `twitter.com`, or any X subdomain.
3. The X Likes MCP Server shall not require, read, or create a `cookies.json` file at any point.
4. The X Likes MCP Server shall route every LLM call through the configured `OPENAI_BASE_URL`; if `OPENAI_BASE_URL` points at a hosted vendor, that is the user's choice, not the server's default.
5. The X Likes MCP Server shall write only to the configured output directory (the tree cache file) and to standard error or its log destination; it shall not write to arbitrary filesystem locations.

### Requirement 11: Test suite without live LLM

**Objective:** As the maintainer, I want the MCP server to have tests that run on a clean checkout without a real LLM, so that CI and pre-commit can validate behavior without external services.

#### Acceptance Criteria

1. When a developer runs `pytest tests/` from the repository root with no `OPENAI_BASE_URL` set, the X Likes MCP Server test suite shall execute and finish without making real HTTP calls to any LLM endpoint.
2. The X Likes MCP Server test suite shall mock `walker.walk` (or the underlying chat-completions helper) at the unit-test layer; it shall not mock `tree.build_tree` or `ranker.rank` because those are pure functions exercised against fixtures and direct inputs.
3. The X Likes MCP Server test suite shall include an integration test that drives all four tools against a fixture export directory containing a small set of `likes_YYYY-MM.md` files and a matching `likes.json`.
4. If a test attempts a real outbound HTTP request, the X Likes MCP Server test suite shall fail that test loudly rather than silently making the call.
5. The X Likes MCP Server test suite shall not require `cookies.json` or a real scrape.
6. The X Likes MCP Server project shall declare its test-only dependencies (`pytest`, `responses`) under the `dev` dependency group, reusing Spec 1's group rather than introducing a parallel one.

### Requirement 12: Registration with an MCP client documented

**Objective:** As a user, I want clear instructions on how to register this server with Claude Code so that I can use it from my normal client without guessing the config shape.

#### Acceptance Criteria

1. The X Likes MCP Server documentation shall include a README section that shows the exact `.mcp.json` entry or `claude mcp add` invocation a user runs to register the server.
2. The X Likes MCP Server documentation shall list the required `.env` variables (`OPENAI_BASE_URL`, `OPENAI_MODEL`) and the optional ones (`OPENAI_API_KEY`, the `RANKER_W_*` weights, `RANKER_RECENCY_HALFLIFE_DAYS`), and the prerequisite that `scrape.sh` has been run at least once.
3. The X Likes MCP Server documentation shall identify the four tools the server exposes and a one-line summary of each.
4. The X Likes MCP Server documentation shall state that the server is stdio-only and that hosted LLM endpoints are not used by default.

### Requirement 13: Graceful failure modes

**Objective:** As a single-user maintainer, I want failures to be visible and actionable so that a misconfigured `.env`, an empty export, or a flaky LLM does not leave the server in a confusing half-working state.

#### Acceptance Criteria

1. If startup configuration is invalid (missing `OPENAI_BASE_URL`, missing output directory, empty `output/by_month/`, non-numeric `RANKER_W_*` value), the X Likes MCP Server shall exit non-zero before announcing tools, with a single error line that names the failing condition.
2. While the server is running and the LLM endpoint becomes unreachable, the X Likes MCP Server shall convert the failure into a tool-level error response for the affected call and shall keep the server process alive for subsequent calls.
3. While the server is running and a per-month Markdown file is added or removed, the X Likes MCP Server shall not silently serve stale data on the next call; the cache invalidation rule from Requirement 3 shall apply on the next startup, and the running process may continue to serve the previously loaded tree until restart (documented as a restart-to-pick-up-changes behavior).
4. If a tool receives a malformed argument, the X Likes MCP Server shall return an input-validation error that identifies the failing field rather than raising an unhandled exception.
