# Requirements Document

## Project Description (Input)

I keep years of liked tweets as per-month Markdown files under `output/by_month/`. Today the only way to find anything is grep or scrolling. I want to ask "what was that thread about kernel scheduling I liked last spring?" and get an answer with tweet IDs and dates. PageIndex's reasoning-based tree fits the shape of that data: the headings are already `## YYYY-MM` then `### @handle`, no embeddings or chunking needed. This spec adds a stdio MCP server that registers with Claude Code (or any MCP client) and exposes the like history through four tools backed by PageIndex over the existing exports. The reasoning step calls a local Anthropic-compatible LLM endpoint configured in `.env`. The server consumes the read API defined in Spec 1 (`codebase-foundation`) and never re-fetches from X.

## Boundary Context

- **In scope**: A stdio MCP server packaged as `x_likes_mcp` with a `python -m x_likes_mcp` entry point and a `pyproject.toml` script entry. Four MCP tools: `search_likes(query, year=None, month_start=None, month_end=None)`, `list_months()`, `get_month(year_month)`, `read_tweet(tweet_id)`. PageIndex setup over `output/by_month/`. A tree cache file alongside the export with mtime-based invalidation. Three new `.env` variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_OPUS_MODEL`) plus updates to `.env.sample`. A small change to `MarkdownFormatter.export` so per-month output skips the file-level h1 boilerplate (PageIndex sees `## YYYY-MM` as the effective top of each tree). README section on registering the server with Claude Code. Tests that mock the LLM call and the PageIndex tree builder.
- **Out of scope**: Re-fetching from X under any code path. HTTP/SSE transport (stdio only for now). A web UI. Authentication or multi-user concerns. Tools that mutate the export. Calls to hosted LLM services (hosted Anthropic, hosted OpenAI, etc.) by default. Pre-computing the index in a separate process. Anything Spec 1 owns beyond the one targeted change to `MarkdownFormatter.export` (the rest of the `x_likes_exporter` package internals, scraper data models, and tests for the lib itself stay out).
- **Adjacent expectations**: Spec 1 (`codebase-foundation`) owns the read API. This spec consumes `load_export(path)` and `iter_monthly_markdown(path)` from `x_likes_exporter` and depends on the `Tweet` data model exposed there. If those signatures or the directory layout under `output/by_month/` change, this spec re-checks. The MCP server requires the user to have run `scrape.sh` at least once so `output/by_month/` and `output/likes.json` exist; an empty or missing export is a documented startup error, not a feature.

## Requirements

### Requirement 1: Stdio MCP server runnable from a fresh checkout

**Objective:** As a single-user maintainer, I want to start the MCP server from the project root with one command after `uv sync`, so that any MCP client connected over stdio can talk to my like history without extra wiring.

#### Acceptance Criteria

1. When a user runs `python -m x_likes_mcp` from the repository root after `uv sync`, the X Likes MCP Server shall start a stdio MCP server that announces its tool surface and stays running until the client disconnects.
2. When a user runs the console script declared in `pyproject.toml` (for example `uv run x-likes-mcp`), the X Likes MCP Server shall start the same stdio server as `python -m x_likes_mcp`.
3. The X Likes MCP Server shall declare itself on startup with a stable server name and version that an MCP client can read.
4. If `uv sync` is run on a fresh checkout, the X Likes MCP Server shall install its runtime dependencies (the MCP Python SDK and PageIndex) without requiring any change to the existing `[project.dependencies]` set used by `scrape.sh`.
5. The X Likes MCP Server shall not require `cookies.json`, network access to X, or a real scrape at startup; it consumes only the existing files under `output/`.

### Requirement 2: Configuration through `.env`

**Objective:** As a single-user maintainer, I want the local LLM endpoint, the export directory, and the cache location to come from `.env`, so that I can point the server at my own infrastructure without editing source.

#### Acceptance Criteria

1. When the X Likes MCP Server starts, the X Likes MCP Server shall read `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_DEFAULT_OPUS_MODEL` from the project's `.env` file (and from the process environment if `.env` is absent).
2. When the X Likes MCP Server starts, the X Likes MCP Server shall read `OUTPUT_DIR` from `.env` (defaulting to `output`) and locate `output/by_month/` and `output/likes.json` underneath it.
3. If `ANTHROPIC_BASE_URL` or `ANTHROPIC_DEFAULT_OPUS_MODEL` is unset or empty at startup, the X Likes MCP Server shall fail loudly with an error message that names the missing variable.
4. The X Likes MCP Server shall treat `ANTHROPIC_BASE_URL` as an Anthropic-compatible base URL (for example `http://10.0.0.59:8317`) and shall not call any hosted LLM service by default.
5. The X Likes MCP Server shall pass `model="anthropic/{ANTHROPIC_DEFAULT_OPUS_MODEL}"` to PageIndex (which routes through LiteLLM internally), and shall ensure `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` are set in the process environment before the LiteLLM call so LiteLLM's Anthropic provider picks them up.
6. The X Likes MCP Server shall extend `.env.sample` to document `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_DEFAULT_OPUS_MODEL`, including a note that the endpoint is Anthropic-compatible and local by default.

### Requirement 3: PageIndex tree built from the per-month Markdown

**Objective:** As a single-user maintainer, I want the server to build a PageIndex tree from the existing per-month Markdown on first run and reuse it across restarts, so that startup is fast after the first time.

#### Acceptance Criteria

1. When the X Likes MCP Server starts and no cache file exists, the X Likes MCP Server shall enumerate `output/by_month/likes_YYYY-MM.md` via the read API from Spec 1, build a PageIndex tree from those files, and persist the tree to a cache file under the output directory.
2. When the X Likes MCP Server starts and a cache file exists whose modification time is newer than every Markdown file under `output/by_month/`, the X Likes MCP Server shall load the cached tree without rebuilding.
3. If any Markdown file under `output/by_month/` has a modification time newer than the cache file, the X Likes MCP Server shall rebuild the tree and overwrite the cache.
4. If `output/by_month/` is missing or contains no `likes_YYYY-MM.md` files, the X Likes MCP Server shall fail loudly with an error message identifying the missing directory or empty content, rather than starting in a half-working state.
5. The X Likes MCP Server shall write the cache file alongside the export (under the configured output directory) and shall not write inside the package directory or the user's home.

### Requirement 4: `search_likes` tool

**Objective:** As an MCP client user, I want to ask a natural-language question about my likes and get back the matching tweets with enough metadata to find them, so that I can answer questions like "what was that thread about kernel scheduling I liked last spring?"

#### Acceptance Criteria

1. When an MCP client calls `search_likes` with a non-empty `query` string, the X Likes MCP Server shall hand the query to PageIndex with the cached tree and return a list of matching tweets.
2. The X Likes MCP Server shall accept three optional structured-filter parameters (`year: int | None`, `month_start: str | None`, `month_end: str | None`) and shall pre-filter the markdown set passed to PageIndex to only the months matching the filter (year alone spans the whole year; year + month_start spans one month; year + month_start + month_end spans the inclusive range). When all three are None, the search covers every month.
3. The X Likes MCP Server shall validate the structured filter at the input layer: month values must match `^\d{2}$` and fall in `01..12`; if `month_end` is set without `month_start`, or `month_start` > `month_end`, or any month parameter is set without `year`, the server shall return an input-validation error naming the offending field.
4. The X Likes MCP Server shall include in each `search_likes` result the tweet ID, the month (`YYYY-MM`), the handle, and a snippet of the tweet text.
5. When an MCP client calls `search_likes` with an empty or whitespace-only query, the X Likes MCP Server shall return an input-validation error rather than calling the LLM.
6. If the underlying LLM call fails (network error, non-2xx response, malformed body), the X Likes MCP Server shall surface a tool error with a message identifying the failure category, and shall not crash the server process.
7. When PageIndex returns no matches for a valid query, the X Likes MCP Server shall return an empty result list rather than an error.
8. The X Likes MCP Server shall declare a JSON schema for `search_likes` input and output that an MCP client can introspect.

### Requirement 5: `list_months` tool

**Objective:** As an MCP client user, I want to see which months of likes are available without scanning the directory myself, so that I can scope follow-up calls.

#### Acceptance Criteria

1. When an MCP client calls `list_months`, the X Likes MCP Server shall return the set of `YYYY-MM` values for which `likes_YYYY-MM.md` exists under `output/by_month/`.
2. The X Likes MCP Server shall return the months in reverse chronological order so the most recent month is first.
3. The X Likes MCP Server shall include the file path and the tweet count for each month when that information is available from the read API; if tweet counts are unavailable, the X Likes MCP Server shall return month and path only without raising.
4. The X Likes MCP Server shall declare a JSON schema for `list_months` input (empty) and output that an MCP client can introspect.

### Requirement 6: `get_month` tool

**Objective:** As an MCP client user, I want the raw Markdown for a single month so that the calling LLM can read it directly when a follow-up question is scoped to that month.

#### Acceptance Criteria

1. When an MCP client calls `get_month` with a `year_month` value matching `^\d{4}-\d{2}$` and a corresponding file exists, the X Likes MCP Server shall return the full Markdown contents of `output/by_month/likes_{year_month}.md`.
2. If `year_month` does not match the `YYYY-MM` pattern, the X Likes MCP Server shall return an input-validation error naming the expected format.
3. If `year_month` matches the pattern but no corresponding file exists, the X Likes MCP Server shall return a not-found tool error identifying the missing month, rather than an empty string.
4. The X Likes MCP Server shall declare a JSON schema for `get_month` input and output that an MCP client can introspect.

### Requirement 7: `read_tweet` tool

**Objective:** As an MCP client user, I want to fetch one tweet by ID with full metadata so that I can drill into a specific result returned by `search_likes`.

#### Acceptance Criteria

1. When an MCP client calls `read_tweet` with a `tweet_id` that matches a tweet present in the loaded export, the X Likes MCP Server shall return that tweet's text, handle, display name, created-at timestamp, view/like/retweet counts where present, and the canonical URL.
2. The X Likes MCP Server shall source `read_tweet` results from the loaded `likes.json` via the Spec 1 read API rather than re-parsing the per-month Markdown.
3. If `tweet_id` does not match any tweet in the loaded export, the X Likes MCP Server shall return a not-found tool error identifying the missing ID.
4. If `tweet_id` is empty or non-numeric, the X Likes MCP Server shall return an input-validation error.
5. The X Likes MCP Server shall declare a JSON schema for `read_tweet` input and output that an MCP client can introspect.

### Requirement 8: Read-only over existing exports

**Objective:** As a single-user maintainer, I want strict assurance that the MCP server cannot modify the export, hit X, or contact a hosted LLM, so that running the server is risk-free relative to my data and credentials.

#### Acceptance Criteria

1. The X Likes MCP Server shall not write any file under `output/by_month/` or modify `output/likes.json`.
2. The X Likes MCP Server shall not import, instantiate, or call any code path in `x_likes_exporter` that performs a network request against `x.com`, `twitter.com`, or any X subdomain.
3. The X Likes MCP Server shall not require, read, or create a `cookies.json` file at any point.
4. The X Likes MCP Server shall route every LLM call through the configured `ANTHROPIC_BASE_URL`; if `ANTHROPIC_BASE_URL` points at a hosted vendor, that is the user's choice, not the server's default.
5. The X Likes MCP Server shall write only to the configured output directory (the tree cache file) and to standard error or its log destination; it shall not write to arbitrary filesystem locations.

### Requirement 9: Test suite without live LLM or live PageIndex

**Objective:** As the maintainer, I want the MCP server to have tests that run on a clean checkout without a real LLM and without recomputing PageIndex trees, so that CI and pre-commit can validate behavior without external services.

#### Acceptance Criteria

1. When a developer runs `pytest tests/` from the repository root with no `ANTHROPIC_BASE_URL` set, the X Likes MCP Server test suite shall execute and finish without making real HTTP calls to any LLM endpoint.
2. The X Likes MCP Server test suite shall mock the LLM call (the function or client method that hits `ANTHROPIC_BASE_URL`) and shall mock the PageIndex tree builder for the unit-test layer.
3. The X Likes MCP Server test suite shall include an integration test that drives all four tools against a fixture export directory containing a small set of `likes_YYYY-MM.md` files and a matching `likes.json`.
4. If a test attempts a real outbound HTTP request, the X Likes MCP Server test suite shall fail that test loudly rather than silently making the call.
5. The X Likes MCP Server test suite shall not require `cookies.json` or a real scrape.
6. The X Likes MCP Server project shall declare its test-only dependencies (the MCP SDK testing helpers if any, plus `pytest`) under the `dev` dependency group, reusing Spec 1's group rather than introducing a parallel one.

### Requirement 10: Registration with an MCP client documented

**Objective:** As a user, I want clear instructions on how to register this server with Claude Code so that I can use it from my normal client without guessing the config shape.

#### Acceptance Criteria

1. The X Likes MCP Server documentation shall include a README section that shows the exact `.mcp.json` entry or `claude mcp add` invocation a user runs to register the server.
2. The X Likes MCP Server documentation shall list the required `.env` variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`) and the prerequisite that `scrape.sh` has been run at least once.
3. The X Likes MCP Server documentation shall identify the four tools the server exposes and a one-line summary of each.
4. The X Likes MCP Server documentation shall state that the server is stdio-only and that hosted LLM endpoints are not used by default.

### Requirement 11: Graceful failure modes

**Objective:** As a single-user maintainer, I want failures to be visible and actionable so that a misconfigured `.env`, an empty export, or a flaky LLM does not leave the server in a confusing half-working state.

#### Acceptance Criteria

1. If startup configuration is invalid (missing `ANTHROPIC_BASE_URL`, missing output directory, empty `output/by_month/`), the X Likes MCP Server shall exit non-zero before announcing tools, with a single error line that names the failing condition.
2. While the server is running and the LLM endpoint becomes unreachable, the X Likes MCP Server shall convert the failure into a tool-level error response for the affected call and shall keep the server process alive for subsequent calls.
3. While the server is running and a per-month Markdown file is added or removed, the X Likes MCP Server shall not silently serve stale data on the next call; the cache invalidation rule from Requirement 3 shall apply on the next startup, and the running process may continue to serve the previously loaded tree until restart (documented as a restart-to-pick-up-changes behavior).
4. If a tool receives a malformed argument, the X Likes MCP Server shall return an input-validation error that identifies the failing field rather than raising an unhandled exception.
