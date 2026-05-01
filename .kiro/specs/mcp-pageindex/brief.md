# Brief: mcp-pageindex

## Problem

Years of liked tweets sit in per-month Markdown files. Today the only way to find anything is grep or scrolling. The goal is to be able to ask "what was that thread about kernel scheduling I liked last spring?" and get an answer with tweet IDs and dates.

A vector store would work, but it's a lot of moving parts for one user reading their own archive. PageIndex's reasoning-based tree is a better fit: it indexes the heading structure (which is already tweet-shaped, `## YYYY-MM` then `### @handle`), and an LLM walks the tree to find what was asked. No embeddings, no chunking, no similarity scoring.

## Current State

- The export already produces `output/by_month/likes_*.md` with hierarchical headings PageIndex can read out of the box.
- No MCP server in the project today.
- codebase-foundation (Spec 1) provides a stable read API for iterating these files. This spec depends on that.

## Desired Outcome

- A stdio MCP server that registers with Claude Code (or any MCP client) and exposes the like history as searchable content.
- Tools:
  - `search_likes(query, year=None, month_start=None, month_end=None)` returns matching tweets with month, handle, ID, and a snippet. The optional structured filter pre-filters which markdown files PageIndex sees, so a 3-month query is roughly an order of magnitude faster than a 26k-tweet open-ended one and doesn't depend on the LLM honoring a date in prose.
  - `list_months()` returns available months.
  - `get_month(year_month)` returns the raw Markdown for one month.
  - `read_tweet(tweet_id)` returns one tweet's full text and metadata.
- The server uses PageIndex pointed at the user's local Anthropic-compatible LLM endpoint configured in `.env`. No calls to hosted services by default.

## Approach

- A new top-level entry point: `python -m x_likes_mcp` (or a script wired into `pyproject.toml` as a console entry point).
- Use the official Python MCP SDK (`mcp` package) for stdio server scaffolding.
- On startup, the server scans `output/by_month/`, hands the files to PageIndex, and caches the resulting tree on disk so subsequent starts are instant. Cache is invalidated by mtime: if any `.md` file is newer than the cache, rebuild.
- PageIndex's reasoning step routes through LiteLLM, which is what PageIndex uses internally. Three new `.env` variables drive the LLM call: `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_OPUS_MODEL`. PageIndex is invoked with `model="anthropic/{ANTHROPIC_DEFAULT_OPUS_MODEL}"` so LiteLLM picks up the base URL and key from env.
- The four MCP tools are thin wrappers over PageIndex queries and the read API from Spec 1.
- Before the index can be built sensibly, the per-month markdown's per-file `# X (Twitter) Liked Tweets` boilerplate (h1 plus export-timestamp metadata) needs to come out of `MarkdownFormatter.export` for `split_by_month=True`. With it gone, the `## YYYY-MM` heading is the effective top of each file's tree and PageIndex doesn't see 131 copies of the same h1.

## Scope

- **In:** MCP server, PageIndex setup, the four tools above, `.env` integration for the LLM endpoint, a tree cache file alongside the export, README section on registering the server with Claude Code.
- **Out:** Re-fetching from X. Building PageIndex trees in a separate process. A web UI. Authentication (single-user local tool). Tools that mutate the export.

## Boundary Candidates

- The MCP server transport. Stdio for now; HTTP/SSE could come later but is not in this spec.
- The PageIndex wrapper. Encapsulating PageIndex behind a thin `Index` class makes it swappable if a different retrieval backend is ever wanted.
- Tree cache invalidation. "Cache file mtime older than newest `.md` file" should be enough; no manifest needed.

## Out of Boundary

- Anything in Spec 1's domain (the lib, tests, read API).
- Pre-computing the index outside the server process.
- Multi-user concerns.

## Upstream / Downstream

- **Upstream:** codebase-foundation (Spec 1). The read API for `output/by_month/` is the integration point.
- **Downstream:** none planned. A web UI could come later, but that's not on this roadmap.

## Existing Spec Touchpoints

- **Extends:** none.
- **Adjacent:** codebase-foundation. The read API is the touchpoint.

## Constraints

- Local Anthropic-compatible LLM endpoint. The user plugs in `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_DEFAULT_OPUS_MODEL` via `.env`. PageIndex routes through LiteLLM, which honors those env vars when the model string is `anthropic/<model>`.
- PageIndex's reasoning quality is bound by whatever model is in front of it. The spec doesn't promise quality past "the model can follow PageIndex's prompts."
- No new dependencies past `mcp` (Python SDK) and `pageindex` (which transitively brings in `litellm`).
- Tests for this spec mock the LLM call and the PageIndex tree builder where possible. End-to-end "real model" verification is documented as a manual step, not gated in CI.
- One small change to Spec 1's `MarkdownFormatter.export` (drop the per-file h1 boilerplate when `split_by_month=True`) is in scope for this spec because the indexing depends on it. It does not require re-opening codebase-foundation; the spec ships its own commit.
