# Brief: mcp-pageindex

## Problem

Years of liked tweets sit in per-month Markdown files. Today the only way to find anything is grep or scrolling. The goal is to be able to ask "what was that thread about kernel scheduling I liked last spring?" and get an answer with tweet IDs and dates.

A vector store would work, but it's a lot of moving parts for one user reading their own archive. I tried PageIndex first — its README sells a reasoning-based tree walker, which sounds right for data that's already heading-shaped (`## YYYY-MM` then `### @handle`). The PyPI package (`pageindex` 0.2.8) turned out to be a thin client for the hosted service at `api.pageindex.ai`, not the OSS reasoning framework. Sending my likes to a hosted service is out — single-user local tool, no third parties.

So I'm dropping PageIndex the dependency and keeping the spec name. The shape is the same: parse the per-month markdown into a tree, walk the tree with the local LLM, return tweet matches. The implementation is mine, in three small modules.

I also tried the obvious alternative — embed everything, MMR-rank with BGE — and the ranking was useless. Semantically-near tweets aren't usefully ordered by cosine similarity when the corpus is one person's likes. So I'm not doing that either.

What I am doing for ranking is borrowing the **feature shape** from `twitter/the-algorithm`'s heavy ranker. Not the whole thing — most of that code is Twitter-scale infrastructure (real-graph, SimClusters, TwHIN) we don't have and don't need. Just the design idea: combine engagement counts, author affinity, recency decay, and a couple of small boosts into one weighted score. Every feature it needs is already on the Tweet objects from Spec 1's loader.

## Current State

- The export already produces `output/by_month/likes_*.md` with `## YYYY-MM` and `### @handle` headings. Per-file h1 has been dropped (Spec 2 task 1.1 already shipped).
- No MCP server in the project today.
- codebase-foundation (Spec 1) provides a stable read API: `load_export`, `iter_monthly_markdown`, plus the `Tweet` dataclass with the engagement fields we need.

## Desired Outcome

- A stdio MCP server that registers with Claude Code (or any MCP client) and exposes the like history as searchable content.
- Tools:
  - `search_likes(query, year=None, month_start=None, month_end=None)` returns matching tweets with month, handle, ID, snippet, score, and a "why" line. The optional structured filter pre-filters which months the walker looks at, so a 3-month query is roughly an order of magnitude faster than an open-ended one and doesn't depend on the LLM honoring a date in prose.
  - `list_months()` returns available months.
  - `get_month(year_month)` returns the raw Markdown for one month.
  - `read_tweet(tweet_id)` returns one tweet's full text and metadata.
- The server uses the OpenAI Python SDK pointed at the user's local OpenAI-compatible LLM endpoint configured in `.env`. No calls to hosted services by default.

## Approach

A new top-level package `x_likes_mcp/` with these modules:

- `tree.py` — pure-Python markdown parser. Walks `output/by_month/likes_YYYY-MM.md` and produces `TreeNode` per tweet (year_month, tweet_id, handle, text, raw_section). No LLM, no network.
- `walker.py` — the LLM-driven semantic walk. For each in-scope month, batch the tweets into chunks (e.g. 30 per chunk) and ask the LLM "user asked X, return JSON `[{id, relevance: 0..1, why}]` for plausibly-related tweets, skip the rest." Single LLM call site for the whole spec.
- `ranker.py` — the heavy-ranker-feature-shape combiner. Takes walker hits + the in-memory Tweet objects, computes a weighted score from `walker_relevance`, log-scaled engagement counts, author affinity, recency decay, and verified/media boosts. Returns scored hits sorted descending.
- `config.py`, `errors.py` — already implemented from this spec's earlier tasks. They stay.
- `index.py` — `TweetIndex` orchestrator. Owns the cached tree, the in-memory Tweet map, the precomputed author affinity. `search` runs resolve→walk→rank.
- `tools.py`, `server.py`, `__main__.py` — MCP scaffolding, four tool handlers.

The OpenAI SDK reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the process environment when its client is constructed. `config.load_config` writes them into `os.environ` before the walker runs, so the SDK picks them up directly. `OPENAI_MODEL` is passed to the chat-completions call. No bridge layer.

Cache: pickle the `TweetTree` next to the export. Invalidate by mtime — if any `.md` is newer than the cache, rebuild on next startup. That's the whole policy.

Author affinity is precomputed once at index-build time: `log1p(count_of_likes_from_this_handle)`. People I like a lot rank higher; people whose tweet I liked once don't get an unfair boost.

Ranker weights are tunable from `.env` as `RANKER_W_*`. Sensible defaults shipped.

## Scope

- **In:** MCP server, the three new modules (tree/walker/ranker), the four tools, `.env` integration for the LLM endpoint, ranker-weight env vars, a tree cache file alongside the export, README section on registering the server with Claude Code.
- **Out:** Re-fetching from X. Vector embeddings, MMR, BGE-style similarity ranking. A web UI. Authentication. Tools that mutate the export. Hosted LLM services by default. Pre-computing the index in a separate process. Real-graph features, SimClusters, TwHIN, anything that needs Twitter-scale infrastructure.

## Boundary Candidates

- The MCP server transport. Stdio for now; HTTP/SSE could come later but is not in this spec.
- The walker module. It is the only LLM call site. Swappable behind a function signature if a different reasoning approach is wanted later.
- Ranker weights. Defaults in code; overrides in `.env`. If a future model wants to tune them per-query, the function signature already takes a weights object.
- Tree cache invalidation. "Cache file mtime older than newest `.md` file" is enough; no manifest needed.

## Out of Boundary

- Anything in Spec 1's domain (the lib, tests, read API).
- Pre-computing the index outside the server process.
- Multi-user concerns.
- A separate retrieval backend other than walker+ranker. We tried PageIndex (hosted, out) and embeddings+MMR (didn't rank well). We're not doing a third one in this spec.

## Upstream / Downstream

- **Upstream:** codebase-foundation (Spec 1). The read API for `output/by_month/`, the `load_export` function, the `Tweet` dataclass with engagement fields are the integration points.
- **Downstream:** none planned. A web UI could come later; not on this roadmap.

## Existing Spec Touchpoints

- **Extends:** none.
- **Adjacent:** codebase-foundation. The read API and the `Tweet` shape are the touchpoints.

## Constraints

- Local OpenAI-compatible LLM endpoint. Plug `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` into `.env`. The user's existing local proxy at `http://10.0.0.59:8317/v1` already serves `/v1/chat/completions`; verified.
- Walker quality is bound by the model in front of it. The spec doesn't promise quality past "the model can return JSON for the prompt template."
- New runtime deps: `mcp` (Python SDK) and `openai` (the SDK itself, since we call it directly now).
- Tests for this spec mock the walker (the LLM call). The tree parser and ranker are pure functions, no mocks needed. End-to-end "real model" verification is documented as a manual step, not gated in CI.
- The per-month Markdown's per-file h1 boilerplate was already dropped in this spec's task 1.1. With it gone, `## YYYY-MM` is the effective top of each per-month file.

## Why Not PageIndex

The PyPI package at `pageindex` 0.2.8 is a hosted-service client. Its `pageindex.PageIndexClient` posts documents to `api.pageindex.ai` and reads back tree structure. The README's "reasoning-based" framing is about the hosted service's behavior, not local code. Nothing in the package builds a tree locally. So pulling it in would mean shipping users' tweets to a third-party endpoint, which contradicts the local-only constraint above.

The OSS PageIndex framework people sometimes mean exists separately and is not on PyPI under that name. Building against it is a much bigger commitment than this spec's scope. Writing our own three-module replacement is smaller than integrating that, and gives us the ranker we want anyway.

## Why Not MMR + BGE Embeddings

I tried this on a smaller corpus first. Cosine similarity over BGE embeddings finds semantically-near tweets, but on one person's likes the ranking is flat — every other liked tweet about kernel internals scores within a few percent of every other one. MMR helps with diversity but doesn't fix the underlying issue: similarity alone, on an already-curated set, does not produce useful ordering. The Twitter heavy-ranker design fixes this by mixing in engagement, recency, and author signals. None of those need an embedding model.
