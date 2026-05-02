# Brief: mcp-fast-search

## Problem

`search_likes` from mcp-pageindex doesn't scale. The walker calls the
LLM once per chunk of ~30 tweets, once per in-scope month, sequentially.
For a 4-month query covering ~2,200 tweets that's 60+ LLM calls
back-to-back. Many minutes, far past any MCP client's tool-call timeout.
I tried the actual search ("pentesting with AI and LLMs", Feb-May 2026)
and the call hung; cancelling it left the server still grinding through
chunks in the background.

The walker conflates retrieval (which tweets are even worth looking at)
and re-ranking (which order should those tweets be shown in). Production
search systems split those: cheap retrieval over the whole corpus, then
expensive re-ranking on a small candidate set. The walker tries to do
the expensive thing on every tweet.

## Current State

- mcp-pageindex shipped: TweetIndex, walker, ranker, four MCP tools,
  stdio server, 98 tests green. The Twitter-shape ranker formula
  (engagement, author affinity, recency, etc.) is sound and stays.
- 7,780 tweets in `likes.json`, 23,775 in the per-month markdown tree.
- Walker is the only LLM call site. Ranker is pure-python and fast.
- `.mcp.json` is registered with Claude Code; the user has confirmed
  the server boots.

## Desired Outcome

`search_likes` returns results in under 10 seconds for any query,
regardless of how many months are in scope. The ranker keeps producing
the engagement-and-affinity-biased ordering that makes results useful.
Tool-call timeouts stop happening.

## Approach

Two-stage retrieval, both layers running locally:

1. **Embedding retrieval** (cheap, runs over the whole corpus). Embed
   every tweet once at index-build time using a small local model
   (`BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`,
   both ~30-90 MB, CPU-friendly). Cache embeddings on disk as a numpy
   `.npy` file alongside the existing `tweet_tree_cache.pkl`. At search
   time: embed the query (~50 ms), cosine similarity vs all stored
   tweet vectors (~50 ms with numpy on 7,780 rows), take top-200
   candidates. No LLM call.

2. **Heavy-ranker re-ranking** (already implemented). Score the 200
   candidates with the existing `ranker.rank` formula. Top-50 returned
   to the MCP client.

3. **Optional LLM explainer**, off by default. A new tool parameter
   like `with_why=True` triggers a single LLM call over the top-20
   ranked results to generate the `why` field. ~5 seconds when on,
   skipped entirely otherwise.

The walker source stays in place. Existing tests stay green. The walker
becomes the implementation of the optional explainer; it's just no
longer in the search hot path.

## Scope

- **In:** New `embeddings.py` module (loads the model, embeds tweets,
  cosine search). Refactor `tools.search_likes` to call embedding
  retrieval first, then the existing ranker. On-disk embedding cache
  with rebuild logic when the model name or the tweets-by-id set
  changes. New `EMBEDDING_MODEL` env var with a sensible default.
  Optional `with_why` parameter on `search_likes`. README MCP section
  updated with the new model dependency and the explainer flag.

- **Out:** Replacing the ranker (it's working). Image embeddings.
  Cross-encoder re-ranking (overkill at this scale). Hybrid BM25 +
  vector (interesting, but later). Removing the walker module
  (preserves the explainer path; deleting the code would also break
  Spec 2's tests for no benefit).

## Boundary Candidates

- Embedding model choice. Default to `bge-small-en-v1.5` (best
  retrieval at the size). Allow override via `EMBEDDING_MODEL`. If a
  user changes models, the on-disk cache is invalidated and rebuilt.
- Cache format. A numpy `.npy` of shape `(N, D)` plus a parallel
  `.json` carrying the tweet-id-to-row index and a model-name header.
  Cheap to mmap, easy to invalidate.
- Cache invalidation. Same mtime-based policy the tree cache uses,
  plus an explicit version field so a model change forces a rebuild
  even when the tweets are unchanged.
- Embedding for the walker explainer. The explainer prompt sees only
  the top-20 ranked results, so chunk_size becomes irrelevant. Single
  LLM call per search.

## Out of Boundary

- Anything codebase-foundation owns (the lib, the read API, Tweet
  shape).
- The MCP server transport, JSON schemas, error wrapper. The tool
  signature stays compatible (one new optional parameter at most).
- The ranker formula and weights. They produced the right ordering
  in unit tests; this spec doesn't second-guess them.
- The PageIndex tree builder (`tree.py`). The tree still feeds the
  optional explainer; nothing about its shape changes.

## Upstream / Downstream

- **Upstream:** mcp-pageindex (Spec 2). This spec consumes
  `TweetIndex.tweets_by_id`, the ranker, the MCP server scaffold, and
  the tools surface. The walker module stays available as the
  explainer.
- **Downstream:** none planned. A future spec could add hybrid BM25 +
  vector recall if pure-vector recall misses too much keyword content,
  but that's speculation.

## Existing Spec Touchpoints

- **Extends:** mcp-pageindex. The new module sits next to walker and
  ranker; `tools.search_likes` gets refactored; `config.py` learns
  `EMBEDDING_MODEL`; `.env.sample` documents it.
- **Adjacent:** codebase-foundation. Tweet shape and `load_export`
  stay fixed.

## Constraints

- Local embeddings only. No hosted embedding APIs (privacy + cost +
  latency).
- New runtime dep: `sentence-transformers` plus `torch` (CPU-only is
  fine). Adds ~200 MB to the install graph. Acceptable trade-off given
  what it buys.
- Embedding the corpus once on first start: ~30-60 seconds for 7,780
  tweets on CPU, ~10 seconds on a modest GPU. Cached afterward.
- End-to-end search latency target: under 10 seconds for any query
  shape. Embedding-only path should typically come back in 1-2 seconds.
- The walker explainer stays opt-in so the default search remains fast.
- Tests must mock the embedding model (no model download in CI). The
  `embeddings.py` module exposes a `_encode` seam the test layer
  patches, mirroring the walker's `_call_chat_completions` pattern.
