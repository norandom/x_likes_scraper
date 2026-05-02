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
  stdio server, 149 tests green. The Twitter-shape ranker formula
  (engagement, author affinity, recency, etc.) is sound and stays.
- 7,780 tweets in `likes.json`, 23,775 in the per-month markdown tree.
- Walker is the only LLM call site. Ranker is pure-python and fast.
- `.mcp.json` is registered with Claude Code; the user has confirmed
  the server boots.
- Platform reality: the maintainer's primary machine is Intel macOS
  x86_64. PyTorch ≥ 2.3 and ONNX Runtime ≥ 1.x dropped wheels for that
  triple, which kills any sentence-transformers / fastembed approach.

## Desired Outcome

`search_likes` returns results in under 10 seconds for any query,
regardless of how many months are in scope. The ranker keeps producing
the engagement-and-affinity-biased ordering that makes results useful.
Results don't get polluted by surface-similar-but-off-topic tweets the
way pure cosine over a mixed corpus tends to. Tool-call timeouts stop
happening.

## Approach

Three-layer search, modeled after how production systems (Elasticsearch
hybrid, Vertex AI Search, Pinecone hybrid) do it:

1. **Hybrid recall** (cheap, runs over the whole corpus). Two
   complementary retrievals run in parallel and their rankings are
   fused.
   - **BM25** over tokenized tweets via `rank_bm25` (pure-python, no
     native deps, ~10 ms on 7,780 short docs). Anchors results to actual
     query terms — the lexical signal that pure cosine routinely
     loses on short text.
   - **Dense vectors** via OpenRouter's `/embeddings` endpoint. Default
     model `nvidia/llama-nemotron-embed-vl-1b-v2:free` (free tier on
     OpenRouter, modern 1B-param vision-language encoder; the
     vision-language part also opens up a future image-search path
     without re-architecting). Vectors persisted to `corpus_embeddings.npy`
     so we only embed the corpus once and the per-query path is one
     network round trip plus a numpy dot product.
   - Rankings fused with **Reciprocal Rank Fusion** (RRF, k=60), the
     standard parameter-light combiner. Top-300 ids out.

2. **Heavy-ranker re-ranking** (already implemented). Score the 300
   candidates with the existing `ranker.rank` formula. Top-50 returned
   to the MCP client.

3. **Optional LLM explainer**, off by default. A new tool parameter
   `with_why=True` triggers a single LLM call over the top-20 ranked
   results to generate the `why` field. ~5 seconds when on, skipped
   entirely otherwise.

The walker source stays in place. Existing walker tests stay green. The
walker becomes the implementation of the optional explainer; it's just
no longer in the search hot path.

## Scope

- **In:** New `embeddings.py` module (OpenRouter HTTP client, batched
  corpus embed with backoff, query embed, cosine top-K). New `bm25.py`
  module (rank_bm25-backed lexical index over tweet text, top-K). New
  `fusion.py` (or inline helper) implementing RRF over the two ranked
  lists. Refactor `tools.search_likes` (and `TweetIndex.search`) to
  drive the hybrid pipeline followed by the existing ranker. On-disk
  embedding cache with structural invalidation (model name, tweet-id
  set, schema version). New env vars: `OPENROUTER_API_KEY`,
  `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`),
  `EMBEDDING_MODEL` (default `nvidia/llama-nemotron-embed-vl-1b-v2:free`).
  Optional `with_why` parameter on `search_likes`. README MCP section
  updated.

- **Out:** Replacing the ranker (it's working). Image embeddings (the
  default model can do them, but plumbing them through is a separate
  spec). Cross-encoder re-ranking (overkill at this scale). Removing
  the walker module (preserves the explainer path; deleting the code
  would also break Spec 2's tests for no benefit). Storing vectors in
  Chroma / Qdrant / FAISS (overengineering for 7,780 docs; numpy is
  the right answer at this scale).

## Boundary Candidates

- Embedding source. Default to OpenRouter with the free Nemotron VL 1B
  model. Allow override via `EMBEDDING_MODEL`. The OpenAI Python SDK
  (already a dep for the walker) handles the `/v1/embeddings` shape;
  pointing it at `OPENROUTER_BASE_URL` is the only configuration.
- Rate limit handling. Free-tier OpenRouter typically caps at ~20 RPM.
  The corpus embed batches inputs (default 32 per request, capped at
  the model's max), retries with exponential backoff on 429, and
  serializes requests so the cap isn't tripped. Per-query embedding is
  one request; not rate-limit-sensitive in normal use.
- Cache format. A numpy `.npy` of shape `(N, D)` plus a parallel
  `.json` carrying the tweet-id-to-row index, the model name, and a
  schema version. Easy to invalidate, easy to inspect.
- BM25 corpus shape. `rank_bm25.BM25Okapi` over per-tweet token lists;
  built in-memory at index time, not persisted (rebuild is cheap).
  Tokenization is plain whitespace + lowercase + `\W` strip; matches
  the casual nature of tweet text without dragging in a stemmer.
- Fusion parameter. RRF with k=60 (the published default). Per-method
  ranks come from BM25 score-sorted order and cosine score-sorted order;
  the fused score is the sum of `1/(k+rank)` across methods.

## Out of Boundary

- Anything codebase-foundation owns (the lib, the read API, Tweet
  shape).
- The MCP server transport, JSON schemas, error wrapper. The tool
  signature stays compatible (one new optional `with_why` parameter).
- The ranker formula and weights. They produced the right ordering in
  unit tests; this spec doesn't second-guess them.
- The PageIndex tree builder (`tree.py`). The tree still feeds the
  optional explainer; nothing about its shape changes.

## Upstream / Downstream

- **Upstream:** mcp-pageindex (Spec 2). This spec consumes
  `TweetIndex.tweets_by_id`, the ranker, the MCP server scaffold, and
  the tools surface. The walker module stays available as the explainer.
- **Downstream:** none planned. A future spec could add image-modal
  search reusing the same VL embedding endpoint; that's speculation.

## Existing Spec Touchpoints

- **Extends:** mcp-pageindex. New modules sit next to walker and ranker;
  `tools.search_likes` gets refactored; `config.py` learns the new env
  vars; `.env.sample` documents them.
- **Adjacent:** codebase-foundation. Tweet shape and `load_export`
  stay fixed.

## Constraints

- The maintainer's primary machine is Intel macOS x86_64, which has no
  modern PyTorch or ONNX Runtime wheels. Any approach that requires
  loading a transformer model in-process is off the table. The dense
  retrieval has to be a network call.
- New runtime deps: `rank_bm25` (pure-python, ~50 KB). The OpenAI
  Python SDK is already a dep. No `sentence-transformers`, no `torch`,
  no `onnxruntime`.
- Embedding the corpus once on first start: ~3-12 minutes on the free
  OpenRouter tier (244 batched requests at ~20 RPM); cached afterwards.
  Rebuilds only when the model name changes or the tweet-id set changes.
- End-to-end search latency target: under 10 seconds for any query
  shape. Hybrid path should typically come back in 200-500 ms (BM25
  ~10 ms + query embed ~80-200 ms LAN/WAN + cosine + RRF + ranker).
- `OPENROUTER_API_KEY` is required for the MCP server to start. Missing
  key fails loud with a message naming the env var.
- The walker explainer stays opt-in so the default search remains fast
  and offline-after-corpus-embed.
- Tests must mock the OpenRouter call (no network in CI). The
  `embeddings.py` module exposes a `_call_embeddings_api` seam the test
  layer patches, mirroring the walker's `_call_chat_completions` pattern.
  The `bm25.py` module is deterministic and runs against in-memory
  fixtures with no external dependencies.

## Why Hybrid (Not Pure Vector)

A 7,780-tweet corpus covers everything the maintainer has liked over
years — broad topical mix, short texts, lots of partial-vocabulary
overlap between unrelated subjects. Pure cosine over a corpus that
shape produces a known failure mode: the top-K fills up with tweets
that share generic vocabulary with the query but aren't on the actual
topic. Tweets about AI music, AI art, and LLM-as-API land next to
"pentesting with AI and LLMs" because the query embedding sees them as
neighbours.

BM25 anchors retrieval to the actual query tokens. RRF means we take
both the lexical view and the semantic view and use whichever ranks a
candidate well; no weight tuning needed. The heavy ranker still owns
final ordering — its engagement, recency, and affinity signals dominate
the visible top-50 — but it can only sort what recall hands it. Hybrid
recall is the part that fixes pollution at the source.
