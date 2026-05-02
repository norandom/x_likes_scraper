# Requirements Document

## Project Description (Input)

`search_likes` from mcp-pageindex doesn't scale. The walker calls a local LLM once per chunk of ~30 tweets, once per in-scope month, sequentially. A 4-month query over ~2,200 tweets is 60+ LLM calls back-to-back; MCP clients time out waiting. This spec replaces the retrieval layer of `search_likes` with a three-layer pipeline: hybrid recall (BM25 + dense vectors via OpenRouter, fused with Reciprocal Rank Fusion) over the entire corpus, then the existing engagement-and-affinity ranker on the survivors, with the walker preserved as an opt-in explainer over the top hits. Hybrid recall is required because pure cosine over a broad mixed corpus of short tweets surfaces topically-similar-but-off-target results; BM25 anchors retrieval to actual query tokens. End-to-end search latency drops from "minutes, often a timeout" to under 10 seconds in any query shape.

## Boundary Context

- **In scope**: A new `embeddings.py` module inside `x_likes_mcp/` that calls OpenRouter's `/v1/embeddings` endpoint via the `openai` SDK, embeds every tweet at index-build time with rate-limit-aware batched requests, persists the corpus embedding matrix to disk alongside the existing tree cache, and offers a cosine top-k retrieval seam. A new `bm25.py` module that builds a `rank_bm25.BM25Okapi` index over tokenized tweet text and exposes a top-k retrieval seam. A new `fusion.py` (or inline helper in `tools.py`) implementing Reciprocal Rank Fusion (RRF) over BM25 and dense rankings. A `_call_embeddings_api` helper inside `embeddings.py` that tests can patch to avoid network calls in CI. The refactor of `tools.search_likes` (and `TweetIndex.search`) to drive the BM25-and-dense-fused pipeline, then the existing `ranker.rank`, then optionally drive the walker as an explainer over the top results. A new optional boolean parameter `with_why` on `search_likes`, default `false`, that gates the explainer call. New env vars `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`), and `EMBEDDING_MODEL` (default `nvidia/llama-nemotron-embed-vl-1b-v2:free`). Disk-format and invalidation rules for the new embedding cache (rebuild on model name change or when the set of tweet ids changes). Updates to `.env.sample` and the README MCP section. New tests under `tests/mcp/` for the embedder, the BM25 module, the RRF helper, and the refactored search path; existing walker, ranker, tree, and config tests stay green.
- **Out of scope**: Replacing the ranker. Removing the walker module (it stays; `with_why=true` keeps it reachable as the explainer). Image embeddings (the default model is vision-language-capable, but plumbing image inputs is a separate spec). Cross-encoder re-ranking. Storing vectors in a vector database (Chroma, Qdrant, FAISS). Any local-transformer-loading approach (the maintainer's primary machine is Intel macOS x86_64, which has no modern PyTorch or ONNX Runtime wheels). Anything `x_likes_exporter` (Spec 1) owns. The MCP server transport, JSON-schema scaffolding, error-wrapper boundary in `server.py` (one new optional field added to the `search_likes` input schema; nothing else changes).
- **Adjacent expectations**: This spec consumes mcp-pageindex (Spec 2) types and modules: `TweetIndex.tweets_by_id`, the `Tweet` shape from Spec 1's read API, `ranker.rank` and `ScoredHit`, the existing tree cache, the `errors` boundary helpers, and the `walker.walk` entry point that becomes the explainer. The walker's `_call_chat_completions` test seam stays; tests must keep mocking it. If Spec 2's `tools.search_likes` signature, the `ScoredHit` shape, or the tree cache path conventions change, this spec re-checks. The runtime gains one small new dep: `rank_bm25` (pure-python, ~50 KB). The OpenAI Python SDK is already a dep for the walker; OpenRouter is reached by pointing it at a different `base_url`. No `sentence-transformers`, no `torch`, no `onnxruntime`.

## Requirements

### Requirement 1: OpenRouter configuration

**Objective:** As a single-user maintainer, I want the embedding endpoint, API key, and model name to come from `.env` with sensible defaults, so that I can swap models without editing source and so that the MCP server fails loudly when a required secret is missing rather than starting in a half-working state.

#### Acceptance Criteria

1. When the X Likes MCP Server starts, the X Likes MCP Server shall read an optional `OPENROUTER_BASE_URL` value from `.env` (and from `os.environ` if `.env` is absent), defaulting to `https://openrouter.ai/api/v1` when unset or empty.
2. When the X Likes MCP Server starts, the X Likes MCP Server shall read an optional `EMBEDDING_MODEL` value from `.env`, defaulting to `nvidia/llama-nemotron-embed-vl-1b-v2:free` when unset or empty.
3. When the X Likes MCP Server starts, the X Likes MCP Server shall read `OPENROUTER_API_KEY` from `.env` (or `os.environ`); if the key is unset or empty the X Likes MCP Server shall fail at index-build time with an error message that names `OPENROUTER_API_KEY` and instructs the user to set it.
4. The X Likes MCP Server shall not require any local transformer model file or weights to be present; the dense retrieval path is entirely network-based.
5. The X Likes MCP Server shall extend `.env.sample` to document `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `EMBEDDING_MODEL`, including the default values, a note that the default model is on OpenRouter's free tier, and a one-line description of the rate-limit-driven first-run cost.

### Requirement 2: Embedding cache built at index time

**Objective:** As a single-user maintainer, I want tweets embedded once via OpenRouter and re-used across restarts, so that startup is fast after the first run and the API only sees tweets it has not seen before.

#### Acceptance Criteria

1. When the X Likes MCP Server starts and no embedding cache file exists, the X Likes MCP Server shall embed every tweet present in `TweetIndex.tweets_by_id` by issuing batched requests to OpenRouter's `/v1/embeddings` endpoint and shall persist the resulting matrix and a metadata header to disk under the configured output directory.
2. The X Likes MCP Server shall use the tweet's primary text content for embedding input and shall keep the embedded order aligned with a tweet-id list so that row N of the matrix corresponds to tweet id position N in the metadata.
3. The X Likes MCP Server shall batch corpus embedding requests so that each HTTP call carries multiple tweets (default 32 inputs per request, capped at the model's stated maximum), shall pause between batches as needed to stay within the free-tier rate limit, and shall retry with exponential backoff (capped at three attempts per batch) on `429 Too Many Requests` and transient `5xx` responses.
4. While the embedding cache file exists and its metadata records (a) the same model name as the currently configured `EMBEDDING_MODEL`, and (b) a tweet-id set equal to the current `set(TweetIndex.tweets_by_id.keys())`, the X Likes MCP Server shall load the cached embeddings without re-embedding any tweet.
5. If the model name in the cache metadata differs from the current `EMBEDDING_MODEL`, the X Likes MCP Server shall rebuild the embedding cache from scratch and overwrite the previous cache files.
6. If the tweet-id set in the cache metadata differs from the current `set(TweetIndex.tweets_by_id.keys())` (added or removed tweets), the X Likes MCP Server shall rebuild the embedding cache from scratch and overwrite the previous cache files.
7. The X Likes MCP Server shall write the embedding cache files alongside the existing tree cache (under the configured output directory) and shall write atomically (temporary file plus rename) so that a crash mid-write does not corrupt either file.
8. The X Likes MCP Server shall fail the index-build step if the embedding cache cannot be written to the configured output directory or if the OpenRouter API ultimately rejects all retries for any batch, rather than continuing without persistence.

### Requirement 3: Embedding cache file format

**Objective:** As a single-user maintainer, I want a file format that is easy to inspect, easy to invalidate, and does not invent its own serialization protocol, so that the cache is debuggable and version changes are explicit.

#### Acceptance Criteria

1. The X Likes MCP Server shall persist the corpus embedding matrix as a numpy `.npy` file at a path under the output directory that uses a stable, documented name (for example `corpus_embeddings.npy`).
2. The X Likes MCP Server shall persist the corpus metadata as a JSON file alongside the `.npy` file (for example `corpus_embeddings.meta.json`) containing at minimum: the model name string, the integer tweet count, the integer embedding dimensionality, the ordered list of tweet ids whose row index matches the matrix, and an integer schema version.
3. If the schema version recorded in the metadata file is not the version this server was built for, the X Likes MCP Server shall rebuild the embedding cache from scratch rather than attempting to migrate the file in place.
4. If either the `.npy` file or the metadata JSON file is missing or unreadable, the X Likes MCP Server shall treat the cache as absent and rebuild from scratch.

### Requirement 4: Dense (cosine) retrieval

**Objective:** As an MCP client user, I want every search query to consider the entire corpus, not just the months the LLM happens to walk, so that relevant tweets aren't missed because they fall outside an arbitrary date window.

#### Acceptance Criteria

1. When `search_likes` is called with a non-empty query, the X Likes MCP Server shall embed the query string by issuing a single request to OpenRouter's `/v1/embeddings` endpoint and shall compute cosine similarity between the query vector and the corpus matrix.
2. The X Likes MCP Server shall return the top-K most similar candidates (K defaulting to 200) as the dense input to the fusion step.
3. While the structured filter (`year`, `month_start`, `month_end`) restricts the search to fewer than K tweets, the X Likes MCP Server shall return every in-scope candidate rather than padding with out-of-scope tweets.
4. The X Likes MCP Server shall apply the structured filter at the candidate stage so that only tweets whose `created_at` resolves to one of the in-scope months can appear in the candidate set; tweets with unparseable `created_at` shall be excluded from filtered queries and included in unfiltered queries.
5. The X Likes MCP Server shall keep cosine similarity computation pure-numpy after the network round trip (no LLM, no further network calls) so the dense retrieval step has predictable per-query latency once the query vector is in hand.

### Requirement 5: BM25 (lexical) retrieval

**Objective:** As an MCP client user, I want my exact query terms to anchor recall, so that "pentesting with AI" returns tweets that actually mention pentesting rather than generic-AI tweets that share embedding-space neighbourhood with the query.

#### Acceptance Criteria

1. When the X Likes MCP Server builds the index, the X Likes MCP Server shall construct an in-memory BM25 index using `rank_bm25.BM25Okapi` over the tokenized text of every tweet in `TweetIndex.tweets_by_id`.
2. The X Likes MCP Server shall tokenize tweet text and the query string with the same deterministic, dependency-free tokenizer (lowercase, split on whitespace, strip leading/trailing non-word characters from each token, drop empty tokens).
3. When `search_likes` is called with a non-empty query, the X Likes MCP Server shall compute BM25 scores against the entire corpus and shall return the top-K candidates (K defaulting to 200) as the lexical input to the fusion step.
4. The X Likes MCP Server shall apply the same structured filter to the BM25 candidate set as it does to the dense candidate set so that only in-scope tweets can appear; tweets with unparseable `created_at` shall be excluded from filtered queries and included in unfiltered queries.
5. The X Likes MCP Server shall not persist the BM25 index to disk; it is rebuilt in-memory at index startup, cheap relative to the embedding cache, and avoids a second invalidation surface.

### Requirement 6: Reciprocal Rank Fusion

**Objective:** As an MCP client user, I want the lexical and semantic signals combined without me having to tune weights, so that the recall layer benefits from both views.

#### Acceptance Criteria

1. When `search_likes` runs, the X Likes MCP Server shall combine the BM25 top-K and the dense top-K into a single ranked list using Reciprocal Rank Fusion: for each candidate `d`, `score(d) = sum_over_methods( 1 / (k_rrf + rank_method(d)) )` with `k_rrf` defaulting to 60.
2. The X Likes MCP Server shall produce a fused candidate set whose size is at most the union of the two input top-K lists and shall pass at most the top-300 fused candidates to the heavy ranker.
3. The X Likes MCP Server shall treat the fused score as a recall signal only; it shall not appear in the response and shall not directly influence the heavy ranker's ordering. The heavy ranker continues to consume the per-candidate dense cosine score as the `relevance` input to its formula.
4. While only one of the two retrievals returns candidates for a query (for example, BM25 returns nothing on a paraphrase, or the dense path is mocked out in tests), the X Likes MCP Server shall still produce the fused set from whichever method returned candidates rather than failing the query.

### Requirement 7: Refactored `search_likes` flow

**Objective:** As an MCP client user, I want `search_likes` to come back in seconds instead of timing out, so that I can use it as a routine tool from any MCP client.

#### Acceptance Criteria

1. When an MCP client calls `search_likes` with a non-empty query, the X Likes MCP Server shall: validate the query and structured filter, resolve the in-scope candidate set, run BM25 top-K and dense top-K (with the dense path issuing one OpenRouter request), fuse the two via RRF, hand the top-300 fused ids to the existing `ranker.rank`, and return the top-N (default 50) `ScoredHit` results.
2. The X Likes MCP Server shall not call the walker on the default `search_likes` path (`with_why=false` or unset), so the default search performs zero LLM calls (the embedding request is not an LLM completion call).
3. When both retrievals return no candidates for a valid query, the X Likes MCP Server shall return an empty result list rather than an error.
4. If dense retrieval fails (OpenRouter unreachable, auth rejected, model returns malformed payload, cache file unreadable mid-run), the X Likes MCP Server shall fall back to BM25-only recall and shall log one stderr line naming the failure; the call still succeeds.
5. If BM25 retrieval fails (extremely unlikely; rank_bm25 is pure-python), the X Likes MCP Server shall fall back to dense-only recall and shall log one stderr line naming the failure; the call still succeeds.
6. If both retrievals fail, the X Likes MCP Server shall return a tool error with `category="upstream_failure"` and shall keep the server process alive for subsequent calls.
7. The X Likes MCP Server shall continue to honor every existing `search_likes` requirement that survives the refactor: filter validation rules, empty-query rejection, JSON-schema declaration, top-N bound, response shape (`tweet_id`, `year_month`, `handle`, `snippet`, `score`, `walker_relevance`, `why`, `feature_breakdown`).
8. The X Likes MCP Server shall populate `walker_relevance` and `why` on every result regardless of `with_why`: when the explainer does not run, the X Likes MCP Server shall set `walker_relevance` from the cosine similarity score normalized to `[0, 1]` (or from the BM25-only fallback path's normalized rank when dense is unavailable) and shall set `why` to a short empty-or-placeholder string so the result shape stays the same as it does today.

### Requirement 8: Optional walker explainer over top hits

**Objective:** As an MCP client user, I want the option to ask "why" the top hits matter, paying the LLM-call cost only when I want it, so that the default search stays fast and the explainer stays available.

#### Acceptance Criteria

1. The X Likes MCP Server shall accept an optional boolean `with_why` parameter on `search_likes`, defaulting to `false` when absent.
2. When `with_why=true`, the X Likes MCP Server shall invoke the walker over at most the top-20 ranked results (after the hybrid-recall + ranker pipeline) using the existing walker module and shall populate the `why` and `walker_relevance` fields on each affected result from the walker's response.
3. The X Likes MCP Server shall preserve the order produced by the ranker even when `with_why=true`; the walker's output shall be used to populate `why` and refresh `walker_relevance` only, not to reorder results.
4. If the walker fails while `with_why=true`, the X Likes MCP Server shall return the hybrid+ranker results without `why` annotations and shall surface a single clear stderr log line describing the failure, rather than failing the entire `search_likes` call.
5. When `with_why=false` or unset, the X Likes MCP Server shall not invoke the walker module's `walk` function, so the default request makes no chat-completions HTTP call (it does still make one embeddings HTTP call for the query).

### Requirement 9: End-to-end latency

**Objective:** As an MCP client user, I want predictable, sub-timeout response times so that `search_likes` is usable as a routine tool from Claude Code or any MCP client.

#### Acceptance Criteria

1. When `search_likes` is called with `with_why=false` against a corpus of approximately 8,000 tweets and a warm embedding cache, the X Likes MCP Server shall return results within 10 seconds end-to-end on a CPU-only laptop-class machine; in the typical case with a healthy network round trip to OpenRouter the call shall return in well under 2 seconds.
2. While the embedding cache is being built for the first time on approximately 8,000 tweets via the OpenRouter free tier, the X Likes MCP Server shall complete the build within roughly 12 minutes (244 batched requests at ~20 RPM); this cost is paid once at first startup and is documented in the README rather than enforced by automated tests.
3. When `search_likes` is called with `with_why=true`, the X Likes MCP Server shall complete within roughly 10 seconds for the hybrid+ranker stage plus one walker call over the top 20 results; the absolute upper bound depends on the configured local LLM endpoint and is not enforced in CI.

### Requirement 10: Tests must mock the OpenRouter API

**Objective:** As the maintainer, I want the test suite to remain CI-friendly so that running `pytest tests/` on a clean checkout completes without hitting the network or requiring an API key.

#### Acceptance Criteria

1. When a developer runs `pytest tests/` from the repository root with no `OPENROUTER_API_KEY` set and no network access, the X Likes MCP Server test suite shall execute and finish without making any HTTP request.
2. The X Likes MCP Server `embeddings` module shall expose a documented test seam (the `_call_embeddings_api` function) that tests patch to return canned vectors for given input strings, mirroring the walker's `_call_chat_completions` mock pattern.
3. The X Likes MCP Server test suite shall include unit tests that exercise: cosine top-k math (with and without id-set restriction), cache-write-and-reload round-trip, model-name change triggers a rebuild, tweet-id set change triggers a rebuild, schema-version mismatch triggers a rebuild, batched embedding with rate-limit retry behavior on a fake transport.
4. The X Likes MCP Server test suite shall include unit tests for the BM25 module (deterministic top-K against a hand-built corpus, structured-filter masking, fallback when query tokenizes to nothing) and for the RRF fusion helper (canonical fused order across two crafted rankings, single-method input, empty-input handling).
5. The X Likes MCP Server test suite shall include integration-level tests that drive `tools.search_likes` end-to-end with the embeddings seam mocked and the walker mocked, asserting the new flow (hybrid recall → ranker → optional walker explainer) without needing real network or model access.

### Requirement 11: Walker module remains in place

**Objective:** As the maintainer, I want the walker code, walker tests, and walker semantics preserved so that this spec extends mcp-pageindex rather than rewriting it.

#### Acceptance Criteria

1. The X Likes MCP Server shall keep the existing `walker.py` module callable as-is (`walker.walk(tree, query, months_in_scope, config)` continues to work).
2. The X Likes MCP Server shall keep every existing `tests/mcp/test_walker.py` test green; this spec shall not break the walker's contract.
3. The X Likes MCP Server shall keep `walker._call_chat_completions` as the LLM mock seam used by the walker tests and any test that exercises the explainer path.
4. The X Likes MCP Server shall route the optional explainer call through `walker.walk` (or a thin wrapper inside `tools.py` that calls into it) so the chat-completions LLM call site count stays at one.

### Requirement 12: Configuration and documentation alignment

**Objective:** As a single-user maintainer, I want the new env vars, the new optional tool parameter, and the new install footprint documented in the same places everything else is, so that the change is discoverable.

#### Acceptance Criteria

1. The X Likes MCP Server shall extend `.env.sample` with commented `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `EMBEDDING_MODEL` entries showing the default values and a one-line description per entry.
2. The X Likes MCP Server documentation shall update the README MCP section to describe the new default search behavior (hybrid BM25 + dense via OpenRouter, fused with RRF, then the existing ranker), the optional `with_why` flag, and the install graph (one new pure-python dep, `rank_bm25`).
3. The X Likes MCP Server documentation shall include the on-disk paths of the new cache files (`corpus_embeddings.npy`, `corpus_embeddings.meta.json`) and a note that they live alongside the existing `tweet_tree_cache.pkl` under the configured output directory.
4. The X Likes MCP Server documentation shall state that the walker is now opt-in via `with_why=true` and remains the only chat-completions LLM call site.
5. The X Likes MCP Server documentation shall note that the maintainer's primary platform (Intel macOS x86_64) lacks modern PyTorch/ONNX wheels and that this is the explicit reason the dense path is hosted rather than local.
