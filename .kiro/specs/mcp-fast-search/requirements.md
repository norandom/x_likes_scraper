# Requirements Document

## Project Description (Input)

`search_likes` from mcp-pageindex doesn't scale. The walker calls a local LLM once per chunk of ~30 tweets, once per in-scope month, sequentially. A 4-month query over ~2,200 tweets is 60+ LLM calls back-to-back; MCP clients time out waiting. This spec replaces the retrieval layer of `search_likes` with a two-stage local pipeline: cheap embedding-based recall over the entire corpus, then the existing engagement-and-affinity ranker on the survivors. The walker stays in the package and becomes an opt-in explainer over the top hits, off by default. End-to-end search latency drops from "minutes, often a timeout" to under 10 seconds in any query shape.

## Boundary Context

- **In scope**: A new `embeddings.py` module inside `x_likes_mcp/` that loads a local sentence-transformers model, embeds every tweet at index-build time, persists the corpus embeddings to disk alongside the existing tree cache, and offers a cosine top-k retrieval seam. A `_encode` helper inside that module that tests can patch to avoid downloading the model in CI. The refactor of `tools.search_likes` (and `TweetIndex.search`) to call embedding retrieval first, then the existing `ranker.rank`, then optionally drive the walker as an explainer over the top results. A new optional boolean parameter `with_why` on `search_likes`, default `false`, that gates the explainer call. A new `EMBEDDING_MODEL` env var with a sensible default. Disk-format and invalidation rules for the new cache (rebuild on model name change or when the set of tweet ids changes). Updates to `.env.sample` and the README MCP section. New tests under `tests/mcp/` for the embedder and the refactored search path; existing walker, ranker, tree, and config tests stay green.
- **Out of scope**: Replacing the ranker. Removing the walker module (it stays; `with_why=true` keeps it reachable as the explainer). Image embeddings. Cross-encoder re-ranking. Hybrid BM25 + vector retrieval. Hosted embedding APIs. Anything `x_likes_exporter` (Spec 1) owns. The MCP server transport, JSON-schema scaffolding, error-wrapper boundary in `server.py` (one new optional field added to the `search_likes` input schema; nothing else changes).
- **Adjacent expectations**: This spec consumes mcp-pageindex (Spec 2) types and modules: `TweetIndex.tweets_by_id`, the `Tweet` shape from Spec 1's read API, `ranker.rank` and `ScoredHit`, the existing tree cache, the `errors` boundary helpers, and the `walker.walk` entry point that becomes the explainer. The walker's `_call_chat_completions` test seam stays; tests must keep mocking it. If Spec 2's `tools.search_likes` signature, the `ScoredHit` shape, or the tree cache path conventions change, this spec re-checks. The runtime gains a non-trivial install-graph dep (`sentence-transformers` plus a CPU-only `torch`); that cost is acceptable here and is documented in the README.

## Requirements

### Requirement 1: Embedding model loading and configuration

**Objective:** As a single-user maintainer, I want the embedding model name to come from `.env` with a sensible default, so that I can swap models without editing source and so that a fresh checkout works without me having to set anything new.

#### Acceptance Criteria

1. When the X Likes MCP Server starts, the X Likes MCP Server shall read an optional `EMBEDDING_MODEL` value from `.env` (and from `os.environ` if `.env` is absent), defaulting to `BAAI/bge-small-en-v1.5` when the variable is unset or empty.
2. The X Likes MCP Server shall load a sentence-transformers model identified by the resolved `EMBEDDING_MODEL` name and shall not require any hosted embedding service to be reachable.
3. If the configured model name cannot be resolved at startup (network unreachable on first install, name typo, incompatible model), the X Likes MCP Server shall fail loudly during index build with an error message that names `EMBEDDING_MODEL` and the underlying cause, rather than starting in a half-working state.
4. The X Likes MCP Server shall extend `.env.sample` to document `EMBEDDING_MODEL`, the default value, the install-graph cost (sentence-transformers plus a CPU-only torch wheel, ~200 MB), and a note that the model is downloaded on first run and cached locally by sentence-transformers.

### Requirement 2: Embedding cache built at index time

**Objective:** As a single-user maintainer, I want tweets embedded once and re-used across restarts, so that startup is fast after the first run and the model only runs over tweets it has not seen before.

#### Acceptance Criteria

1. When the X Likes MCP Server starts and no embedding cache file exists, the X Likes MCP Server shall embed every tweet present in `TweetIndex.tweets_by_id` using the configured embedding model and shall persist the resulting matrix and a metadata header to disk under the configured output directory.
2. The X Likes MCP Server shall use the tweet's primary text content for embedding input and shall keep the embedded order aligned with a tweet-id list so that row N of the matrix corresponds to tweet id position N in the metadata.
3. While the embedding cache file exists and its metadata records (a) the same model name as the currently configured `EMBEDDING_MODEL`, and (b) a tweet-id set equal to the current `set(TweetIndex.tweets_by_id.keys())`, the X Likes MCP Server shall load the cached embeddings without re-embedding any tweet.
4. If the model name in the cache metadata differs from the current `EMBEDDING_MODEL`, the X Likes MCP Server shall rebuild the embedding cache from scratch and overwrite the previous cache files.
5. If the tweet-id set in the cache metadata differs from the current `set(TweetIndex.tweets_by_id.keys())` (added or removed tweets), the X Likes MCP Server shall rebuild the embedding cache from scratch and overwrite the previous cache files.
6. The X Likes MCP Server shall write the embedding cache files alongside the existing tree cache (under the configured output directory) and shall write atomically (temporary file plus rename) so that a crash mid-write does not corrupt either file.
7. The X Likes MCP Server shall fail the index-build step if the embedding cache cannot be written to the configured output directory, rather than continuing without persistence.

### Requirement 3: Embedding cache file format

**Objective:** As a single-user maintainer, I want a file format that is easy to inspect, easy to invalidate, and does not invent its own serialization protocol, so that the cache is debuggable and version changes are explicit.

#### Acceptance Criteria

1. The X Likes MCP Server shall persist the corpus embedding matrix as a numpy `.npy` file at a path under the output directory that uses a stable, documented name (for example `corpus_embeddings.npy`).
2. The X Likes MCP Server shall persist the corpus metadata as a JSON file alongside the `.npy` file (for example `corpus_embeddings.meta.json`) containing at minimum: the model name string, the integer tweet count, the ordered list of tweet ids whose row index matches the matrix, and an integer schema version.
3. If the schema version recorded in the metadata file is not the version this server was built for, the X Likes MCP Server shall rebuild the embedding cache from scratch rather than attempting to migrate the file in place.
4. If either the `.npy` file or the metadata JSON file is missing or unreadable, the X Likes MCP Server shall treat the cache as absent and rebuild from scratch.

### Requirement 4: Cosine similarity retrieval

**Objective:** As an MCP client user, I want every search query to consider the entire corpus, not just the months the LLM happens to walk, so that relevant tweets aren't missed because they fall outside an arbitrary date window.

#### Acceptance Criteria

1. When `search_likes` is called with a non-empty query, the X Likes MCP Server shall embed the query string using the configured embedding model and shall compute cosine similarity between the query vector and the corpus matrix.
2. The X Likes MCP Server shall return the top-K most similar candidates (K defaulting to 200) as the input to the existing ranker.
3. While the structured filter (`year`, `month_start`, `month_end`) restricts the search to fewer than K tweets, the X Likes MCP Server shall return every in-scope candidate rather than padding with out-of-scope tweets.
4. The X Likes MCP Server shall apply the structured filter at the candidate stage so that only tweets whose `created_at` resolves to one of the in-scope months can appear in the candidate set; tweets with unparseable `created_at` shall be excluded from filtered queries and included in unfiltered queries.
5. The X Likes MCP Server shall keep cosine similarity computation pure-numpy (no LLM, no network) so the retrieval step has predictable per-query latency.

### Requirement 5: Refactored `search_likes` flow

**Objective:** As an MCP client user, I want `search_likes` to come back in seconds instead of timing out, so that I can use it as a routine tool from any MCP client.

#### Acceptance Criteria

1. When an MCP client calls `search_likes` with a non-empty query, the X Likes MCP Server shall: validate the query and structured filter, resolve the in-scope candidate set, compute the embedding-based top-K, run the existing `ranker.rank` over those candidates, and return the top-N (default 50) `ScoredHit` results.
2. The X Likes MCP Server shall not call the walker on the default `search_likes` path (`with_why=false` or unset), so the default search performs zero LLM calls.
3. When the embedding retrieval returns no candidates for a valid query, the X Likes MCP Server shall return an empty result list rather than an error.
4. If embedding retrieval fails (model unloadable at runtime, numpy error, cache file unreadable mid-run), the X Likes MCP Server shall return a tool error with `category="upstream_failure"` and shall keep the server process alive for subsequent calls.
5. The X Likes MCP Server shall continue to honor every existing `search_likes` requirement that survives the refactor: filter validation rules, empty-query rejection, JSON-schema declaration, top-N bound, response shape (`tweet_id`, `year_month`, `handle`, `snippet`, `score`, `walker_relevance`, `why`, `feature_breakdown`).
6. The X Likes MCP Server shall populate `walker_relevance` and `why` on every result regardless of `with_why`: when the explainer does not run, the X Likes MCP Server shall set `walker_relevance` from the cosine similarity score normalized to `[0, 1]` and shall set `why` to a short cosine-derived label (for example, an empty string or a stable placeholder) so the result shape stays the same as it does today.

### Requirement 6: Optional walker explainer over top hits

**Objective:** As an MCP client user, I want the option to ask "why" the top hits matter, paying the LLM-call cost only when I want it, so that the default search stays fast and the explainer stays available.

#### Acceptance Criteria

1. The X Likes MCP Server shall accept an optional boolean `with_why` parameter on `search_likes`, defaulting to `false` when absent.
2. When `with_why=true`, the X Likes MCP Server shall invoke the walker over at most the top-20 ranked results (after the cosine-then-ranker pipeline) using the existing walker module and shall populate the `why` and `walker_relevance` fields on each affected result from the walker's response.
3. The X Likes MCP Server shall preserve the order produced by the ranker even when `with_why=true`; the walker's output shall be used to populate `why` and refresh `walker_relevance` only, not to reorder results.
4. If the walker fails while `with_why=true`, the X Likes MCP Server shall return the cosine-then-ranker results without `why` annotations and shall surface a single clear stderr log line describing the failure, rather than failing the entire `search_likes` call.
5. When `with_why=false` or unset, the X Likes MCP Server shall not import or invoke the OpenAI SDK as part of the request path, so the default request makes no outbound HTTP.

### Requirement 7: End-to-end latency

**Objective:** As an MCP client user, I want predictable, sub-timeout response times so that `search_likes` is usable as a routine tool from Claude Code or any MCP client.

#### Acceptance Criteria

1. When `search_likes` is called with `with_why=false` against a corpus of approximately 8,000 tweets and a warm embedding cache, the X Likes MCP Server shall return results within 10 seconds end-to-end on a CPU-only laptop-class machine.
2. While the embedding cache is being built for the first time on approximately 8,000 tweets, the X Likes MCP Server shall complete the build within roughly 60 seconds on a CPU-only laptop-class machine; this cost is paid once at first startup and is documented in the README rather than enforced by automated tests.
3. When `search_likes` is called with `with_why=true`, the X Likes MCP Server shall complete within roughly 10 seconds for the cosine-then-ranker stage plus one walker call over the top 20 results; the absolute upper bound depends on the configured local LLM endpoint and is not enforced in CI.

### Requirement 8: Tests must mock the embedding model

**Objective:** As the maintainer, I want the test suite to remain CI-friendly so that running `pytest tests/` on a clean checkout completes without downloading models or hitting the network.

#### Acceptance Criteria

1. When a developer runs `pytest tests/` from the repository root with no `EMBEDDING_MODEL` or model files cached locally, the X Likes MCP Server test suite shall execute and finish without downloading any sentence-transformers model.
2. The X Likes MCP Server `embeddings` module shall expose a documented test seam (the `_encode` function) that tests patch to return canned vectors for given input strings, mirroring the walker's `_call_chat_completions` mock pattern.
3. The X Likes MCP Server test suite shall include unit tests that exercise: cosine top-k math, cache-write-and-reload round-trip, model-name change triggers a rebuild, tweet-id set change triggers a rebuild, schema-version mismatch triggers a rebuild.
4. The X Likes MCP Server test suite shall include integration-level tests that drive `tools.search_likes` end-to-end with the embedding seam mocked and assert the new flow (cosine retrieval → ranker → optional walker explainer) without needing a real model load.

### Requirement 9: Walker module remains in place

**Objective:** As the maintainer, I want the walker code, walker tests, and walker semantics preserved so that this spec extends mcp-pageindex rather than rewriting it.

#### Acceptance Criteria

1. The X Likes MCP Server shall keep the existing `walker.py` module callable as-is (`walker.walk(tree, query, months_in_scope, config)` continues to work).
2. The X Likes MCP Server shall keep every existing `tests/mcp/test_walker.py` test green; this spec shall not break the walker's contract.
3. The X Likes MCP Server shall keep `walker._call_chat_completions` as the LLM mock seam used by the walker tests and any test that exercises the explainer path.
4. The X Likes MCP Server shall route the optional explainer call through `walker.walk` (or a thin wrapper inside `tools.py` that calls into it) so the LLM call site count stays at one.

### Requirement 10: Configuration and documentation alignment

**Objective:** As a single-user maintainer, I want the new env var, the new optional tool parameter, and the new install footprint documented in the same places everything else is, so that the change is discoverable.

#### Acceptance Criteria

1. The X Likes MCP Server shall extend `.env.sample` with a commented `EMBEDDING_MODEL` entry showing the default value and a one-line description.
2. The X Likes MCP Server documentation shall update the README MCP section to describe the new default search behavior (cosine retrieval + ranker), the optional `with_why` flag, and the install-graph cost of `sentence-transformers`.
3. The X Likes MCP Server documentation shall include the on-disk paths of the new cache files (`corpus_embeddings.npy`, `corpus_embeddings.meta.json`) and a note that they live alongside the existing `tweet_tree_cache.pkl` under the configured output directory.
4. The X Likes MCP Server documentation shall state that the walker is now opt-in via `with_why=true` and remains the only LLM call site.
