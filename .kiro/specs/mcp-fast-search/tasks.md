# Implementation Plan

- [ ] 1. Foundation: dependency and config plumbing
- [x] 1.1 Add `rank_bm25` dep and document the new env vars
  - Add `rank_bm25>=0.2` to `[project.dependencies]` in `pyproject.toml`. The `openai` SDK is already a dep; do not add it again. Do not add `sentence-transformers`, `torch`, or `onnxruntime` — they have no Intel macOS x86_64 wheels and are explicitly out of scope.
  - Append a "Fast-search retrieval (Spec 3 / mcp-fast-search)" block to `.env.sample` with commented `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`), and `EMBEDDING_MODEL` (default `nvidia/llama-nemotron-embed-vl-1b-v2:free`) entries, plus a one-line description per entry and a note about the ~12 min first-run embedding cost on the free tier.
  - Run `uv sync` so the lockfile picks up `rank_bm25`. The earlier rolled-back `sentence-transformers` attempt should leave no trace; verify the lockfile contains `rank_bm25` and no `torch`.
  - Observable completion: `uv sync` finishes without error; `python -c "import rank_bm25; print(rank_bm25.__version__)"` succeeds in the project venv; `.env.sample` contains the three new variables under the Spec 3 header.
  - _Requirements: 1.5, 12.1_
  - _Boundary: pyproject.toml, .env.sample_

- [x] 1.2 Extend `config.Config` with OpenRouter and embedding-model fields
  - Add `openrouter_api_key: str | None = None`, `openrouter_base_url: str = "https://openrouter.ai/api/v1"`, and `embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"` to the `Config` dataclass. Use module-level default constants imported from `embeddings.py` (or a shared `_defaults` constant block) so a single source of truth governs both modules.
  - In `load_config`, read `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `EMBEDDING_MODEL` from the resolved env dict; empty/unset for the URL and model fall back to defaults; empty/unset for the API key resolves to `None` (surfaced at index-build time, not config-load time, so config tests don't need a key).
  - Update `tests/mcp/test_config.py`: assert each new field is populated from env, that the URL and model fall back to documented defaults, that the API key is `None` when unset, and that the default constants match what `embeddings.py` exports.
  - Observable completion: `load_config(env={...})` returns a `Config` whose three new fields match the env (or defaults), and the new test cases pass.
  - _Requirements: 1.1, 1.2, 1.3_
  - _Boundary: x_likes_mcp/config.py, tests/mcp/test_config.py_

- [ ] 2. Core retrieval modules
- [x] 2.1 Implement the `Embedder` class with the OpenRouter HTTP seam
  - Create `x_likes_mcp/embeddings.py` with `EmbeddingError`, `CorpusEmbeddings`, `Embedder`, the module-level constants (`CACHE_SCHEMA_VERSION`, `DEFAULT_EMBEDDING_MODEL`, `DEFAULT_BASE_URL`, `DEFAULT_TOP_K`, `DEFAULT_BATCH_SIZE`, `DEFAULT_MAX_RETRIES`), and the shapes described in the design.
  - `Embedder.__init__(api_key, base_url, model_name, batch_size)` records configuration; the `openai.OpenAI(api_key=..., base_url=...)` client is constructed lazily on first `_call_embeddings_api` invocation so tests that patch the seam never construct a real client.
  - `Embedder._call_embeddings_api(texts)` issues one `client.embeddings.create(model=self.model_name, input=texts)` call, sorts the response by `index`, and returns `list[list[float]]`. On `429` and transient `5xx` it retries up to `DEFAULT_MAX_RETRIES` times with exponential backoff (1s, 2s, 4s); after the cap it raises `EmbeddingError` naming the failure cause. `401`/`403` propagate immediately as `EmbeddingError` naming `OPENROUTER_API_KEY`. If `api_key` is empty/`None` at first invocation, raise `EmbeddingError` immediately without an HTTP call.
  - `embed_query(query)` calls `_call_embeddings_api([query])`, takes row 0, L2-normalizes, returns a `(D,)` `np.float32` vector.
  - `embed_corpus(ordered_ids, texts)` chunks `texts` into batches of `self.batch_size`, calls `_call_embeddings_api(batch)` for each batch, concatenates, L2-normalizes per row, returns `(N, D)` `np.float32` aligned with `ordered_ids`. Inserts a small `time.sleep(0.05)` between batches to avoid burst-rate edges.
  - Observable completion: with `_call_embeddings_api` patched to return canned vectors, `embed_query` and `embed_corpus` produce the documented shapes and dtypes; a stub that raises a transient error twice then succeeds completes in three attempts; a stub that raises persistently raises `EmbeddingError`; `EmbeddingError` is raised when `api_key` is empty.
  - _Requirements: 1.3, 1.4, 2.3, 10.2_
  - _Boundary: x_likes_mcp/embeddings.py_

- [x] 2.2 Implement `cosine_top_k` with optional id-set masking
  - Add `Embedder.cosine_top_k(query_vec, corpus, k=200, restrict_to_ids=None)` to `embeddings.py`.
  - Compute cosine similarity as a single matrix-vector dot product on the already L2-normalized inputs.
  - When `restrict_to_ids` is `None`, take the top-k indices over the whole matrix. When it is provided, gather only the matching rows before taking the top-k; if the restricted scope is smaller than k, return every restricted candidate.
  - Return `list[tuple[str, float]]` in descending score order.
  - Observable completion: unit tests in `test_embeddings.py` cover ordered top-k, masked top-k, and the small-restricted-scope-returns-all case; all pass.
  - _Requirements: 4.1, 4.2, 4.3, 4.5_
  - _Boundary: x_likes_mcp/embeddings.py_
  - _Depends: 2.1_

- [x] 2.3 Implement the on-disk cache: format, atomic writes, validation
  - Add `_save_cache(cache_npy, cache_meta, matrix, ordered_ids, model_name)` and `_load_cache(cache_npy, cache_meta, expected_model, expected_ids, expected_schema_version)` helpers in `embeddings.py`.
  - Save the matrix with `np.save` and the metadata with `json.dump` (`schema_version`, `model_name`, `n_tweets`, `embedding_dim`, `tweet_ids_in_order`). Both writes go to `*.tmp` files that are then `os.replace`d onto the canonical paths.
  - `_load_cache` returns a `CorpusEmbeddings` only when all three checks pass (model name, id set equality, schema version equality). Missing or unreadable files return `None`.
  - When the cache directory is unwritable, the save helper raises `EmbeddingError` with the path in the message.
  - Observable completion: round-trip tests in `test_embeddings.py` (save then load) recover the same matrix and ordered_ids; mutated metadata triggers the documented rebuild path; unwritable directory raises `EmbeddingError`.
  - _Requirements: 2.7, 2.8, 3.1, 3.2, 3.3, 3.4_
  - _Boundary: x_likes_mcp/embeddings.py_
  - _Depends: 2.1_

- [x] 2.4 Implement `open_or_build_corpus` orchestrator
  - Add `open_or_build_corpus(embedder, tweets_by_id, cache_dir)` to `embeddings.py`.
  - Cache hit path: `_load_cache(...)` returns a `CorpusEmbeddings` -> return it.
  - Cache miss path: `ordered_ids = sorted(tweets_by_id.keys())`, `texts = [tweets_by_id[i].text or "" for i in ordered_ids]`, `matrix = embedder.embed_corpus(ordered_ids, texts)`, then `_save_cache(...)`.
  - Empty `tweets_by_id` returns an empty `CorpusEmbeddings(matrix=np.zeros((0, 0), dtype=np.float32), ordered_ids=[], model_name=...)` and does not write the cache.
  - Observable completion: with a stubbed `_call_embeddings_api`, `open_or_build_corpus` builds and persists on the first call and skips the build (verified by `_call_embeddings_api` not being called) on subsequent calls when nothing changed.
  - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6_
  - _Boundary: x_likes_mcp/embeddings.py_
  - _Depends: 2.1, 2.2, 2.3_

- [x] 2.5 Implement the BM25 module
  - Create `x_likes_mcp/bm25.py` with the deterministic `tokenize(text)` function (`re.split(r"\s+", text.lower())` + strip leading/trailing non-word chars + drop empties), the `BM25Index` dataclass, and `BM25Index.build(tweets_by_id)` and `BM25Index.top_k(query, k=200, restrict_to_ids=None)` methods as specified in the design.
  - `BM25Index.build`: `ordered_ids = sorted(tweets_by_id.keys())`, `tokenized = [tokenize(tweets_by_id[i].text or "") for i in ordered_ids]`, `bm25 = BM25Okapi(tokenized)`. Return the dataclass.
  - `BM25Index.top_k`: tokenize the query; if it produces no tokens, return `[]`. Compute `scores = bm25.get_scores(query_tokens)`. If `restrict_to_ids` is provided, set non-restricted positions to `-inf` before taking top-k. Return `list[tuple[str, float]]` in descending score order; when the restricted scope is smaller than `k`, return every restricted candidate that scored above `-inf`.
  - Observable completion: `tokenize` produces the documented output on canned inputs (case, punctuation, whitespace, empty); `BM25Index.build` over a hand-built corpus returns the expected top-K for a query that hits one document strongly; `restrict_to_ids` masks correctly; an all-empty-after-tokenize query returns `[]`. All assertions live in `test_bm25.py`.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - _Boundary: x_likes_mcp/bm25.py_

- [x] 2.6 Implement the Reciprocal Rank Fusion helper
  - Create `x_likes_mcp/fusion.py` with `DEFAULT_K_RRF = 60`, `DEFAULT_FUSED_TOP = 300`, and `reciprocal_rank_fusion(rankings, k_rrf=60, top=300) -> list[str]`.
  - For each ranking, iterate 1-indexed (rank 1, 2, 3, ...). For each id in each ranking, accumulate `score[d] += 1.0 / (k_rrf + rank)`.
  - Empty rankings are silently ignored. If all rankings are empty, return `[]`.
  - Sort survivors by descending score; ties broken by deterministic insertion order (first ranking that mentioned the id, then earlier rank in that ranking, then lexicographic id).
  - Truncate to `top` and return the id list (no scores in the return — RRF score is recall-only).
  - Observable completion: `test_fusion.py` covers two crafted rankings producing the documented fused order; single-method input (one `[]`) returns the other ranking's order; both-empty returns `[]`; ties resolved deterministically across runs.
  - _Requirements: 6.1, 6.2, 6.4_
  - _Boundary: x_likes_mcp/fusion.py_

- [ ] 3. Index integration
- [x] 3.1 Wire embedder, corpus, and BM25 into `TweetIndex.open_or_build`
  - Add `embedder: Embedder`, `corpus: CorpusEmbeddings`, and `bm25: BM25Index` fields on the `TweetIndex` dataclass.
  - In `TweetIndex.open_or_build`, after the existing tree + tweets-by-id + author-affinity steps:
    1. Construct `Embedder(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url, model_name=config.embedding_model)`.
    2. Call `open_or_build_corpus(embedder, tweets_by_id, config.output_dir)` and store the returned `CorpusEmbeddings`.
    3. Call `BM25Index.build(tweets_by_id)` and store the result.
  - Observable completion: a fresh `TweetIndex.open_or_build` against the test fixture produces an index with a populated `corpus.matrix` shape `(N, D)` matching `len(tweets_by_id)` and a populated `bm25.ordered_ids` matching `sorted(tweets_by_id.keys())`. A second call against the same export reuses the embedding cache (verified by counting `_call_embeddings_api` calls on the patched embedder) and rebuilds the BM25 index in-memory.
  - _Requirements: 2.1, 5.1_
  - _Boundary: x_likes_mcp/index.py_
  - _Depends: 2.4, 2.5_

- [x] 3.2 Resolve candidate id sets from the structured filter
  - Add `TweetIndex._candidate_ids(year, month_start, month_end) -> set[str] | None`.
  - When the filter is fully unset, return `None` (whole corpus). Otherwise call the existing `_resolve_filter` to get the in-scope `YYYY-MM` list, then walk `self.tweets_by_id` and select ids whose `Tweet.get_created_datetime().strftime('%Y-%m')` is in that set. Tweets with unparseable `created_at` are excluded only when a filter is active.
  - Observable completion: unit tests assert that an unfiltered call returns `None`, a year-only filter returns ids only for that year, and tweets with unparseable `created_at` are excluded from filtered results but included in unfiltered ones.
  - _Requirements: 4.4, 5.4_
  - _Boundary: x_likes_mcp/index.py_

- [x] 3.3 Refactor `TweetIndex.search` to drive the hybrid pipeline
  - Replace the walker call in `TweetIndex.search` with the hybrid path:
    1. `candidate_ids = self._candidate_ids(...)`.
    2. Run dense and BM25 retrieval. Each path is wrapped in its own try/except: dense failure logs once and yields `[]`; BM25 failure logs once and yields `[]`. If both yield `[]`, raise `EmbeddingError` (or a new `RetrievalError`) up to `tools.py` which maps to `upstream_failure`. Otherwise proceed with whichever returned candidates.
    3. `dense_ids = [tid for tid, _ in dense_ranking]`, `bm25_ids = [tid for tid, _ in bm25_ranking]`.
    4. `fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids], k_rrf=60, top=300)`.
    5. Build a `dense_score_by_id` map; for each fused id, look up its dense cosine score (default to `0.0` when only BM25 had it) and synthesize `WalkerHit(tweet_id=tid, relevance=score, why="")`.
    6. `scored = ranker.rank(synthetic_hits, self.tweets_by_id, self.author_affinity, self.weights, anchor=_compute_anchor(...))`.
    7. Return `scored[:top_n]`.
  - Walker imports and calls are removed from `index.py`; the walker is now only invoked from `tools.py`.
  - Observable completion: with the embedder seam patched and `walker.walk` patched to raise on invocation, `TweetIndex.search` returns ranker-shaped `ScoredHit` instances and never raises from the walker patch on the default path; empty fused results return an empty list; dense-down is recoverable when BM25 has candidates and vice versa.
  - _Requirements: 4.1, 4.2, 4.3, 5.3, 7.1, 7.3, 7.4, 7.5, 7.6_
  - _Boundary: x_likes_mcp/index.py_
  - _Depends: 3.1, 3.2, 2.6_

- [ ] 4. Tool layer rewrite
- [ ] 4.1 Add `with_why` to `tools.search_likes` with hybrid + fallback semantics
  - Extend `tools.search_likes` with `with_why: bool = False`.
  - Validate `with_why`: `None` or absent treats as `False`; non-`bool` raises `errors.invalid_input("with_why", ...)`.
  - When `with_why=False`, return the `_shape_hit(...)` results from `index.search` directly. Set `walker_relevance` from the synthetic `WalkerHit.relevance` (cosine score, or `0.0` when only BM25 had the id) and `why` to an empty string.
  - When `with_why=True`, call `_call_walker_explainer(top_20_results, query, index)` and merge the returned `tweet_id -> WalkerHit` map onto the matching results: refresh `why` and `walker_relevance` from the walker output, leave order untouched.
  - Translate any non-`ToolError` exception bubbling out of `index.search` (notably `EmbeddingError` raised when both retrievals failed) to `errors.upstream_failure(...)`.
  - Observable completion: `test_tools.py` cases assert (a) default call returns shaped dicts and never invokes the walker, (b) `with_why=true` invokes the walker exactly once and merges its output, (c) non-bool `with_why` raises `invalid_input`, (d) `EmbeddingError` from `index.search` becomes `upstream_failure`, (e) dense-down with BM25 candidates returns shaped dicts (call still succeeds).
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 7.7, 7.8, 8.1, 8.2, 8.3, 8.5_
  - _Boundary: x_likes_mcp/tools.py_
  - _Depends: 3.3_

- [ ] 4.2 Implement `_call_walker_explainer` over a synthetic single-chunk tree
  - Build a small in-memory `TweetTree`-shaped object whose `nodes_by_month` contains exactly one month key and a list of up to 20 `TreeNode`s drawn from `index.tree.nodes_by_id` for the top-20 ranked tweet ids.
  - Call `walker.walk(synthetic_tree, query, months_in_scope=[<the synthetic month>], config=index.config, chunk_size=20)` so the walker issues exactly one chat-completions call.
  - Build a `tweet_id -> WalkerHit` map from the result.
  - On any exception (including `WalkerError`), log a single stderr line naming the failure and return an empty map; do not re-raise.
  - Observable completion: a unit test patches `walker.walk` to return canned hits and asserts the returned map keys match the requested top-20; a second test patches `walker.walk` to raise and asserts the helper returns `{}` and the `search_likes` call still succeeds.
  - _Requirements: 8.2, 8.3, 8.4, 11.1, 11.4_
  - _Boundary: x_likes_mcp/tools.py_

- [ ] 4.3 Extend the `search_likes` MCP input schema with `with_why`
  - In `server._search_likes_tool`, add a `with_why` boolean property to the input schema with `"default": false` and a one-line description matching the design.
  - Thread `arguments.get("with_why", False)` through `_dispatch` into `tools.search_likes`.
  - Update the `test_server_integration.py` schema-shape assertion so the new field is required to be present.
  - Observable completion: an in-process MCP `call_tool("search_likes", {"query": "x", "with_why": true})` reaches the tools handler with `with_why=True`; the schema declared by the server includes the new property; existing `search_likes` integration tests still pass.
  - _Requirements: 7.7, 8.1_
  - _Boundary: x_likes_mcp/server.py, tests/mcp/test_server_integration.py_

- [ ] 5. Test layer
- [ ] 5.1 (P) Add embedder mock guard to `tests/mcp/conftest.py`
  - Add an autouse fixture that patches `x_likes_mcp.embeddings.Embedder._call_embeddings_api` to a deterministic vectorizer (e.g. hashed-bag-of-words producing a stable `list[list[float]]` of dimension `D`). Tests that need a different vectorizer override the fixture.
  - Choose a small fixed dimension (e.g. `D=16`) and document it as a test-only constant in conftest.
  - Observable completion: `pytest tests/mcp/` runs to completion on a clean checkout with no `OPENROUTER_API_KEY` set and no network access; no real HTTP request is observed.
  - _Requirements: 10.1, 10.2_
  - _Boundary: tests/mcp/conftest.py_

- [ ] 5.2 (P) Add `tests/mcp/test_embeddings.py`
  - Cover `embed_query`, `embed_corpus` (including batch boundaries), `cosine_top_k` (with and without `restrict_to_ids`, including the smaller-than-k case), `_save_cache` + `_load_cache` round-trip, and the four invalidation paths (model-name mismatch, tweet-id set mismatch, schema-version mismatch, missing files).
  - Cover `open_or_build_corpus`: cold call invokes `_call_embeddings_api`; warm call against the same input does not.
  - Cover retry behavior: a stub that raises a transient error twice then returns vectors yields the vectors in three calls; a stub that raises persistently raises `EmbeddingError` after the documented retry cap.
  - Cover `EmbeddingError` raised on an unwritable cache directory and on empty `api_key`.
  - Observable completion: every test in `test_embeddings.py` passes; counter assertions confirm `_call_embeddings_api` is called exactly the expected number of times in each case.
  - _Requirements: 10.3_
  - _Boundary: tests/mcp/test_embeddings.py_
  - _Depends: 2.4_

- [ ] 5.3 (P) Add `tests/mcp/test_bm25.py`
  - Cover `tokenize` on canned inputs (mixed case, leading/trailing punctuation, internal whitespace, empty string, all-punctuation string).
  - Cover `BM25Index.build` over a small hand-built corpus where one tweet contains the query terms strongly and others do not; assert top-K ordering.
  - Cover `restrict_to_ids` masking; assert smaller-than-k restricted scope returns every restricted candidate.
  - Cover empty-after-tokenize query returns `[]`.
  - Observable completion: every test in `test_bm25.py` passes.
  - _Requirements: 10.4_
  - _Boundary: tests/mcp/test_bm25.py_
  - _Depends: 2.5_

- [ ] 5.4 (P) Add `tests/mcp/test_fusion.py`
  - Cover RRF over two crafted rankings with overlapping and non-overlapping ids; assert the documented fused order.
  - Cover single-method input (`[]` for one ranking) returns the other ranking's order.
  - Cover both-empty returns `[]`.
  - Cover deterministic tie-breaking across runs.
  - Cover `top` truncation.
  - Observable completion: every test in `test_fusion.py` passes.
  - _Requirements: 10.4_
  - _Boundary: tests/mcp/test_fusion.py_
  - _Depends: 2.6_

- [ ] 5.5 Update `tests/mcp/test_index.py`
  - Add tests for `TweetIndex._candidate_ids` (unset filter returns `None`; year-only returns the year's ids; tweets with unparseable `created_at` excluded under filter, included without).
  - Add tests for `TweetIndex.open_or_build` building and reusing the embedding cache (assert via the `_call_embeddings_api` call counter) and building the BM25 index.
  - Add tests for `TweetIndex.search` running both retrievals with the right `restrict_to_ids` and never invoking the walker on the default path (patch `walker.walk` to raise on invocation). Add a dense-down case (patch `_call_embeddings_api` to raise persistently) that succeeds via BM25 alone.
  - Observable completion: the additions pass; existing `test_index.py` tests continue to pass with no behavior changes other than the new hooks.
  - _Requirements: 4.4, 5.4, 7.4, 10.5_
  - _Boundary: tests/mcp/test_index.py_
  - _Depends: 3.1, 3.2, 3.3_

- [ ] 5.6 Update `tests/mcp/test_tools.py`
  - Replace existing walker-driven `search_likes` cases with hybrid-driven ones: assert the default call returns ranker-shaped dicts, never calls `walker.walk`, and uses the cosine score (or `0.0` when only BM25 had the id) as `walker_relevance`.
  - Add cases for `with_why=true`: walker is invoked exactly once over the top-20 ranked ids; merged `why`/`walker_relevance` reach the response; walker failure during the explainer is logged but not fatal.
  - Add `with_why` validation cases: non-bool raises `invalid_input`; absent and `False` are equivalent.
  - Add an `EmbeddingError -> upstream_failure` translation case (both retrievals fail).
  - Add a dense-down-with-BM25-up case asserting the call still succeeds.
  - Observable completion: every `test_tools.py::test_search_likes_*` case passes; a counter on the patched `walker.walk` confirms zero invocations on the default path and exactly one invocation on the explainer path.
  - _Requirements: 7.1, 7.2, 7.4, 7.6, 7.7, 7.8, 8.1, 8.2, 8.4, 8.5, 10.5_
  - _Boundary: tests/mcp/test_tools.py_
  - _Depends: 4.1, 4.2_

- [ ] 5.7 Verify walker module preservation and test-suite continuity
  - Run `pytest tests/mcp/test_walker.py` and confirm every existing case passes unchanged after the spec's edits land. The walker module should not be edited beyond optional docstring nudges; the `_call_chat_completions` mock seam stays in place as the walker tests use it.
  - Add a single regression assertion in `test_walker.py` (or a small new test) that grepping `x_likes_mcp/walker.py` still defines `_call_chat_completions`, and that no module under `x_likes_mcp/` other than `walker.py` issues `client.chat.completions.create` (a textual grep is sufficient).
  - Observable completion: `pytest tests/mcp/test_walker.py` is green and the regression assertion passes.
  - _Requirements: 11.1, 11.2, 11.3_
  - _Boundary: tests/mcp/test_walker.py_

- [ ] 5.8 Update `tests/mcp/test_server_integration.py`
  - Drive `search_likes` through the in-process MCP server with the embedder seam patched and the walker patched to a canned response. Assert the JSON-schema for `search_likes` exposes `with_why`; assert the response shape is unchanged when `with_why=false`; assert the `why` field populates when `with_why=true`.
  - Confirm existing integration assertions (the four registered tools, ToolError -> error response, server stays alive on upstream failure) still hold.
  - Observable completion: the server-integration test file passes end-to-end with the new field exercised in both states.
  - _Requirements: 7.7, 8.1, 10.5_
  - _Boundary: tests/mcp/test_server_integration.py_
  - _Depends: 4.3_

- [ ] 6. Documentation and manual verification
- [ ] 6.1 Update the README MCP section
  - Add a paragraph describing the hybrid recall (BM25 + dense via OpenRouter, fused with RRF) + ranker default path, the `with_why` opt-in, the three new env vars (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `EMBEDDING_MODEL`) with defaults, and the install graph (one new pure-python dep, `rank_bm25`).
  - Document the new on-disk caches: `output/corpus_embeddings.npy` and `output/corpus_embeddings.meta.json`, alongside the existing `tweet_tree_cache.pkl`.
  - State that the walker remains the only chat-completions LLM call site and is now opt-in.
  - State the platform reason for hosted dense embeddings: Intel macOS x86_64 has no modern PyTorch / ONNX Runtime wheels, so an in-process transformer is not possible on the maintainer's primary machine; OpenRouter is the pragmatic alternative.
  - Document the first-run cost (~12 minutes on the OpenRouter free tier for ~7,780 tweets) and warm-cache cost (sub-second startup, sub-2-second queries in the typical case).
  - Observable completion: the README MCP section mentions every item above and the `.env.sample` reference resolves.
  - _Requirements: 12.2, 12.3, 12.4, 12.5_
  - _Boundary: README.md_

- [ ] 6.2 Manual smoke verification (off-CI)
  - On a real checkout, set `OPENROUTER_API_KEY` (and optionally `EMBEDDING_MODEL`) in `.env`, run `python -m x_likes_mcp` once and confirm the cold-start embedding pass completes (~12 min on the free tier for 7,780 tweets) and the two cache files appear under `output/`.
  - Re-run `python -m x_likes_mcp` and confirm sub-second startup (cache hit).
  - From an MCP client, call `search_likes("pentesting with AI and LLMs")` and confirm a sub-2-second response with no chat-completions call observed at the local proxy and one embeddings call observed at OpenRouter.
  - Call the same query with `with_why=true` and confirm exactly one chat-completions call at the local proxy and a populated `why` field on the top results.
  - Observable completion: a short note added to the spec's `research.md` (or an inline log) recording the wall-clock numbers observed; this task is verification, not gated by CI.
  - _Requirements: 9.1, 9.2, 9.3_
  - _Boundary: manual verification_
  - _Depends: 4.1, 4.3, 6.1_

## Implementation Notes

(empty; populated by kiro-impl as cross-cutting learnings emerge)
