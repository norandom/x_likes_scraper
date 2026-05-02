# Implementation Plan

- [ ] 1. Foundation: dependency and config plumbing
- [ ] 1.1 Add `sentence-transformers` runtime dependency and document the new env var
  - Add `sentence-transformers>=2.7` to `[project.dependencies]` in `pyproject.toml`. `numpy` comes in transitively; do not add it explicitly.
  - Append a commented `# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` block to `.env.sample` under a new "Fast-search retrieval" header, with a one-line description and the install-graph cost note.
  - Run `uv sync` (or document the equivalent on the CI image) so the lockfile picks up the new dep.
  - Observable completion: `uv sync` finishes without error and `python -c "import sentence_transformers"` succeeds in the project venv.
  - _Requirements: 1.4, 10.1_
  - _Boundary: pyproject.toml, .env.sample_

- [ ] 1.2 Extend `config.Config` with `embedding_model`
  - Add `embedding_model: str` to the `Config` dataclass with the default value `"BAAI/bge-small-en-v1.5"` (taken from a module-level constant so the embeddings module can import it).
  - Read `EMBEDDING_MODEL` from the resolved env dict in `load_config`; empty / unset falls back to the default.
  - Update `tests/mcp/test_config.py` to assert the new field is populated from env, falls back to the default, and that the default constant matches what `embeddings.py` exposes.
  - Observable completion: `load_config(env={...})` returns a `Config` whose `embedding_model` matches the env value, and the new test cases pass.
  - _Requirements: 1.1_
  - _Boundary: x_likes_mcp/config.py, tests/mcp/test_config.py_

- [ ] 2. Core retrieval module
- [ ] 2.1 Implement the `Embedder` class and the `_encode` test seam
  - Create `x_likes_mcp/embeddings.py` with the `EmbeddingError`, `CorpusEmbeddings`, and `Embedder` shapes described in the design.
  - `Embedder.__init__` records the model name; the underlying `SentenceTransformer` is constructed lazily on the first `_encode` call so tests that patch `_encode` never trigger a model load.
  - `embed_query(query)` calls `_encode([query])`, takes row 0, L2-normalizes, returns a `(D,)` `np.float32` vector.
  - `embed_corpus(ordered_ids, texts)` calls `_encode(texts)` once and returns the `(N, D)` `np.float32` matrix with L2-normalized rows.
  - Observable completion: with `_encode` patched to return canned vectors, `embed_query` and `embed_corpus` produce the documented shapes and dtypes, and `EmbeddingError` is raised when the underlying model load actually fails (verified with a patch that raises).
  - _Requirements: 1.2, 1.3, 8.2_
  - _Boundary: x_likes_mcp/embeddings.py_

- [ ] 2.2 Implement `cosine_top_k` with optional id-set masking
  - Add `Embedder.cosine_top_k(query_vec, corpus, k=200, restrict_to_ids=None)` to `embeddings.py`.
  - Compute cosine similarity as a single matrix-vector dot product on the already L2-normalized inputs.
  - When `restrict_to_ids` is `None`, take the top-k indices over the whole matrix. When it is provided, gather only the matching rows before taking the top-k; if the restricted scope is smaller than k, return every restricted candidate.
  - Return `list[tuple[str, float]]` in descending score order.
  - Observable completion: unit tests in `test_embeddings.py` cover ordered top-k, masked top-k, and the small-restricted-scope-returns-all case; all pass.
  - _Requirements: 4.1, 4.2, 4.3, 4.5_
  - _Boundary: x_likes_mcp/embeddings.py_

- [ ] 2.3 Implement the on-disk cache: format, atomic writes, validation
  - Add `_save_cache(cache_npy, cache_meta, matrix, ordered_ids, model_name)` and `_load_cache(cache_npy, cache_meta, expected_model, expected_ids, expected_schema_version)` helpers.
  - Save the matrix with `np.save` and the metadata with `json.dump` (`schema_version`, `model_name`, `n_tweets`, `tweet_ids_in_order`). Both writes go to `*.tmp` files that are then `os.replace`d onto the canonical paths.
  - `_load_cache` returns a `CorpusEmbeddings` only when all three checks pass (model name, id set equality, schema version equality). Missing or unreadable files return `None`.
  - When the cache directory is unwritable, the save helper raises `EmbeddingError` with the path in the message.
  - Observable completion: round-trip tests in `test_embeddings.py` (save then load) recover the same matrix and ordered_ids; mutated metadata triggers the documented rebuild path.
  - _Requirements: 2.6, 2.7, 3.1, 3.2, 3.3, 3.4_
  - _Boundary: x_likes_mcp/embeddings.py_
  - _Depends: 2.1_

- [ ] 2.4 Implement `open_or_build_corpus` orchestrator
  - Add `open_or_build_corpus(embedder, tweets_by_id, cache_dir)` to `embeddings.py`.
  - Cache hit path: `_load_cache(...)` returns a `CorpusEmbeddings` -> return it.
  - Cache miss path: `ordered_ids = sorted(tweets_by_id.keys())`, `texts = [tweets_by_id[i].text or "" for i in ordered_ids]`, `matrix = embedder.embed_corpus(ordered_ids, texts)`, then `_save_cache(...)`.
  - Empty `tweets_by_id` returns an empty `CorpusEmbeddings` and does not write the cache.
  - Observable completion: with a stubbed `_encode`, `open_or_build_corpus` builds and persists on the first call and skips the build (verified by `_encode` not being called) on subsequent calls when nothing changed.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - _Boundary: x_likes_mcp/embeddings.py_
  - _Depends: 2.1, 2.2, 2.3_

- [ ] 3. Index integration
- [ ] 3.1 Wire the embedder into `TweetIndex.open_or_build`
  - Add `embedder: Embedder` and `corpus: CorpusEmbeddings` fields on the `TweetIndex` dataclass.
  - In `TweetIndex.open_or_build`, after the existing tree + tweets-by-id + author-affinity steps, construct `Embedder(config.embedding_model)` and call `open_or_build_corpus(embedder, tweets_by_id, config.output_dir)`. Store the returned `CorpusEmbeddings` on the instance.
  - Observable completion: a fresh `TweetIndex.open_or_build` against the test fixture produces an index with a populated `corpus.matrix` shape `(N, D)` matching `len(tweets_by_id)`; a second call against the same export reuses the cache (verified by counting `_encode` calls on the patched embedder).
  - _Requirements: 2.1_
  - _Boundary: x_likes_mcp/index.py_
  - _Depends: 2.4_

- [ ] 3.2 Resolve candidate id sets from the structured filter
  - Add `TweetIndex._candidate_ids(year, month_start, month_end) -> set[str] | None`.
  - When the filter is fully unset, return `None` (whole corpus). Otherwise call the existing `_resolve_filter` to get the in-scope `YYYY-MM` list, then walk `self.tweets_by_id` and select ids whose `Tweet.get_created_datetime().strftime('%Y-%m')` is in that set. Tweets with unparseable `created_at` are excluded only when a filter is active.
  - Observable completion: unit tests assert that an unfiltered call returns `None`, a year-only filter returns ids only for that year, and tweets with unparseable `created_at` are excluded from filtered results but included in unfiltered ones.
  - _Requirements: 4.4_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.3 Refactor `TweetIndex.search` to call the cosine path
  - Replace the walker call in `TweetIndex.search` with: `candidate_ids = self._candidate_ids(...)`, `query_vec = self.embedder.embed_query(query)`, `cosine_hits = self.embedder.cosine_top_k(query_vec, self.corpus, k=200, restrict_to_ids=candidate_ids)`.
  - Build synthetic `WalkerHit(tweet_id=tid, relevance=score, why="")` entries from the cosine hits and pass them to `ranker.rank` exactly as before. The recency anchor logic stays.
  - Walker imports and calls are removed from `index.py` (the walker is now only invoked from `tools.py`).
  - Observable completion: with the embedder's `_encode` patched, `TweetIndex.search` returns ranker-shaped `ScoredHit` instances and never calls `walker.walk` (assert via a patched walker that raises if invoked); empty cosine results return an empty list.
  - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.3_
  - _Boundary: x_likes_mcp/index.py_
  - _Depends: 3.1, 3.2_

- [ ] 4. Tool layer rewrite
- [ ] 4.1 Add `with_why` to `tools.search_likes` and the explainer helper
  - Extend `tools.search_likes` with `with_why: bool = False`.
  - Validate `with_why`: `None` or absent treats as `False`; non-`bool` raises `errors.invalid_input("with_why", ...)`.
  - When `with_why=False`, return the `_shape_hit(...)` results from `index.search` directly. Set `walker_relevance` to the cosine score (already on the synthetic `WalkerHit`) and `why` to an empty string.
  - When `with_why=True`, call `_call_walker_explainer(top_20_results, query, index)` and merge the returned `tweet_id -> WalkerHit` map onto the matching results: refresh `why` and `walker_relevance` from the walker output, leave order untouched.
  - Translate any non-`ToolError` exception from `index.search` (notably `EmbeddingError`) to `errors.upstream_failure(...)`.
  - Observable completion: `test_tools.py` cases assert (a) default call returns shaped dicts and never invokes the walker, (b) `with_why=true` invokes the walker exactly once and merges its output, (c) non-bool `with_why` raises `invalid_input`, (d) `EmbeddingError` from `index.search` becomes `upstream_failure`.
  - _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.5_
  - _Boundary: x_likes_mcp/tools.py_
  - _Depends: 3.3_

- [ ] 4.2 Implement `_call_walker_explainer` over a synthetic single-chunk tree
  - Build a small in-memory `TweetTree`-shaped object whose `nodes_by_month` contains exactly one month key and a list of up to 20 `TreeNode`s drawn from `index.tree.nodes_by_id` for the top-20 ranked tweet ids.
  - Call `walker.walk(synthetic_tree, query, months_in_scope=[<the synthetic month>], config=index.config, chunk_size=20)` so the walker issues exactly one LLM call.
  - Build a `tweet_id -> WalkerHit` map from the result.
  - On any exception (including `WalkerError`), log a single stderr line naming the failure and return an empty map; do not re-raise.
  - Observable completion: a unit test patches `walker.walk` to return canned hits and asserts the returned map keys match the requested top-20; a second test patches `walker.walk` to raise and asserts the helper returns `{}` and the `search_likes` call still succeeds.
  - _Requirements: 6.2, 6.3, 6.4, 9.1, 9.4_
  - _Boundary: x_likes_mcp/tools.py_

- [ ] 4.3 Extend the `search_likes` MCP input schema with `with_why`
  - In `server._search_likes_tool`, add a `with_why` boolean property to the input schema with `"default": false` and a one-line description matching the design.
  - Thread `arguments.get("with_why", False)` through `_dispatch` into `tools.search_likes`.
  - Update the `test_server_integration.py` schema-shape assertion so the new field is required to be present.
  - Observable completion: an in-process MCP `call_tool("search_likes", {"query": "x", "with_why": true})` reaches the tools handler with `with_why=True`; the schema declared by the server includes the new property; existing `search_likes` integration tests still pass.
  - _Requirements: 5.5, 6.1_
  - _Boundary: x_likes_mcp/server.py, tests/mcp/test_server_integration.py_

- [ ] 5. Test layer
- [ ] 5.1 (P) Add embedder mock guard to `tests/mcp/conftest.py`
  - Add an autouse fixture that patches `x_likes_mcp.embeddings.Embedder._encode` to a deterministic vectorizer (e.g. hashed-bag-of-words producing a stable `(D,)` `np.float32` vector). Tests that need a different vectorizer override the fixture.
  - Choose a small fixed dimension (e.g. `D=16`) that the deterministic stub uses; document it as a test-only constant in conftest.
  - Observable completion: `pytest tests/mcp/` runs to completion on a clean checkout with no `~/.cache/huggingface/` entries for the default model and no real-model download is observed.
  - _Requirements: 8.1, 8.2_
  - _Boundary: tests/mcp/conftest.py_

- [ ] 5.2 (P) Add `tests/mcp/test_embeddings.py`
  - Cover `embed_query`, `embed_corpus`, `cosine_top_k` (with and without `restrict_to_ids`, including the smaller-than-k case), `_save_cache` + `_load_cache` round-trip, and the four invalidation paths (model-name mismatch, tweet-id set mismatch, schema-version mismatch, missing files).
  - Cover `open_or_build_corpus`: cold call invokes `_encode`; warm call against the same input does not.
  - Cover `EmbeddingError` raised on an unwritable cache directory (use `tempfile`/`os.chmod` or a non-existent path).
  - Observable completion: every test in `test_embeddings.py` passes; counter assertions confirm `_encode` is called exactly the expected number of times in each case.
  - _Requirements: 8.3_
  - _Boundary: tests/mcp/test_embeddings.py_
  - _Depends: 2.4_

- [ ] 5.3 Update `tests/mcp/test_index.py`
  - Add tests for `TweetIndex._candidate_ids` (unset filter returns `None`; year-only returns the year's ids; tweets with unparseable `created_at` excluded under filter, included without).
  - Add tests for `TweetIndex.open_or_build` building and reusing the embedding cache (assert via the `_encode` call counter).
  - Add tests for `TweetIndex.search` calling `cosine_top_k` with the right `restrict_to_ids` and never invoking the walker on the default path (patch `walker.walk` to raise on invocation).
  - Observable completion: the additions pass; existing `test_index.py` tests continue to pass with no behavior changes other than the embedder hook.
  - _Requirements: 8.4, 4.4, 5.2_
  - _Boundary: tests/mcp/test_index.py_
  - _Depends: 3.1, 3.2, 3.3_

- [ ] 5.4 Update `tests/mcp/test_tools.py`
  - Replace existing walker-driven `search_likes` cases with cosine-driven ones: assert the default call returns ranker-shaped dicts, never calls `walker.walk`, and uses the cosine score as `walker_relevance`.
  - Add cases for `with_why=true`: walker is invoked exactly once over the top-20 ranked ids; merged `why`/`walker_relevance` reach the response; walker failure during the explainer is logged but not fatal.
  - Add `with_why` validation cases: non-bool raises `invalid_input`; absent and `False` are equivalent.
  - Add an `EmbeddingError -> upstream_failure` translation case.
  - Observable completion: every `test_tools.py::test_search_likes_*` case passes; a counter on the patched `walker.walk` confirms zero invocations on the default path and exactly one invocation on the explainer path.
  - _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6, 6.1, 6.2, 6.4, 6.5, 8.4_
  - _Boundary: tests/mcp/test_tools.py_
  - _Depends: 4.1, 4.2_

- [ ] 5.5 Verify walker module preservation and test-suite continuity
  - Run `pytest tests/mcp/test_walker.py` and confirm every existing case passes unchanged after the spec's edits land. The walker module should not be edited beyond optional docstring nudges; the `_call_chat_completions` mock seam stays in place as the walker tests use it.
  - Add a single regression assertion in `test_walker.py` that grepping `x_likes_mcp/walker.py` still defines `_call_chat_completions` and that no module under `x_likes_mcp/` other than `walker.py` imports `openai`.
  - Observable completion: `pytest tests/mcp/test_walker.py` is green and the regression assertion passes; no new file under `x_likes_mcp/` (other than `walker.py`) imports `openai`.
  - _Requirements: 9.1, 9.2, 9.3_
  - _Boundary: tests/mcp/test_walker.py_

- [ ] 5.6 Update `tests/mcp/test_server_integration.py`
  - Drive `search_likes` through the in-process MCP server with the embedder seam patched and the walker patched to a canned response. Assert the JSON-schema for `search_likes` exposes `with_why`; assert the response shape is unchanged when `with_why=false`; assert the `why` field populates when `with_why=true`.
  - Confirm existing integration assertions (the four registered tools, ToolError -> error response, server stays alive on upstream failure) still hold.
  - Observable completion: the server-integration test file passes end-to-end with the new field exercised in both states.
  - _Requirements: 5.5, 6.1, 8.4_
  - _Boundary: tests/mcp/test_server_integration.py_
  - _Depends: 4.3_

- [ ] 6. Documentation and manual verification
- [ ] 6.1 Update the README MCP section
  - Add a paragraph describing the cosine-then-ranker default path, the `with_why` opt-in, the `EMBEDDING_MODEL` env var with default, and the install-graph cost (~200 MB for `sentence-transformers` + CPU `torch`).
  - Document the new on-disk caches: `output/corpus_embeddings.npy` and `output/corpus_embeddings.meta.json`, alongside the existing `tweet_tree_cache.pkl`.
  - State that the walker remains the only LLM call site and is now opt-in.
  - Observable completion: the README MCP section mentions every item above and the `.env.sample` reference resolves.
  - _Requirements: 10.2, 10.3, 10.4_
  - _Boundary: README.md_

- [ ] 6.2 Manual smoke verification (off-CI)
  - On a real checkout, set `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, run `python -m x_likes_mcp` once and confirm the cold-start embedding pass completes (~30–60 s on a CPU-only laptop) and the two cache files appear under `output/`.
  - Re-run `python -m x_likes_mcp` and confirm sub-second startup (cache hit).
  - From an MCP client, call `search_likes("pentesting with AI and LLMs")` and confirm a sub-10 s response with no LLM call observed at the local proxy.
  - Call the same query with `with_why=true` and confirm exactly one LLM call at the local proxy and a populated `why` field on the top results.
  - Observable completion: a short note added to the spec's `research.md` (or an inline log) recording the wall-clock numbers observed; this task is verification, not gated by CI.
  - _Requirements: 7.1, 7.2, 7.3_
  - _Boundary: manual verification_
  - _Depends: 4.1, 4.3, 6.1_
