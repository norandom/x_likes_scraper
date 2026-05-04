# Implementation Plan

> Parallel-capable tasks are marked with `(P)`. They have no data, file, or boundary conflicts with their immediate peers and Foundation work has already completed when they run. Cross-group dependencies, when relevant, are declared via `_Depends:_`.

- [ ] 1. Foundation: dependencies, configuration, fence-marker plumbing, and shared scaffolding
- [x] 1.1 Pin new third-party dependencies and adjust gitignore
  - Add `dspy-ai>=2.6`, `httpx>=0.27`, and `pydantic>=2` to the project's runtime dependencies
  - Add `output/url_cache/` and `output/synthesis_compiled/` to `.gitignore` so cached bodies and compiled programs never enter git
  - `uv sync` succeeds with the new pins and `pip show dspy-ai httpx pydantic` reports the expected versions
  - _Requirements: 6.1, 6.3, 6.4, 11.1_
  - _Boundary: project metadata_

- [x] 1.2 Extend the configuration loader with synthesis-report env vars
  - Surface `crawl4ai_base_url`, `url_cache_dir`, `url_cache_ttl_days`, `synthesis_max_hops`, `synthesis_per_source_bytes`, `synthesis_total_context_bytes`, `synthesis_round_two_k`, and `url_fetch_allowed_private_cidrs` on the existing config object with documented defaults
  - Parse the comma-separated CIDR allowlist into IP-network objects at load time so malformed values fail loudly before any fetch
  - Preserve the existing `OPENAI_BASE_URL` / `OPENAI_MODEL` reads as the default synthesis LM endpoint; do not introduce a separate synthesis endpoint
  - Loading the config from a `.env` that omits every new variable still produces working defaults; loading with a malformed CIDR raises immediately
  - _Requirements: 4.3, 12.1, 12.2, 12.3_
  - _Boundary: config_

- [x] 1.3 Add the four new fence-marker families to the shared sanitize module
  - Define `URL_BODY`, `ENTITY`, `KG_NODE`, and `KG_EDGE` open/close markers and add them to the existing all-fences set so neutralization scrubs every family from every body
  - Provide per-family fence helpers that reuse the existing sanitize and marker-neutralization passes
  - Unit tests show that a body containing any of the eight new markers is neutralized before it is wrapped in a different family's fence (no marker can prematurely close a fence and reopen another)
  - _Requirements: 7.1, 7.2_
  - _Boundary: sanitize_

- [x] 1.4 Create the `synthesis` subpackage skeleton with shared types and per-shape config
  - Lay down the empty modules for the orchestrator, leaf modules, and the public surface
  - Define `ReportShape` (brief / synthesis / trend), `ReportOptions`, `ReportResult`, and the supporting dataclasses (`FetchedUrl`, `Entity`, `Claim`, `Section`, `MonthSummary`) the design pins
  - Encode each shape's length and section directives plus the `MAX_MINDMAP_DEPTH=4` cap in one place so the renderer and synthesizer read the same numbers
  - Importing `from x_likes_mcp.synthesis import ReportShape, ReportOptions, ReportResult` succeeds and an unknown shape value raises immediately
  - _Requirements: 1.2, 1.3, 2.4, 8.2_
  - _Boundary: synthesis package public surface_

- [x] 1.5 Stand up the synthesis test scaffolding
  - Add a per-package `conftest.py` under `tests/mcp/synthesis/` that autouse-blocks real URL fetching and real DSPy LM calls
  - Provide a `FakeDspyLM` that returns canned responses keyed by signature name and input hash so signature tests stay offline
  - Reserve a `real_lm` pytest marker that is collected only when an explicit opt-in flag is passed
  - Running `pytest tests/mcp/synthesis -q` against the empty package collects zero failures and the autouse fixtures are exercised by a placeholder smoke test
  - _Requirements: 6.2, 7.3_
  - _Boundary: synthesis test scaffolding_

- [ ] 2. Core leaf modules — security, persistence, KG, extraction, and rendering
- [x] 2.1 (P) Implement the SSRF guard with the two-tier blocklist and CIDR allowlist
  - Resolve the URL hostname once, walk the address candidates, and pin the first non-blocked IP so the connection cannot be rebound mid-fetch
  - Treat loopback, cloud-metadata addresses (AWS / GCP / Azure / DigitalOcean / OCI), broadcast / multicast, and IANA-reserved ranges as unconditional blocks that no allowlist can override
  - Block RFC1918, IPv4 link-local (minus the metadata IP), IPv6 link-local, and IPv6 ULA unless the resolved address falls inside the operator-supplied private-range CIDR allowlist
  - Re-validation API accepts a redirect target and runs the same checks so the fetcher can call it on every redirect hop
  - Tests cover: 169.254.169.254 always blocked, RFC1918 blocked by default but allowed when its CIDR is on the allowlist, public IPv4 / IPv6 pass, missing host raises, non-HTTP(S) scheme raises
  - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - _Boundary: ssrf_guard_

- [x] 2.2 (P) Implement the sha256-keyed URL cache
  - Provide get / put / expire operations keyed on `sha256(url)` under the configured cache root, with atomic writes via temp-file + rename
  - Persist only post-sanitize markdown plus the metadata fields documented in the design (`url`, `final_url`, `content_type`, `fetched_at`); raw HTML / PDF bytes never touch disk
  - Treat entries younger than the configured TTL (default 30 days) as fresh and re-fetch on miss or stale; create the cache directory on first write
  - Tests cover: missing directory creation, stable hash key across runs, TTL boundary (29-day-old entry hits, 31-day-old entry misses), atomic write survives a simulated mid-write crash
  - _Requirements: 11.1, 11.2, 11.3, 11.4_
  - _Boundary: url_cache_

- [x] 2.3 (P) Build the in-memory mini knowledge graph
  - Provide node and edge types covering at least query, tweet, handle, hashtag, domain, and concept nodes and the authored_by, cites, mentions, and recall_for edges
  - Namespace IDs (`tweet:<id>`, `handle:<screen_name>`, `hashtag:<tag>`, `domain:<host>`, `concept:<lower-snake-case>`) so cross-source collisions are impossible
  - Expose `top_entities(kind, n)` and a neighbor lookup so the multihop fan-out and the mindmap can both read the same structure
  - Persist nothing; the graph lives only for the duration of a single report run
  - Tests cover: namespaced ID stability, weight-ranked top entities, neighbor filtering by edge kind, two builds on the same input produce identical graphs
  - _Requirements: 5.3, 5.4_
  - _Boundary: kg_

- [x] 2.4 (P) Implement regex entity extraction with a DSPy-fallback hook
  - Run cheap regex / counter passes for handles, hashtags, URL domains, and recurring noun phrases over the hit text and any fetched URL bodies before any LM call
  - Expose a fallback entry point that the orchestrator can wire to the DSPy `ExtractEntities` predictor; the fallback fires only for hits where the regex pass returned zero entities
  - Tests cover: regex extracts handles / hashtags / domains, regex returns empty for tweets that contain only stopwords, fallback hook is called exactly once per empty-regex hit and not at all for regex-covered hits
  - _Requirements: 5.1, 5.2_
  - _Boundary: entities_

- [x] 2.5 (P) Build the depth-capped mermaid mindmap renderer
  - Generate a mermaid `mindmap` block whose root is the user query and whose level-1 children are the entity categories present in the KG (Authors, Sources, Themes, Hashtags) — only categories with non-empty children are emitted
  - Cap the rendered depth at the documented maximum (4) so GitHub, Obsidian, and VS Code preview render legibly
  - Filter node labels to the safe character subset before emission so mermaid's parser never trips on quotes, brackets, slashes, or `@` / `:`
  - Tests cover: depth cap honored even on a deep KG, unsafe characters stripped from labels, empty KG renders an empty mindmap block (no crash)
  - _Requirements: 8.1, 8.2_
  - _Boundary: mindmap_

- [ ] 3. LLM modules: DSPy signatures, asserts, and compiled-program persistence
- [x] 3.1 Define the DSPy signatures, ChainOfThought modules, and claim-source assert
  - Declare typed signatures for `SynthesizeBrief`, `SynthesizeNarrative`, and `SynthesizeTrend` with Pydantic-typed structured outputs (`Claim`, `Section`, `MonthSummary`) and a `dspy.Predict(ExtractEntities)` for the entity-extraction fallback
  - Carry the three system-prompt rules verbatim in each synthesis signature's docstring so DSPy attaches them to every call
  - Configure the DSPy LM via litellm using the existing `OPENAI_BASE_URL` / `OPENAI_MODEL`, failing fast with a clear error if neither is set
  - Wire `dspy.Assert` so any claim whose `sources` are not a subset of the known-source-ID set triggers one corrective retry; the second failure raises a structured `SynthesisValidationError`
  - Tests (using the FakeDspyLM seam) cover: docstring rules present, missing LM endpoint raises, claim-source assert rejects unknown source IDs, claim-source assert passes when every cited source is in the known set
  - _Requirements: 5.2, 6.1, 6.2, 6.6, 7.3_
  - _Boundary: dspy_modules_

- [x] 3.2 Implement compiled-program load / save and the optimizer entry point
  - Resolve the per-shape compiled-program path under the configured directory (default `output/synthesis_compiled/{shape}.json`) and return `None` from the loader when the file is missing or stale so the orchestrator can fall back to the un-optimized signature
  - Save compiled programs atomically (temp file + replace) and create the parent directory on first write
  - Provide a `run_optimizer(shape, labeled_examples, optimizer="BootstrapFewShot")` entry point that sanitizes and fences each demo's input fields before the optimizer sees them, then writes the compiled program to the configured path
  - Tests cover: missing file returns `None`, stored program round-trips through load, demo inputs are sanitized + fenced before being handed to the optimizer (the optimizer is stubbed in unit tests), optimizer end-to-end test is marked `slow` and skipped by default
  - _Requirements: 6.3, 6.4, 6.5_
  - _Boundary: compiled_

- [ ] 4. Pipeline modules: context assembly, fetcher, multihop, and report renderer
- [x] 4.1 Assemble the fenced synthesis context with per-source caps and a total-budget enforcer
  - Build the fenced blob by sanitizing and fencing each tweet body, fetched URL link, fetched URL body, entity string, KG node label, and KG edge caption with the matching marker family
  - Apply per-source byte caps (defaults `per_tweet_bytes=280`, `per_url_body_bytes=4096`, `per_entity_bytes=64`, `per_kg_label_bytes=64`) before fencing so a single huge source cannot crowd out the rest
  - Enforce a total context budget (default `total_bytes=32768`) by dropping the lowest-rank URL bodies first, then the lowest-rank tweets, until the assembled blob fits; never drop the system prompt or the user query
  - Return both the fenced blob and the set of known source IDs (`tweet:<id>` and `url:<final_url>`) the synthesizer's claim-source assert will check against
  - Tests cover: each marker family neutralized on every body before fencing, per-source cap truncates correctly, total-budget enforcement drops in the documented order, known-source-ID set matches the fenced sources exactly
  - _Requirements: 6.5, 7.1, 7.2, 7.4_
  - _Boundary: context_
  - _Depends: 1.3, 2.3, 3.1_

- [x] 4.2 Implement the crawl4ai HTTP fetcher with SSRF, redirect, content-type, and cache discipline
  - Probe the configured crawl4ai endpoint at startup when fetching is enabled and surface a `ContainerUnreachable` error that names the endpoint and its env-var override
  - Per URL: SSRF-validate the host, POST to the documented crawl4ai endpoint with `follow_redirects=False`, manually walk up to three redirect hops while re-validating each new host (with the same allowlist), enforce the 5-second timeout, drop URLs whose declared content type is not in the allowlist (text/html, text/plain, application/json, application/pdf), take the `markdown` field from the JSON response, sanitize it, truncate to the configured per-URL byte cap, and persist the post-sanitize body through the cache
  - Treat any per-URL skip (timeout, blocked IP, bad content type, 5xx, oversize body) as a soft drop that returns no result for that URL while the rest of the run continues
  - Tests (with a stubbed crawl4ai HTTP layer) cover: container probe failure raises, happy-path HTML fetch caches and returns sanitized markdown, PDF response uses the same `markdown` field, mid-redirect host change is blocked by the SSRF re-check, content-type allowlist drops Office types silently, cache hit short-circuits the network, per-URL timeout drops only that URL
  - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.6, 4.4, 4.5, 4.6, 11.1, 11.3, 12.4_
  - _Boundary: fetcher_
  - _Depends: 2.1, 2.2_

- [x] 4.3 Implement the multihop fan-out
  - Run the round-1 search via the existing index search and return the round-1 hits unchanged when `hops==1`
  - When `hops==2`, mine the top-K entities from the round-1 KG (default K=5, override `SYNTHESIS_ROUND_TWO_K`), issue one search per entity in parallel through a thread pool capped at four workers, and apply the same `year` / `month_start` / `month_end` filters supplied to round-1
  - Fuse round-1 and round-2 hits by `tweet_id` so a tweet that appears in both rounds is rendered once, preserving round-1 ordering for shared IDs and appending round-2-only hits at the end
  - Reject any value of `hops` greater than the configured maximum (default 2) before any search call
  - Tests cover: `hops==1` issues exactly one index search, `hops==2` issues exactly K parallel round-2 searches with the same filters as round-1, fusion deduplicates by `tweet_id` deterministically, `hops==3` is rejected before any search
  - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.4_
  - _Boundary: multihop_
  - _Depends: 2.3_

- [x] 4.4 Implement the markdown report renderer for all three shapes
  - Render `brief` to a ~300-word concept brief with top entities and 5–10 anchor tweets; render `synthesis` to a longer narrative with a mermaid mindmap and per-cluster tweet list; render `trend` to a month-bucketed timeline (using each tweet's `created_at`) plus a mindmap with temporal weighting
  - Render every anchor tweet as a clickable link to its canonical `https://x.com/{handle}/status/{id}` URL via the existing status-URL helper
  - Group `trend` anchor tweets into chronologically-ordered month buckets derived from `created_at`
  - Render an "empty corpus" report that names the query and states that no matching tweets were found, without invoking the synthesizer, when the orchestrator passes an empty hit list
  - Run the assembled markdown through the shared sanitize pass once before returning so an LM-emitted ANSI / control / bidi codepoint cannot reach the file
  - Tests cover: `brief` length envelope, `synthesis` includes a `mermaid mindmap` block and a per-cluster tweet list, `trend` orders months chronologically, anchor links use the canonical x.com URL, empty-corpus report produces no LM call, output passes through sanitize
  - _Requirements: 1.2, 7.5, 8.1, 8.3, 8.4, 9.4_
  - _Boundary: report_render_
  - _Depends: 2.5_

- [ ] 5. Integration: orchestrator and the MCP / CLI surface
- [x] 5.1 Implement the synthesis orchestrator
  - Validate `ReportOptions` (shape, hops, filter values, paths) and reject unknown shapes / out-of-range hops before any side effect
  - Drive the pipeline: round-1 search → optional round-2 fan-out → optional URL fetch (only when `fetch_urls=True`, including the crawl4ai probe) → KG build → fenced-context assembly → DSPy synthesis → markdown render
  - Load the per-shape compiled program when present and fall back to the un-optimized signature when absent; never fail because a compiled program is missing
  - Wire the entity extractor's DSPy fallback so the fallback fires only for hits where the regex pass returned no entities
  - Translate downstream errors (LM unreachable, synthesis-validation failure, crawl4ai unreachable when fetching is on) into a structured exception type the CLI / MCP boundary can map to a non-zero exit / `upstream_failure`
  - Make zero network calls when `fetch_urls=False` and the LM endpoint is reachable; the run's only outbound call in that mode is the LM endpoint
  - Tests cover: `hops==1` skips the round-2 entry point, `hops==2` triggers exactly the documented number of parallel round-2 searches, `fetch_urls=False` never reaches the fetcher, empty corpus skips the LM call, claim-source assert failure surfaces as a structured error, the orchestrator never writes to disk itself
  - _Requirements: 1.1, 1.3, 1.4, 5.2, 6.3, 9.4, 12.4_
  - _Boundary: orchestrator_
  - _Depends: 2.4, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4_

- [x] 5.2 Add the `synthesize_likes` MCP tool boundary
  - Validate inputs (`query`, `report_shape`, optional `fetch_urls` defaulting to false, optional `hops` defaulting to 1, optional date filter fields) and reject unknown shape values with a structured `invalid_input` error before any search or LM call
  - Refuse to contact the crawl4ai endpoint when the caller did not pass `fetch_urls=true`, regardless of any other configuration
  - Translate orchestrator failures into `upstream_failure`, `invalid_input`, or `not_found` envelopes; on success return `{markdown, shape, used_hops, fetched_url_count}` with the markdown already sanitized by the renderer
  - Tests cover: invalid shape returns `invalid_input` and never reaches the orchestrator, missing `fetch_urls` defaults to `False`, response shape pins the four documented fields, empty corpus returns the documented `not_found`-style envelope without an LM call
  - _Requirements: 1.3, 4.7, 9.4, 10.1, 10.2, 10.3, 10.4_
  - _Boundary: tools.synthesize_likes_
  - _Depends: 5.1_

- [x] 5.3 Register the new tool with the MCP server
  - Add the input schema for `synthesize_likes` to the existing tool-definitions builder and route the dispatch path so a call with the tool name reaches the new tool entry
  - Tests cover: the running server advertises `synthesize_likes` in its tool list, dispatching a `call_tool` for `synthesize_likes` reaches the boundary function, dispatching with an invalid shape returns the `invalid_input` envelope produced by the boundary
  - _Requirements: 10.1_
  - _Boundary: server_
  - _Depends: 5.2_

- [x] 5.4 Add the CLI `--report` mode
  - Extend the existing argparse setup with `--report {brief,synthesis,trend}`, `--query`, `--out`, `--fetch-urls`, `--hops`, and `--report-optimize`, sharing `--limit`, `--year`, `--month-start`, and `--month-end` with the existing search mode
  - Build `ReportOptions` from the parsed args, call the orchestrator, and write the resulting markdown to `--out` or to stdout when `--out` is omitted
  - Wire `--report-optimize` to the compiled-program optimizer entry point so the operator can refresh the per-shape compiled program from the labeled-example set
  - Exit with status `0` on success and `2` on any failure (LM unreachable, crawl4ai unreachable while `--fetch-urls` was set, malformed shape, synthesis-validation error)
  - Tests cover: `--report brief --query "..." --out path.md` writes the file and exits 0, `--out` omitted prints to stdout, missing crawl4ai container with `--fetch-urls` exits 2 with a clear endpoint-and-env-var message, malformed shape exits 2 before any search, `--report-optimize` invokes the optimizer with sanitized + fenced demos
  - _Requirements: 1.1, 6.4, 9.1, 9.2, 9.3, 9.5_
  - _Boundary: __main___
  - _Depends: 3.2, 5.1_

- [ ] 6. Validation: end-to-end coverage of the integrated feature
- [x] 6.1 End-to-end orchestrator test with the autouse blockers in place
  - Drive `run_report` from a synthetic in-memory index across all three shapes with the FakeDspyLM and the URL-fetch blocker active
  - Assert that `hops=1` issues exactly one index search, `hops=2` issues exactly K parallel round-2 searches, and `fetch_urls=False` never instantiates the fetcher
  - Assert that the returned markdown is sanitized and that every claim cites a known source ID
  - _Requirements: 1.1, 1.4, 2.1, 2.2, 6.6, 7.5_
  - _Boundary: synthesis end-to-end_
  - _Depends: 5.1_

- [x] 6.2 End-to-end CLI test
  - Invoke the CLI with `--report synthesis --query "..." --out tmp.md` against a fixture index and assert that `tmp.md` exists, contains a `mermaid mindmap` fenced block, and the process exits 0
  - Invoke with `--fetch-urls` while the crawl4ai endpoint is unreachable and assert exit status 2 with an error message that names the endpoint and the override env var
  - _Requirements: 9.1, 9.5, 3.3_
  - _Boundary: __main__ end-to-end_
  - _Depends: 5.4_

- [x] 6.3 End-to-end MCP tool test
  - Call `synthesize_likes` through the MCP dispatcher with a valid shape and assert the response shape pins `{markdown, shape, used_hops, fetched_url_count}` and the markdown passes through sanitize
  - Call with an invalid shape and assert the dispatcher returns the `invalid_input` envelope without invoking the orchestrator
  - Call with `fetch_urls` omitted and assert the orchestrator runs with fetching disabled even if the crawl4ai endpoint is configured
  - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - _Boundary: server end-to-end_
  - _Depends: 5.3_
