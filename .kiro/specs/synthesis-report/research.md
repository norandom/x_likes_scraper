# Gap Analysis: synthesis-report

Generated against the requirements in `requirements.md`. Brownfield project; the
existing MCP server already covers the search and walker layers, so a substantial
share of the pipeline can extend rather than replace.

## 1. Existing assets

### Search and indexing

- **`x_likes_mcp/index.py`** — `TweetIndex.open_or_build(config, weights)` with a
  full hybrid `search(query, year, month_start, month_end, top_n=50)` method
  that already returns the ranked `ScoredHit` list the orchestrator needs for
  round-1 (and for each round-2 entity query). Author-affinity, recency-anchor,
  and the structured filter resolver (`_resolve_filter`, `_check_filter_deps`,
  `_parse_month_range`) are all reusable.
- **`x_likes_mcp/embeddings.py`** — `Embedder` with `embed_query` /
  `cosine_top_k`, on-disk corpus cache (model + tweet-id-set + schema versioned),
  retry classifier (`_classify_call_error`). New synthesis work does not touch
  this; it consumes whatever round-1 returns.
- **`x_likes_mcp/bm25.py`** — `BM25Index.top_k` with mask + slice helpers;
  unchanged.
- **`x_likes_mcp/fusion.py`** — `reciprocal_rank_fusion` already wired into
  `index.search`.

### Tool / dispatch / shaping

- **`x_likes_mcp/tools.py`** — `search_likes`, `read_tweet`, `list_months`,
  `get_month`. Ranking, hit-shaping, and `_build_status_url` (handle and id
  regex check, `i/status` fallback) already exist. The synthesis tool will live
  alongside them.
- **`x_likes_mcp/server.py`** — `_build_tool_definitions` declares the four
  current MCP tool schemas; `_dispatch` is the inner handler. Adding a fifth
  tool is a small additive change.
- **`x_likes_mcp/__main__.py`** — argparse-driven CLI entry. `--init` and
  `--search` modes already use a mutually exclusive group; a `--report` mode is
  the same shape.

### Sanitize / fence layer

- **`x_likes_mcp/sanitize.py`** — owns `sanitize_text`, `fence_for_llm`,
  `fence_url_for_llm`, `safe_http_url`, and the marker family
  (`LLM_FENCE_OPEN/CLOSE`, `URL_FENCE_OPEN/CLOSE`) plus
  `_neutralize_fence_markers`. The synthesis pipeline reuses every helper as-is
  and extends `_ALL_FENCES` with the four new families
  (`URL_BODY` / `ENTITY` / `KG_NODE` / `KG_EDGE`). Format codepoints are built
  from `chr(...)` so the source ASCII never carries literal bidi chars; the
  pattern stays.

### Walker / LM transport

- **`x_likes_mcp/walker.py`** — `_call_chat_completions` is the existing test
  seam; it builds a JSON-mode-or-not OpenAI call and parses one chunk at a
  time. The synthesizer is a different shape (one big call, structured output)
  so it gets its own module rather than reshaping the walker. The walker stays
  hand-written for now per the requirements.

### Config and env

- **`x_likes_mcp/config.py`** — `load_config` resolves shell env over `.env`,
  reads `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY` / `OPENROUTER_*`
  and the `RANKER_W_*` weights. New vars (`CRAWL4AI_BASE_URL` etc.) extend the
  same loader. Validation uses simple `_f` / `or None` patterns; copying that
  shape avoids inventing a new config style.

### Tests, lint, type discipline

- 369 tests pass. Test file naming mirrors the module
  (`tests/mcp/test_<module>.py`). Autouse fixtures in `tests/mcp/conftest.py`
  block real LLM and embedding calls — every new module that talks to a
  network endpoint adds a `_call_*` seam plus a fixture that fakes it.
- Pre-commit runs ruff + mypy. The synthesis modules must land already typed
  (`x_likes_mcp/` is the strict mypy scope).

## 2. Requirement-to-asset map

Tagging:
- ✅ Existing covers it
- ➕ Extend an existing module
- 🆕 New module/file required
- ❓ Research needed

| Req | Need | Status | Asset / new component |
|-----|------|--------|-----------------------|
| 1 Report orchestration | Top-level pipeline | 🆕 | New `x_likes_mcp/synthesis/orchestrator.py` |
| 1 Shape dispatch (`brief` / `synthesis` / `trend`) | Routing | 🆕 | `x_likes_mcp/synthesis/shapes.py` |
| 2 Multi-hop search | Round-2 fan-out | ➕ + 🆕 | Reuse `TweetIndex.search`; new `synthesis/multihop.py` builds round-2 queries |
| 2 Date-filter passthrough | Filter args | ✅ | `index._resolve_filter` already validates and resolves |
| 3 Container-backed URL fetch | crawl4ai HTTP client | 🆕 | New `x_likes_mcp/synthesis/fetcher.py` |
| 3 PDF/Office to markdown | markitdown | 🆕 + ❓ | New `synthesis/markitdown_adapter.py`; ❓ design decides in-container vs host subprocess |
| 3 Sanitize on the way back | Reuse | ✅ | `sanitize.sanitize_text` |
| 4 SSRF defenses | Hostname resolution + IP blocklist | 🆕 | New `synthesis/ssrf_guard.py`; ❓ research stdlib `ipaddress` + `socket.getaddrinfo` for pre-connect resolve, plus DNS rebinding mitigation |
| 4 HTTPS-only / scheme allowlist | Reuse | ✅ | `sanitize.safe_http_url` already blocks non-HTTP(S) |
| 4 Redirect re-validation | Custom httpx transport | 🆕 + ❓ | Build on httpx; ❓ research whether httpx's redirect hook is enough or if we need a manual loop |
| 5 Entity extraction (regex pass) | Tokenizer + counter | 🆕 | `synthesis/entities.py`; reuse `bm25.tokenize` for the cheap path |
| 5 DSPy entity fallback | DSPy module | 🆕 + ❓ | New `synthesis/dspy_modules.py`; ❓ DSPy version pin |
| 5 In-memory KG | Dict-of-dicts | 🆕 | `synthesis/kg.py`; no `networkx` dep needed |
| 6 DSPy synthesis modules | DSPy ChainOfThought | 🆕 + ❓ | `synthesis/dspy_modules.py`; ❓ DSPy structured output (native vs Pydantic) |
| 6 Compiled-program load/save | Disk I/O | 🆕 | `synthesis/compiled.py` |
| 6 BootstrapFewShot optimizer | DSPy optimizer | 🆕 + ❓ | Same module; ❓ optimizer API stability |
| 6 dspy.Assert source validator | DSPy assertion | 🆕 | Wired in `synthesis/dspy_modules.py` |
| 7 Six fence families | Marker neutralization | ➕ | Extend `sanitize._ALL_FENCES`; add `URL_BODY`, `ENTITY`, `KG_NODE`, `KG_EDGE` |
| 7 Synthesis context assembly | Build fenced blob | 🆕 | `synthesis/context.py` (caps, fencing, ordering) |
| 7 Output sanitization on return | Reuse | ✅ | `sanitize.sanitize_text` |
| 8 Mermaid mindmap render | String builder | 🆕 + ❓ | `synthesis/mindmap.py`; ❓ depth cap and how to label nodes safely (no fence markers leak into the diagram) |
| 8 Month-bucketed `trend` | `created_at` parsing | ➕ | Reuse `Tweet.get_created_datetime` |
| 8 `tweet_url` in report | Reuse | ✅ | `tools._build_status_url` |
| 9 CLI surface | New mode in `__main__` | ➕ | Extend the `argparse` mutually-exclusive group with `--report`; reuse the existing filter args |
| 10 MCP `synthesize_likes` | New tool | ➕ | Register in `server._build_tool_definitions`, dispatch in `_dispatch`; logic in `tools.synthesize_likes` |
| 11 URL cache | sha256-keyed JSON | 🆕 | `synthesis/url_cache.py` |
| 12 Config | Env vars | ➕ | Extend `Config` and `load_config` with `CRAWL4AI_BASE_URL`, `URL_CACHE_DIR`, `URL_CACHE_TTL_DAYS`, `SYNTHESIS_MAX_HOPS`, etc. |

## 3. Implementation approach options

### Option A: Pure extension of `x_likes_mcp/`

Drop everything into the existing flat layout: `x_likes_mcp/synthesis_*.py`
files alongside `bm25.py`, `embeddings.py`, etc.

- ✅ Matches the current "one module per concern, all flat" layout.
- ✅ No import path changes; mypy/ruff config covers it for free.
- ❌ Adds ~10 new files to the flat namespace, harder to scan.
- ❌ Couples synthesis lifecycle to the MCP server module (harder to ship as a
  separate library later).

### Option B: New `x_likes_mcp/synthesis/` subpackage

Dedicated subpackage with `orchestrator.py`, `multihop.py`, `fetcher.py`,
`ssrf_guard.py`, `entities.py`, `kg.py`, `dspy_modules.py`, `mindmap.py`,
`context.py`, `compiled.py`, `url_cache.py`, `shapes.py`, `markitdown_adapter.py`.
The existing `tools.py` exposes a thin `synthesize_likes` wrapper that calls the
subpackage.

- ✅ Clean separation; the synthesis pipeline is one package the user can
  reason about end-to-end.
- ✅ Easier to test in isolation (subpackage-level conftest + fixtures).
- ✅ Keeps the flat top-level small (`bm25`, `embeddings`, `index`, `tools`,
  `server`, `walker` stay easy to scan).
- ❌ More files to create. Roughly 12 new modules + their tests.
- ❌ Have to remember the import path stays `x_likes_mcp.synthesis.foo`.

### Option C: Hybrid — subpackage + small extensions

Same as B for the bulk of the work, plus three small in-place edits:

1. `sanitize.py` gains `URL_BODY` / `ENTITY` / `KG_NODE` / `KG_EDGE` marker
   constants and `fence_url_body_for_llm` / `fence_entity_for_llm` helpers.
2. `tools.py` gains `synthesize_likes(...)` next to `search_likes`,
   `read_tweet`, etc.
3. `server.py` registers the new MCP tool in `_build_tool_definitions` and
   `_dispatch`.
4. `__main__.py` adds `--report` to the existing mutually exclusive mode group
   and a `_run_report` helper alongside `_run_search` / `_print_init_summary`.
5. `config.py` gains the new env var fields and defaults.

The subpackage owns everything novel; the existing modules pick up the new
seams without changing their shape.

- ✅ Best fit for this codebase. Minimal disruption to existing modules; new
  surface area lives in one place.
- ✅ Each small extension is a 5-20 line edit that reviewers can read at a
  glance.
- ❌ Reviewers have to look in two places (subpackage + the small edits).

## 4. Effort and risk

| Area | Effort | Risk | Note |
|------|--------|------|------|
| Subpackage scaffolding + shapes router (Req 1) | S | Low | Pattern matches existing `tools.py` style |
| Multi-hop orchestration + dedupe (Req 2) | S | Low | `index.search` already does the heavy lifting |
| crawl4ai HTTP client + sanitize wiring (Req 3) | M | Medium | New container dependency; image + endpoint contract must be pinned |
| markitdown adapter + content-type routing (Req 3) | S | Medium | markitdown supports many formats; PDF size cap and encoding edge cases need test coverage |
| SSRF guard (Req 4) | M | High | Pre-connect IP resolve + DNS rebinding mitigation is the most security-sensitive piece in the spec; needs careful test cases against private ranges |
| Entity extraction regex + DSPy fallback (Req 5) | S | Low | Reuses `bm25.tokenize`; DSPy fallback is one signature |
| In-memory KG (Req 5) | S | Low | Plain dict-of-dicts |
| DSPy synthesis modules + compiled-program load/save (Req 6) | M | Medium | New library, structured-output mechanism needs to be picked once and stuck to; LM endpoint already configured |
| Fence family extension + assembly (Req 7) | S | Low | Existing `_neutralize_fence_markers` already handles a marker family; add four constants |
| Mindmap renderer (Req 8) | S | Low | Pure string builder; depth cap is local |
| CLI surface (Req 9) | S | Low | Same shape as the existing `--search` extension |
| MCP tool surface (Req 10) | S | Low | Schema entry + dispatch arm |
| URL cache (Req 11) | S | Low | sha256 + JSON file; mtime check for TTL |
| Config (Req 12) | S | Low | Mirrors existing `load_config` pattern |
| **Total (Option C, full scope)** | **~M-L** (5-9 days) | **Medium** | SSRF guard and DSPy structured output are the two pacing items |

## 5. Research items to carry into design

- **DSPy version pin** and structured-output mechanism (native DSPy output
  schemas vs Pydantic attached to the signature). Pick one; consistency across
  all `Synthesize*` signatures matters for the validator code path.
- **DSPy assertions API**: confirm `dspy.Assert` (or its current equivalent)
  supports retry-with-feedback and the failure mode the orchestrator needs
  when a claim cites an unknown source.
- **crawl4ai container contract**: pin the request shape (`POST /crawl`
  payload) and the response shape (where the markdown lives in the JSON) for
  the version we target. Document the exact image tag in the README.
- **markitdown placement**: bundle into the crawl4ai image (one container,
  one network surface) vs run as a host subprocess (no extra container, but
  the host gets the file bytes). Both are plausible; design phase decides
  based on packaging cost.
- **SSRF resolver**: confirm `socket.getaddrinfo` returns enough information
  to apply the IP blocklist before httpx opens a connection, and confirm
  httpx's redirect-disabled-then-manual-loop pattern is the right way to
  re-validate after each redirect (vs. using its event hooks).
- **DNS rebinding**: decide whether the resolver pins the IP for the duration
  of the fetch (resolve once, then connect to the IP, with `Host:` header
  preserved) or accepts a small TOCTOU window. Pin is safer.
- **Mermaid mindmap depth**: confirm GitHub's renderer (and Obsidian's, and
  VS Code preview's) tolerate 4 levels reliably; the design phase picks the
  cap.
- **Optimizer cost**: estimate the LM cost of one `BootstrapFewShot` run on
  this corpus (5-10 demos, 3-5 trials) for the default OpenRouter model so
  the README can warn the user before they invoke `--report-optimize`.
- **Output schema retry policy**: when a claim cites an unknown source,
  retry-with-feedback (one retry) vs. hard-fail. Design phase pins the policy
  and the user-visible error.
- **Per-source byte caps**: pick concrete numbers for URL body, tweet body,
  and entity list that fit the smallest supported model context window. The
  walker's existing `_PROMPT_TEXT_MAX_CHARS = 280` is a reference point but
  too small for synthesis.

## 6. Recommendation for design phase

- **Approach**: Option C (subpackage + small in-place edits to `sanitize.py`,
  `tools.py`, `server.py`, `__main__.py`, `config.py`).
- **Suggested module layout**:
  ```
  x_likes_mcp/synthesis/
    __init__.py
    orchestrator.py       # top-level run_report
    shapes.py             # brief / synthesis / trend dispatch
    multihop.py           # round-2 fan-out
    entities.py           # regex extraction + DSPy fallback wiring
    kg.py                 # in-memory knowledge graph
    fetcher.py            # crawl4ai HTTP client
    ssrf_guard.py         # pre-connect IP resolve + redirect re-validation
    markitdown_adapter.py # PDF/Office -> markdown
    url_cache.py          # sha256-keyed disk cache
    dspy_modules.py       # signatures + ChainOfThought + Asserts
    compiled.py           # load/save compiled DSPy programs
    context.py            # build the fenced synthesis context
    mindmap.py            # mermaid mindmap renderer
    report_render.py      # final markdown assembly per shape
  ```
- **Test layout**: `tests/mcp/synthesis/test_<module>.py` mirroring source.
  Reuse the existing `_block_real_llm` / `_block_real_embeddings` autouse
  fixtures and add `_block_real_url_fetch` for the new network seam.
- **Sequencing**: SSRF guard and DSPy structured output get designed first
  (they pace everything else). The crawl4ai client and the synthesis context
  builder consume those decisions.

---

## Discovery Findings (design phase)

Light-discovery process executed; most external dependency choices were already
fixed in `requirements.md`. The findings recorded here are the design-phase
synthesis: each open item from earlier in this file gets a decision and a brief
rationale.

### Generalizations identified

- All three report shapes (`brief`, `synthesis`, `trend`) share the same
  upstream pipeline (search → fetch → entity → KG → fenced context). Only the
  DSPy signature, the per-shape Pydantic output type, and the renderer differ.
  Generalization: one `run_report(options)` orchestrator + per-shape
  signature / renderer; no duplicated pipeline code.
- Entity extraction (regex first, DSPy fallback for empty hits) and the
  walker explainer share the "cheap path first, LM fallback" pattern. We do
  not generalize across them in v1; the walker stays hand-written. The shared
  pattern is informative, not a forced abstraction.

### Build-vs-adopt decisions

- **Adopt**: `crawl4ai` (HTML, container), `markitdown` (PDF / Office,
  host-side subprocess), `dspy-ai` (synthesis prompt as signature),
  `litellm` (LM transport, transitive via DSPy), `pydantic` (structured
  output type).
- **Build**: `ssrf_guard` (no Python lib covers RFC1918 + IPv6 ULA + cloud
  metadata + post-redirect re-validation cleanly), `url_cache` (sha256-keyed
  JSON files, simpler than any cache lib for this shape), in-memory KG
  (dict-of-dicts, no `networkx` dependency). The walker stays hand-written;
  migrating it to DSPy is a follow-up.

### Simplifications applied

- KG is a flat dict-of-dicts with namespaced IDs (`tweet:<id>`,
  `handle:<screen_name>`). No `networkx`, no graph DB, no triples.
- One LM endpoint reused (`OPENAI_BASE_URL` / `OPENAI_MODEL`) instead of a
  dedicated synthesizer config. A separate `SYNTHESIS_*` env pair becomes a
  revalidation trigger if a future model needs more context window.
- Pydantic models for DSPy structured output (consistent across all three
  signatures) instead of DSPy native schemas (extra mental model, less
  familiar).
- Markitdown placement: host-side subprocess (no extra container build,
  bytes already came back through the crawl4ai container's HTTP path).
- Mindmap depth cap at 4 levels (legible in GitHub / Obsidian / VS Code).
- One corrective retry on a hallucinated source, then hard-fail (vs.
  multi-step refinement).

### Decisions on each open item from the original list

| Item | Decision | Rationale |
|------|----------|-----------|
| One synthesis LLM endpoint vs. separate config | Reuse walker's `OPENAI_BASE_URL` / `OPENAI_MODEL` | Avoid premature config split; revalidation trigger if a future synthesis model needs different limits |
| Chunk synthesis vs. full context | Full context with budget-driven dropping (lowest-rank URL bodies first) | Single LM call is simpler; budget keeps it bounded |
| Cache-key shape | sha256(url) | Simplest correct key; the cache value records `final_url` for redirect tracking |
| Mindmap depth cap | 4 levels | Legible across GitHub / Obsidian / VS Code |
| crawl4ai container deployment | Documented `docker run` in README; no compose file in v1 | Operator runs the container; we document the command |
| markitdown placement | Host-side subprocess | No extra container; bytes already came through crawl4ai |
| Docker network shape | Custom bridge default; egress proxy as recommended hardening with documented threat model | We do not ship the proxy; we document the gap |
| Synthesis fencing budget | tweet=280, url_body=4096, entity=64, kg_label=64, total=32768 | Fits any 8K-token model with margin |
| Structured-output schema | Pydantic models (`Claim`, `Section`, `MonthSummary`) attached to DSPy signature | Consistent with project's `pydantic>=2` runtime |
| Validation failure policy | One corrective retry via `dspy.Assert`, then hard-fail | Bounds cost; surfaces real problems |
| DSPy module per stage | `Predict` for entity fallback, `ChainOfThought` for synthesis | Reasoning helps narrative coherence; entity fallback is a simple classification |
| DSPy structured output mechanism | Pydantic | Same reason as the schema decision above |
| Optimizer choice | `BootstrapFewShot` | Simplest demo-only optimizer; MIPROv2 is a follow-up if needed |
| Demo storage | Gitignored `output/synthesis_compiled/{shape}.json` | Demo content includes tweet text; rebuild locally |
| Walker migration to DSPy | Out of scope; walker stays hand-written | Separate spec; reduces this spec's blast radius |
| SSRF resolver | Resolve once via `getaddrinfo`, pin IP, send original `Host:` header; manual redirect loop with re-validation per hop | Mitigates DNS rebinding; closes the redirect-rebinding gap |
| DNS rebinding | Pin policy as above | Captured in `ssrf_guard.resolve_and_check` |
| Per-source byte caps | tweet=280 (matches existing walker `_PROMPT_TEXT_MAX_CHARS`), url_body=4096, entity=64, kg_label=64 | Conservative; designs phase confirmed total=32768 fits 8K-token contexts |
| Optimizer cost estimate | ~$0.10-1 per run on the default OpenRouter model with 5-10 demos and a small trial budget | README will warn before pointing users at `--report-optimize` |

### Carried forward into implementation

- Verify the targeted `unclecode/crawl4ai` image tag's exact request and
  response shapes against the design contract; capture the verified version
  in `Modified Files: README.md` during the implementation phase.
- Pin the `dspy-ai` minor version after confirming `dspy.Assert` and
  Pydantic-typed outputs work as documented.
- Confirm `httpx` 0.27 supports the manual-redirect-loop pattern with
  `follow_redirects=False` plus `Response.next_request`.

---

## Validate-design pass 1 repairs

Three critical issues raised by `/kiro-validate-design` plus one user clarification (zero-trust networks) led to the following design.md and requirements.md edits.

### Repair 1: PDF flow simplified to crawl4ai native

User clarified: PDFs only, Office formats irrelevant. crawl4ai handles PDF
extraction natively via the same `markdown` response field used for HTML.

- **Requirements 3.5** rewritten to state PDFs go through crawl4ai's native
  extraction; Office types out of scope for v1.
- **Requirements Project Description (step 2)** updated: "HTML / static pages
  and PDFs" via crawl4ai; Office formats moved to "out of scope for v1".
- **Requirements 4.6** content-type allowlist: dropped Office types.
- **Design**: `markitdown_adapter` component removed (file, summary row,
  detail block, Tech Stack row, Modified Files dep, architecture-diagram
  edge). `fetcher` now uses crawl4ai's `markdown` field uniformly for HTML
  and PDF. Office formats are explicitly dropped at the content-type
  allowlist; a future spec can add a converter if needed.

### Repair 2: Round-2 K and executor cap pinned

- `multihop` gets a detail block with `K=5`, `max_workers=4`, and an
  entity-ranking weight formula `count(occurrences) * affinity_boost(NodeKind)`.
  Operator overrides via `SYNTHESIS_ROUND_TWO_K` env var.

### Repair 3: DSPy test seam pinned

- `dspy_modules` adds a "Test seam" subsection: `FakeDspyLM(dspy.LM)` lives
  in `tests/mcp/synthesis/conftest.py`; an autouse `_stub_dspy_lm` fixture
  installs it via `dspy.configure(lm=FakeDspyLM(...))` for every test in
  the package; tests opt into a real LM with `@pytest.mark.real_lm`
  collected only when `--run-real-lm` is passed.

### Zero-trust network accommodation (CIDR allowlist)

User raised: zero-trust meshes with TLS-offload sidecars give services
RFC1918 IPs (e.g. `10.100.x.x`); a strict RFC1918 block refuses legitimate
destinations. Resolution:

- **Two-tier blocklist** in `ssrf_guard`:
  - **Unconditional tier**: loopback, cloud metadata IPs (169.254.169.254 +
    per-cloud equivalents), broadcast, multicast, IANA-reserved. Cannot be
    overridden.
  - **Private tier**: RFC1918, IPv6 ULA, IPv6 link-local, IPv4 link-local
    minus the metadata IP. Each candidate is checked against the operator's
    `URL_FETCH_ALLOWED_PRIVATE_CIDRS` (comma-separated CIDRs); allowlisted
    addresses pass.
- **Default**: empty allowlist preserves the strict default.
- **Requirements 4.2 split** into two criteria:
  - 4.2 (unconditional block, cannot be overridden)
  - 4.3 (private block, with operator allowlist) — new criterion
  - Existing 4.3-4.6 renumbered to 4.4-4.7.
- **Requirements 12.3** lists the new env var.
- **Design `ssrf_guard` block** documents the two-tier model, the
  `parse_allowlist` / `is_unconditional_blocked` / `is_blocked_address` API,
  and a "Threat-model commitment" subsection that explicitly states what
  the default does and does not cover.
- **Threat model commitment**: cloud metadata exfiltration and loopback
  reach are blocked unconditionally and cannot be misconfigured. Private
  range reach is blocked by default, opt-in per CIDR. Operators with
  zero-trust infrastructure document their internal CIDRs in `.env`; the
  loud-on-bad-input parser fails at config load if the CIDR is malformed.
