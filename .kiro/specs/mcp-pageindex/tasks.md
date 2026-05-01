# Implementation Plan

## 1. Foundation: dependencies, env sample, and the formatter prerequisite

- [x] 1.1 Add the `omit_global_header` parameter to `MarkdownFormatter.export` and route the per-month branch through it
  - In `x_likes_exporter/formatters.py`, add `omit_global_header: bool = False` as a keyword parameter to `MarkdownFormatter.export`
  - Wrap the four `md_lines.append(...)` calls that emit the global h1, the `**Exported:**` timestamp, the `**Total Tweets:**` line, and the trailing `---` inside `if not omit_global_header:`
  - In `x_likes_exporter/exporter.py`, the per-month branch of `XLikesExporter.export_markdown` (the `for year_month in sorted(...)` loop, the `formatter.export(...)` call inside it) passes `omit_global_header=True`
  - The non-split-by-month branch (`formatter.export(self.tweets, str(output_file), include_media=include_media)` near the end) is left unchanged so single-file output is byte-identical to before
  - Observable completion: calling `MarkdownFormatter().export(tweets, "out.md")` (default args) produces a file containing `# X (Twitter) Liked Tweets` and `**Exported:**`; calling `MarkdownFormatter().export(tweets, "out.md", omit_global_header=True)` produces a file where neither string appears, while `## YYYY-MM` and the per-tweet blocks remain
  - _Requirements: 3.1_
  - _Boundary: x_likes_exporter/formatters.py, x_likes_exporter/exporter.py_

- [x] 1.2 Update `tests/test_formatters.py` to cover both shapes of the new parameter
  - Add a test that asserts `# X (Twitter) Liked Tweets`, `**Exported:**`, and `**Total Tweets:**` are present when `MarkdownFormatter().export(...)` is called with default arguments
  - Add a test that asserts those three strings are absent when called with `omit_global_header=True`, while `## 2025-03`, `## 2025-01`, the per-tweet `[@alice]` block, and the stats line all remain present in the output
  - Existing `test_markdown_formatter_basic` and `test_markdown_formatter_unknown_routing` continue to assert default-argument behavior and continue to pass without modification
  - Observable completion: `pytest tests/test_formatters.py -v` shows green; the two new tests both pass; the suite count grows by exactly two
  - _Requirements: 3.1_
  - _Depends: 1.1_
  - _Boundary: tests/test_formatters.py_

- [ ] 1.3 Add MCP runtime deps and console script entry to `pyproject.toml`
  - Append `mcp>=1.0,<2.0` and `pageindex` (with whatever version pin is current at impl time) to `[project.dependencies]`
  - Add `x-likes-mcp = "x_likes_mcp.__main__:main"` under `[project.scripts]`
  - Extend `[tool.hatch.build.targets.wheel].packages` to include `x_likes_mcp`
  - Run `uv sync` and confirm both new deps resolve and install
  - Observable completion: `uv pip show mcp pageindex` lists both with version numbers; importing both from a Python REPL succeeds without errors
  - _Requirements: 1.4, 9.6_
  - _Boundary: pyproject.toml_

- [ ] 1.4 Create the `x_likes_mcp/` package skeleton
  - Create `x_likes_mcp/__init__.py` defining `__version__ = "0.1.0"` and a one-line module docstring
  - Create empty (placeholder) module files: `config.py`, `errors.py`, `index.py`, `tools.py`, `server.py`, `__main__.py`
  - `__main__.py` contains `def main() -> int: return 0` and the `if __name__ == "__main__": sys.exit(main())` guard so the module is runnable
  - Observable completion: `python -m x_likes_mcp` exits with code 0 and produces no output; `from x_likes_mcp import __version__` returns `"0.1.0"`
  - _Requirements: 1.1, 1.2_
  - _Boundary: x_likes_mcp/_

- [ ] 1.5 Extend `.env.sample` with the three new Anthropic-compatible variables
  - Append `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_OPUS_MODEL` entries with comments stating the endpoint is local Anthropic-compatible by default and that PageIndex routes through LiteLLM with model string `anthropic/<model_name>`
  - Default value for `ANTHROPIC_BASE_URL` is `http://localhost:8080`; default for `ANTHROPIC_AUTH_TOKEN` is empty; default for `ANTHROPIC_DEFAULT_OPUS_MODEL` is a placeholder model name (e.g. `claude-opus-4-5`)
  - Observable completion: `grep ANTHROPIC_BASE_URL .env.sample`, `grep ANTHROPIC_AUTH_TOKEN .env.sample`, and `grep ANTHROPIC_DEFAULT_OPUS_MODEL .env.sample` all match
  - _Requirements: 2.5, 2.4_
  - _Boundary: .env.sample_

## 2. Configuration and error layers

- [ ] 2.1 (P) Implement `config.py` with `.env` reader and `Config` dataclass
  - Define frozen `Config` dataclass with the fields documented in design.md (`output_dir`, `by_month_dir`, `likes_json`, `cache_path`, `anthropic_base_url`, `anthropic_auth_token`, `anthropic_model`, `litellm_model_string`)
  - Implement a stdlib `.env` reader (split on `=`, strip comments and whitespace, no shell-quote handling)
  - Implement `load_config(env_path=None, env=None)` that reads `.env` from cwd by default, falls back to `os.environ`, and validates that `ANTHROPIC_BASE_URL` and `ANTHROPIC_DEFAULT_OPUS_MODEL` are set and non-empty
  - Raise `ConfigError` naming the missing variable when validation fails
  - `litellm_model_string` is `f"anthropic/{anthropic_model}"`, computed once
  - Default `OUTPUT_DIR` to `"output"`; derive `by_month_dir`, `likes_json`, `cache_path` from `output_dir`
  - Side effect: write `ANTHROPIC_BASE_URL` (and `ANTHROPIC_AUTH_TOKEN` if set) into `os.environ` before returning so LiteLLM picks them up at PageIndex call time. If LiteLLM, at impl time, expects a different env var name internally for the auth token, set both names
  - Observable completion: importing `load_config` and calling it against an in-memory `env` dict returns a populated `Config` whose `litellm_model_string` starts with `"anthropic/"`; calling without `ANTHROPIC_BASE_URL` raises `ConfigError` whose message contains the string `ANTHROPIC_BASE_URL`; after a successful call, `os.environ["ANTHROPIC_BASE_URL"]` matches the configured value
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 11.1_
  - _Boundary: x_likes_mcp/config.py_

- [ ] 2.2 (P) Implement `errors.py` with `ToolError` and category helpers
  - Define `ToolError(Exception)` carrying `category: str` and `message: str` attributes
  - Implement `invalid_input(field, message)`, `not_found(what, identifier)`, `upstream_failure(detail)` factory functions returning `ToolError` instances with the right category strings
  - Observable completion: `errors.invalid_input("query", "must be non-empty").category == "invalid_input"`; the same for `not_found` and `upstream_failure`; `str(err)` includes the field name and the message
  - _Requirements: 4.3, 4.4, 6.2, 6.3, 7.3, 7.4, 11.2, 11.4_
  - _Boundary: x_likes_mcp/errors.py_

## 3. Indexing layer

- [ ] 3.1 Implement `Index` data containers and the build-or-load skeleton
  - Define `SearchHit` and `MonthInfo` frozen dataclasses with the fields documented in design.md
  - Define `IndexError(Exception)`
  - Implement `Index.__init__` accepting the tree, side-table, tweet map, paths-by-month map, and config; mark instance attributes as read-only by convention
  - Implement `Index.open_or_build(config)` that enumerates files via `iter_monthly_markdown(config.by_month_dir)`, raises `IndexError("output/by_month/ is empty or missing")` when no files yield
  - Compute `newest_md_mtime` and the cache freshness check; load cached `(tree, side_table)` via `pickle.load` on hit; call `_build_tree(...)` on miss; write the cache atomically (`.tmp` + `os.replace`)
  - Call `load_export(config.likes_json)` and store as `dict[str, Tweet]` keyed on `tweet.id`; also retain the original `list[Tweet]` for `list_months` counts
  - Build a `paths_by_month: dict[str, Path]` map by parsing `YYYY-MM` from each `likes_YYYY-MM.md` filename; this is what `list_months` and `get_month_markdown` consult
  - Leave `_build_tree` and `_query` as stubs that raise `NotImplementedError`; Tasks 3.2 and 3.3 fill them in
  - Observable completion: with `_build_tree` monkeypatched to return `(sentinel_tree, {})`, `Index.open_or_build` against the fixture export returns an `Index` whose internal tweet map has the expected number of entries and whose `paths_by_month` maps each fixture month to its file; the cache file appears under `config.cache_path`
  - _Requirements: 3.1, 3.4, 3.5, 7.2, 8.5_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.2 Implement the PageIndex tree builder seam (`_build_tree`)
  - Implement `_build_tree(paths, model_string)` calling PageIndex's tree-build entry point with the file paths and the LiteLLM-style model string `anthropic/<model>`; return `(tree, side_table)`
  - Build the side-table by walking the resulting tree's leaf sections and matching their heading text against the in-memory `Tweet` list to record `(node_key -> tweet_id)`
  - When a leaf cannot be matched to a tweet, skip it (do not raise); log a single line to stderr
  - Observable completion: against the fixture export with PageIndex's tree-build mocked to return a known-shape fake tree, `_build_tree` returns a tuple where the side-table maps at least one leaf to one of the fixture tweet IDs; the exact `model_string` passed to PageIndex is `"anthropic/<the configured model>"`
  - _Requirements: 3.1, 4.1, 4.2, 8.4_
  - _Depends: 3.1_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.3 Implement the structured-filter resolver and the query seam (`_query`, `_resolve_filter`, `search`)
  - Implement `_resolve_filter(year, month_start, month_end) -> list[str] | None` enforcing the filter rules from design.md (`year` required if `month_start` set; `month_start` required if `month_end` set; `month_start <= month_end`; year-only spans the whole year; year + single month means that month)
  - Validation errors raise `ValueError` with a clear message; `tools.search_likes` translates that into `errors.invalid_input("filter", ...)`
  - Implement `_query(tree_or_subtree, query)` calling PageIndex's query entry point and returning a list of matches
  - Implement `search(query, year=None, month_start=None, month_end=None)` that calls `_resolve_filter`, narrows the cached tree to the in-range subtree (or assembles an ad-hoc tree from per-month subtrees, whichever PageIndex's API supports — choose at impl time), calls `_query`, and maps each match through the side-table to a `SearchHit`
  - When the filter is `None`, `search` passes the full tree
  - Returns `[]` when there are no matches
  - Observable completion: with PageIndex's query mocked, `Index.search("anything")` returns a list of `SearchHit` with the expected `tweet_id`/`year_month`/`handle`; `Index.search("anything", year=2025, month_start="01", month_end="02")` results in `_query` being called with a tree limited to the two months (assert by inspecting the captured argument or by spying on the resolved month list); `_resolve_filter(year=None, month_start="01", month_end=None)` raises `ValueError`
  - _Requirements: 3.2, 3.3, 4.1, 4.2, 4.5_
  - _Depends: 3.1, 3.2_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.4 (P) Implement `Index.lookup_tweet`, `list_months`, `get_month_markdown`
  - `lookup_tweet(tweet_id)` returns `self._tweets.get(tweet_id)` (`None` when missing)
  - `list_months()` consults `paths_by_month`, parses `YYYY-MM` from each filename, groups the in-memory tweet list by `Tweet.get_created_datetime()` to produce counts (tweets with unparseable `created_at` are skipped from counts), returns `MonthInfo` list sorted reverse-chronologically
  - `get_month_markdown(year_month)` reads `by_month_dir / f"likes_{year_month}.md"` if the file exists; returns `None` otherwise (the tools layer translates that into a not-found error)
  - Observable completion: against the fixture export, `Index.lookup_tweet` returns the right `Tweet` for a known fixture ID and `None` for `"missing"`; `Index.list_months` returns the fixture months in reverse order with correct counts; `Index.get_month_markdown("2025-01")` returns the file content; `Index.get_month_markdown("2099-12")` returns `None`
  - _Requirements: 5.1, 5.2, 5.3, 7.1, 7.3_
  - _Depends: 3.1_
  - _Boundary: x_likes_mcp/index.py_

## 4. Tool handlers

- [ ] 4.1 (P) Implement `tools.search_likes` and `tools.list_months`
  - `search_likes(index, query, year=None, month_start=None, month_end=None)` strips the query, raises `errors.invalid_input("query", ...)` when empty
  - Validates filter shape (`year` is integer in valid range, `month_start`/`month_end` match `^(0[1-9]|1[0-2])$`); shape errors raise `errors.invalid_input("filter", ...)`
  - Calls `index.search(query, year, month_start, month_end)`; catches `ValueError` from the resolver and re-raises as `errors.invalid_input("filter", ...)`
  - Catches any non-`ToolError` exception from `index.search` and re-raises as `errors.upstream_failure(...)` (the LLM-failure path)
  - Returns a list of dicts shaped `{"tweet_id", "year_month", "handle", "snippet"}`
  - `list_months(index)` returns a list of dicts shaped `{"year_month", "path", "tweet_count"}`; `tweet_count` may be `None`; ordering is whatever `Index.list_months` produced
  - Observable completion: with a mocked `Index`, `tools.search_likes(index, "  ")` raises `ToolError(category="invalid_input")`; `tools.search_likes(index, "x", year=2025, month_start="01", month_end="02")` calls `index.search` with exactly those four positional/keyword arguments; `tools.search_likes(index, "x", year=None, month_start="01")` raises `invalid_input`; `index.search` raising `RuntimeError("LLM down")` becomes `ToolError(category="upstream_failure")`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3_
  - _Depends: 2.2, 3.3, 3.4_
  - _Boundary: x_likes_mcp/tools.py_

- [ ] 4.2 (P) Implement `tools.get_month` and `tools.read_tweet`
  - `get_month(index, year_month)` validates `year_month` against `^\d{4}-\d{2}$`; raises `errors.invalid_input("year_month", ...)` when the pattern fails; calls `index.get_month_markdown`; raises `errors.not_found("month", year_month)` when the result is `None`; otherwise returns the Markdown string
  - `read_tweet(index, tweet_id)` validates `tweet_id` is non-empty and matches `^\d+$`; raises `errors.invalid_input("tweet_id", ...)` otherwise; calls `index.lookup_tweet`; raises `errors.not_found("tweet", tweet_id)` when missing; otherwise returns a dict with `tweet_id`, `handle`, `display_name`, `text`, `created_at`, `view_count`, `like_count`, `retweet_count`, `url` (omitting fields the source `Tweet` does not have)
  - Observable completion: with a mocked `Index`, `tools.get_month(index, "2025/01")` raises `invalid_input`; `tools.get_month(index, "2099-12")` (where `Index.get_month_markdown` returns `None`) raises `not_found`; `tools.read_tweet(index, "")` raises `invalid_input`; `tools.read_tweet(index, "abc")` raises `invalid_input`; `tools.read_tweet(index, "999")` against an `Index` that returns `None` raises `not_found`
  - _Requirements: 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 7.4_
  - _Depends: 2.2, 3.4_
  - _Boundary: x_likes_mcp/tools.py_

## 5. MCP server wiring

- [ ] 5.1 Implement `server.build_server(index)` with tool registration and JSON schemas
  - Construct an MCP `Server` instance with name `"x-likes-mcp"` and `version=__version__`
  - Register the four tools using the SDK's tool-registration API; declare input/output JSON schemas inline
  - `search_likes` schema: `query` required string, `year` optional integer with min 2006 and max equal to current year, `month_start` and `month_end` optional strings with `pattern: "^(0[1-9]|1[0-2])$"`; output is array of `{tweet_id, year_month, handle, snippet}`
  - `list_months` schema: empty input; output is array of `{year_month, path, tweet_count}`
  - `get_month` schema: `year_month` required string with `pattern: "^\\d{4}-\\d{2}$"`; output is string
  - `read_tweet` schema: `tweet_id` required string with `pattern: "^\\d+$"`; output is the documented metadata object
  - Implement the boundary error wrapper: `ToolError` becomes an MCP error response with `category` and `message`; other exceptions become a stderr log plus a generic `upstream_failure` response
  - Implement `server.run(index)` calling the SDK's stdio entry point; returns when the client disconnects
  - Observable completion: a unit test that calls `build_server(index)` and inspects the registered tool list finds exactly four tools with names `search_likes`, `list_months`, `get_month`, `read_tweet` and non-empty input schemas; the `search_likes` schema includes the `year` / `month_start` / `month_end` fields with the documented patterns
  - _Requirements: 1.1, 1.3, 4.4, 4.6, 5.4, 6.4, 7.5, 11.2_
  - _Depends: 4.1, 4.2_
  - _Boundary: x_likes_mcp/server.py_

- [ ] 5.2 Implement `__main__.main()` startup pipeline
  - Replace the placeholder `main()` with the real pipeline: `load_config()` → `Index.open_or_build(config)` → `server.run(index)` → return 0
  - Catch `ConfigError`, `IndexError`, `FileNotFoundError` at the top of `main`; print one stderr line naming the failing condition; return exit code 2
  - Other exceptions during startup propagate (intentional: real bugs surface as tracebacks)
  - Observable completion: with a fixture export and a mocked LLM, running `python -m x_likes_mcp` against a temp `.env` (`ANTHROPIC_BASE_URL=http://fake`, `ANTHROPIC_DEFAULT_OPUS_MODEL=fake-model`) starts the SDK stdio loop without raising; running it without `ANTHROPIC_BASE_URL` exits non-zero and prints a stderr line containing `ANTHROPIC_BASE_URL`
  - _Requirements: 1.1, 1.2, 1.5, 11.1_
  - _Depends: 2.1, 3.1, 5.1_
  - _Boundary: x_likes_mcp/__main__.py_

## 6. Test infrastructure and tests

- [ ] 6.1 Create `tests/mcp/` test tree with conftest, fixtures, and the network/LLM guard
  - Create `tests/mcp/__init__.py`, `tests/mcp/conftest.py`, and `tests/mcp/fixtures/` directory
  - Hand-build `tests/mcp/fixtures/by_month/likes_2025-01.md`, `likes_2025-02.md`, `likes_2025-03.md` matching the post-change formatter layout (`## YYYY-MM`, `### @handle`, the per-tweet block; no global h1); under 50 lines each
  - Hand-build `tests/mcp/fixtures/likes.json` with four tweets across the three months whose IDs match what is referenced in the per-month files
  - In `conftest.py`, declare an autouse fixture that monkeypatches the LLM call entry point (the wrapper around LiteLLM that PageIndex calls during `_build_tree` and `_query`) to raise `RealLLMCallAttempted` so any unmocked test fails loudly
  - Declare a fixture `fake_export(tmp_path)` that copies `tests/mcp/fixtures/` into a temp dir and returns the resulting `Config` (with `ANTHROPIC_BASE_URL="http://fake"`, `ANTHROPIC_DEFAULT_OPUS_MODEL="fake-model"`)
  - Declare an autouse fixture that asserts no `cookies.json` access happens during a test run (set an env var that `Config` honors as a tests-mode hint, or patch a known path read)
  - Observable completion: `pytest tests/mcp -k nothing` collects with no errors and zero tests run; `RealLLMCallAttempted` is importable from `tests.mcp.conftest`; the `fake_export` fixture, when used in a smoke test, returns a `Config` whose `by_month_dir` contains exactly the three fixture files
  - _Requirements: 9.1, 9.2, 9.4, 9.5_
  - _Boundary: tests/mcp/_

- [ ] 6.2 (P) Write `test_config.py`
  - Test that `load_config(env={"ANTHROPIC_BASE_URL": "x", "ANTHROPIC_DEFAULT_OPUS_MODEL": "m"})` returns a `Config` with the expected paths and `litellm_model_string == "anthropic/m"`
  - Test that `load_config(env={"ANTHROPIC_BASE_URL": "x"})` raises `ConfigError` whose message contains `"ANTHROPIC_DEFAULT_OPUS_MODEL"`
  - Test that `load_config(env={"ANTHROPIC_DEFAULT_OPUS_MODEL": "m"})` raises `ConfigError` whose message contains `"ANTHROPIC_BASE_URL"`
  - Test that `OUTPUT_DIR` defaults to `"output"` when absent
  - Test that the `.env` file path code path reads a temp `.env` correctly (file with the three Anthropic vars plus comment lines)
  - Test that after a successful `load_config`, `os.environ["ANTHROPIC_BASE_URL"]` matches the configured value
  - Observable completion: `pytest tests/mcp/test_config.py -v` shows green across all six tests
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 11.1_
  - _Depends: 2.1, 6.1_
  - _Boundary: tests/mcp/test_config.py_

- [ ] 6.3 (P) Write `test_index.py`
  - Test `Index.open_or_build` against the fixture export with `_build_tree` mocked to return a known-shape fake tree: cache absent → builds and writes cache; cache fresh (mtime newer than all `.md`) → loads cache and does not call the builder; cache stale (touch one `.md` newer than cache) → rebuilds (assert via call counter on the mock)
  - Test `Index.open_or_build` against an empty `by_month/` raises `IndexError`
  - Test `_build_tree` receives a `model_string` of the form `"anthropic/<model>"`
  - Test `Index.search("anything")` (filter unset) calls `_query` with the full tree and returns the mocked `SearchHit` list
  - Test `Index.search("anything", year=2025, month_start="01", month_end="02")` causes `_query` to be called with a tree narrowed to the two in-range months (assert by spying on the captured tree argument or on the resolved month list)
  - Test `Index.search("anything", year=2025)` resolves to the full year's months
  - Test `Index._resolve_filter` raises `ValueError` for invalid combinations (`month_start` set but `year` not; `month_end` set but `month_start` not; `month_start > month_end`)
  - Test `Index.lookup_tweet` returns the right `Tweet` for a fixture ID and `None` for `"missing"`
  - Test `Index.list_months` returns `MonthInfo` list reverse-chronologically with correct counts
  - Test `Index.get_month_markdown("2025-01")` returns the file content; `get_month_markdown("2099-12")` returns `None`
  - Observable completion: `pytest tests/mcp/test_index.py -v` shows green; the cache-stale test verifies the builder was invoked (call counter on the mock)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.5, 5.1, 5.2, 5.3, 7.1, 7.3, 8.4_
  - _Depends: 3.1, 3.2, 3.3, 3.4, 6.1_
  - _Boundary: tests/mcp/test_index.py_

- [ ] 6.4 (P) Write `test_tools.py`
  - For each of the four tools, test the happy path with a mocked `Index` and the relevant error paths (invalid input, not found, upstream failure)
  - `search_likes`: empty/whitespace `query` → `invalid_input`; valid `query` with mocked matches → list of dicts with the four expected keys; valid `query` with full filter triple → `index.search` called with exactly those arguments; year-only filter → `index.search` called with `month_start=None, month_end=None`; `month_start` without `year` → `invalid_input` (the resolver raises `ValueError`, the handler translates); `index.search` raising `RuntimeError` → `upstream_failure`
  - `list_months`: returns dict list with `year_month`, `path`, `tweet_count` (some may be `None`)
  - `get_month`: bad pattern → `invalid_input`; missing month → `not_found`; valid → returns the string
  - `read_tweet`: empty/non-numeric → `invalid_input`; unknown id → `not_found`; valid → returns the metadata dict
  - Observable completion: `pytest tests/mcp/test_tools.py -v` shows green across all four tools' happy and error paths
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1, 7.3, 7.4, 11.4_
  - _Depends: 4.1, 4.2, 6.1_
  - _Boundary: tests/mcp/test_tools.py_

- [ ] 6.5 Write `test_server_integration.py`
  - Build the MCP server in-process via `server.build_server(index)` against the fixture export with PageIndex mocked
  - Drive each of the four tools through the SDK's tool-call dispatch (programmatic, not stdio) and assert the response shape matches the declared output schema
  - Verify a `ToolError` raised inside a handler becomes an MCP error response with the right category and the server does not propagate the exception
  - Verify the registered tool list is exactly the four tool names; verify the `search_likes` schema lists `year`, `month_start`, `month_end` as optional fields with the documented patterns
  - Verify a simulated LLM failure (`_query` raising `RuntimeError`) results in an `upstream_failure` tool error and the server stays alive for subsequent calls
  - Observable completion: `pytest tests/mcp/test_server_integration.py -v` shows green; the empty-query test asserts the response payload contains `"category": "invalid_input"`; a follow-up `list_months` call after the simulated LLM failure still returns successfully
  - _Requirements: 1.1, 1.3, 1.5, 4.6, 5.4, 6.4, 7.5, 9.3, 11.2, 11.4_
  - _Depends: 5.1, 6.1, 6.3, 6.4_
  - _Boundary: tests/mcp/test_server_integration.py_

## 7. Documentation

- [ ] 7.1 Add MCP Server section to `README.md`
  - Add a new section after the existing usage section titled "MCP Server"
  - Include `.mcp.json` snippet showing the `command`/`args` shape Claude Code expects (`uv run x-likes-mcp` or `python -m x_likes_mcp`)
  - Include the equivalent `claude mcp add` invocation
  - List the three new `.env` variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_OPUS_MODEL`) and state that the endpoint is local Anthropic-compatible by default and that PageIndex routes through LiteLLM with model string `anthropic/<model_name>`
  - List the prerequisite that `scrape.sh` has been run at least once and that, after upgrading from a previous version, `./scrape.sh --no-media --format markdown` should be re-run once so per-month files reflect the new (h1-less) shape
  - List the four tools with one-line summaries; for `search_likes`, document the optional `year` / `month_start` / `month_end` filter fields and explain that the filter pre-selects which markdown PageIndex sees
  - State that the server is stdio-only and that hosted LLM endpoints are not used by default
  - Document the manual real-LLM verification path (start a local Anthropic-compatible LLM, set env, run server, ask a sample question with and without the structured filter)
  - Observable completion: `grep "MCP Server" README.md` matches; `grep "claude mcp add" README.md` matches; the section names all four tools; the section names the three Anthropic env variables
  - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - _Depends: 5.2_
  - _Boundary: README.md_

## 8. Final integration check

- [ ] 8.1 Run the full test suite end-to-end and verify Spec 1's tests still pass alongside this spec's
  - Run `pytest` from the repo root and confirm both `tests/` (Spec 1) and `tests/mcp/` collect and pass
  - Run `python -m x_likes_mcp` against the existing `output/` directory with a temp `.env` that has `ANTHROPIC_BASE_URL=http://localhost:1234`, `ANTHROPIC_DEFAULT_OPUS_MODEL=any` (no real server needed; startup ends at the SDK stdio loop, send EOF on stdin to exit cleanly)
  - Verify `cookies.json` is not opened during the test run (rely on the conftest guard)
  - Confirm `sentrux scan` still passes against `.sentrux/rules.toml` (no new boundary violations)
  - Observable completion: `pytest` exit code 0 with both test trees green; `python -m x_likes_mcp` starts the loop and exits cleanly on EOF; no `cookies.json` access during tests; `sentrux scan` produces no new boundary violations relative to the pre-spec baseline
  - _Requirements: 1.5, 8.1, 8.2, 8.3, 9.1, 9.4, 9.5_
  - _Depends: 1.2, 6.2, 6.3, 6.4, 6.5, 7.1_
  - _Boundary: integration check_

## 9. Manual smoke (not gated in CI)

- [ ] 9.1 Manual end-to-end smoke against a real local Anthropic-compatible LLM
  - Re-run `./scrape.sh --no-media --format markdown` once so `output/by_month/` reflects the new (h1-less) shape
  - Start a local Anthropic-compatible LLM endpoint (whatever the user runs locally — llama.cpp/ollama/etc., as long as it speaks Anthropic's API)
  - Set `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_OPUS_MODEL` in `.env`
  - Run `python -m x_likes_mcp` from the project root and register it with Claude Code via `claude mcp add` or `.mcp.json`
  - Issue at least three queries from the MCP client: an open-ended `search_likes(query)`, a year-scoped `search_likes(query, year=2025)`, and a 3-month range `search_likes(query, year=2025, month_start="03", month_end="05")`; confirm sensible answers and visibly faster responses on the filtered queries
  - Issue one `list_months`, one `get_month`, and one `read_tweet` and confirm the responses match what is on disk
  - Observable completion: each of the six manual queries returns a response that pattern-matches the documented output shape; no server crashes; the cache file under `output/pageindex_cache.pkl` is created on first run and reused on the second run (mtime older than any `.md` would force a rebuild)
  - _Requirements: 8.4, 10.1, 10.2, 10.3, 10.4_
  - _Depends: 8.1_
  - _Boundary: manual integration check_
