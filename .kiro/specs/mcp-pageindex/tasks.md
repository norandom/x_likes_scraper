# Implementation Plan

## 1. Project scaffolding and dependencies

- [ ] 1.1 Add runtime deps and console script entry to `pyproject.toml`
  - Append `mcp` and `pageindex` to `[project.dependencies]` with conservative version pins (`mcp>=1.0,<2.0`, `pageindex>=0.1`)
  - Add `x-likes-mcp = "x_likes_mcp.__main__:main"` under `[project.scripts]`
  - Extend `[tool.hatch.build.targets.wheel].packages` to include `x_likes_mcp`
  - Run `uv sync` and confirm both new deps resolve and install
  - Observable completion: `uv run x-likes-mcp --help` (or equivalent SDK-default help) prints without ImportError; `uv pip show mcp pageindex` lists both
  - _Requirements: 1.4, 9.6_
  - _Boundary: pyproject.toml_

- [ ] 1.2 Create `x_likes_mcp/` package skeleton
  - Create `x_likes_mcp/__init__.py` defining `__version__ = "0.1.0"` and a docstring naming the package
  - Create empty (placeholder) module files: `config.py`, `errors.py`, `index.py`, `tools.py`, `server.py`, `__main__.py`
  - `__main__.py` contains `def main() -> int: return 0` plus the `if __name__ == "__main__": sys.exit(main())` guard so the module is runnable
  - Observable completion: `python -m x_likes_mcp` exits with code 0 and produces no output
  - _Requirements: 1.1, 1.2_
  - _Boundary: x_likes_mcp/_

- [ ] 1.3 Extend `.env.sample` with the two new variables
  - Append `LLM_API_BASE` and `LLM_API_KEY` entries with comments stating the endpoint is OpenAI-compatible and local by default
  - Observable completion: `grep LLM_API_BASE .env.sample` and `grep LLM_API_KEY .env.sample` both match
  - _Requirements: 2.5, 2.4_
  - _Boundary: .env.sample_

## 2. Configuration and error layers

- [ ] 2.1 Implement `config.py` with `.env` reader and `Config` dataclass (P)
  - Define frozen `Config` dataclass with the fields documented in design.md
  - Implement stdlib `.env` reader (split on `=`, strip comments and whitespace, no shell-quote handling)
  - Implement `load_config(env_path=None, env=None)` that reads `.env` from cwd by default, falls back to `os.environ`, and validates `LLM_API_BASE` is non-empty
  - Raise `ConfigError` naming the missing variable when validation fails
  - Default `OUTPUT_DIR` to `"output"`; derive `by_month_dir`, `likes_json`, `cache_path` from `output_dir`
  - Observable completion: importing `load_config` and calling it against an in-memory env dict returns a populated `Config`; calling without `LLM_API_BASE` raises `ConfigError` whose message contains the string `LLM_API_BASE`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 11.1_
  - _Boundary: x_likes_mcp/config.py_

- [ ] 2.2 Implement `errors.py` with `ToolError` and category helpers (P)
  - Define `ToolError(Exception)` with `category: str` and `message: str` attributes
  - Implement `invalid_input(field, message)`, `not_found(what, identifier)`, `upstream_failure(detail)` factory functions returning `ToolError` instances with the right category strings
  - Observable completion: `errors.invalid_input("query", "must be non-empty").category == "invalid_input"` and the same for `not_found` and `upstream_failure`
  - _Requirements: 4.3, 4.4, 6.2, 6.3, 7.3, 7.4, 11.2, 11.4_
  - _Boundary: x_likes_mcp/errors.py_

## 3. Indexing layer

- [ ] 3.1 Implement `Index` data containers and the build-or-load skeleton
  - Define `SearchHit` and `MonthInfo` frozen dataclasses with the fields documented in design.md
  - Define `IndexError(Exception)`
  - Implement `Index.__init__` accepting the tree, side-table, tweet map, and config; mark instance attributes as read-only by convention
  - Implement `Index.open_or_build(config)` enumerating files via `iter_monthly_markdown`, raising `IndexError("output/by_month/ is empty or missing")` when no files yield
  - Compute `newest_md_mtime` and the cache freshness check; load cached `(tree, side_table)` via `pickle.load` on hit; call `_build_tree(...)` on miss; write cache atomically (`.tmp` + `os.replace`)
  - Call `load_export(config.likes_json)` and store as `dict[str, Tweet]` keyed on `tweet.id`
  - Leave `_build_tree` as a stub that raises `NotImplementedError` for now; Task 3.2 fills it in
  - Observable completion: with `_build_tree` monkeypatched to return a sentinel tuple, `Index.open_or_build` against the fixture export returns an `Index` instance whose internal tweet map has the expected number of entries, and writes the cache file under `config.cache_path`
  - _Requirements: 3.1, 3.4, 3.5, 7.2, 8.5_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.2 Implement PageIndex tree builder seam in `Index._build_tree`
  - Implement `_build_tree(paths, llm_api_base, llm_api_key)` calling the PageIndex tree-build entry point with the file paths and LLM config, returning `(tree, side_table)`
  - Build the side-table by walking the resulting tree's leaf sections and matching their heading text against the in-memory `Tweet` list to record `(node_key -> tweet_id)`
  - When a leaf can't be matched to a tweet, skip it (do not raise); log a single line to stderr
  - Observable completion: against the fixture export, `_build_tree` returns a tuple where the side-table maps at least one leaf to one of the fixture tweet IDs (verified by a unit test that mocks PageIndex's tree-build to return a known-shape fake tree)
  - _Requirements: 3.1, 4.1, 4.2_
  - _Depends: 3.1_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.3 Implement `Index.search`, `lookup_tweet`, `list_months`, `get_month_markdown` (P)
  - `search(query)` calls PageIndex's query entry point with the cached tree and the query, then maps each match through the side-table to a `SearchHit`; returns `[]` when there are no matches
  - `lookup_tweet(tweet_id)` returns `self._tweets.get(tweet_id)` (`None` when missing)
  - `list_months()` derives months from `iter_monthly_markdown`, parses `YYYY-MM` from each filename, groups the in-memory tweet list by `Tweet.get_created_datetime()` to produce counts, returns `MonthInfo` list sorted reverse-chronologically
  - `get_month_markdown(year_month)` reads `by_month_dir / f"likes_{year_month}.md"`; returns `None` when the file doesn't exist (the tools layer translates that into a not-found error)
  - Observable completion: against the fixture export with PageIndex mocked, `search("anything")` returns a list of `SearchHit` with the expected `tweet_id`/`year_month`/`handle` values; `lookup_tweet` returns the right `Tweet` for a known fixture ID and `None` for `"missing"`; `list_months` returns the fixture months in reverse order
  - _Requirements: 3.2, 3.3, 4.1, 4.2, 4.5, 5.1, 5.2, 5.3, 7.1, 7.3_
  - _Depends: 3.1, 3.2_
  - _Boundary: x_likes_mcp/index.py_

## 4. Tool handlers

- [ ] 4.1 Implement `tools.search_likes` and `tools.list_months` (P)
  - `search_likes(index, query)` strips the query, raises `errors.invalid_input("query", ...)` when empty, otherwise calls `index.search(query)` and returns a list of dicts shaped `{"tweet_id", "year_month", "handle", "snippet"}`
  - Wrap the underlying call so any exception raised by `index.search` other than `ToolError` is converted via `errors.upstream_failure(...)` (the LLM-failure path)
  - `list_months(index)` returns a list of dicts shaped `{"year_month", "path", "tweet_count"}`; `tweet_count` may be `None`; ordering is whatever `Index.list_months` produced
  - Observable completion: with a mocked `Index`, calling `tools.search_likes(index, "  ")` raises `ToolError(category="invalid_input")`; calling with a valid query returns a list of dicts whose first entry has all four keys
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3_
  - _Depends: 2.2, 3.3_
  - _Boundary: x_likes_mcp/tools.py_

- [ ] 4.2 Implement `tools.get_month` and `tools.read_tweet` (P)
  - `get_month(index, year_month)` validates `year_month` against `^\d{4}-\d{2}$`; raises `errors.invalid_input("year_month", ...)` when the pattern fails; calls `index.get_month_markdown`; raises `errors.not_found("month", year_month)` when the result is `None`; otherwise returns the Markdown string
  - `read_tweet(index, tweet_id)` validates `tweet_id` is non-empty and numeric; raises `errors.invalid_input("tweet_id", ...)` otherwise; calls `index.lookup_tweet`; raises `errors.not_found("tweet", tweet_id)` when missing; otherwise returns a dict with text, handle, display_name, created_at, view_count, like_count, retweet_count, url (omitting fields the source `Tweet` doesn't have)
  - Observable completion: with a mocked `Index`, `tools.get_month(index, "2025/01")` raises `invalid_input`; `tools.get_month(index, "2099-12")` raises `not_found`; `tools.read_tweet(index, "")` raises `invalid_input`; `tools.read_tweet(index, "999")` against an `Index` that returns `None` raises `not_found`
  - _Requirements: 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 7.4_
  - _Depends: 2.2, 3.3_
  - _Boundary: x_likes_mcp/tools.py_

## 5. MCP server wiring

- [ ] 5.1 Implement `server.build_server(index)` with tool registration and JSON schemas
  - Construct an MCP `Server` instance with name `"x-likes-mcp"` and `version=__version__`
  - Register the four tools using the SDK's tool-registration API; declare input/output JSON schemas inline matching the patterns in design.md (`^\d{4}-\d{2}$` for `year_month`, etc.)
  - Implement the boundary error wrapper: `ToolError` -> MCP error response with `category` and `message`; other exceptions -> stderr log + generic `upstream_failure` response
  - Implement `server.run(index)` that calls the SDK's stdio entry point and returns when the client disconnects
  - Observable completion: a unit test that calls `build_server(index)` and inspects the registered tool list finds exactly four tools with names `search_likes`, `list_months`, `get_month`, `read_tweet` and non-empty input schemas
  - _Requirements: 1.1, 1.3, 4.4, 4.6, 5.4, 6.4, 7.5, 11.2_
  - _Depends: 4.1, 4.2_
  - _Boundary: x_likes_mcp/server.py_

- [ ] 5.2 Implement `__main__.main()` startup pipeline
  - Replace the placeholder `main()` with the real pipeline: `load_config()` -> `Index.open_or_build(config)` -> `server.run(index)` -> return 0
  - Catch `ConfigError`, `IndexError`, `FileNotFoundError` at the top of `main`; print one stderr line naming the failing condition; return exit code 2
  - Other exceptions during startup propagate (intentional: real bugs surface as tracebacks)
  - Observable completion: with a fixture export and a mocked LLM, running `python -m x_likes_mcp` against a temp `.env` (`LLM_API_BASE=http://fake`) starts the SDK stdio loop without raising; running it without `LLM_API_BASE` exits non-zero and prints a stderr line containing `LLM_API_BASE`
  - _Requirements: 1.1, 1.2, 1.5, 11.1_
  - _Depends: 2.1, 3.1, 5.1_
  - _Boundary: x_likes_mcp/__main__.py_

## 6. Test infrastructure and fixtures

- [ ] 6.1 Create `tests/mcp/` test tree with conftest and fixtures (P)
  - Create `tests/mcp/__init__.py`, `tests/mcp/conftest.py`, and `tests/mcp/fixtures/` directory
  - Hand-build `tests/mcp/fixtures/by_month/likes_2025-01.md` and `likes_2025-02.md` matching the formatter's layout (`## YYYY-MM`, `### @handle`, the per-tweet block); under 50 lines each
  - Hand-build `tests/mcp/fixtures/likes.json` with three tweets whose IDs match what's referenced in the per-month files
  - In `conftest.py`, declare an autouse fixture that monkeypatches the LLM-call entry point (in `Index._build_tree` and in PageIndex's query path) to raise `RealLLMCallAttempted` so any unmocked test fails loudly
  - Declare a fixture `fake_export(tmp_path)` that copies `tests/mcp/fixtures/` into a temp dir and returns the resulting `Config` (with `LLM_API_BASE="http://fake"`)
  - Observable completion: `pytest tests/mcp -k nothing` collects with no errors and zero tests run; `RealLLMCallAttempted` is importable from `tests.mcp.conftest`
  - _Requirements: 9.1, 9.2, 9.4, 9.5_
  - _Boundary: tests/mcp/_

- [ ] 6.2 Write `test_config.py` (P)
  - Test that `load_config(env={"LLM_API_BASE": "x", "LLM_API_KEY": "y"})` returns a `Config` with the expected paths and values
  - Test that `load_config(env={})` raises `ConfigError` whose message contains `"LLM_API_BASE"`
  - Test that `OUTPUT_DIR` defaults to `"output"` when absent
  - Test that the `.env` file path code path reads a temp `.env` correctly (file with `LLM_API_BASE=http://fake` plus comment lines)
  - Observable completion: `pytest tests/mcp/test_config.py -v` shows four green tests
  - _Requirements: 2.1, 2.2, 2.3, 11.1_
  - _Depends: 2.1, 6.1_
  - _Boundary: tests/mcp/test_config.py_

- [ ] 6.3 Write `test_index.py` (P)
  - Test `Index.open_or_build` against the fixture export with PageIndex's tree-builder mocked: cache absent → builds and writes cache; cache fresh (mtime newer than all `.md`) → loads cache and does not call the builder; cache stale (one `.md` mtime touched newer than cache) → rebuilds
  - Test `Index.open_or_build` against an empty `by_month/` directory → raises `IndexError`
  - Test `Index.search` returns `SearchHit` list when the mocked PageIndex query returns matches; returns `[]` when no matches
  - Test `Index.lookup_tweet` returns the right `Tweet` for a fixture ID and `None` for a missing ID
  - Test `Index.list_months` against the fixture: returns `MonthInfo` list in reverse chronological order with correct counts
  - Test `Index.get_month_markdown` returns the file content for an existing month and `None` for a missing one
  - Observable completion: `pytest tests/mcp/test_index.py -v` shows green; the cache-stale test verifies the builder was invoked (call counter on the mock)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.5, 5.1, 5.2, 5.3, 7.1, 7.3_
  - _Depends: 3.1, 3.2, 3.3, 6.1_
  - _Boundary: tests/mcp/test_index.py_

- [ ] 6.4 Write `test_tools.py` (P)
  - For each of the four tools, test the happy path with a mocked `Index` and the relevant error paths (invalid input, not found, upstream failure)
  - `search_likes`: empty/whitespace query → `invalid_input`; valid query with mocked matches → list of dicts with the four expected keys; `index.search` raising `RuntimeError` → `upstream_failure`
  - `list_months`: returns dict list with `year_month`, `path`, `tweet_count` (some `None`)
  - `get_month`: bad pattern → `invalid_input`; missing month → `not_found`; valid → returns the string
  - `read_tweet`: empty/non-numeric → `invalid_input`; unknown id → `not_found`; valid → returns the metadata dict
  - Observable completion: `pytest tests/mcp/test_tools.py -v` shows green across all four tools' happy and error paths
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1, 7.3, 7.4, 11.4_
  - _Depends: 4.1, 4.2, 6.1_
  - _Boundary: tests/mcp/test_tools.py_

- [ ] 6.5 Write `test_server_integration.py`
  - Build the MCP server in-process via `server.build_server(index)` against the fixture export with PageIndex mocked
  - Drive each of the four tools through the SDK's tool-call dispatch (programmatic, not stdio) and assert the response shape matches the declared output schema
  - Verify a `ToolError` raised inside a handler becomes an MCP error response with the right category, and that the server doesn't propagate the exception
  - Verify the registered tool list is exactly the four tool names
  - Observable completion: `pytest tests/mcp/test_server_integration.py -v` shows green; the test that calls `search_likes` with empty query asserts the response contains `"category": "invalid_input"`
  - _Requirements: 1.1, 1.3, 1.5, 4.6, 5.4, 6.4, 7.5, 9.3, 11.2, 11.4_
  - _Depends: 5.1, 6.1, 6.3, 6.4_
  - _Boundary: tests/mcp/test_server_integration.py_

## 7. Documentation

- [ ] 7.1 Add MCP Server section to `README.md`
  - Add a new section after the existing usage section titled "MCP Server"
  - Include `.mcp.json` snippet showing the `command`/`args` shape Claude Code expects
  - Include the equivalent `claude mcp add` invocation
  - List the two new `.env` variables (`LLM_API_BASE`, `LLM_API_KEY`) and state that the endpoint is OpenAI-compatible and local by default
  - List the prerequisite that `scrape.sh` has been run at least once
  - List the four tools with one-line summaries
  - State that the server is stdio-only and that hosted LLM endpoints are not used by default
  - Document the manual real-LLM verification path (start a local LLM, set env, run server, ask a question)
  - Observable completion: `grep "MCP Server" README.md` matches; `grep "claude mcp add" README.md` matches; the section names all four tools
  - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - _Depends: 5.2_
  - _Boundary: README.md_

## 8. Final integration check

- [ ] 8.1 Run the full test suite and verify Spec 1's tests still pass alongside this spec's
  - Run `pytest` from the repo root and confirm both `tests/` (Spec 1) and `tests/mcp/` collect and pass
  - Run `python -m x_likes_mcp` against the existing `output/` directory with a temp `.env` that has `LLM_API_BASE=http://localhost:1234/v1` (no real server needed; startup ends at the SDK stdio loop, send EOF on stdin to exit cleanly)
  - Verify `cookies.json` is not opened during the test run (e.g. by inspecting an `strace`-style log or by relying on the conftest guard)
  - Observable completion: `pytest` exit code 0 with both test trees green; `python -m x_likes_mcp` starts the loop and exits cleanly on EOF; no `cookies.json` access during tests
  - _Requirements: 1.5, 8.1, 8.2, 8.3, 9.1, 9.4, 9.5_
  - _Depends: 6.2, 6.3, 6.4, 6.5, 7.1_
  - _Boundary: integration check_
