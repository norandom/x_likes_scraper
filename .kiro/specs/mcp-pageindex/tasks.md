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

- [x] 1.3 Add MCP runtime deps and console script entry to `pyproject.toml`
  - Append `mcp>=1.0,<2.0` and `pageindex` (with whatever version pin is current at impl time) to `[project.dependencies]`
  - Add `x-likes-mcp = "x_likes_mcp.__main__:main"` under `[project.scripts]`
  - Extend `[tool.hatch.build.targets.wheel].packages` to include `x_likes_mcp`
  - Run `uv sync` and confirm both new deps resolve and install
  - Observable completion: `uv pip show mcp pageindex` lists both with version numbers; importing both from a Python REPL succeeds without errors
  - _Requirements: 1.4, 11.6_
  - _Boundary: pyproject.toml_
  - _Note: This task originally added pageindex; task 1.6 below removes it now that we are not using PageIndex._

- [x] 1.4 Create the `x_likes_mcp/` package skeleton
  - Create `x_likes_mcp/__init__.py` defining `__version__ = "0.1.0"` and a one-line module docstring
  - Create empty (placeholder) module files: `config.py`, `errors.py`, `index.py`, `tools.py`, `server.py`, `__main__.py`
  - `__main__.py` contains `def main() -> int: return 0` and the `if __name__ == "__main__": sys.exit(main())` guard so the module is runnable
  - Observable completion: `python -m x_likes_mcp` exits with code 0 and produces no output; `from x_likes_mcp import __version__` returns `"0.1.0"`
  - _Requirements: 1.1, 1.2_
  - _Boundary: x_likes_mcp/_

- [x] 1.5 Extend `.env.sample` with the three new OpenAI-compatible variables
  - Append `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` entries with comments stating the endpoint is local OpenAI-compatible by default and that the walker uses the OpenAI Python SDK directly (the SDK reads `OPENAI_BASE_URL` from the process environment automatically)
  - Default value for `OPENAI_BASE_URL` is `http://localhost:8080/v1`; default for `OPENAI_API_KEY` is empty; default for `OPENAI_MODEL` is a placeholder model name
  - Observable completion: `grep OPENAI_BASE_URL .env.sample`, `grep OPENAI_API_KEY .env.sample`, and `grep OPENAI_MODEL .env.sample` all match
  - _Requirements: 2.7, 2.4_
  - _Boundary: .env.sample_

- [x] 1.6 Drop `pageindex` from `pyproject.toml` and add `openai`; create `tree.py`, `walker.py`, `ranker.py` skeleton files
  - In `pyproject.toml`, remove `pageindex` from `[project.dependencies]`; add `openai>=1.0` (the walker calls the SDK directly)
  - Run `uv sync` and confirm `pageindex` is gone from the lockfile / installed env, and `openai` resolves
  - Create `x_likes_mcp/tree.py`, `x_likes_mcp/walker.py`, `x_likes_mcp/ranker.py` as empty placeholder files (one-line docstring each). Tasks 3.2, 3.3b, 3.3c will fill them in.
  - Observable completion: `grep -n pageindex pyproject.toml` returns nothing; `uv pip show openai` lists a version; `python -c "from x_likes_mcp import tree, walker, ranker"` succeeds without error
  - _Requirements: 1.4, 11.6, 4.5, 5.6_
  - _Boundary: pyproject.toml, x_likes_mcp/_

- [x] 1.7 Extend `.env.sample` with the ranker weight variables
  - Append `RANKER_W_RELEVANCE`, `RANKER_W_FAVORITE`, `RANKER_W_RETWEET`, `RANKER_W_REPLY`, `RANKER_W_VIEW`, `RANKER_W_AFFINITY`, `RANKER_W_RECENCY`, `RANKER_W_VERIFIED`, `RANKER_W_MEDIA`, and `RANKER_RECENCY_HALFLIFE_DAYS` entries (all commented-out by default so the in-code defaults apply unless the user opts in)
  - Include a brief comment block stating the score formula and the documented default values
  - Observable completion: `grep RANKER_W_RELEVANCE .env.sample` matches; the formula comment is present
  - _Requirements: 2.7, 2.6_
  - _Boundary: .env.sample_

## 2. Configuration and error layers

- [x] 2.1 (P) Implement `config.py` with `.env` reader and `Config` dataclass
  - Define frozen `Config` dataclass with the fields documented in design.md (`output_dir`, `by_month_dir`, `likes_json`, `cache_path`, `openai_base_url`, `openai_api_key`, `openai_model`)
  - Implement a stdlib `.env` reader (split on `=`, strip comments and whitespace, no shell-quote handling)
  - Implement `load_config(env_path=None, env=None)` that reads `.env` from cwd by default, falls back to `os.environ`, and validates that `OPENAI_BASE_URL` and `OPENAI_MODEL` are set and non-empty
  - Raise `ConfigError` naming the missing variable when validation fails
  - Default `OUTPUT_DIR` to `"output"`; derive `by_month_dir`, `likes_json`, `cache_path` from `output_dir`
  - Side effect: write `OPENAI_BASE_URL` (and `OPENAI_API_KEY` if set) into `os.environ` before returning so the OpenAI SDK picks them up at client-construction time
  - Observable completion: importing `load_config` and calling it against an in-memory `env` dict returns a populated `Config` whose `openai_model` equals the configured model name; calling without `OPENAI_BASE_URL` raises `ConfigError` whose message contains the string `OPENAI_BASE_URL`; after a successful call, `os.environ["OPENAI_BASE_URL"]` matches the configured value
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 13.1_
  - _Boundary: x_likes_mcp/config.py_

- [x] 2.2 (P) Implement `errors.py` with `ToolError` and category helpers
  - Define `ToolError(Exception)` carrying `category: str` and `message: str` attributes
  - Implement `invalid_input(field, message)`, `not_found(what, identifier)`, `upstream_failure(detail)` factory functions returning `ToolError` instances with the right category strings
  - Observable completion: `errors.invalid_input("query", "must be non-empty").category == "invalid_input"`; the same for `not_found` and `upstream_failure`; `str(err)` includes the field name and the message
  - _Requirements: 6.3, 6.5, 8.2, 8.3, 9.3, 9.4, 13.2, 13.4_
  - _Boundary: x_likes_mcp/errors.py_

- [x] 2.3 Extend `config.py` with `RankerWeights` and `load_ranker_weights`
  - Add a frozen `RankerWeights` dataclass with the nine weight fields plus `recency_halflife_days`, defaults matching the design (`relevance=10.0`, `favorite=2.0`, `retweet=2.5`, `reply=1.0`, `view=0.5`, `affinity=3.0`, `recency=1.5`, `verified=0.5`, `media=0.3`, `recency_halflife_days=180.0`)
  - Implement `load_ranker_weights(env: dict[str, str]) -> RankerWeights` that reads `RANKER_W_RELEVANCE`, `RANKER_W_FAVORITE`, `RANKER_W_RETWEET`, `RANKER_W_REPLY`, `RANKER_W_VIEW`, `RANKER_W_AFFINITY`, `RANKER_W_RECENCY`, `RANKER_W_VERIFIED`, `RANKER_W_MEDIA`, `RANKER_RECENCY_HALFLIFE_DAYS`. Missing keys take defaults; non-numeric values raise `ConfigError` with a message naming the variable.
  - `load_config` continues to return `Config`. Either expose `load_ranker_weights` as a separate function callers invoke, or extend `load_config` to return `(Config, RankerWeights)` — implementer's choice. The integration in `__main__` (task 5.2) is the consumer.
  - Observable completion: `load_ranker_weights({"RANKER_W_RELEVANCE": "12.0"}).relevance == 12.0`; `load_ranker_weights({"RANKER_W_RELEVANCE": "abc"})` raises `ConfigError` whose message contains `RANKER_W_RELEVANCE`; `load_ranker_weights({}).relevance == 10.0`
  - _Requirements: 2.6, 5.3, 13.1_
  - _Depends: 2.1_
  - _Boundary: x_likes_mcp/config.py_

## 3. Tree, walker, ranker, index

- [x] 3.1 Implement `TweetIndex` data containers and the build-or-load skeleton
  - Define `MonthInfo` frozen dataclass with `year_month`, `path`, `tweet_count` fields
  - Define `IndexError(Exception)`
  - Implement `TweetIndex.__init__` accepting the `TweetTree`, the `tweets_by_id: dict[str, Tweet]`, the `paths_by_month: dict[str, Path]`, the `author_affinity: dict[str, float]`, the `Config`, and the `RankerWeights`; mark instance attributes as read-only by convention
  - Implement `TweetIndex.open_or_build(config, weights)` that enumerates files via `iter_monthly_markdown(config.by_month_dir)`, raises `IndexError("output/by_month/ is empty or missing")` when no files yield
  - Compute `newest_md_mtime` and the cache freshness check; load cached `TweetTree` via `pickle.load` on hit; call `tree.build_tree(config.by_month_dir)` on miss; write the cache atomically (`.tmp` + `os.replace`)
  - Call `load_export(config.likes_json)` and store as `dict[str, Tweet]` keyed on `tweet.id`; also retain the `list[Tweet]` for `list_months` counts
  - Build a `paths_by_month: dict[str, Path]` map by parsing `YYYY-MM` from each `likes_YYYY-MM.md` filename
  - Compute `author_affinity = ranker.compute_author_affinity(tweets)` (task 3.3c provides this function; for this task `compute_author_affinity` may temporarily live as a small helper inline and move into `ranker.py` when 3.3c lands)
  - Leave `search` (task 3.3d) and `_resolve_filter` (task 3.3a) as method stubs that raise `NotImplementedError`
  - Observable completion: against the fixture export with `tree.build_tree` monkeypatched to return a known-shape fake `TweetTree`, `TweetIndex.open_or_build` returns a `TweetIndex` whose internal tweet map has the expected number of entries and whose `paths_by_month` maps each fixture month to its file; the cache file appears under `config.cache_path`; `author_affinity[<known_handle>]` is the documented `log1p(count)` value
  - _Requirements: 3.1, 3.4, 3.5, 5.7, 9.2, 10.5_
  - _Boundary: x_likes_mcp/index.py_

- [x] 3.2 Implement `tree.py` with `build_tree`, `TreeNode`, `TweetTree`
  - Define frozen `TreeNode` dataclass with `year_month`, `tweet_id`, `handle`, `text`, `raw_section` fields
  - Define frozen `TweetTree` dataclass with `nodes_by_month: dict[str, list[TreeNode]]` and `nodes_by_id: dict[str, TreeNode]`
  - Implement `build_tree(by_month_dir: Path) -> TweetTree`:
    - Walk `by_month_dir` for files matching `likes_YYYY-MM.md` (regex on filename)
    - For each file, read its text, split on `\n### ` boundaries (after the file's `## YYYY-MM` heading) to get per-tweet sections
    - For each section, extract `tweet_id` from the canonical `🔗 [View on X](https://x.com/{handle}/status/{id})` link via regex; if no match, log to stderr and skip the section
    - Extract `handle` from the section heading (`### [@handle]` or `### @handle`); fall back to the link-line handle if heading parse fails
    - `text` is the section body minus the heading and link line; `raw_section` is the section text including the heading
    - Build `nodes_by_month` (ordered list per month, in source order) and `nodes_by_id`
  - Pure function: same input directory, same `TweetTree` (up to dataclass equality)
  - No LLM call, no network access, no OpenAI SDK import
  - Observable completion: `tree.build_tree(tests/mcp/fixtures/by_month)` returns a `TweetTree` whose `nodes_by_month` keys are exactly `{"2025-01", "2025-02", "2025-03"}` and whose `nodes_by_id` contains the four fixture tweet IDs; `grep -n "openai" x_likes_mcp/tree.py` returns nothing
  - _Requirements: 3.1, 3.6_
  - _Depends: 1.6_
  - _Boundary: x_likes_mcp/tree.py_

- [x] 3.3a Implement `_resolve_filter` on `TweetIndex`
  - Implement `_resolve_filter(year, month_start, month_end) -> list[str] | None` enforcing:
    - `month_start` set requires `year` set (else `ValueError("filter: month_start requires year")`)
    - `month_end` set requires `month_start` set (else `ValueError("filter: month_end requires month_start")`)
    - When both `month_start` and `month_end` set, `month_start <= month_end` (else `ValueError("filter: month_start must be <= month_end")`)
    - All three `None` → return `None` (meaning "all months")
    - `year` only → return `[f"{year}-01", ..., f"{year}-12"]`
    - `year` + `month_start` only → return `[f"{year}-{month_start}"]`
    - `year` + `month_start` + `month_end` → return `[f"{year}-{m:02d}" for m in range(int(month_start), int(month_end)+1)]`
  - Validation errors raise `ValueError` with messages identifying the offending field; `tools.search_likes` translates that into `errors.invalid_input("filter", ...)`
  - Observable completion: `_resolve_filter(None, None, None)` returns `None`; `_resolve_filter(2025, "01", "03")` returns `["2025-01", "2025-02", "2025-03"]`; `_resolve_filter(None, "01", None)` raises `ValueError`; `_resolve_filter(2025, "03", "01")` raises `ValueError`
  - _Requirements: 6.2, 6.3_
  - _Depends: 3.1_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.3b Implement `walker.py` with `walk`, `WalkerHit`, `WalkerError`
  - Define frozen `WalkerHit` dataclass with `tweet_id: str`, `relevance: float`, `why: str`
  - Define `WalkerError(RuntimeError)`
  - Implement `walk(tree: TweetTree, query: str, months_in_scope: list[str] | None, config: Config, chunk_size: int = 30) -> list[WalkerHit]`:
    - If `months_in_scope is None`, iterate every key in `tree.nodes_by_month` (sorted ascending). Otherwise iterate only the listed months that exist in the tree.
    - For each month, partition `tree.nodes_by_month[month]` into chunks of `chunk_size`.
    - For each chunk, build the prompt: a system prompt explaining the JSON output contract (return only plausibly-relevant tweets, including indirect/thematic relevance), and a user prompt listing each tweet as `[id={tweet_id}] @{handle}: {text}` (truncate `text` to a reasonable length to keep the prompt small).
    - Call the OpenAI SDK: `client = OpenAI()` (the SDK reads `OPENAI_BASE_URL`/`OPENAI_API_KEY` from `os.environ`); `client.chat.completions.create(model=config.openai_model, messages=[...])`. Use JSON mode (`response_format={"type": "json_object"}`) when supported; otherwise instruct the model to return raw JSON and parse tolerantly.
    - Parse the response. Drop entries whose `id` is not in the chunk. Drop entries whose `relevance` is not a finite number in `[0, 1]`. Truncate `why` to ~240 chars.
    - On per-chunk LLM failure (HTTP error, malformed JSON that cannot be salvaged), raise `WalkerError(detail)`.
  - Implement an internal helper that the test layer can mock (e.g. `_call_chat_completions(client, model, messages) -> str`). Tests replace either `walker.walk` or this helper.
  - Observable completion: with the chat-completions helper mocked to return canned JSON, `walker.walk` against a fixture `TweetTree` and `months_in_scope=["2025-01"]` returns the expected `WalkerHit`s; `walker.walk` raises `WalkerError` when the helper raises; `grep -n "openai" x_likes_mcp/walker.py` is the only file in the package besides `pyproject.toml`'s lockfile that imports the SDK
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 6.1_
  - _Depends: 1.6, 3.2_
  - _Boundary: x_likes_mcp/walker.py_

- [x] 3.3c Implement `ranker.py` with `rank`, `ScoredHit`, `compute_author_affinity`
  - Define frozen `ScoredHit` dataclass with `tweet_id: str`, `score: float`, `walker_relevance: float`, `why: str`, `feature_breakdown: dict[str, float]`
  - Implement `compute_author_affinity(tweets: list[Tweet]) -> dict[str, float]`: count occurrences of each `tweet.user.screen_name`, return `{handle: math.log1p(count)}`
  - Implement `rank(walker_hits, tweets_by_id, author_affinity, weights, anchor)`:
    - For each `WalkerHit`, look up the `Tweet` in `tweets_by_id`; skip if missing
    - Compute the per-feature contributions:
      - `relevance = walker_hit.relevance * weights.relevance`
      - `favorite = math.log1p(tweet.favorite_count) * weights.favorite`
      - `retweet = math.log1p(tweet.retweet_count) * weights.retweet`
      - `reply = math.log1p(tweet.reply_count) * weights.reply`
      - `view = math.log1p(tweet.view_count) * weights.view`
      - `affinity = author_affinity.get(tweet.user.screen_name, 0.0) * weights.affinity`
      - `recency = recency_decay(tweet.created_at, anchor, weights.recency_halflife_days) * weights.recency` (when `created_at` is unparseable, contribute 0 for the recency term and log once)
      - `verified = (1.0 if tweet.user.verified else 0.0) * weights.verified`
      - `media = (1.0 if tweet.media else 0.0) * weights.media`
    - `score = sum of all contributions`
    - `feature_breakdown` is a `dict[str, float]` with the nine keys above
  - Sort the resulting `ScoredHit` list descending by `score`, ties broken by `walker_relevance` descending then `tweet_id` ascending for determinism
  - Pure function: no I/O, no LLM, no network
  - Implement helper `recency_decay(created_at: str, anchor: datetime, halflife_days: float) -> float`: parse `created_at` (raise on failure to let the caller skip the term), compute `days = max(0, (anchor - created).total_seconds() / 86400)`, return `math.exp(-days / halflife_days)`
  - Observable completion: `rank` against hand-built inputs produces `ScoredHit` lists where `feature_breakdown` keys sum to `score`; monotonicity holds (higher walker_relevance → higher score); recency_decay returns 1.0 for `days=0` and `~0.5` at `days=halflife`; `grep -n "openai" x_likes_mcp/ranker.py` returns nothing
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - _Depends: 1.6, 2.3_
  - _Boundary: x_likes_mcp/ranker.py_

- [ ] 3.3d Implement `TweetIndex.search` orchestrating resolve → walk → rank
  - Implement `search(query, year=None, month_start=None, month_end=None, top_n=50)` that:
    - Calls `self._resolve_filter(year, month_start, month_end)` → `months_in_scope or None`
    - Computes the recency anchor: end of `month_end` if both `month_start` and `month_end` set; end of `month_start`'s month if `month_start` set without `month_end`; end of `year` if only `year` set; `datetime.now(timezone.utc)` otherwise
    - Calls `walker.walk(self._tree, query, months_in_scope, self._config)` (returns `list[WalkerHit]`)
    - Calls `ranker.rank(walker_hits, self._tweets_by_id, self._author_affinity, self._weights, anchor)` (returns sorted `list[ScoredHit]`)
    - Returns `result[:top_n]`
  - Walker and ranker exceptions propagate; the tools layer shapes them
  - Observable completion: with `walker.walk` mocked to return canned hits, `TweetIndex.search("anything")` calls walker with `months_in_scope=None`; `TweetIndex.search("anything", year=2025, month_start="01", month_end="02")` calls walker with `months_in_scope=["2025-01", "2025-02"]` and uses an anchor at the end of February 2025; the returned list is sorted descending by `score`
  - _Requirements: 6.1, 6.2, 6.7, 5.4_
  - _Depends: 3.1, 3.2, 3.3a, 3.3b, 3.3c_
  - _Boundary: x_likes_mcp/index.py_

- [ ] 3.4 (P) Implement `TweetIndex.lookup_tweet`, `list_months`, `get_month_markdown`
  - `lookup_tweet(tweet_id)` returns `self._tweets_by_id.get(tweet_id)` (`None` when missing)
  - `list_months()` consults `paths_by_month`, parses `YYYY-MM` from each filename, groups the in-memory tweet list by `Tweet.get_created_datetime()` to produce counts (tweets with unparseable `created_at` are skipped from counts), returns `MonthInfo` list sorted reverse-chronologically
  - `get_month_markdown(year_month)` reads `by_month_dir / f"likes_{year_month}.md"` if the file exists; returns `None` otherwise
  - Observable completion: against the fixture export, `TweetIndex.lookup_tweet` returns the right `Tweet` for a known fixture ID and `None` for `"missing"`; `TweetIndex.list_months` returns the fixture months in reverse order with correct counts; `TweetIndex.get_month_markdown("2025-01")` returns the file content; `TweetIndex.get_month_markdown("2099-12")` returns `None`
  - _Requirements: 7.1, 7.2, 7.3, 9.1, 9.3_
  - _Depends: 3.1_
  - _Boundary: x_likes_mcp/index.py_

## 4. Tool handlers

- [ ] 4.1 (P) Implement `tools.search_likes` and `tools.list_months`
  - `search_likes(index, query, year=None, month_start=None, month_end=None)` strips the query, raises `errors.invalid_input("query", ...)` when empty
  - Validates filter shape (`year` is integer in valid range, `month_start`/`month_end` match `^(0[1-9]|1[0-2])$`); shape errors raise `errors.invalid_input("filter", ...)`
  - Calls `index.search(query, year, month_start, month_end)`; catches `ValueError` from the resolver and re-raises as `errors.invalid_input("filter", ...)`
  - Catches any non-`ToolError` exception from `index.search` (notably `WalkerError`) and re-raises as `errors.upstream_failure(...)`
  - Returns a list of dicts shaped `{"tweet_id", "year_month", "handle", "snippet", "score", "walker_relevance", "why", "feature_breakdown"}`. The snippet is drawn from the loaded `Tweet.text` (or whatever Spec 1's `Tweet` exposes), truncated to ~240 chars. `year_month` derives from `Tweet.get_created_datetime()` when parseable, otherwise from the matching `TreeNode.year_month`.
  - `list_months(index)` returns a list of dicts shaped `{"year_month", "path", "tweet_count"}`; `tweet_count` may be `None`; ordering is whatever `TweetIndex.list_months` produced
  - Observable completion: with a mocked `TweetIndex`, `tools.search_likes(index, "  ")` raises `ToolError(category="invalid_input")`; `tools.search_likes(index, "x", year=2025, month_start="01", month_end="02")` calls `index.search` with exactly those four arguments; `tools.search_likes(index, "x", year=None, month_start="01")` raises `invalid_input`; `index.search` raising `WalkerError("LLM down")` becomes `ToolError(category="upstream_failure")`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.2, 7.3_
  - _Depends: 2.2, 3.3d, 3.4_
  - _Boundary: x_likes_mcp/tools.py_

- [ ] 4.2 (P) Implement `tools.get_month` and `tools.read_tweet`
  - `get_month(index, year_month)` validates `year_month` against `^\d{4}-\d{2}$`; raises `errors.invalid_input("year_month", ...)` when the pattern fails; calls `index.get_month_markdown`; raises `errors.not_found("month", year_month)` when the result is `None`; otherwise returns the Markdown string
  - `read_tweet(index, tweet_id)` validates `tweet_id` is non-empty and matches `^\d+$`; raises `errors.invalid_input("tweet_id", ...)` otherwise; calls `index.lookup_tweet`; raises `errors.not_found("tweet", tweet_id)` when missing; otherwise returns a dict with `tweet_id`, `handle`, `display_name`, `text`, `created_at`, `view_count`, `like_count`, `retweet_count`, `url` (omitting fields the source `Tweet` does not have)
  - Observable completion: with a mocked `TweetIndex`, `tools.get_month(index, "2025/01")` raises `invalid_input`; `tools.get_month(index, "2099-12")` (where `TweetIndex.get_month_markdown` returns `None`) raises `not_found`; `tools.read_tweet(index, "")` raises `invalid_input`; `tools.read_tweet(index, "abc")` raises `invalid_input`; `tools.read_tweet(index, "999")` against an `TweetIndex` that returns `None` raises `not_found`
  - _Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 9.4_
  - _Depends: 2.2, 3.4_
  - _Boundary: x_likes_mcp/tools.py_

## 5. MCP server wiring

- [ ] 5.1 Implement `server.build_server(index)` with tool registration and JSON schemas
  - Construct an MCP `Server` instance with name `"x-likes-mcp"` and `version=__version__`
  - Register the four tools using the SDK's tool-registration API; declare input/output JSON schemas inline
  - `search_likes` schema: `query` required string, `year` optional integer with min 2006 and max equal to current year, `month_start` and `month_end` optional strings with `pattern: "^(0[1-9]|1[0-2])$"`; output is array of objects with `tweet_id`, `year_month`, `handle`, `snippet`, `score`, `walker_relevance`, `why`, `feature_breakdown`
  - `list_months` schema: empty input; output is array of `{year_month, path, tweet_count}`
  - `get_month` schema: `year_month` required string with `pattern: "^\\d{4}-\\d{2}$"`; output is string
  - `read_tweet` schema: `tweet_id` required string with `pattern: "^\\d+$"`; output is the documented metadata object
  - Implement the boundary error wrapper: `ToolError` becomes an MCP error response with `category` and `message`; other exceptions become a stderr log plus a generic `upstream_failure` response
  - Implement `server.run(index)` calling the SDK's stdio entry point; returns when the client disconnects
  - Observable completion: a unit test that calls `build_server(index)` and inspects the registered tool list finds exactly four tools with names `search_likes`, `list_months`, `get_month`, `read_tweet` and non-empty input schemas; the `search_likes` schema includes the `year` / `month_start` / `month_end` fields with the documented patterns
  - _Requirements: 1.1, 1.3, 6.6, 6.8, 7.4, 8.4, 9.5, 13.2_
  - _Depends: 4.1, 4.2_
  - _Boundary: x_likes_mcp/server.py_

- [ ] 5.2 Implement `__main__.main()` startup pipeline
  - Replace the placeholder `main()` with the real pipeline: `load_config()` → `load_ranker_weights()` → `TweetIndex.open_or_build(config, weights)` → `server.run(index)` → return 0
  - Catch `ConfigError`, `IndexError`, `FileNotFoundError` at the top of `main`; print one stderr line naming the failing condition; return exit code 2
  - Other exceptions during startup propagate (intentional: real bugs surface as tracebacks)
  - Observable completion: with a fixture export and `walker.walk` mocked, running `python -m x_likes_mcp` against a temp `.env` (`OPENAI_BASE_URL=http://fake/v1`, `OPENAI_MODEL=fake-model`) starts the SDK stdio loop without raising; running it without `OPENAI_BASE_URL` exits non-zero and prints a stderr line containing `OPENAI_BASE_URL`; running it with `RANKER_W_RELEVANCE=abc` exits non-zero and prints a stderr line containing `RANKER_W_RELEVANCE`
  - _Requirements: 1.1, 1.2, 1.5, 13.1_
  - _Depends: 2.1, 2.3, 3.1, 3.3d, 5.1_
  - _Boundary: x_likes_mcp/__main__.py_

## 6. Test infrastructure and tests

- [ ] 6.1 Create `tests/mcp/` test tree with conftest, fixtures, and the network/LLM guard
  - Create `tests/mcp/__init__.py`, `tests/mcp/conftest.py`, and `tests/mcp/fixtures/` directory
  - Hand-build `tests/mcp/fixtures/by_month/likes_2025-01.md`, `likes_2025-02.md`, `likes_2025-03.md` matching the post-task-1.1 formatter layout (`## YYYY-MM`, `### [@handle]`, the per-tweet block including the canonical `🔗 [View on X](https://x.com/{handle}/status/{id})` link; no global h1); under 50 lines each
  - Hand-build `tests/mcp/fixtures/likes.json` with four tweets across the three months whose IDs match the per-month file references; engagement counts are non-zero so ranker tests can exercise the formula
  - In `conftest.py`, declare an autouse fixture that monkeypatches `walker.walk` (or its underlying chat-completions helper) to raise `RealLLMCallAttempted` so any unmocked test fails loudly
  - Declare a fixture `fake_export(tmp_path)` that copies `tests/mcp/fixtures/` into a temp dir and returns the resulting `Config` (with `OPENAI_BASE_URL="http://fake/v1"`, `OPENAI_MODEL="fake-model"`)
  - Declare an autouse fixture that asserts no `cookies.json` access happens during a test run
  - Observable completion: `pytest tests/mcp -k nothing` collects with no errors and zero tests run; `RealLLMCallAttempted` is importable from `tests.mcp.conftest`; the `fake_export` fixture, when used in a smoke test, returns a `Config` whose `by_month_dir` contains exactly the three fixture files
  - _Requirements: 11.1, 11.2, 11.4, 11.5_
  - _Boundary: tests/mcp/_

- [ ] 6.2 (P) Write `test_config.py`
  - Test that `load_config(env={"OPENAI_BASE_URL": "x", "OPENAI_MODEL": "m"})` returns a `Config` with the expected paths and `openai_model == "m"`
  - Test that `load_config(env={"OPENAI_BASE_URL": "x"})` raises `ConfigError` whose message contains `"OPENAI_MODEL"`
  - Test that `load_config(env={"OPENAI_MODEL": "m"})` raises `ConfigError` whose message contains `"OPENAI_BASE_URL"`
  - Test that `OUTPUT_DIR` defaults to `"output"` when absent
  - Test that the `.env` file path code path reads a temp `.env` correctly (file with the three vars plus comment lines)
  - Test that after a successful `load_config`, `os.environ["OPENAI_BASE_URL"]` matches the configured value
  - Test `load_ranker_weights({"RANKER_W_RELEVANCE": "12.0"}).relevance == 12.0`
  - Test `load_ranker_weights({}).relevance == 10.0` (default holds)
  - Test `load_ranker_weights({"RANKER_W_RELEVANCE": "abc"})` raises `ConfigError` whose message contains `RANKER_W_RELEVANCE`
  - Observable completion: `pytest tests/mcp/test_config.py -v` shows green
  - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 13.1_
  - _Depends: 2.1, 2.3, 6.1_
  - _Boundary: tests/mcp/test_config.py_

- [ ] 6.3 (P) Write `test_tree.py`
  - Test `tree.build_tree(tests/mcp/fixtures/by_month)` returns a `TweetTree` whose `nodes_by_month` keys are exactly the three fixture months
  - Test each `TreeNode` has the right `tweet_id`, `handle`, non-empty `text`, non-empty `raw_section`
  - Test that adding a malformed section to a fixture file (missing the link line) causes that section to be skipped without raising
  - Test that `nodes_by_id` covers every node in `nodes_by_month`
  - Test that the parser is pure: calling `build_tree` twice on the same directory yields equal `TweetTree` objects (compare via dataclass equality)
  - Test that `tree.py` does not import the OpenAI SDK (`grep` check via reading the file's source as a string)
  - Observable completion: `pytest tests/mcp/test_tree.py -v` shows green; the no-LLM grep assertion passes
  - _Requirements: 3.6_
  - _Depends: 3.2, 6.1_
  - _Boundary: tests/mcp/test_tree.py_

- [ ] 6.4 (P) Write `test_walker.py`
  - Mock `walker._call_chat_completions` (or equivalent helper). Tests never make real HTTP.
  - Test that `walk(tree, "q", months_in_scope=None, config)` iterates every month
  - Test that `walk(tree, "q", months_in_scope=["2025-01", "2025-03"], config)` iterates only those months (count helper invocations)
  - Test that the helper is called once per chunk; with `chunk_size=2` and a 3-tweet month, the helper is called twice
  - Test that the model gets `model=config.openai_model`
  - Test that JSON entries with unknown ids are dropped
  - Test that JSON entries with `relevance` outside `[0, 1]` or non-numeric are dropped
  - Test that a helper that raises produces `WalkerError` from `walker.walk`
  - Observable completion: `pytest tests/mcp/test_walker.py -v` shows green; no real HTTP calls (network guard would catch them)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_
  - _Depends: 3.3b, 6.1_
  - _Boundary: tests/mcp/test_walker.py_

- [ ] 6.5 (P) Write `test_ranker.py`
  - Pure-function tests with hand-built `WalkerHit` lists, hand-built `Tweet` objects (constructed directly via the dataclass), hand-built `author_affinity` dicts
  - Test `compute_author_affinity` returns `{handle: log1p(count)}` for a list with two tweets from one handle and one from another
  - Test `rank` produces exactly one `ScoredHit` per `WalkerHit` whose tweet is in the map; missing tweets are dropped
  - Test monotonicity: increasing `walker_relevance` (all else equal) increases `score`
  - Test that engagement counts contribute via `log1p`: one tweet with `favorite_count=10` and one with `favorite_count=100` differ by `log1p(100) - log1p(10)` times the weight
  - Test recency decay: `recency_decay(now, anchor=now, halflife=180) == 1.0`; `recency_decay(now-180days, anchor=now, halflife=180)` is approximately `0.5`
  - Test verified and media flags add their constant amounts
  - Test `feature_breakdown` keys sum to `score` (within float tolerance)
  - Test sort order: descending by `score`, deterministic ties
  - Test `ranker.py` does not import the OpenAI SDK
  - Observable completion: `pytest tests/mcp/test_ranker.py -v` shows green
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - _Depends: 3.3c, 6.1_
  - _Boundary: tests/mcp/test_ranker.py_

- [ ] 6.6 (P) Write `test_index.py`
  - Test `TweetIndex.open_or_build` against the fixture export with `tree.build_tree` mocked to return a known-shape fake `TweetTree`: cache absent → builds and writes cache; cache fresh (mtime newer than all `.md`) → loads cache and does not call the builder; cache stale (touch one `.md` newer than cache) → rebuilds (assert via call counter on the mock)
  - Test `TweetIndex.open_or_build` against an empty `by_month/` raises `IndexError`
  - Test `TweetIndex._resolve_filter` with the rule matrix from task 3.3a (year-only, year+month_start, year+range, the three error cases)
  - Test `TweetIndex.search("anything")` (filter unset) calls `walker.walk` (mocked) with `months_in_scope=None`
  - Test `TweetIndex.search("anything", year=2025, month_start="01", month_end="02")` causes `walker.walk` to be called with `months_in_scope=["2025-01", "2025-02"]`
  - Test `TweetIndex.search("anything", year=2025)` resolves to the full year's months
  - Test that the recency anchor passed to `ranker.rank` matches the documented selection rule
  - Test `TweetIndex.lookup_tweet` returns the right `Tweet` for a fixture ID and `None` for `"missing"`
  - Test `TweetIndex.list_months` returns `MonthInfo` list reverse-chronologically with correct counts
  - Test `TweetIndex.get_month_markdown("2025-01")` returns the file content; `get_month_markdown("2099-12")` returns `None`
  - Test that `author_affinity` was precomputed at build time (assert via attribute access)
  - Observable completion: `pytest tests/mcp/test_index.py -v` shows green; the cache-stale test verifies the builder was invoked (call counter on the mock)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.7, 6.1, 6.2, 6.7, 7.1, 7.2, 7.3, 9.1, 9.3_
  - _Depends: 3.1, 3.2, 3.3a, 3.3b, 3.3c, 3.3d, 3.4, 6.1_
  - _Boundary: tests/mcp/test_index.py_

- [ ] 6.7 (P) Write `test_tools.py`
  - For each of the four tools, test the happy path with a mocked `TweetIndex` and the relevant error paths (invalid input, not found, upstream failure)
  - `search_likes`: empty/whitespace `query` → `invalid_input`; valid `query` with mocked matches → list of dicts with the eight expected keys; valid `query` with full filter triple → `index.search` called with exactly those arguments; year-only filter → `index.search` called with `month_start=None, month_end=None`; `month_start` without `year` → `invalid_input` (the resolver raises `ValueError`, the handler translates); `index.search` raising `WalkerError` → `upstream_failure`
  - `list_months`: returns dict list with `year_month`, `path`, `tweet_count` (some may be `None`)
  - `get_month`: bad pattern → `invalid_input`; missing month → `not_found`; valid → returns the string
  - `read_tweet`: empty/non-numeric → `invalid_input`; unknown id → `not_found`; valid → returns the metadata dict
  - Observable completion: `pytest tests/mcp/test_tools.py -v` shows green across all four tools' happy and error paths
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.2, 7.3, 8.1, 8.2, 8.3, 9.1, 9.3, 9.4, 13.4_
  - _Depends: 4.1, 4.2, 6.1_
  - _Boundary: tests/mcp/test_tools.py_

- [ ] 6.8 Write `test_server_integration.py`
  - Build the MCP server in-process via `server.build_server(index)` against the fixture export with `walker.walk` mocked
  - Drive each of the four tools through the SDK's tool-call dispatch (programmatic, not stdio) and assert the response shape matches the declared output schema
  - Verify a `ToolError` raised inside a handler becomes an MCP error response with the right category and the server does not propagate the exception
  - Verify the registered tool list is exactly the four tool names; verify the `search_likes` schema lists `year`, `month_start`, `month_end` as optional fields with the documented patterns
  - Verify a simulated walker failure (`walker.walk` raising `WalkerError`) results in an `upstream_failure` tool error and the server stays alive for subsequent calls
  - Observable completion: `pytest tests/mcp/test_server_integration.py -v` shows green; the empty-query test asserts the response payload contains `"category": "invalid_input"`; a follow-up `list_months` call after the simulated walker failure still returns successfully
  - _Requirements: 1.1, 1.3, 1.5, 6.6, 6.8, 7.4, 8.4, 9.5, 11.3, 13.2, 13.4_
  - _Depends: 5.1, 6.1, 6.6, 6.7_
  - _Boundary: tests/mcp/test_server_integration.py_

## 7. Documentation

- [ ] 7.1 Add MCP Server section to `README.md`
  - Add a new section after the existing usage section titled "MCP Server"
  - Include `.mcp.json` snippet showing the `command`/`args` shape Claude Code expects (`uv run x-likes-mcp` or `python -m x_likes_mcp`)
  - Include the equivalent `claude mcp add` invocation
  - List the three OpenAI env variables (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`) and state that the endpoint is local OpenAI-compatible by default and that the walker uses the OpenAI Python SDK (which reads `OPENAI_BASE_URL` from the process environment automatically)
  - List the optional ranker weight overrides (`RANKER_W_*`) plus `RANKER_RECENCY_HALFLIFE_DAYS`, with the documented defaults and the score formula
  - List the prerequisite that `scrape.sh` has been run at least once and that, after upgrading from a previous version, `./scrape.sh --no-media --format markdown` should be re-run once so per-month files reflect the new (h1-less) shape
  - List the four tools with one-line summaries; for `search_likes`, document the optional `year` / `month_start` / `month_end` filter fields and explain that the filter pre-selects which markdown the walker looks at
  - State that the server is stdio-only and that hosted LLM endpoints are not used by default
  - Document the manual real-LLM verification path (start a local OpenAI-compatible LLM, set env, run server, ask a sample question with and without the structured filter)
  - Observable completion: `grep "MCP Server" README.md` matches; `grep "claude mcp add" README.md` matches; the section names all four tools; the section names the three OpenAI env variables and the ranker weight variables
  - _Requirements: 12.1, 12.2, 12.3, 12.4_
  - _Depends: 5.2_
  - _Boundary: README.md_

## 8. Final integration check

- [ ] 8.1 Run the full test suite end-to-end and verify Spec 1's tests still pass alongside this spec's
  - Run `pytest` from the repo root and confirm both `tests/` (Spec 1) and `tests/mcp/` collect and pass
  - Run `python -m x_likes_mcp` against the existing `output/` directory with a temp `.env` that has `OPENAI_BASE_URL=http://localhost:1234/v1`, `OPENAI_MODEL=any` (no real server needed; startup ends at the SDK stdio loop, send EOF on stdin to exit cleanly)
  - Verify `cookies.json` is not opened during the test run (rely on the conftest guard)
  - Confirm `sentrux scan` still passes against `.sentrux/rules.toml` (no new boundary violations)
  - Confirm `pageindex` is not in the lockfile / installed env: `uv pip show pageindex` exits non-zero
  - Observable completion: `pytest` exit code 0 with both test trees green; `python -m x_likes_mcp` starts the loop and exits cleanly on EOF; no `cookies.json` access during tests; `sentrux scan` produces no new boundary violations relative to the pre-spec baseline; `pageindex` is gone
  - _Requirements: 1.5, 10.1, 10.2, 10.3, 11.1, 11.4, 11.5_
  - _Depends: 1.6, 1.2, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 7.1_
  - _Boundary: integration check_

## 9. Manual smoke (not gated in CI)

- [ ] 9.1 Manual end-to-end smoke against a real local OpenAI-compatible LLM
  - Re-run `./scrape.sh --no-media --format markdown` once if the existing `output/by_month/` was generated before task 1.1 (per-file h1 needs to be gone)
  - Start a local OpenAI-compatible LLM endpoint
  - Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` in `.env`; optionally set `RANKER_W_*` overrides
  - Run `python -m x_likes_mcp` from the project root and register it with Claude Code via `claude mcp add` or `.mcp.json`
  - Issue at least three queries from the MCP client: an open-ended `search_likes(query)`, a year-scoped `search_likes(query, year=2025)`, and a 3-month range `search_likes(query, year=2025, month_start="03", month_end="05")`; confirm sensible answers and visibly faster responses on the filtered queries
  - Issue one `list_months`, one `get_month`, and one `read_tweet` and confirm the responses match what is on disk
  - Inspect the `feature_breakdown` field on a few `search_likes` results to sanity-check the ranker is doing what the formula says
  - Observable completion: each of the manual queries returns a response that pattern-matches the documented output shape; no server crashes; the cache file under `output/tweet_tree_cache.pkl` is created on first run and reused on the second run (mtime older than any `.md` would force a rebuild)
  - _Requirements: 10.4, 12.1, 12.2, 12.3, 12.4_
  - _Depends: 8.1_
  - _Boundary: manual integration check_
