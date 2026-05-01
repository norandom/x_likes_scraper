# Implementation Plan

This plan implements the design in three layers: refactor production code so it is testable and the read API exists, set up the test infrastructure, then write the tests one source module at a time. Each task touches one boundary; cross-boundary work is called out as integration.

The ordering follows the Refactor Sequencing section of the design. Steps 1-4 are the production-code refactor (must keep `scrape.sh` working at every step). Step 5 sets up dev dependencies. Steps 6-9 build the test suite. Step 10 records the sentrux signal.

- [x] 1. Add the date-parse helper as a leaf module
- [x] 1.1 Create `dates.py` with `parse_x_datetime`
  - Add `x_likes_exporter/dates.py` with the `X_CREATED_AT_FORMAT` constant and a `parse_x_datetime(value: str) -> Optional[datetime]` function that uses `datetime.strptime` and returns `None` on any failure (empty string, wrong format, non-string input).
  - The helper does not raise; all error paths return `None`.
  - Observable: `python -c "from x_likes_exporter.dates import parse_x_datetime; print(parse_x_datetime('Sun Nov 09 11:05:17 +0000 2025'))"` prints a `datetime` object, and the same call with `''` prints `None`.
  - _Requirements: 8.1, 8.2_
  - _Boundary: dates_

- [x] 2. Add the pure parser module
- [x] 2.1 Lift parsing functions out of `client.py` into `parser.py`
  - Create `x_likes_exporter/parser.py` with module-level functions `extract_tweets`, `parse_tweet`, `extract_cursor`, and `parse_response` whose bodies are the existing `_extract_tweets`, `_parse_tweet`, `_extract_cursor` logic with `self` removed.
  - Functions never raise on missing keys: a malformed response yields `[]` from `extract_tweets` and `None` from `extract_cursor`; a per-entry failure is skipped.
  - Replace the bodies of `XAPIClient._extract_tweets`, `_parse_tweet`, `_extract_cursor` with two-line passthroughs that delegate to `parser`.
  - Observable: running `scrape.sh` against a real account produces the same `likes.json` it produced before this task; running `python -c "from x_likes_exporter.parser import extract_tweets; print(extract_tweets({}))"` prints `[]` without raising.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - _Boundary: parser, client_

- [x] 3. Replace the four duplicated date-parse blocks
- [x] 3.1 Route `models.py`, `exporter.py`, and `formatters.py` through `parse_x_datetime`
  - In `models.py:Tweet.get_created_datetime`, replace the `dateutil.parser.parse` call with `parse_x_datetime(self.created_at)`. Translate a `None` return into a raise so the existing public contract of this method is preserved.
  - In `exporter.py:export_markdown`, replace the inline `datetime.strptime(tweet.created_at, "%a %b %d ...")` block in the per-month grouping with `parse_x_datetime(tweet.created_at)`; route `None` to the `unknown` group.
  - In `formatters.py:MarkdownFormatter.export`, replace the per-month grouping that calls `tweet.get_created_datetime()` with `parse_x_datetime(tweet.created_at)`; route `None` to the `unknown` group.
  - In `formatters.py:MarkdownFormatter._format_tweet`, replace the per-tweet date-string block with `parse_x_datetime(tweet.created_at)`; on `None`, fall back to the raw `tweet.created_at` string (current behavior).
  - Observable: `grep -rn "strptime" x_likes_exporter/` returns no results (the format string lives only in `dates.py`); `grep -rn "dateutil.parser" x_likes_exporter/` returns no results; `scrape.sh` against a real account produces the same per-month files and the same Markdown rendering as before this task.
  - _Requirements: 8.3, 8.4_
  - _Boundary: dates, models, exporter, formatters_

- [x] 4. Add the public read API
- [x] 4.1 Implement `loader.py` with `load_export` and `iter_monthly_markdown`
  - Create `x_likes_exporter/loader.py` with `load_export(path: str | Path) -> list[Tweet]` that reads a `likes.json`, reconstructs `Tweet` objects (and nested `User`, `Media`, `quoted_tweet`, `retweeted_tweet`), and returns the list.
  - Add `iter_monthly_markdown(path: str | Path) -> Iterator[Path]` that yields files matching `likes_YYYY-MM.md` under the given directory in reverse-chronological order; non-matching files are skipped.
  - Raise `FileNotFoundError` (with the missing path in the message) when the input path does not exist; raise `ValueError` (with the failing field in the message) when the JSON does not match the expected shape.
  - The loader does not import from `cookies.py`, `auth.py`, `client.py`, or `exporter.py`. It depends only on `models.py` and stdlib.
  - Observable: `python -c "from x_likes_exporter import load_export; tweets = load_export('output/likes.json'); print(len(tweets), type(tweets[0]).__name__)"` prints a count and `Tweet` without ever reading `cookies.json`.
  - _Requirements: 7.1, 7.2, 7.4, 7.5_
  - _Boundary: loader_

- [x] 4.2 Re-export the read API from the package top level
  - Update `x_likes_exporter/__init__.py` to import `load_export` and `iter_monthly_markdown` and add them to `__all__`.
  - Observable: `python -c "from x_likes_exporter import load_export, iter_monthly_markdown; print('ok')"` prints `ok` from a fresh interpreter.
  - _Requirements: 7.3, 7.6_
  - _Boundary: loader, __init___

- [x] 5. Set up the dev dependency group
- [x] 5.1 Add `[dependency-groups].dev` to `pyproject.toml`
  - Add a `[dependency-groups]` section with a `dev` array containing `pytest>=8.0` and `responses>=0.25`. Do not modify `[project.dependencies]`.
  - Run `uv sync --group dev` and confirm both packages install into the dev environment.
  - Observable: `uv pip list | grep -E "pytest|responses"` prints both names; `grep -A 5 "\[project\]" pyproject.toml | grep dependencies` shows the same runtime dependencies as before.
  - _Requirements: 1.1, 11.3_
  - _Boundary: pyproject_

- [ ] 6. Set up the test infrastructure
- [x] 6.1 Create `tests/` skeleton with `conftest.py` and a network guard
  - Create `tests/__init__.py` and `tests/conftest.py`.
  - In `conftest.py`, configure an autouse session fixture that activates `responses` in strict mode so any unregistered URL raises `ConnectionError`.
  - Add a session-scoped fixture that asserts no `cookies.json` exists at the project root during the test run, or patches `CookieManager._load_cookies` to a no-op so accidental construction does not read the file.
  - Observable: `pytest --collect-only` succeeds and reports zero tests; running a throwaway test that calls `requests.get('https://example.com')` fails with a `ConnectionError` rather than reaching the network.
  - _Requirements: 1.2, 1.3, 1.4, 11.1, 11.2_
  - _Boundary: tests_

- [x] 6.2 Record and scrub the test fixtures
  - Create `tests/fixtures/` with: `likes_page_success.json` (a full Likes page with one or more tweet entries), `likes_page_empty.json` (a Likes timeline with zero entries), `likes_page_malformed.json` (a response missing the `data.user.result.timeline` chain), `home_page.html` (an `x.com/home` body containing a recognized `main.<hash>.js` link), `main_script.js` (a script body containing one Bearer token literal and one `queryId/operationName` pair for `Likes`), `cookies_valid.json`, `cookies_missing_ct0.json`, `likes_export.json` (a small valid export the loader will consume).
  - Apply the scrubbing rules from `design.md`: replace any `auth_token`, `ct0`, `guest_id`, bearer token, real user id, and third-party `screen_name` with placeholder values (`REDACTED`, `Bearer REDACTED`, the documented test user id, `test_user`).
  - Add `tests/fixtures/README.md` documenting how each fixture was captured and the scrubbing checklist (the `grep -r REDACTED` and `grep -r <known-token-prefix>` checks).
  - Observable: `grep -rn "REDACTED" tests/fixtures/` returns at least one hit per credential-bearing fixture; `python -c "import json; json.load(open('tests/fixtures/likes_page_success.json'))"` succeeds.
  - _Requirements: 1.5, 2.1, 2.2, 2.3, 2.4_
  - _Boundary: tests/fixtures_

- [ ] 7. Write unit tests for the leaf modules
- [x] 7.1 (P) `test_dates.py`
  - Cases: a known-good X-format string parses to a timezone-aware `datetime` matching the expected components; an empty string returns `None`; a string in an unrelated format (e.g. ISO 8601) returns `None`; a non-string input returns `None` rather than raising.
  - Observable: `pytest tests/test_dates.py -v` reports all cases passing.
  - _Requirements: 8.1, 8.2_
  - _Boundary: dates_

- [x] 7.2 (P) `test_parser.py`
  - Cases: `extract_tweets(load_fixture('likes_page_success.json'))` returns the expected number of `Tweet` objects with expected ids and `screen_name`; `extract_tweets(load_fixture('likes_page_empty.json'))` returns `[]`; `extract_tweets(load_fixture('likes_page_malformed.json'))` returns `[]` and does not raise; `extract_cursor` returns the bottom cursor when present and `None` when absent; hand-built dicts exercise the per-entry edge cases (missing `legacy`, missing `core`, non-numeric `views.count`, retweet variant with `retweeted_status_result`, quote variant with `quoted_status_result`).
  - Observable: `pytest tests/test_parser.py -v` reports all cases passing; one case explicitly asserts `view_count == 0` for a tweet whose fixture has `views.count = "abc"`.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - _Boundary: parser_

- [x] 7.3 (P) `test_cookies.py`
  - Cases: `CookieManager('tests/fixtures/cookies_valid.json').validate()` returns `True` and `get_csrf_token()` returns the placeholder ct0; `CookieManager('tests/fixtures/cookies_missing_ct0.json').validate()` returns `False`.
  - Observable: `pytest tests/test_cookies.py -v` reports both cases passing.
  - _Requirements: 5.1, 5.2_
  - _Boundary: cookies_

- [x] 7.4 (P) `test_checkpoint.py`
  - Cases: save a small `Tweet` list with a cursor and user id under `tmp_path`, then construct a fresh `Checkpoint` and assert `load()` returns the same tweets, cursor, and user id; `clear()` removes both files and `exists()` returns `False`; `is_valid('matching_uid')` returns `True` and `is_valid('other_uid')` returns `False`.
  - Observable: `pytest tests/test_checkpoint.py -v` reports all cases passing without writing anywhere outside `tmp_path`.
  - _Requirements: 6.1, 6.2, 6.3_
  - _Boundary: checkpoint_

- [x] 7.5 (P) `test_loader.py`
  - Cases: `load_export('tests/fixtures/likes_export.json')` returns a `list[Tweet]` whose elements' `to_dict()` equals the source JSON entries; `load_export('does/not/exist.json')` raises `FileNotFoundError` with the path in the message; `load_export(<path to a JSON that is a string, not a list>)` raises `ValueError`; `iter_monthly_markdown(tmp_path)` over a directory containing `likes_2024-03.md`, `likes_2024-01.md`, `likes_2025-02.md`, and `notes.md` yields paths in the order `2025-02, 2024-03, 2024-01` and skips `notes.md`; `iter_monthly_markdown('does/not/exist')` raises `FileNotFoundError`; `from x_likes_exporter import load_export, iter_monthly_markdown` succeeds.
  - Observable: `pytest tests/test_loader.py -v` reports all cases passing.
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - _Boundary: loader_

- [x] 7.6 (P) `test_formatters.py`
  - Build a small hand-written `Tweet` list of three tweets covering: a plain tweet with one media item, a retweet, and a tweet whose `created_at` is an unparseable string. Run each formatter against the list under `tmp_path`.
  - JSON: re-load the written file and assert it equals `[t.to_dict() for t in tweets]`.
  - Markdown: assert the output contains a section header with `unknown` for the unparseable-date tweet, sections for the parseable months in reverse chronological order, and one tweet block per tweet with handle, name, date, text, and stats present.
  - HTML: assert the output is a single document with `<div class='tweet'>` appearing exactly three times and the user `screen_name` text appearing for each tweet.
  - Pandas: assert the returned DataFrame has three rows and the documented column set (`tweet_id`, `text`, `created_at`, `user_screen_name`, ...).
  - Observable: `pytest tests/test_formatters.py -v` reports all cases passing; the `unknown` group routing is asserted in the Markdown case.
  - _Requirements: 8.3, 9.1, 9.2, 9.3, 9.4, 9.5_
  - _Boundary: formatters_

- [ ] 8. Write the network-mocked tests for the auth and client layers
- [ ] 8.1 `test_auth.py`
  - Cases (each uses `@responses.activate`): register `https://x.com/home` returning `home_page.html` and the script URL it references returning `main_script.js`; assert `XAuthenticator.get_bearer_token()` returns the placeholder bearer literal in the script; register a home page with no `<link>` matching the pattern and assert the call raises with a clear message; register a main script with no bearer literal and assert the call raises with a clear message; call `get_bearer_token()` twice and assert exactly one network round-trip occurred (cache reuse); call `get_query_id('Likes')` twice and assert the same.
  - Observable: `pytest tests/test_auth.py -v` reports all cases passing; the cache-reuse case asserts `len(responses.calls) == 2` (one home, one script) after the second `get_bearer_token` invocation.
  - _Requirements: 5.3, 5.4, 5.5, 5.6_
  - _Boundary: auth_

- [ ] 8.2 `test_client.py`
  - Cases (each uses `@responses.activate`, with `XAPIClient` constructed against a `CookieManager` pointing at `tests/fixtures/cookies_valid.json` and an authenticator stubbed to return placeholder bearer / queryId without network calls): register the Likes endpoint returning `likes_page_success.json` with rate-limit headers populated and assert `fetch_likes` returns the expected tweets, next cursor, and rate-limit info; register the Likes endpoint returning `likes_page_empty.json` and assert an empty list with no raise; register the endpoint returning HTTP 429 and assert `fetch_likes` raises with a rate-limit message; register HTTP 401 and assert it raises with an authentication message; register a sequence of two pages followed by a no-cursor terminator and assert `fetch_all_likes` stops after the terminator and returns the merged tweet list; register a response whose `x-rate-limit-remaining` header is `0` and assert the wait-and-checkpoint branch is exercised (assert by stubbing `time.sleep` and `checkpoint_callback` and checking they were invoked).
  - Observable: `pytest tests/test_client.py -v` reports all cases passing; the pagination case asserts the request count matches the expected page count; no test triggers a real `ConnectionError` from the conftest network guard.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - _Boundary: client_

- [ ] 9. Cover the exporter resume path
- [ ] 9.1 `test_exporter_resume.py`
  - Cases: pre-populate a checkpoint under `tmp_path` for `user_id="A"`; instantiate `XLikesExporter` with cookies pointing at `tests/fixtures/cookies_valid.json`, an `output_dir=tmp_path`, and `XAPIClient.fetch_all_likes` stubbed to return a list of new tweets including one whose id matches a tweet already in the checkpoint; call `fetch_likes(user_id="A", resume=True, download_media=False)` and assert the returned list has no duplicate ids and that the original cursor was passed into the stubbed `fetch_all_likes`. Repeat with `user_id="B"` (different from the checkpoint's `"A"`) and assert the checkpoint was cleared and the fetch started fresh.
  - Observable: `pytest tests/test_exporter_resume.py -v` reports both cases passing; the duplicate-id assertion is explicit.
  - _Requirements: 6.4, 6.5_
  - _Boundary: exporter, checkpoint_

- [ ] 10. Validate the sentrux signal direction
- [ ] 10.1 Run sentrux scan and record the result
  - Run `sentrux scan` from the repo root after all preceding tasks land.
  - Compare the redundancy axis to the pre-spec baseline of 7625 and the overall signal to the pre-spec baseline of 6469.
  - Record the new numbers (e.g. in the spec's research notes or a follow-up commit message); do not silently treat a regression on either axis as acceptable.
  - Observable: a recorded sentrux output exists with both numbers; if redundancy improved (lower number means less duplication, depending on the tool's polarity — confirm against the tool's docs), note that explicitly; if either axis regressed, note the cause and decide whether to address it before declaring the spec done.
  - _Requirements: 10.1, 10.2, 10.3_
  - _Boundary: project-wide_

- [ ] 11. Manual smoke check (optional, manual gate)
- [ ] 11.1* Run `scrape.sh` against a real account and confirm output is unchanged
  - With valid cookies, run `scrape.sh` and confirm `output/likes.json`, `output/likes.csv`, `output/likes.html`, and `output/by_month/likes_YYYY-MM.md` are produced and their content shape matches what the same command produced before the refactor (modulo timestamps and any new likes since the previous run).
  - This is the final manual gate for Requirement 11.4. Cannot be automated without real cookies and is documented as a manual check, marked optional.
  - Observable: a maintainer signs off that the output files are present and structurally identical to the pre-refactor output.
  - _Requirements: 11.4_
  - _Boundary: project-wide_
