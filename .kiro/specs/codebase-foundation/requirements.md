# Requirements Document

## Project Description (Input)

The X Likes Exporter library has no automated tests. Every change so far has been validated by running the scraper against a real X account and looking at the output. Recent refactors have leaned on "looks reasonable" plus a manual run as the gate, which won't scale once the lib has more than one consumer. The library also has no clean public surface for "I have a `likes.json`, give me Tweet objects": every external consumer has to instantiate `XLikesExporter`, which validates cookies it does not actually need to read JSON. That's blocking the MCP server work in Spec 2 (mcp-pageindex).

This spec brings the lib to a state where it is testable, tested on the paths that have actually broken, and exposes a small read API that a downstream consumer can call without a cookies file or a real scrape. It also collapses the four near-identical date-parse-with-fallback blocks into one helper so the redundancy axis of the sentrux signal moves the right direction.

## Boundary Context

- **In scope**: A pytest-based test suite that runs without real cookies or network; recorded and scrubbed HTTP fixtures; a public read API for loading an existing export; a single date-parse helper that replaces the four duplicates; a refactor of `client.py` parsing into pure functions so they can be tested directly; a `dev` dependency group declared in `pyproject.toml` for test-only packages.
- **Out of scope**: New scraper features, new export formats, async, a logging framework, performance tuning, any change that alters the output `scrape.sh` produces today. Test coverage of `cli.py`, `download_media.py`, `split_md_by_month.py`, `update_json_with_local_paths.py`, and `examples/` is excluded; those are integration glue.
- **Adjacent expectations**: `mcp-pageindex` (Spec 2) is the first consumer of the read API. The seam this spec defines is the contract Spec 2 builds against. If Spec 2 finds the API awkward, that is feedback against this spec, not a Spec 2 bolt-on.

## Requirements

### Requirement 1: Test infrastructure runnable on a fresh checkout

**Objective:** As the maintainer, I want `pytest` to run on any clean checkout after `uv sync --group dev`, so that I and any future contributor can validate changes without setting up cookies or hitting the X API.

#### Acceptance Criteria

1. When a developer runs `uv sync --group dev` on a fresh checkout, the X Likes Exporter project shall install pytest and the HTTP-mocking library into the dev environment without modifying the runtime dependency set used by `scrape.sh`.
2. When a developer runs `pytest` from the repository root with no `cookies.json` present and no network access, the X Likes Exporter test suite shall execute and finish without raising connection errors, missing-file errors, or skips caused by absent credentials.
3. The X Likes Exporter test suite shall organize tests by source module under a top-level `tests/` directory rather than colocated with source files.
4. If a test attempts a real outbound HTTP request, the X Likes Exporter test suite shall fail that test loudly rather than silently making the call.
5. The X Likes Exporter test suite shall load API response fixtures from a `tests/fixtures/` directory that is checked into the repository.

### Requirement 2: Fixtures recorded from real responses and scrubbed of secrets

**Objective:** As the maintainer, I want the HTTP fixtures used by the tests to be derived from actual X API responses but with all real tokens, cookies, and account identifiers replaced by placeholders, so that the fixtures stay representative without leaking credentials into the repository.

#### Acceptance Criteria

1. The X Likes Exporter test suite shall ship at least one recorded fixture for a successful Likes timeline page, one for an empty Likes timeline page, and one for a malformed or unexpected response shape.
2. The X Likes Exporter test fixtures shall not contain real `auth_token`, `ct0`, bearer-token, or account-identifier values; placeholder values shall be used instead.
3. Where a fixture is recorded from a live response, the X Likes Exporter project shall document in the fixtures directory how that fixture was captured and how scrubbing was applied, so future fixtures can be produced the same way.
4. If a recorded fixture is changed, the X Likes Exporter test suite shall continue to run without requiring any other change in the test code than the fixture replacement, when the new fixture preserves the same structural shape.

### Requirement 3: Scraper and pagination tested against mocked HTTP

**Objective:** As the maintainer, I want the X API client to have meaningful tests on its core fetch paths and on the failure paths I have actually seen in practice, so that future refactors of the client cannot silently break pagination, rate-limit handling, or response parsing.

#### Acceptance Criteria

1. When the X API client is invoked against a mocked successful Likes response, the X API client shall return the expected `Tweet` objects, the expected next cursor, and a populated rate-limit info object.
2. When the X API client is invoked against a mocked Likes response with no entries, the X API client shall return an empty tweet list and shall not raise.
3. When the X API client receives a 429 status from the mocked endpoint, the X API client shall surface the rate-limit condition to the caller rather than silently returning an empty list.
4. When the X API client receives a 401 status from the mocked endpoint, the X API client shall surface an authentication failure to the caller.
5. When the X API client iterates pages with a mocked sequence of responses, the X API client shall stop when the next cursor is absent or the page returns zero entries, and shall not request a further page.
6. When the X API client iterates pages and the mocked rate-limit headers indicate the limit is reached, the X API client shall trigger its wait-and-checkpoint behavior rather than continuing to fetch immediately.

### Requirement 4: Response parsing isolated and tested directly

**Objective:** As the maintainer, I want the response parsing logic to be testable without instantiating the API client, so that changes in the X API response shape can be caught by feeding fixture dicts directly to the parser.

#### Acceptance Criteria

1. The X Likes Exporter library shall expose a parsing path that takes a parsed JSON response (a dict) and returns the same `Tweet` list and next cursor that the API client would return for that response.
2. When the parsing path receives a response missing the expected `data.user.result.timeline` chain, the X Likes Exporter library shall return an empty tweet list and shall not raise.
3. When the parsing path receives a tweet entry missing required `legacy` or `core` sub-objects, the X Likes Exporter library shall skip that entry and continue parsing the remaining entries.
4. When the parsing path receives a tweet with a non-numeric or absent `views.count`, the X Likes Exporter library shall produce a `Tweet` with `view_count` set to the documented default rather than raising.
5. When the parsing path receives a quoted or retweeted variant, the X Likes Exporter library shall set the corresponding `is_quote` or `is_retweet` flag on the returned `Tweet`.

### Requirement 5: Authentication and cookies tested without live network

**Objective:** As the maintainer, I want the cookie loader and the bearer-token / query-id extractors to have tests that exercise their failure modes against mocked HTML responses, so that an X-side change in the script URL pattern or the bearer-token regex is detected by the suite rather than by a failed scrape.

#### Acceptance Criteria

1. When the cookie loader is given a JSON file containing the required cookies, the cookie loader shall return a dict containing those cookies and the `validate()` call shall succeed.
2. When the cookie loader is given a JSON file missing `ct0` or `auth_token`, the cookie loader shall report the cookies as invalid via `validate()`.
3. When the authenticator is invoked against a mocked X home page that contains a recognized main-script URL and a bearer token in the script body, the authenticator shall return that bearer token.
4. If the authenticator is invoked against a mocked X home page where the main-script URL pattern is absent, the authenticator shall raise a clear error rather than returning silently.
5. If the authenticator is invoked against a mocked main-script body where the bearer-token regex does not match, the authenticator shall raise a clear error.
6. When the authenticator is invoked twice for the same operation, the authenticator shall reuse the cached token and query id rather than issuing a second pair of HTTP requests.

### Requirement 6: Checkpoint round-trip and resume tested

**Objective:** As the maintainer, I want the checkpoint save/load cycle and the resume path to be tested, so that a regression in the resume logic does not silently lose previously fetched tweets.

#### Acceptance Criteria

1. When the checkpoint manager saves a list of tweets, a cursor, and a user id, and then a fresh checkpoint manager is constructed against the same directory and asked to load, the loaded payload shall round-trip the original tweets, cursor, and user id.
2. When `clear()` is called on the checkpoint manager, the checkpoint files shall be removed, and a subsequent `exists()` call shall report no checkpoint.
3. When the checkpoint manager is asked whether a checkpoint is valid for a given user id, the checkpoint manager shall return true only if the saved checkpoint's user id matches.
4. When the exporter is invoked with `resume=True` and a checkpoint exists for the same user id, the exporter shall start the next fetch from the saved cursor and shall merge new tweets with existing checkpoint tweets without duplicating ids.
5. When the exporter is invoked with `resume=True` and a checkpoint exists for a different user id, the exporter shall discard the existing checkpoint and start fresh.

### Requirement 7: Public read API for an existing export

**Objective:** As a downstream consumer (the `mcp-pageindex` MCP server is the first such consumer), I want to load an existing `likes.json` and iterate the per-month Markdown directory without instantiating `XLikesExporter` and without supplying a cookies file, so that a read-only consumer does not have to depend on scraper internals or fake credentials.

#### Acceptance Criteria

1. When a consumer calls the public read API with a path to a `likes.json` produced by this library, the X Likes Exporter library shall return a list of `Tweet` objects equivalent to what `XLikesExporter.fetch_likes` would have produced for the same data.
2. When a consumer calls the public read API and no `cookies.json` is present in the project, the X Likes Exporter library shall return the loaded tweets without raising and without prompting for cookies.
3. When a consumer calls the public read API with a path to an `output/by_month/` directory, the X Likes Exporter library shall yield the per-month Markdown file paths in a deterministic order so the consumer can iterate them without re-implementing directory discovery.
4. If the public read API is given a path that does not exist, the X Likes Exporter library shall raise a clear error identifying the missing path.
5. If the public read API is given a path that exists but does not contain valid JSON in the expected shape, the X Likes Exporter library shall raise a clear error identifying the parse failure.
6. The public read API shall be importable from the package's top level so a consumer does not have to know which submodule it lives in.

### Requirement 8: Single date-parse helper replacing the four duplicates

**Objective:** As the maintainer, I want one helper that turns an X-format `created_at` string into a `datetime` (or a documented sentinel on failure), so that the four near-identical try/except blocks currently spread across `models.py`, `exporter.py`, and `formatters.py` collapse into one tested function.

#### Acceptance Criteria

1. When the date-parse helper is called with a valid X-format `created_at` string (e.g. `"Sun Nov 09 11:05:17 +0000 2025"`), the helper shall return a `datetime` object representing that moment.
2. If the date-parse helper is called with an empty string or a string that does not match the X format, the helper shall return the documented fallback (either `None` or a sentinel) without raising.
3. The X Likes Exporter library shall route every `created_at` parse it performs through this single helper, including the per-month grouping in the exporter, the per-month grouping in the Markdown formatter, the per-tweet date rendering in the Markdown formatter, and the `Tweet.get_created_datetime()` method.
4. When the helper's behavior is changed, the X Likes Exporter library shall reflect that change in every consumer site without further per-callsite modification.

### Requirement 9: Formatters tested on representative tweet samples

**Objective:** As the maintainer, I want the JSON, Markdown, and HTML formatters to have tests that lock in their output shape on a small sample, so that an accidental change in the rendering of a tweet (date, media block, stats line) is caught by the suite.

#### Acceptance Criteria

1. When the JSON formatter is invoked on a representative `Tweet` list, the X Likes Exporter library shall produce JSON that round-trips back through `json.load` to the same logical structure each `Tweet.to_dict()` produced.
2. When the Markdown formatter is invoked on a representative `Tweet` list, the X Likes Exporter library shall group tweets into month-titled sections in reverse chronological order and render the per-tweet block (handle, name, date, text, stats, URL) consistently with the existing layout.
3. When the Markdown formatter is invoked on a `Tweet` whose `created_at` does not parse, the X Likes Exporter library shall route that tweet to the `unknown` group rather than raising.
4. When the HTML formatter is invoked on a representative `Tweet` list, the X Likes Exporter library shall produce a single HTML document containing one tweet block per input tweet with the user, text, and URL rendered.
5. The Pandas formatter shall produce a DataFrame with one row per tweet and the documented columns, when given a representative `Tweet` list.

### Requirement 10: Sentrux signal direction

**Objective:** As the maintainer, I want the sentrux quality signal to move up after this spec lands, primarily by reducing the redundancy axis through the date-parse helper consolidation, so that the structural-quality direction of travel matches the intent of the refactor.

#### Acceptance Criteria

1. After the date-parse helper consolidation lands, the X Likes Exporter project shall have exactly one implementation of the X-format `created_at` parsing logic, and a sentrux scan shall report a redundancy axis that has improved relative to the pre-spec baseline of 7625 (the direction of "improvement" is per the sentrux tool's polarity for that axis; the intent is fewer duplicated patterns than before).
2. After this spec lands, a sentrux scan of the X Likes Exporter project shall report an overall quality signal that is at least the pre-spec baseline of 6469.
3. If a sentrux scan after the spec lands reports a regression on either axis, the X Likes Exporter project shall treat that as a finding to investigate before declaring the spec done, rather than as a metric to silently override.

### Requirement 11: No real cookies, no real network, no new runtime dependencies

**Objective:** As the maintainer, I want the test suite and the new read API to add no runtime cost and no credential cost, so that the library's deployment surface for users running `scrape.sh` is unchanged by this spec.

#### Acceptance Criteria

1. The X Likes Exporter test suite shall not require, read, or create a `cookies.json` file at any point in its execution.
2. The X Likes Exporter test suite shall not perform any real network I/O against `x.com`, `twitter.com`, `abs.twimg.com`, `pbs.twimg.com`, or any other live host.
3. The X Likes Exporter project shall declare every new test-only dependency under a `dev` dependency group in `pyproject.toml`, and shall not add any new entry to the runtime `dependencies` list.
4. When `scrape.sh` is run after this spec lands, the X Likes Exporter project shall produce the same output files (`likes.json`, `likes.csv`, `likes.html`, and the per-month Markdown directory) as before, modulo timestamps and re-fetched data.
