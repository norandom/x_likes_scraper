# Brief: codebase-foundation

## Problem

The library has no tests. Every change so far has been validated by running the scraper against a real X account and looking at the output. That's slow, requires valid cookies, and means anything I touch could silently break. Recent refactors have leaned on "looks reasonable" plus a manual run as the gate, which won't scale once the lib has more than one consumer.

It also means the lib doesn't have a public surface. Anyone wanting to use it from outside `cli.py` reaches into the internals (`exporter.tweets`, `exporter.api_client`, etc.). That's blocking the MCP server work in Spec 2.

## Current State

- 22 Python files, ~3,470 LoC, single package `x_likes_exporter/`.
- Sentrux quality signal: 6469. Bottleneck is modularity (3333) because every internal file imports from siblings; that's largely a function of the small project size. Redundancy: 7625 (~24% repeated patterns, mostly the four near-identical date-parse-with-fallback blocks).
- All code paths exercised only via running `scrape.sh`. No `tests/` directory.
- No clean way to open an existing `likes.json` and iterate it without instantiating `XLikesExporter`, which validates cookies it doesn't actually need to read JSON.

## Desired Outcome

When this spec is done:

- `pytest` passes against a mocked X API. The scraper, the response parser, the auth flow, the checkpoint round-trip, and each formatter have meaningful tests on their core paths and on the failure paths actually seen in practice (rate limits, empty pages, malformed responses, expired cookies).
- A small public read API exists for "I have a `likes.json`, give me Tweet objects" and "iterate `output/by_month/`" without forcing a cookies file or a real scrape.
- The sentrux signal goes up. Modularity won't move much (the project stays small), but redundancy should drop as the duplicated date-parse blocks collapse into one helper.

## Approach

- pytest plus `responses` for HTTP mocking. Fixtures load saved API responses from `tests/fixtures/` (recorded once, scrubbed of real tokens).
- Tests organized by module, not by feature: `tests/test_client.py`, `tests/test_formatters.py`, etc.
- One refactor pass for testability: extract the response-parsing logic in `client._extract_tweets` and `_parse_tweet` into pure functions that take dicts and return `Tweet` objects. They're already mostly pure, so this is mechanical.
- One refactor pass for extension: add a `load_export(path)` module-level function (or `XLikesExporter.from_export(path)` classmethod) that returns the same `Tweet` list `fetch_likes()` does, without going through `XLikesExporter.__init__` and its cookie validation.
- Pull the four date-parse-with-fallback blocks into a single helper.

## Scope

- **In:** Test suite, mocking infrastructure, scrubbed fixtures from real responses, the read API, the date-parse helper, type hint completion where it's missing.
- **Out:** New features. New export formats. Async. Logging framework. Performance work. Anything that changes the behavior `scrape.sh` produces today.

## Boundary Candidates

- The read API. Lives in a `loader.py` module, or as classmethods on `XLikesExporter`. Whichever we pick is the public seam Spec 2 builds on.
- The fixtures directory. Tests should run from any checkout without setup beyond `uv sync`.
- The date-parse helper. Single home, every consumer imports it.

## Out of Boundary

- Re-fetching tweets from X under any code path other than `fetch_likes`.
- The CLI's argument parsing and `scrape.sh`'s shell glue. Those work; touching them invites scope creep.
- Test coverage of `cli.py`, `download_media.py`, `split_md_by_month.py`, `update_json_with_local_paths.py`, and `examples/`. They are integration glue, not the testable surface.

## Upstream / Downstream

- **Upstream:** none. This spec is the foundation.
- **Downstream:** mcp-pageindex (Spec 2) consumes the read API.

## Existing Spec Touchpoints

- **Extends:** none.
- **Adjacent:** none.

## Constraints

- Tests must not require real cookies or a real network call.
- No coverage threshold. The success bar is "the functions that matter (scraper, parser, auth, checkpoint) have tests that would catch the kinds of breakage seen recently." Sentrux signal direction (up, not down) is the secondary check.
- No new runtime dependencies. Test deps go under a `dev` group in `pyproject.toml`.
