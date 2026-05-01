# Test Fixtures

This directory holds the recorded-and-scrubbed HTTP and on-disk artefacts the
test suite uses instead of touching the live X API or a real `cookies.json`.
The conftest at `tests/conftest.py` blocks all real network I/O and never reads
a real cookies file; every test that needs a response shape, a credential
file, or a previously exported `likes.json` reaches into this directory.

## File inventory

Each fixture carries the minimum bytes its consumer needs. Comments and
extraneous fields from the live responses have been removed where doing so
does not change the structural shape the parser or regex relies on.

| File | Purpose | Captured from | Consumed by |
|------|---------|---------------|-------------|
| `likes_page_success.json` | Full Likes timeline page with 3 tweet entries (one plain, one with a photo, one quote) plus Top and Bottom cursor entries. | Hand-built to mirror the shape of a real Likes GraphQL response (the same shape `full_response.json` at the repo root captures from a live run with valid cookies). All ids and credentials are synthetic. | `tests/test_parser.py` (via `extract_tweets`, `extract_cursor`, `parse_response`); `tests/test_client.py` (registered as the body of a mocked Likes endpoint). |
| `likes_page_empty.json` | Likes page with zero tweet entries, only Top and Bottom cursor entries. | Constructed by deleting all `TimelineTimelineItem` entries from a copy of `likes_page_success.json`. | `tests/test_parser.py`, `tests/test_client.py`. |
| `likes_page_malformed.json` | Likes response missing the `data.user.result.timeline` chain entirely (an `errors[]` payload of the kind X returns when the account is suspended or unreachable). | Constructed from the literal shape X returns for `UserUnavailable` results. | `tests/test_parser.py` (asserts `extract_tweets` returns `[]` and does not raise). |
| `home_page.html` | Scrubbed `https://x.com/home` body. Only structurally relevant content: a `<link rel="preload" ... href="https://abs.twimg.com/responsive-web/client-web/main.<hash>.js" ...>` tag that the regex in `x_likes_exporter/auth.py` looks for. | Captured from a live `https://x.com/home` response; everything except the `<link>` tag and surrounding `<head>` skeleton has been replaced with placeholders. | `tests/test_auth.py` (registered as the body of a mocked `https://x.com/home` GET). |
| `main_script.js` | Scrubbed slice of the X main client-web bundle. Contains exactly one `"Bearer ..."` literal and one `{queryId:"...",operationName:"Likes"}` literal so both regexes in `x_likes_exporter/auth.py` match. | Captured from the live bundle URL the home page links to (the first ~5MB of minified webpack output is irrelevant to the auth tests; only the two literals matter). | `tests/test_auth.py` (registered as the body of a mocked `https://abs.twimg.com/.../main.<hash>.js` GET). |
| `cookies_valid.json` | Browser-export cookies file containing `auth_token`, `ct0`, and `guest_id`. All values are `REDACTED` placeholders. | Hand-built to match the shape an extension like "EditThisCookie" produces. | `tests/test_cookies.py` (`CookieManager(...).validate()` should return `True`). |
| `cookies_missing_ct0.json` | Browser-export cookies file containing `auth_token` and `guest_id` but no `ct0`. Used to exercise the negative path of `CookieManager.validate()`. | Constructed from `cookies_valid.json` by removing the `ct0` entry. | `tests/test_cookies.py` (`CookieManager(...).validate()` should return `False`). |
| `likes_export.json` | Small valid export (3 tweets, one with media, one with a `quoted_tweet`) in the shape `JSONFormatter` writes and `loader.load_export` consumes. | Hand-built from the same synthetic tweets used in `likes_page_success.json`, run through `Tweet.to_dict()` shape. | `tests/test_loader.py` (round-trips through `load_export`). |

## How a fixture was captured (recording procedure)

The procedure below is the runbook for adding or refreshing a fixture. It is
the procedure the existing fixtures were produced with.

1. **Capture from a live run.** With a valid `cookies.json` at the repo root,
   run `scrape.sh`. The repo also keeps `full_response.json` and
   `response_debug.json` from past runs at the project root; those are the
   raw material for the JSON fixtures.
2. **Slice.** Copy the relevant chunk of the live response into a new file
   under `tests/fixtures/`. For the success fixture, take a single Likes
   page and trim entries to two or three representative tweets (one plain,
   one with media, optionally one quote/retweet variant). Keep the Bottom
   cursor entry; it is what `extract_cursor` is tested against.
3. **Scrub.** Apply the substitutions in the next section.
4. **Verify shape.** Run the smoke checks at the bottom of this file.
5. **Verify scrub.** Run the grep checks in the next section.

The empty and malformed JSON fixtures are constructed by editing a success
fixture: remove every `TimelineTimelineItem` entry to make
`likes_page_empty.json`; remove the `data.user.result.timeline` chain
entirely to make `likes_page_malformed.json`.

## Scrubbing rules

Every fixture in this directory has been run through these substitutions
before being committed:

| Field | Real value (live response) | Placeholder (fixture) |
|-------|---------------------------|-----------------------|
| `auth_token` cookie value | a long opaque hex string | `REDACTED` |
| `ct0` cookie value | a long opaque hex string | `REDACTED` |
| `guest_id` cookie value | `v1%3A<hex>` | `v1%3AREDACTED` |
| Bearer token in `main_script.js` | `Bearer AAAA...` (long opaque base64-ish string) | `Bearer REDACTEDBEARERTOKEN0000` |
| Real GraphQL queryId | a long opaque hex string | `REDACTEDQUERYID000000Likes` (and similar) |
| The maintainer's numeric X user id | the real id | the documented test id `1234567890` |
| Third-party `screen_name` values | real screen names from the live timeline | `test_user`, `test_user_two`, `test_user_three`, `test_user_four` |
| Tweet ids, conversation ids | real X ids | synthetic monotonically increasing ids (`1000000000000000001`, `1000000000000000002`, ...) |
| Profile image URL hash, `t.co` short links, `pbs.twimg.com` media URL hash, CSP nonces, CSRF token, tracing trace ids | real values | `REDACTED` (or, for SRI hashes, `sha384-REDACTED`) |

If a fixture is changed (refreshed from a new live capture, or a new tweet
shape is added), re-apply the scrubbing rules above and re-run the
verification commands below.

## Verification checklist

Run from the repo root before committing any change to `tests/fixtures/`.

### 1. Every credential-bearing fixture has at least one `REDACTED` placeholder

```bash
grep -rn "REDACTED" tests/fixtures/
```

Expected: at least one hit in each of `cookies_valid.json`,
`cookies_missing_ct0.json`, `home_page.html`, `main_script.js`,
`likes_page_success.json`, `likes_page_malformed.json`, and
`likes_export.json`. (`likes_page_empty.json` carries cursor strings that
include `REDACTED` too.)

### 2. No known-token prefix appears anywhere

X bearer tokens captured from live responses begin with the literal
`AAAAAAAA` (eight `A`s). If any of those leak into a fixture, they are easy
to grep for:

```bash
grep -rn "Bearer AAAAAAAA" tests/fixtures/
grep -rn "AAAAAAAANRILgAAAA" tests/fixtures/   # the prefix the public web
                                                # bearer token has held for
                                                # several years
```

Both should return zero hits. If either matches, the fixture has a real
token in it; re-scrub and re-commit.

A second class of leaks is the maintainer's own numeric user id (see
`x_account.md` in the user's memory) appearing in fixtures it should not
appear in. The substitution above replaces it with `1234567890`; a
post-edit sanity check is:

```bash
grep -rn "14252145" tests/fixtures/
```

Expected: zero hits.

### 3. Every JSON fixture parses

```bash
for f in tests/fixtures/*.json; do
  uv run python -c "import json, sys; json.load(open(sys.argv[1]))" "$f" || echo "BAD: $f"
done
```

Expected: silent success for every file.

### 4. The HTML and JS fixtures match the auth regexes

```bash
uv run python -c "
import re
html = open('tests/fixtures/home_page.html').read()
js   = open('tests/fixtures/main_script.js').read()
assert re.search(r'<link[^>]+href=\"(https://abs\.twimg\.com/responsive-web/client-web/main\.[^\"]+\.js)\"', html), 'home_page.html: main script link not found'
assert re.search(r'\"(Bearer [\w%]+)\"', js), 'main_script.js: bearer token regex did not match'
assert re.search(r'{queryId:\"([^\"]+)\",operationName:\"Likes\"', js), 'main_script.js: Likes queryId/operationName pair not found'
print('OK')
"
```

Expected: prints `OK`.

### 5. The success fixture parses through the parser

```bash
uv run python -c "
from x_likes_exporter.parser import extract_tweets, extract_cursor
import json
r = json.load(open('tests/fixtures/likes_page_success.json'))
ts = extract_tweets(r)
assert len(ts) >= 1, 'expected at least one tweet'
assert extract_cursor(r) is not None, 'expected a Bottom cursor'
print('parsed', len(ts), 'tweets, cursor present')
"
```

### 6. The empty fixture yields `[]`, the malformed fixture yields `[]` without raising

```bash
uv run python -c "
from x_likes_exporter.parser import extract_tweets
import json
assert extract_tweets(json.load(open('tests/fixtures/likes_page_empty.json'))) == []
assert extract_tweets(json.load(open('tests/fixtures/likes_page_malformed.json'))) == []
print('OK')
"
```

### 7. The export fixture loads through the loader

```bash
uv run python -c "
from x_likes_exporter.loader import load_export
ts = load_export('tests/fixtures/likes_export.json')
assert len(ts) > 0
print('loaded', len(ts), 'tweets')
"
```

If any of the seven checks fails, the fixture is broken and must be fixed
before merge.
