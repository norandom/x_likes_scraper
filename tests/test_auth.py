"""Tests for ``x_likes_exporter.auth.XAuthenticator``.

Covers Requirements 5.3-5.6 of the codebase-foundation spec:

- 5.3: When invoked against a mocked X home page that contains a recognized
  main-script URL and a bearer token in the script body, the authenticator
  shall return that bearer token.
- 5.4: If invoked against a mocked X home page where the main-script URL
  pattern is absent, the authenticator shall raise a clear error rather than
  returning silently.
- 5.5: If invoked against a mocked main-script body where the bearer-token
  regex does not match, the authenticator shall raise a clear error.
- 5.6: When invoked twice for the same operation, the authenticator shall
  reuse the cached token and query id rather than issuing a second pair of
  HTTP requests.

All HTTP traffic is mocked via the ``responses`` library. The autouse session
fixture ``_block_real_network`` in ``conftest.py`` keeps ``responses`` active
in strict mode for the whole session, so any URL that is not registered here
raises ``ConnectionError`` rather than reaching the live network. Each test
adds a per-test scope on top via ``@responses.activate``, which resets
registered URLs and the call list between tests.

Cookies are not exercised in this module: ``XAuthenticator.get_bearer_token``
and ``get_query_id`` only forward the cookie dict on the request to the home
page, and the bearer/query-id extraction reads the response body. The
session-level ``_no_real_cookies`` fixture supplies a placeholder
``CookieManager`` so no disk I/O occurs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

from x_likes_exporter.auth import XAuthenticator
from x_likes_exporter.cookies import CookieManager

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HOME_PAGE_PATH = FIXTURES_DIR / "home_page.html"
MAIN_SCRIPT_PATH = FIXTURES_DIR / "main_script.js"

HOME_URL = "https://x.com/home"
# The main-script URL referenced by ``home_page.html``. The auth regex matches
# the literal ``main.<hash>.js`` filename in a ``<link>`` tag's ``href``.
SCRIPT_URL = "https://abs.twimg.com/responsive-web/client-web/main.ABCDEF12.js"

# Expected extraction results for the unmodified fixtures.
EXPECTED_BEARER = "Bearer REDACTEDBEARERTOKEN0000"
EXPECTED_LIKES_QUERY_ID = "REDACTEDQUERYID000000Likes"


@pytest.fixture
def home_page_html() -> str:
    """Read the scrubbed X.com home page fixture as text."""
    return HOME_PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def main_script_js() -> str:
    """Read the scrubbed main-script bundle fixture as text."""
    return MAIN_SCRIPT_PATH.read_text(encoding="utf-8")


@pytest.fixture
def authenticator() -> XAuthenticator:
    """Build a fresh ``XAuthenticator`` backed by a placeholder ``CookieManager``.

    The session-level ``_no_real_cookies`` autouse fixture patches
    ``CookieManager._load_cookies`` to return a placeholder dict, so the file
    path passed here is never actually opened. A fresh authenticator per test
    guarantees the per-instance ``_bearer_token`` / ``_query_ids`` caches start
    empty, which is essential for the cache-reuse assertions below.
    """
    cm = CookieManager(str(FIXTURES_DIR / "cookies_valid.json"))
    return XAuthenticator(cm)


@responses.activate
def test_get_bearer_token_success(
    authenticator: XAuthenticator,
    home_page_html: str,
    main_script_js: str,
) -> None:
    """5.3: bearer literal in the script body is returned verbatim."""
    responses.add(
        responses.GET,
        HOME_URL,
        body=home_page_html,
        content_type="text/html",
        status=200,
    )
    responses.add(
        responses.GET,
        SCRIPT_URL,
        body=main_script_js,
        content_type="application/javascript",
        status=200,
    )

    token = authenticator.get_bearer_token()

    assert token == EXPECTED_BEARER
    # Sanity check: exactly one home fetch + one script fetch.
    assert len(responses.calls) == 2


@responses.activate
def test_get_bearer_token_no_link_raises(
    authenticator: XAuthenticator,
) -> None:
    """5.4: home page lacking the ``<link>`` to ``main.<hash>.js`` raises."""
    html_without_link = (
        "<!doctype html><html><head>"
        "<title>X</title>"
        '<link rel="stylesheet" href="https://abs.twimg.com/responsive-web/client-web/shared.REDACTED.css">'
        "</head><body></body></html>"
    )
    responses.add(
        responses.GET,
        HOME_URL,
        body=html_without_link,
        content_type="text/html",
        status=200,
    )

    with pytest.raises(Exception) as exc_info:
        authenticator.get_bearer_token()

    # The error message should clearly identify the failure point. ``auth.py``
    # wraps the inner failure with an "Error getting Bearer token" prefix, so
    # both the wrapper and the underlying "main script URL" phrase appear.
    message = str(exc_info.value)
    assert "main script URL" in message
    # The script URL was never registered, so we must have failed before
    # attempting to fetch it.
    assert len(responses.calls) == 1


@responses.activate
def test_get_bearer_token_no_bearer_in_script_raises(
    authenticator: XAuthenticator,
    home_page_html: str,
) -> None:
    """5.5: a script body lacking the bearer literal raises a clear error."""
    script_without_bearer = (
        "/* scrubbed script with no bearer literal */\n"
        'var __MARKER_QUERIES__ = [{queryId:"X",operationName:"Likes"}];\n'
    )
    responses.add(
        responses.GET,
        HOME_URL,
        body=home_page_html,
        content_type="text/html",
        status=200,
    )
    responses.add(
        responses.GET,
        SCRIPT_URL,
        body=script_without_bearer,
        content_type="application/javascript",
        status=200,
    )

    with pytest.raises(Exception) as exc_info:
        authenticator.get_bearer_token()

    message = str(exc_info.value)
    assert "Bearer token" in message
    # We did get to the script fetch, so both URLs were called.
    assert len(responses.calls) == 2


@responses.activate
def test_get_bearer_token_caches(
    authenticator: XAuthenticator,
    home_page_html: str,
    main_script_js: str,
) -> None:
    """5.6: a second ``get_bearer_token()`` call must not hit the network."""
    responses.add(
        responses.GET,
        HOME_URL,
        body=home_page_html,
        content_type="text/html",
        status=200,
    )
    responses.add(
        responses.GET,
        SCRIPT_URL,
        body=main_script_js,
        content_type="application/javascript",
        status=200,
    )

    first = authenticator.get_bearer_token()
    second = authenticator.get_bearer_token()

    assert first == second == EXPECTED_BEARER
    # Exactly one home fetch + one script fetch. The second invocation served
    # from the in-memory ``_bearer_token`` cache.
    assert len(responses.calls) == 2


@responses.activate
def test_get_query_id_success(
    authenticator: XAuthenticator,
    home_page_html: str,
    main_script_js: str,
) -> None:
    """5.3 (query-id variant): the ``Likes`` query id is extracted verbatim."""
    responses.add(
        responses.GET,
        HOME_URL,
        body=home_page_html,
        content_type="text/html",
        status=200,
    )
    responses.add(
        responses.GET,
        SCRIPT_URL,
        body=main_script_js,
        content_type="application/javascript",
        status=200,
    )

    query_id = authenticator.get_query_id("Likes")

    assert query_id == EXPECTED_LIKES_QUERY_ID
    assert len(responses.calls) == 2


@responses.activate
def test_get_query_id_caches(
    authenticator: XAuthenticator,
    home_page_html: str,
    main_script_js: str,
) -> None:
    """5.6 (query-id variant): repeated ``get_query_id`` reuses the cache."""
    responses.add(
        responses.GET,
        HOME_URL,
        body=home_page_html,
        content_type="text/html",
        status=200,
    )
    responses.add(
        responses.GET,
        SCRIPT_URL,
        body=main_script_js,
        content_type="application/javascript",
        status=200,
    )

    first = authenticator.get_query_id("Likes")
    second = authenticator.get_query_id("Likes")

    assert first == second == EXPECTED_LIKES_QUERY_ID
    # Cache reuse: still exactly one home fetch + one script fetch.
    assert len(responses.calls) == 2
