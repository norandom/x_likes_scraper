"""Tests for ``x_likes_exporter.client.XAPIClient``.

Covers Requirements 3.1-3.6 of the codebase-foundation spec:

- 3.1: ``fetch_likes`` issues a GraphQL request to the Likes endpoint and
  returns parsed tweets, the next-page cursor, and rate-limit info derived
  from the response headers.
- 3.2: An empty page (no tweet entries, only cursors) yields an empty tweet
  list without raising.
- 3.3: An HTTP 429 response surfaces as an exception whose message indicates
  rate-limiting.
- 3.4: An HTTP 401 response surfaces as an exception whose message indicates
  authentication failure.
- 3.5: ``fetch_all_likes`` paginates through successive pages and stops once
  a page returns no next cursor.
- 3.6: When a page reports ``x-rate-limit-remaining: 0``, the wait-and-
  checkpoint branch in ``fetch_all_likes`` is exercised: ``time.sleep`` is
  called for the computed wait and the ``checkpoint_callback`` is invoked
  with the in-flight cursor before the wait.

All tests use ``@responses.activate`` to register HTTP mocks, so no real
network call escapes the autouse ``_block_real_network`` guard from
``conftest.py``. The autouse ``_no_real_cookies`` guard means we can pass any
path to ``CookieManager`` and still get the placeholder cookie dict.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import responses

from x_likes_exporter.client import XAPIClient
from x_likes_exporter.cookies import CookieManager

FIXTURES_DIR = Path(__file__).parent / "fixtures"
COOKIES_FIXTURE = FIXTURES_DIR / "cookies_valid.json"
LIKES_SUCCESS_FIXTURE = FIXTURES_DIR / "likes_page_success.json"
LIKES_EMPTY_FIXTURE = FIXTURES_DIR / "likes_page_empty.json"

# Authenticator stub values. The real authenticator would scrape ``main.js``
# to derive these; here we substitute placeholders so no network call is made
# during authentication.
FAKE_BEARER = "Bearer FAKE_BEARER_TOKEN"
FAKE_QUERY_ID = "FAKE_QUERY_ID"

# The client builds the URL as
# ``https://x.com/i/api/graphql/{query_id}/Likes``. Match exactly so the
# strict ``responses`` mode does not raise ConnectionError.
LIKES_URL = f"https://x.com/i/api/graphql/{FAKE_QUERY_ID}/Likes"

# Cursor values present in the success fixture (parser extracts the Bottom
# cursor as the next-page cursor).
EXPECTED_BOTTOM_CURSOR = "DAABCgABREDACTEDBOTTOMCURSOR"

USER_ID = "14252145"


def _load_fixture(path: Path) -> dict:
    """Load a JSON fixture from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def client() -> XAPIClient:
    """Build an ``XAPIClient`` with a stubbed authenticator.

    The autouse ``_no_real_cookies`` fixture in ``conftest.py`` patches
    ``CookieManager._load_cookies`` to return a placeholder dict, so the path
    we pass here does not need to exist on disk for cookie loading. We point
    at the valid fixture for documentation only.

    The ``authenticator`` attribute is replaced with a ``MagicMock`` whose
    ``get_bearer_token`` and ``get_query_id`` methods return placeholder
    strings. This isolates the client from the real ``XAuthenticator`` (which
    would otherwise issue HTTP calls to scrape the bearer token and query id
    from x.com).
    """
    cookie_manager = CookieManager(str(COOKIES_FIXTURE))
    api_client = XAPIClient(cookie_manager)

    fake_authenticator = MagicMock()
    fake_authenticator.get_bearer_token.return_value = FAKE_BEARER
    fake_authenticator.get_query_id.return_value = FAKE_QUERY_ID
    api_client.authenticator = fake_authenticator

    return api_client


# ---------------------------------------------------------------------------
# Single-page fetch tests (Requirements 3.1, 3.2, 3.3, 3.4)
# ---------------------------------------------------------------------------


@responses.activate
def test_fetch_likes_success(client: XAPIClient) -> None:
    """A 200 response with the success fixture yields tweets, cursor, rate info."""
    body = _load_fixture(LIKES_SUCCESS_FIXTURE)
    responses.add(
        responses.GET,
        LIKES_URL,
        json=body,
        status=200,
        headers={
            "x-rate-limit-limit": "500",
            "x-rate-limit-remaining": "499",
            "x-rate-limit-reset": "1700000000",
        },
    )

    tweets, next_cursor, rate_info = client.fetch_likes(user_id=USER_ID)

    assert len(tweets) > 0, "Success fixture should yield at least one tweet"
    assert next_cursor == EXPECTED_BOTTOM_CURSOR
    assert rate_info.limit == 500
    assert rate_info.remaining == 499
    assert rate_info.reset == 1700000000


@responses.activate
def test_fetch_likes_empty(client: XAPIClient) -> None:
    """An empty page returns no tweets and does not raise.

    The empty fixture has only cursor entries (no tweet items), so the parser
    yields ``[]`` for tweets. The Bottom cursor is still present, so the
    parser returns it; ``fetch_likes`` does not treat that as terminal --
    that decision is made by ``fetch_all_likes``.
    """
    body = _load_fixture(LIKES_EMPTY_FIXTURE)
    responses.add(
        responses.GET,
        LIKES_URL,
        json=body,
        status=200,
        headers={
            "x-rate-limit-limit": "500",
            "x-rate-limit-remaining": "498",
            "x-rate-limit-reset": "1700000000",
        },
    )

    tweets, _next_cursor, rate_info = client.fetch_likes(user_id=USER_ID)

    assert tweets == []
    assert rate_info.limit == 500
    assert rate_info.remaining == 498


@responses.activate
def test_fetch_likes_rate_limit_429(client: XAPIClient) -> None:
    """An HTTP 429 surfaces as an exception with a rate-limit message."""
    responses.add(
        responses.GET,
        LIKES_URL,
        json={"errors": [{"message": "Rate limit exceeded"}]},
        status=429,
        headers={
            "x-rate-limit-limit": "500",
            "x-rate-limit-remaining": "0",
            "x-rate-limit-reset": "1700000000",
        },
    )

    with pytest.raises(Exception) as exc_info:
        client.fetch_likes(user_id=USER_ID)

    # The client wraps HTTPError(429) as ``Rate limit exceeded. Please wait
    # before retrying.`` and then re-wraps it via the bare ``Exception`` arm
    # of ``except Exception`` as ``Error fetching likes: ...``. Either way
    # the substring ``Rate limit`` must appear.
    assert "Rate limit" in str(exc_info.value)


@responses.activate
def test_fetch_likes_auth_401(client: XAPIClient) -> None:
    """An HTTP 401 surfaces as an exception with an authentication message."""
    responses.add(
        responses.GET,
        LIKES_URL,
        json={"errors": [{"message": "Unauthorized"}]},
        status=401,
        headers={
            "x-rate-limit-limit": "500",
            "x-rate-limit-remaining": "499",
            "x-rate-limit-reset": "1700000000",
        },
    )

    with pytest.raises(Exception) as exc_info:
        client.fetch_likes(user_id=USER_ID)

    # The client raises ``Authentication failed. Please check your cookies.``
    # which the outer ``except Exception`` then wraps as ``Error fetching
    # likes: Authentication failed. ...``. Match on the stable substring.
    assert "Authentication" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Pagination tests (Requirements 3.5, 3.6)
# ---------------------------------------------------------------------------


def _success_body_with_cursor(cursor_value: str) -> dict:
    """Return the success fixture body with the Bottom cursor swapped.

    ``fetch_all_likes`` paginates by feeding the previous response's Bottom
    cursor as the next page's ``cursor`` variable. To assert that pagination
    is actually happening (and not just hitting the same response twice), we
    rewrite the Bottom cursor on each page so each registration is
    distinguishable.
    """
    body = _load_fixture(LIKES_SUCCESS_FIXTURE)
    entries = body["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][0]["entries"]
    for entry in entries:
        content = entry.get("content", {})
        if (
            content.get("entryType") == "TimelineTimelineCursor"
            and content.get("cursorType") == "Bottom"
        ):
            content["value"] = cursor_value
    return body


def _success_body_without_cursor() -> dict:
    """Return the success fixture body with the Bottom cursor entry removed.

    ``parser.extract_cursor`` returns ``None`` when no Bottom cursor entry is
    present; ``fetch_all_likes`` treats ``next_cursor is None`` as the
    pagination terminator.
    """
    body = _load_fixture(LIKES_SUCCESS_FIXTURE)
    instructions = body["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][0]
    instructions["entries"] = [
        entry
        for entry in instructions["entries"]
        if entry.get("content", {}).get("cursorType") != "Bottom"
    ]
    return body


@responses.activate
def test_fetch_all_likes_pagination(client: XAPIClient, monkeypatch) -> None:
    """``fetch_all_likes`` paginates through pages and stops on no-cursor.

    Three responses are registered against the same URL; ``responses`` consumes
    them in order. Page 1 and page 2 each return the success body with a
    distinct Bottom cursor; page 3 returns the success body with the Bottom
    cursor entry removed, so the parser yields ``next_cursor = None`` and
    ``fetch_all_likes`` terminates.
    """
    # Skip the inter-request delay to keep the test fast.
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    page1 = _success_body_with_cursor("CURSOR_PAGE_1")
    page2 = _success_body_with_cursor("CURSOR_PAGE_2")
    page3 = _success_body_without_cursor()

    # Healthy rate-limit headers so the wait-and-checkpoint branch is NOT
    # taken in this test (that path is exercised separately below).
    healthy_headers = {
        "x-rate-limit-limit": "500",
        "x-rate-limit-remaining": "400",
        "x-rate-limit-reset": "1700000000",
    }

    for body in (page1, page2, page3):
        responses.add(
            responses.GET,
            LIKES_URL,
            json=body,
            status=200,
            headers=healthy_headers,
        )

    all_tweets = client.fetch_all_likes(user_id=USER_ID)

    # Each fixture page contributes 3 tweets (per the success fixture
    # entries). Three pages -> 9 tweets total.
    assert len(all_tweets) == 9
    # Exactly three HTTP requests were issued.
    assert len(responses.calls) == 3


@responses.activate
def test_fetch_all_likes_rate_limit_branch(client: XAPIClient, monkeypatch) -> None:
    """A response with ``x-rate-limit-remaining: 0`` triggers wait + checkpoint.

    ``RateLimitInfo.should_wait()`` returns ``True`` when ``remaining <= 1``,
    so we register the first page with ``remaining = 0`` and a reset 5
    seconds in the future. ``fetch_all_likes`` is expected to:

    1. Save a checkpoint (the current cursor) before sleeping.
    2. Call ``time.sleep`` with the computed wait time.

    A second response with no Bottom cursor terminates the loop. We patch
    ``time.sleep`` (and ``time.time`` so the wait-time computation is stable)
    via ``monkeypatch`` and pass a ``MagicMock`` as the checkpoint callback
    so we can assert both branches were hit.
    """
    # Pin "now" so ``get_wait_time`` returns a deterministic value:
    # reset (now + 5) - now + 5s buffer = 10 seconds.
    fake_now = 1_700_000_000
    monkeypatch.setattr(time, "time", lambda: fake_now)

    sleep_mock = MagicMock()
    monkeypatch.setattr(time, "sleep", sleep_mock)

    checkpoint_callback = MagicMock()

    # Page 1: remaining = 0 (forces wait branch), Bottom cursor present so
    # ``fetch_all_likes`` proceeds to the rate-limit handling block before
    # the next iteration.
    page1 = _success_body_with_cursor("CURSOR_PAGE_1")
    responses.add(
        responses.GET,
        LIKES_URL,
        json=page1,
        status=200,
        headers={
            "x-rate-limit-limit": "500",
            "x-rate-limit-remaining": "0",
            "x-rate-limit-reset": str(fake_now + 5),
        },
    )

    # Page 2: no Bottom cursor -> terminator. Healthy headers so we do not
    # re-enter the wait branch.
    page2 = _success_body_without_cursor()
    responses.add(
        responses.GET,
        LIKES_URL,
        json=page2,
        status=200,
        headers={
            "x-rate-limit-limit": "500",
            "x-rate-limit-remaining": "400",
            "x-rate-limit-reset": str(fake_now + 1000),
        },
    )

    all_tweets = client.fetch_all_likes(
        user_id=USER_ID,
        checkpoint_callback=checkpoint_callback,
    )

    # Pagination still merges all pages (3 tweets per fixture page * 2 pages).
    assert len(all_tweets) == 6

    # The wait-and-checkpoint branch was exercised:
    # - ``time.sleep`` invoked with the computed wait (10s) at minimum.
    # - ``checkpoint_callback`` invoked at least once with (tweets, cursor)
    #   before the sleep.
    assert sleep_mock.called, "time.sleep should be invoked when remaining=0"
    sleep_args = [call.args[0] for call in sleep_mock.call_args_list]
    assert 10 in sleep_args, f"Expected sleep(10) for rate-limit wait, got {sleep_args}"

    assert checkpoint_callback.called, (
        "checkpoint_callback should be invoked before the rate-limit sleep"
    )
    # The checkpoint call before the sleep should carry the cursor that
    # triggered the wait (i.e. CURSOR_PAGE_1, the Bottom cursor of page 1).
    cursors_passed = [call.args[1] for call in checkpoint_callback.call_args_list]
    assert "CURSOR_PAGE_1" in cursors_passed, (
        f"Expected checkpoint with CURSOR_PAGE_1, got {cursors_passed}"
    )
