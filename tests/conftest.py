"""
Shared pytest fixtures for the X Likes Exporter test suite.

This conftest installs two safety nets that apply to every test in the suite:

1. ``_block_real_network`` (autouse, session-scoped):
   Activates the ``responses`` library in its default strict mode for the
   duration of the test session. Any HTTP request made via ``requests`` to a
   URL that has not been explicitly registered on a ``responses`` mock will
   raise ``ConnectionError`` rather than reaching the live network. Tests that
   need to mock specific URLs use ``@responses.activate`` (or
   ``responses.add(...)`` against the session-level mock) to register them;
   any test that does not register a URL but accidentally calls out is failed
   loudly. This implements Requirements 1.4 and 11.2 (no real network I/O
   against ``x.com``, ``twitter.com``, ``abs.twimg.com``, ``pbs.twimg.com``,
   or any other live host).

2. ``_no_real_cookies`` (autouse, session-scoped):
   Patches ``x_likes_exporter.cookies.CookieManager._load_cookies`` so that
   constructing a ``CookieManager`` never reads a real ``cookies.json`` from
   disk. The patched loader returns a placeholder dict containing valid-looking
   ``ct0`` and ``auth_token`` values so ``CookieManager.validate()`` returns
   ``True`` without any disk I/O. This implements Requirement 11.1 (the test
   suite shall not require, read, or create a ``cookies.json`` file at any
   point in its execution).

Tests that need to exercise the real ``_load_cookies`` behavior against fixture
files (e.g. ``tests/fixtures/cookies_valid.json``,
``tests/fixtures/cookies_missing_ct0.json``) explicitly opt out by
unpatching within the test or by using ``monkeypatch.undo()`` on the affected
attribute. The default for the session is "no disk reads of cookies".
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import responses

# Placeholder cookies that pass ``CookieManager.validate()``. Used by the
# autouse cookies-guard fixture so that any accidental construction of
# ``CookieManager`` during tests does not touch the disk and still yields a
# valid-looking object for downstream code.
_PLACEHOLDER_COOKIES = {
    "ct0": "TEST_CT0",
    "auth_token": "TEST_AUTH",
    "guest_id": "TEST_GUEST",
}


@pytest.fixture(autouse=True, scope="session")
def _block_real_network():
    """Activate ``responses`` in strict mode for the whole test session.

    Any HTTP request to a URL that has not been explicitly registered on a
    ``responses`` mock will raise ``ConnectionError``. Tests that need to
    register URLs do so via ``@responses.activate`` decorators or by calling
    ``responses.add(...)`` against the running session mock.
    """
    responses.start()
    try:
        yield
    finally:
        # ``stop()`` deactivates the patch; ``reset()`` clears any registered
        # responses so state does not leak across sessions (this is the last
        # thing that runs at session teardown).
        responses.stop()
        responses.reset()


@pytest.fixture(autouse=True, scope="session")
def _no_real_cookies():
    """Patch ``CookieManager._load_cookies`` to avoid disk reads.

    Returns a placeholder cookie dict that satisfies
    ``CookieManager.validate()`` so any test that incidentally constructs a
    ``CookieManager`` (directly or via ``XAPIClient`` / ``XLikesExporter``)
    proceeds without ever touching ``cookies.json`` on disk.
    """
    with patch(
        "x_likes_exporter.cookies.CookieManager._load_cookies",
        return_value=dict(_PLACEHOLDER_COOKIES),
    ):
        yield
