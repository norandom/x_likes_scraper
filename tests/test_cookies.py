"""Tests for ``x_likes_exporter.cookies.CookieManager``.

Covers Requirements 5.1 and 5.2 of the codebase-foundation spec:

- 5.1: A valid ``cookies.json`` (list of objects with ``name``/``value`` pairs
  containing both ``ct0`` and ``auth_token``) loads successfully and exposes
  ``ct0`` via :meth:`CookieManager.get_csrf_token`.
- 5.2: A ``cookies.json`` that is missing ``ct0`` is detected by
  :meth:`CookieManager.validate` returning ``False``.

These tests must exercise the *real* :meth:`CookieManager._load_cookies` against
the on-disk fixtures under ``tests/fixtures/``. The session-scoped autouse
``_no_real_cookies`` fixture in ``conftest.py`` patches ``_load_cookies`` to a
placeholder for the entire suite, so each test here explicitly restores the
original method via ``monkeypatch.setattr`` before constructing the manager.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from x_likes_exporter.cookies import CookieManager

# Capture the unpatched ``_load_cookies`` at import time, before the session
# autouse fixture in ``conftest.py`` swaps it out. Using ``__dict__`` returns
# the underlying function object on the class without descriptor binding, which
# we can then re-bind with ``monkeypatch.setattr`` to undo the patch for a
# single test.
_REAL_LOAD_COOKIES = CookieManager.__dict__["_load_cookies"]


FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_FIXTURE = FIXTURES_DIR / "cookies_valid.json"
MISSING_CT0_FIXTURE = FIXTURES_DIR / "cookies_missing_ct0.json"


@pytest.fixture
def restore_real_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse ``_no_real_cookies`` patch for the current test.

    Re-binds the original ``_load_cookies`` method onto ``CookieManager`` so
    constructing a manager actually reads the JSON fixture from disk.
    ``monkeypatch`` reverts the change automatically at test teardown,
    restoring the session-level patch for subsequent tests.
    """
    monkeypatch.setattr(CookieManager, "_load_cookies", _REAL_LOAD_COOKIES)


def _expected_ct0_from_fixture(path: Path) -> str:
    """Return the ``ct0`` value declared in the JSON fixture at ``path``."""
    with path.open("r", encoding="utf-8") as f:
        cookies_list = json.load(f)
    for cookie in cookies_list:
        if cookie["name"] == "ct0":
            return cookie["value"]
    raise AssertionError(f"fixture {path} has no ct0 cookie")


def test_cookies_valid(restore_real_cookies: None) -> None:
    """A valid cookies.json validates True and exposes the placeholder ct0.

    The real ``_load_cookies`` is restored for this test, so constructing the
    manager actually parses ``tests/fixtures/cookies_valid.json``. The fixture
    contains both ``ct0`` and ``auth_token``, so ``validate()`` is True and
    ``get_csrf_token()`` returns the placeholder ct0 value declared in the
    fixture.
    """
    expected_ct0 = _expected_ct0_from_fixture(VALID_FIXTURE)

    manager = CookieManager(str(VALID_FIXTURE))

    assert manager.validate() is True
    assert manager.get_csrf_token() == expected_ct0


def test_cookies_missing_ct0(restore_real_cookies: None) -> None:
    """A cookies.json without ct0 is rejected by ``validate()``.

    The real ``_load_cookies`` is restored for this test, so constructing the
    manager actually parses ``tests/fixtures/cookies_missing_ct0.json``. The
    fixture has ``auth_token`` but no ``ct0``, so ``validate()`` must return
    False per Requirement 5.2.
    """
    manager = CookieManager(str(MISSING_CT0_FIXTURE))

    assert manager.validate() is False
