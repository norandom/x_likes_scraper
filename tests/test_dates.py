"""Tests for ``x_likes_exporter.dates.parse_x_datetime``.

Covers Requirements 8.1 and 8.2 of the codebase-foundation spec:

- 8.1: A valid X ``created_at`` string parses to a ``datetime`` representing
  that moment.
- 8.2: An empty string, a string in another format, or any non-string input
  returns the documented fallback (``None``) without raising.

The helper is pure (no I/O, no network, no cookies), so these tests run in
isolation under the autouse network/cookie guards in ``conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from x_likes_exporter.dates import parse_x_datetime


def test_parses_valid_x_format() -> None:
    """A known-good X ``created_at`` string parses to a tz-aware datetime."""
    result = parse_x_datetime("Sun Nov 09 11:05:17 +0000 2025")

    assert result is not None
    assert isinstance(result, datetime)
    # Component-by-component check pins the exact moment.
    assert result.year == 2025
    assert result.month == 11
    assert result.day == 9
    assert result.hour == 11
    assert result.minute == 5
    assert result.second == 17
    # Timezone-aware comparison: the offset must be UTC (+0000).
    assert result.tzinfo is not None
    assert result.utcoffset() == UTC.utcoffset(result)


def test_returns_none_on_empty_string() -> None:
    """An empty string yields ``None`` rather than raising."""
    assert parse_x_datetime("") is None


def test_returns_none_on_iso_8601() -> None:
    """A string in ISO 8601 (wrong format for X) yields ``None``."""
    # ISO 8601 is a perfectly valid datetime string in another format; the
    # helper must reject it because it does not match the X created_at format
    # ("%a %b %d %H:%M:%S %z %Y") and must do so without raising.
    assert parse_x_datetime("2025-11-09T11:05:17+00:00") is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        12345,
        [],
        {},
    ],
    ids=["None", "int", "list", "dict"],
)
def test_returns_none_on_non_string(value: object) -> None:
    """Non-string inputs return ``None`` without raising ``TypeError``."""
    # ``parse_x_datetime`` is typed as taking ``str``, but the helper is the
    # safety net for the four legacy callsites that previously each did their
    # own try/except. Defensive callers (e.g. JSON-loaded data of unknown
    # shape) must be able to pass arbitrary values without crashing.
    assert parse_x_datetime(value) is None  # type: ignore[arg-type]
