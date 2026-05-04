"""Tests for the sha256-keyed URL cache (task 2.2 of synthesis-report).

These tests cover the contract described in design.md's ``url_cache``
component section and exercise the four acceptance criteria of
Requirement 11 (cache lookup before fetch, 30-day TTL, post-sanitize
storage only, lazy directory creation).

The implementation under test is :mod:`x_likes_mcp.synthesis.url_cache`,
which provides a small ``UrlCache`` service plus the ``CachedUrl`` value
object. All tests are stdlib-only, isolated under ``tmp_path``, and
neither read nor write outside the test directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from x_likes_mcp.synthesis.url_cache import CachedUrl, UrlCache


def _make_entry(
    url: str = "https://example.com/a",
    final_url: str | None = None,
    content_type: str = "text/html",
    sanitized_markdown: str = "hello",
    fetched_at: float | None = None,
) -> CachedUrl:
    """Build a :class:`CachedUrl` with sensible defaults for tests."""

    return CachedUrl(
        url=url,
        final_url=final_url if final_url is not None else url,
        content_type=content_type,
        sanitized_markdown=sanitized_markdown,
        fetched_at=fetched_at if fetched_at is not None else time.time(),
    )


# ---------------------------------------------------------------------------
# get / put round-trip
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    """A fresh cache (no files at all) returns ``None`` from ``get``."""

    cache = UrlCache(tmp_path / "url_cache")
    assert cache.get("https://example.com/missing") is None


def test_put_then_get_round_trip(tmp_path: Path) -> None:
    """``put`` followed by ``get(same_url)`` returns an equal :class:`CachedUrl`."""

    cache = UrlCache(tmp_path / "url_cache")
    entry = _make_entry(url="https://example.com/a", sanitized_markdown="body text")
    cache.put(entry)

    got = cache.get("https://example.com/a")
    assert got == entry


def test_get_handles_corrupt_file_as_miss(tmp_path: Path) -> None:
    """A non-JSON cache file is treated as a miss; no exception escapes."""

    root = tmp_path / "url_cache"
    root.mkdir(parents=True)
    url = "https://example.com/a"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    (root / f"{digest}.json").write_text("xyz", encoding="utf-8")

    cache = UrlCache(root)
    assert cache.get(url) is None


# ---------------------------------------------------------------------------
# Lazy directory creation
# ---------------------------------------------------------------------------


def test_put_creates_directory(tmp_path: Path) -> None:
    """``put`` creates the cache root (lazily) on the first write.

    The constructor must not eagerly create the directory; only ``put``
    materializes it. This matches Req 11.4.
    """

    root = tmp_path / "deep" / "url_cache"
    assert not root.exists()

    cache = UrlCache(root)
    # Constructor must not have created the directory.
    assert not root.exists()

    cache.put(_make_entry(url="https://example.com/lazy"))
    assert root.is_dir()


# ---------------------------------------------------------------------------
# sha256 key shape
# ---------------------------------------------------------------------------


def test_sha256_key_is_stable_across_runs(tmp_path: Path) -> None:
    """The cache filename for a given URL is exactly ``sha256(url).hexdigest() + .json``."""

    root = tmp_path / "url_cache"
    cache = UrlCache(root)
    url = "https://example.com/path?q=1"
    cache.put(_make_entry(url=url))

    expected_name = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"
    assert (root / expected_name).is_file()


# ---------------------------------------------------------------------------
# TTL boundaries
# ---------------------------------------------------------------------------


def test_ttl_within_default_returns_entry(tmp_path: Path) -> None:
    """29-day-old entry still hits with the default 30-day TTL."""

    cache = UrlCache(tmp_path / "url_cache")
    entry = _make_entry(
        url="https://example.com/fresh",
        fetched_at=time.time() - 29 * 86400,
    )
    cache.put(entry)

    got = cache.get("https://example.com/fresh")
    assert got is not None
    assert got.url == entry.url


def test_ttl_beyond_default_returns_none(tmp_path: Path) -> None:
    """31-day-old entry misses with the default 30-day TTL."""

    cache = UrlCache(tmp_path / "url_cache")
    entry = _make_entry(
        url="https://example.com/stale",
        fetched_at=time.time() - 31 * 86400,
    )
    cache.put(entry)

    assert cache.get("https://example.com/stale") is None


def test_explicit_ttl_overrides_default(tmp_path: Path) -> None:
    """An explicit ``ttl_days=1`` rejects an entry that is 2 days old."""

    cache = UrlCache(tmp_path / "url_cache", ttl_days=1)
    entry = _make_entry(
        url="https://example.com/short-ttl",
        fetched_at=time.time() - 2 * 86400,
    )
    cache.put(entry)

    assert cache.get("https://example.com/short-ttl") is None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_no_partial_files_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``os.replace`` crashes mid-write, no committed ``<sha>.json`` exists.

    Simulates a crash between writing the temp file and promoting it.
    The committed cache filename must NOT exist (a temp file may linger,
    but it must not match the ``<sha>.json`` pattern the loader checks).
    """

    root = tmp_path / "url_cache"
    cache = UrlCache(root)

    # Patch ``os.replace`` *as imported by the url_cache module* so the
    # replacement step blows up after the temp file has been written.
    from x_likes_mcp.synthesis import url_cache as url_cache_mod

    def _boom(src: str, dst: str) -> None:
        raise OSError("simulated mid-write crash")

    monkeypatch.setattr(url_cache_mod.os, "replace", _boom, raising=True)

    url = "https://example.com/atomic"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    final_path = root / f"{digest}.json"

    with pytest.raises(OSError, match="simulated mid-write crash"):
        cache.put(_make_entry(url=url, sanitized_markdown="will not commit"))

    # The committed cache file must not exist.
    assert not final_path.exists()


# ---------------------------------------------------------------------------
# On-disk persistence shape
# ---------------------------------------------------------------------------


def test_put_does_not_persist_raw_html(tmp_path: Path) -> None:
    """Sanity check: only the documented ``CachedUrl`` fields land on disk.

    No raw HTML, headers, or other body bytes leak into the cache file.
    Storing only post-sanitize markdown is Req 11.3.
    """

    root = tmp_path / "url_cache"
    cache = UrlCache(root)
    url = "https://example.com/strict-shape"
    entry = _make_entry(
        url=url,
        final_url="https://example.com/redirected",
        content_type="text/html",
        sanitized_markdown="post-sanitize body",
        fetched_at=1735000000.0,
    )
    cache.put(entry)

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    on_disk = json.loads((root / f"{digest}.json").read_text(encoding="utf-8"))

    assert set(on_disk.keys()) == {
        "url",
        "final_url",
        "content_type",
        "sanitized_markdown",
        "fetched_at",
    }
    assert on_disk["url"] == url
    assert on_disk["final_url"] == "https://example.com/redirected"
    assert on_disk["content_type"] == "text/html"
    assert on_disk["sanitized_markdown"] == "post-sanitize body"
    assert on_disk["fetched_at"] == 1735000000.0


# ---------------------------------------------------------------------------
# Expire
# ---------------------------------------------------------------------------


def test_expire_older_than_removes_stale_entries(tmp_path: Path) -> None:
    """``expire_older_than(30)`` deletes 35d and 65d entries; the 5d entry survives."""

    cache = UrlCache(tmp_path / "url_cache")
    now = time.time()
    fresh = _make_entry(url="https://example.com/fresh", fetched_at=now - 5 * 86400)
    stale_one = _make_entry(url="https://example.com/stale1", fetched_at=now - 35 * 86400)
    stale_two = _make_entry(url="https://example.com/stale2", fetched_at=now - 65 * 86400)
    for entry in (fresh, stale_one, stale_two):
        cache.put(entry)

    deleted = cache.expire_older_than(30)
    assert deleted == 2

    # Fresh entry survives; stale entries are gone.
    assert cache.get("https://example.com/fresh") is not None
    fresh_digest = hashlib.sha256(b"https://example.com/fresh").hexdigest()
    stale1_digest = hashlib.sha256(b"https://example.com/stale1").hexdigest()
    stale2_digest = hashlib.sha256(b"https://example.com/stale2").hexdigest()
    root = tmp_path / "url_cache"
    assert (root / f"{fresh_digest}.json").exists()
    assert not (root / f"{stale1_digest}.json").exists()
    assert not (root / f"{stale2_digest}.json").exists()


def test_expire_older_than_no_dir_returns_zero(tmp_path: Path) -> None:
    """``expire_older_than`` on a never-created cache root returns 0, not an error."""

    cache = UrlCache(tmp_path / "never_created")
    assert cache.expire_older_than(30) == 0


# ---------------------------------------------------------------------------
# Read-only ``get``
# ---------------------------------------------------------------------------


def test_get_does_not_mutate_disk(tmp_path: Path) -> None:
    """``get`` must not rewrite the file on read (mtime + bytes unchanged)."""

    root = tmp_path / "url_cache"
    cache = UrlCache(root)
    url = "https://example.com/readonly"
    cache.put(_make_entry(url=url, sanitized_markdown="hello"))

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    file_path = root / f"{digest}.json"
    bytes_before = file_path.read_bytes()
    mtime_before = os.stat(file_path).st_mtime_ns

    # Backdate the mtime so a stray write would visibly change it.
    past = mtime_before - 10 * 1_000_000_000  # 10 seconds earlier
    os.utime(file_path, ns=(past, past))
    mtime_before = os.stat(file_path).st_mtime_ns

    got = cache.get(url)
    assert got is not None

    assert os.stat(file_path).st_mtime_ns == mtime_before
    assert file_path.read_bytes() == bytes_before
