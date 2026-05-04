"""sha256-keyed URL cache for the synthesis-report feature.

Provides ``get`` / ``put`` / ``expire`` operations keyed on
``sha256(url)`` under the configured cache root, with atomic writes via
``tempfile.NamedTemporaryFile`` + :func:`os.replace` (matching the
existing ``embeddings.py`` pattern). Persists only post-sanitize
markdown plus the documented metadata fields; raw HTML / PDF bytes
never touch disk.

Design notes:

* The constructor is intentionally cheap — it does **not** create the
  cache directory. ``put`` materializes the directory lazily on the
  first write; this matches Req 11.4 ("If the cache directory does not
  exist, the synthesis-report orchestrator shall create it before its
  first write").
* The cache key is ``sha256(url.encode("utf-8")).hexdigest()``. The
  ``final_url`` after redirects is stored *inside* the value so the
  same input URL always hits the same cache file regardless of where
  it ultimately resolved.
* ``get`` treats malformed JSON the same as a miss: returning ``None``
  lets the caller re-fetch and the next ``put`` overwrites the bad
  file via the same atomic-replace path.
* ``put`` writes to a ``tempfile.NamedTemporaryFile`` in the same
  directory as the destination so :func:`os.replace` is a same-volume
  rename (atomic on POSIX and on Windows). On any exception the temp
  file is best-effort unlinked so failed writes do not leave
  ``tmpXXXXXX`` debris behind.
* ``expire_older_than`` is a maintenance hook for a future cron-style
  cleanup. It walks every ``*.json`` under the root, deletes files
  whose ``fetched_at`` is older than the cutoff, and returns the
  count. A missing root returns ``0`` rather than erroring (the cache
  is allowed to never have been written to).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["CachedUrl", "UrlCache"]


# Seconds in a day. Hoisted so the TTL math reads as
# ``ttl_days * _SECONDS_PER_DAY`` rather than the bare ``86400`` literal.
_SECONDS_PER_DAY: int = 86400


@dataclass(frozen=True)
class CachedUrl:
    """One cached URL fetch result.

    Attributes:
        url: The original request URL (the cache key derives from this).
        final_url: The URL after any redirects. Same as ``url`` when no
            redirect happened.
        content_type: The response's ``Content-Type`` (without parameters).
        sanitized_markdown: The post-sanitize markdown body. Raw HTML
            or PDF bytes never reach this field — Req 11.3.
        fetched_at: Unix time (seconds) when the fetch completed. Used
            for TTL checks against :func:`time.time`.
    """

    url: str
    final_url: str
    content_type: str
    sanitized_markdown: str
    fetched_at: float


class UrlCache:
    """Sha256-keyed disk cache for sanitized URL bodies.

    One file per URL under ``root``: ``root / f"{sha256(url)}.json"``.
    The directory is created lazily on the first ``put``. ``get``
    returns ``None`` on every flavor of miss (file absent, JSON
    malformed, entry beyond TTL).
    """

    def __init__(self, root: Path, ttl_days: int = 30) -> None:
        """Record the cache root and TTL. Does not touch the filesystem."""

        self.root = root
        self.ttl_days = ttl_days

    # ------------------------------------------------------------------
    # Internal helpers

    def _path_for(self, url: str) -> Path:
        """Return the on-disk path for ``url`` (without checking existence)."""

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    # ------------------------------------------------------------------
    # Public API

    def get(self, url: str) -> CachedUrl | None:
        """Return the cached entry for ``url`` if present and fresh.

        Returns ``None`` when:

        * The cache file does not exist.
        * The file contents are not valid JSON (treat as a miss; the
          next ``put`` will rewrite it via the atomic-replace path).
        * The entry is older than ``self.ttl_days``.

        ``get`` is read-only: it never rewrites the file on disk.
        """

        path = self._path_for(url)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            # Any other read failure (permission denied, decode error)
            # is also surfaced as a cache miss — the caller can re-fetch
            # and the next ``put`` will overwrite the bad file.
            return None

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        if not isinstance(data, dict):
            return None

        try:
            entry = CachedUrl(
                url=str(data["url"]),
                final_url=str(data["final_url"]),
                content_type=str(data["content_type"]),
                sanitized_markdown=str(data["sanitized_markdown"]),
                fetched_at=float(data["fetched_at"]),
            )
        except (KeyError, TypeError, ValueError):
            # Missing fields, wrong types, or non-numeric ``fetched_at``
            # all collapse to "miss".
            return None

        if time.time() - entry.fetched_at > self.ttl_days * _SECONDS_PER_DAY:
            return None

        return entry

    def put(self, entry: CachedUrl) -> None:
        """Atomically write ``entry`` to the cache.

        Behavior:

        1. Lazy-mkdir ``self.root`` (``parents=True, exist_ok=True``).
        2. Compute the destination path from ``entry.url``.
        3. Write the JSON payload to a ``NamedTemporaryFile`` in the
           same directory (so :func:`os.replace` stays a same-volume
           rename), ``flush`` + ``fsync`` it, then promote with
           :func:`os.replace`.
        4. On any exception, best-effort unlink the temp file so failed
           writes do not leave ``tmpXXXXXX`` debris behind.

        The on-disk JSON shape is exactly the five :class:`CachedUrl`
        fields — no headers, no raw bytes, no extra metadata.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        final_path = self._path_for(entry.url)

        payload = json.dumps(asdict(entry), ensure_ascii=False, indent=2)

        # ``delete=False`` so we control the lifecycle: the temp file
        # is either renamed (success) or unlinked (failure) below.
        # ``dir=self.root`` keeps the temp on the same filesystem as
        # the destination so ``os.replace`` is atomic. The ``with``
        # block guarantees the underlying file descriptor is closed
        # even on a write/flush exception; promotion via ``os.replace``
        # then runs against a closed-but-still-present file.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.root,
            prefix=".url_cache.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            try:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            except BaseException:
                # Write/flush/fsync failure: drop the temp file and
                # re-raise. The ``with`` block still runs ``close``
                # for us via its __exit__.
                with contextlib.suppress(OSError):
                    Path(tmp_name).unlink(missing_ok=True)
                raise

        try:
            os.replace(tmp_name, final_path)
        except BaseException:
            # Promotion failure: drop the orphaned temp so a crashed
            # rename does not leave ``.url_cache.*.tmp`` debris. The
            # caller's exception (the original ``OSError`` from
            # ``os.replace``) is preserved.
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)
            raise

    def expire_older_than(self, days: int) -> int:
        """Delete cache files older than ``days``. Return the number deleted.

        Walks every ``*.json`` under :attr:`root`. For each file, parses
        the JSON and checks ``fetched_at`` against ``time.time()``;
        files older than ``days * 86400`` are unlinked. Files that fail
        to parse or are missing the ``fetched_at`` field are left in
        place (a future ``put`` will overwrite them).

        If :attr:`root` does not exist, returns ``0`` without error.
        """

        if not self.root.exists():
            return 0

        cutoff_seconds = days * _SECONDS_PER_DAY
        now = time.time()
        deleted = 0

        for path in self.root.glob("*.json"):
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                fetched_at = float(data["fetched_at"])
            except (
                OSError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                # Corrupt or unreadable entries are skipped here; a
                # later ``put`` against the same URL replaces them.
                continue

            if now - fetched_at > cutoff_seconds:
                with contextlib.suppress(OSError):
                    path.unlink()
                    deleted += 1

        return deleted
