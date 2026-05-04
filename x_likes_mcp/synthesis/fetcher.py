"""crawl4ai HTTP fetcher for the synthesis-report feature.

This module owns the per-URL fetch pipeline: cache lookup, scheme +
SSRF validation, the POST to the configured crawl4ai container,
re-validation of any redirect chain reported by the container, content
type filtering, sanitization, byte truncation, and cache write.

Design contract (see ``.kiro/specs/synthesis-report/design.md``,
``fetcher`` component section):

* Connection failures during the startup probe surface as
  :class:`ContainerUnreachable`. Per-URL failures (timeout, blocked
  host, bad content type, 5xx, oversize body, malformed response) are
  *soft drops* — :func:`fetch_url` returns ``None`` and
  :func:`fetch_all` skips that URL while the rest of the run continues.
* The crawl4ai container handles HTTP redirects internally toward the
  real target; the host code re-validates the URL the container
  reports as ``final_url`` (and every hop of any ``redirect_chain``)
  through :func:`x_likes_mcp.synthesis.ssrf_guard.resolve_and_check`
  so a 30x chain that smuggles in an internal hostname is caught
  before its body reaches the LM.
* Bodies are sanitized via :func:`x_likes_mcp.sanitize.sanitize_text`
  before they touch the cache. Raw HTML / PDF bytes never reach disk.
* Truncation runs after sanitize and is UTF-8 safe (it never splits a
  multibyte codepoint).
* The cache key is the *original* request URL (so a redirect chain
  cannot fork the same input into two cache entries).

The HTTP client is injectable so tests pass an ``httpx.MockTransport``
without touching real DNS or the network. The SSRF resolver is also
injectable for the same reason.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from x_likes_mcp.sanitize import safe_http_url, sanitize_text

from .ssrf_guard import (
    _DEFAULT_RESOLVER,
    IPNetwork,
    SsrfBlocked,
    resolve_and_check,
)
from .types import FetchedUrl
from .url_cache import CachedUrl, UrlCache

__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "DEFAULT_TIMEOUT",
    "MAX_REDIRECTS",
    "ContainerUnreachable",
    "FetchError",
    "fetch_all",
    "fetch_url",
    "probe_container",
]


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level contract constants
# ---------------------------------------------------------------------------


# Content types crawl4ai is allowed to return. Office formats are *not*
# on the allowlist for v1 (Req 4.6); when crawl4ai returns one we drop
# the URL silently. The allowlist is matched against the response's
# declared type with parameters (``charset=...``) stripped.
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/html",
        "text/plain",
        "application/json",
        "application/pdf",
    }
)


# Per-URL request timeout, in seconds (Req 4.5). Includes connect +
# read; a slow crawl4ai response is treated as an unreachable URL and
# soft-dropped.
DEFAULT_TIMEOUT: float = 5.0


# Maximum number of redirect hops re-validated against the SSRF guard
# (Req 4.4). A reported redirect chain longer than this caps the URL
# as a soft drop.
MAX_REDIRECTS: int = 3


# Probe-specific timeout. Shorter than the per-URL fetch budget because
# the probe runs once at startup and we want a hung container to fail
# fast instead of stalling the orchestrator's first request.
_PROBE_TIMEOUT: float = 2.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Base class for fetcher-side errors that escape :func:`fetch_url`."""


class ContainerUnreachable(FetchError):
    """Raised by :func:`probe_container` when crawl4ai cannot be reached.

    The message names both the configured endpoint and the
    ``CRAWL4AI_BASE_URL`` env var so an operator looking at the error
    sees the override knob without grepping the docs.
    """


# ---------------------------------------------------------------------------
# Client typing
# ---------------------------------------------------------------------------


@runtime_checkable
class _SupportsPost(Protocol):
    """Duck-typed shape of an httpx-style client.

    We never use the full ``httpx.Client`` surface; the fetcher only
    needs ``post`` (for crawl4ai) and ``get`` (for the probe). Defining
    a Protocol here lets tests inject a hand-rolled object whose
    methods raise ``AssertionError`` to assert "the network was not
    touched" without implementing every ``httpx.Client`` method.
    """

    def post(
        self,
        url: str,
        *,
        json: Any | None = ...,
        timeout: float | None = ...,
    ) -> httpx.Response: ...

    def get(self, url: str, *, timeout: float | None = ...) -> httpx.Response: ...


_Resolver = Callable[..., Sequence[tuple[Any, ...]]]


# ---------------------------------------------------------------------------
# probe_container
# ---------------------------------------------------------------------------


def probe_container(
    base_url: str,
    *,
    timeout: float = _PROBE_TIMEOUT,
    client: _SupportsPost | None = None,
) -> None:
    """Verify the crawl4ai container is reachable.

    Issues a single ``GET <base_url>/`` against the configured endpoint.
    Any HTTP-level response (2xx through 5xx) is treated as "the
    container is up" — even a 4xx means it is running and rejected
    *this* particular path. Only transport failures (connection
    refused, DNS error, timeout) raise :class:`ContainerUnreachable`.

    The error message names both ``base_url`` and the
    ``CRAWL4AI_BASE_URL`` env var so an operator who hits the failure
    sees the override knob at a glance.
    """

    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=timeout)

    try:
        try:
            client.get(f"{base_url.rstrip('/')}/", timeout=timeout)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ContainerUnreachable(
                f"crawl4ai container unreachable at {base_url}; "
                "set CRAWL4AI_BASE_URL to override"
            ) from exc
        except httpx.HTTPError as exc:
            # Other transport-level httpx errors (e.g. ProxyError) are
            # still "container unreachable" from the orchestrator's
            # point of view. Catching the broad ``HTTPError`` keeps the
            # probe robust against future httpx exception subclasses.
            raise ContainerUnreachable(
                f"crawl4ai container unreachable at {base_url}; "
                "set CRAWL4AI_BASE_URL to override"
            ) from exc
    finally:
        if owns_client and isinstance(client, httpx.Client):
            client.close()


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------


def fetch_url(
    url: str,
    *,
    crawl4ai_base_url: str,
    cache: UrlCache,
    max_body_bytes: int,
    private_allowlist: Sequence[IPNetwork] = (),
    timeout: float = DEFAULT_TIMEOUT,
    client: _SupportsPost | None = None,
    _resolver: _Resolver = _DEFAULT_RESOLVER,
) -> FetchedUrl | None:
    """Fetch one URL through the crawl4ai container.

    Pipeline:

    1. Cache lookup. A fresh, parseable cache entry short-circuits the
       network entirely (Req 11.1).
    2. Scheme guard via :func:`safe_http_url`. Anything else is a soft
       drop (Req 4.1).
    3. SSRF guard on the *original* hostname (Req 4.2 / 4.3). Private
       ranges may be punched through with ``private_allowlist``.
    4. POST ``{"urls":[url]}`` to ``<crawl4ai_base_url>/crawl`` with
       ``follow_redirects=False`` (the crawl4ai container handles
       redirects internally on its side).
    5. SSRF re-validate the ``final_url`` and every entry in
       ``redirect_chain`` reported by the container (Req 4.4). A chain
       longer than :data:`MAX_REDIRECTS` is a soft drop.
    6. Content-type allowlist (Req 4.6). The declared content type is
       split on ``;`` so ``text/html; charset=utf-8`` matches.
    7. Sanitize + truncate (Req 3.4 / 3.6). The truncation is UTF-8
       safe.
    8. Cache the post-sanitize body (Req 11.3).

    Returns ``None`` for every soft-drop reason; the caller never sees
    a failure, only an absent URL in the result list.
    """

    # 1) Cache hit short-circuits everything.
    cached = cache.get(url)
    if cached is not None:
        return _fetched_from_cache(cached)

    # 2) Scheme guard.
    cleaned_url = safe_http_url(url)
    if cleaned_url is None:
        return None

    # 3) SSRF guard on the original hostname.
    try:
        resolve_and_check(
            cleaned_url,
            private_allowlist=private_allowlist,
            resolver=_resolver,
        )
    except SsrfBlocked:
        return None

    # 4) POST to crawl4ai. Manage the client lifecycle here so callers
    # who pass ``None`` get a default sync client and tests that pass a
    # ``MockTransport``-backed client retain ownership.
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=timeout)

    try:
        crawl_endpoint = f"{crawl4ai_base_url.rstrip('/')}/crawl"
        try:
            response = client.post(
                crawl_endpoint,
                json={
                    "urls": [cleaned_url],
                    "screenshot": False,
                    "extract_markdown": True,
                },
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError):
            return None

        if response.status_code >= 400:
            return None

        try:
            envelope = response.json()
        except (ValueError, json.JSONDecodeError):
            return None

        result = _first_result(envelope)
        if result is None:
            return None

        final_url = str(result.get("final_url") or cleaned_url)
        content_type_raw = str(result.get("content_type") or "text/html")
        markdown_raw = result.get("markdown")
        redirect_chain = result.get("redirect_chain") or []

        # 5) Redirect chain re-validation. Cap at MAX_REDIRECTS.
        if isinstance(redirect_chain, list):
            if len(redirect_chain) > MAX_REDIRECTS:
                return None
            for hop in redirect_chain:
                if not isinstance(hop, str):
                    return None
                if not _ssrf_ok(hop, private_allowlist, _resolver):
                    return None

        # Always re-check the reported final URL (some crawl4ai
        # versions report ``final_url`` without populating
        # ``redirect_chain``).
        if final_url != cleaned_url and not _ssrf_ok(final_url, private_allowlist, _resolver):
            return None

        # 6) Content-type allowlist.
        content_type = content_type_raw.split(";")[0].strip().lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            return None

        # 7) Markdown body must be a non-empty string.
        if not isinstance(markdown_raw, str) or not markdown_raw:
            if content_type == "application/pdf":
                _log.debug("crawl4ai returned empty markdown for PDF %s", cleaned_url)
            return None

        sanitized = sanitize_text(markdown_raw)
        if not sanitized:
            return None

        truncated = _truncate_utf8(sanitized, max_body_bytes)

        # 8) Cache + return.
        entry = CachedUrl(
            url=cleaned_url,
            final_url=final_url,
            content_type=content_type,
            sanitized_markdown=truncated,
            fetched_at=time.time(),
        )
        try:
            cache.put(entry)
        except OSError:
            # A cache write failure must not turn into a hard error;
            # the body is still usable for this run.
            _log.debug("url cache write failed for %s", cleaned_url)

        return FetchedUrl(
            url=cleaned_url,
            final_url=final_url,
            content_type=content_type,
            sanitized_markdown=truncated,
            size_bytes=len(truncated.encode("utf-8")),
        )
    finally:
        if owns_client and isinstance(client, httpx.Client):
            client.close()


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def fetch_all(
    urls: Sequence[str],
    *,
    crawl4ai_base_url: str,
    cache: UrlCache,
    max_body_bytes: int,
    private_allowlist: Sequence[IPNetwork] = (),
    timeout: float = DEFAULT_TIMEOUT,
    client: _SupportsPost | None = None,
    _resolver: _Resolver = _DEFAULT_RESOLVER,
) -> list[FetchedUrl]:
    """Fetch every URL sequentially; soft-drops never break the run.

    De-duplicates the input list while preserving first-occurrence
    order so a tweet that lists the same URL twice does not generate
    two crawl4ai POSTs. Per-URL failures (timeout, SSRF block, bad
    content type, 5xx) drop that URL and leave the rest untouched
    (Req 4.5: "treat a timeout the same as an unreachable URL — skip
    the URL, continue the report").

    The HTTP client is injected once and reused across every URL so a
    test fixture's ``MockTransport`` sees every request through a
    single seam.
    """

    if not urls:
        return []

    # De-dupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in urls:
        if not isinstance(raw, str):
            continue
        if raw in seen:
            continue
        seen.add(raw)
        deduped.append(raw)

    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=timeout)

    results: list[FetchedUrl] = []
    try:
        for candidate in deduped:
            fetched = fetch_url(
                candidate,
                crawl4ai_base_url=crawl4ai_base_url,
                cache=cache,
                max_body_bytes=max_body_bytes,
                private_allowlist=private_allowlist,
                timeout=timeout,
                client=client,
                _resolver=_resolver,
            )
            if fetched is not None:
                results.append(fetched)
    finally:
        if owns_client and isinstance(client, httpx.Client):
            client.close()

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetched_from_cache(cached: CachedUrl) -> FetchedUrl:
    """Translate a :class:`CachedUrl` into a :class:`FetchedUrl`."""

    return FetchedUrl(
        url=cached.url,
        final_url=cached.final_url,
        content_type=cached.content_type,
        sanitized_markdown=cached.sanitized_markdown,
        size_bytes=len(cached.sanitized_markdown.encode("utf-8")),
    )


def _first_result(envelope: object) -> dict[str, Any] | None:
    """Return the first ``results[0]`` dict from a crawl4ai response."""

    if not isinstance(envelope, dict):
        return None
    results = envelope.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    return first


def _ssrf_ok(
    url: str,
    private_allowlist: Sequence[IPNetwork],
    resolver: _Resolver,
) -> bool:
    """Run the SSRF guard for one URL; return ``False`` on any block."""

    cleaned = safe_http_url(url)
    if cleaned is None:
        return False
    try:
        resolve_and_check(
            cleaned,
            private_allowlist=private_allowlist,
            resolver=resolver,
        )
    except SsrfBlocked:
        return False
    return True


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Return ``text`` truncated so its UTF-8 encoding fits ``max_bytes``.

    Walks back from the cap until a UTF-8 codepoint boundary is found.
    Never returns a string whose encoding exceeds the limit; never
    splits a multibyte codepoint mid-sequence.
    """

    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""
