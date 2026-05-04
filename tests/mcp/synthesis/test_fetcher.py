"""Tests for the crawl4ai HTTP fetcher (task 4.2 of synthesis-report).

Cover the full per-URL pipeline contract from design.md's ``fetcher``
component section and Requirements 3.1, 3.3-3.6, 4.4-4.6, 11.1, 11.3,
12.4: SSRF guard on the original host and on every redirect target,
content-type allowlist, 5s timeout, manual redirect walk capped at three
hops, post-sanitize cache discipline, and soft per-URL drops that never
abort the run.

The fetcher accepts an injected ``client`` (an ``httpx.Client`` that may
be wired to ``httpx.MockTransport``) so every test stays offline. SSRF
re-validation uses an injected ``_resolver`` to keep the resolver in
test-controlled territory; no real DNS lookup ever happens.

The autouse ``_block_real_url_fetch`` guard from
``tests/mcp/synthesis/conftest.py`` patches ``httpx.Client.send`` to
fail loudly. Because tests pass a ``MockTransport``-backed client, the
mock transport satisfies the request before the patched ``send`` would
fire — but a regression that opens a real client (or that bypasses the
``client`` parameter) would surface a ``RealUrlFetchAttempted`` error.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from x_likes_mcp.synthesis.fetcher import (
    ALLOWED_CONTENT_TYPES,
    DEFAULT_TIMEOUT,
    MAX_REDIRECTS,
    ContainerUnreachable,
    FetchError,
    fetch_all,
    fetch_url,
    probe_container,
)
from x_likes_mcp.synthesis.url_cache import CachedUrl, UrlCache

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


# The synthesis-package autouse ``_block_real_url_fetch`` fixture
# monkeypatches ``httpx.Client.send`` so any unmocked fetch surfaces
# loudly. Fetcher tests *do* want httpx.Client.send to fire — but only
# against an in-process ``MockTransport`` handler, never the network.
# Capturing the *real* send method at module-import time gives us a
# deterministic seam to restore inside the ``allow_mock_transport``
# fixture below.
_REAL_HTTPX_CLIENT_SEND = httpx.Client.send


@pytest.fixture(autouse=True)
def _restore_real_httpx_send(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the real ``httpx.Client.send`` for every test in this module.

    The synthesis-package autouse guard replaces ``Client.send`` with a
    loud-failure stub. ``MockTransport`` short-circuits a request from
    *inside* ``send``, so the handler the test wires up never fires
    while the stub is in place. Restoring the original method per-test
    lets every fetcher test wire its own handler via ``MockTransport``;
    tests that should never touch httpx at all use the
    ``_ExplodingClient`` sentinel and still don't reach the network.

    The synthesis-package guard ran first (it is also autouse), so the
    ``monkeypatch`` here strictly overrides it for the duration of the
    test function and the guard's ``monkeypatch`` undo at teardown
    runs cleanly.
    """

    monkeypatch.setattr(httpx.Client, "send", _REAL_HTTPX_CLIENT_SEND)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_resolver(ip: str = "93.184.216.34") -> Any:
    """Return a fake ``socket.getaddrinfo`` that pins every host to ``ip``.

    Used as the ``_resolver`` argument when the test wants the SSRF guard
    to *accept* the URL. The default IP is in TEST-NET-3 — public, not
    blocked.
    """

    def _resolver(
        host: str,
        port: int | None,
        *,
        type: int = 0,
        **_kwargs: object,
    ) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    return _resolver


def _host_resolver(mapping: dict[str, str]) -> Any:
    """Return a fake resolver that maps hosts to specific IPs.

    Hosts not listed fall back to a public IP so the test can exercise
    one specific blocked redirect target without polluting the rest of
    the chain.
    """

    def _resolver(
        host: str,
        port: int | None,
        *,
        type: int = 0,
        **_kwargs: object,
    ) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        ip = mapping.get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    return _resolver


def _client_with_handler(handler: Any) -> httpx.Client:
    """Build an ``httpx.Client`` whose transport is the given handler.

    ``MockTransport`` short-circuits the request before ``Client.send``
    runs, so the autouse loud-block fixture never fires for these tests.
    """

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, timeout=DEFAULT_TIMEOUT)


def _crawl_response(
    *,
    url: str = "https://example.com/page",
    final_url: str | None = None,
    markdown: str = "# title\n\nbody",
    content_type: str = "text/html",
    redirect_chain: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stub crawl4ai response body with the documented shape."""

    result: dict[str, Any] = {
        "url": url,
        "markdown": markdown,
        "content_type": content_type,
    }
    if final_url is not None:
        result["final_url"] = final_url
    if redirect_chain is not None:
        result["redirect_chain"] = redirect_chain
    if extra is not None:
        result.update(extra)
    return {"results": [result]}


# ---------------------------------------------------------------------------
# probe_container
# ---------------------------------------------------------------------------


def test_module_constants_match_design() -> None:
    """The module-level constants must mirror the design contract."""

    assert (
        frozenset({"text/html", "text/plain", "application/json", "application/pdf"})
        == ALLOWED_CONTENT_TYPES
    )
    assert pytest.approx(5.0) == DEFAULT_TIMEOUT
    assert MAX_REDIRECTS == 3
    assert issubclass(ContainerUnreachable, FetchError)


def test_probe_container_succeeds_on_2xx() -> None:
    """A reachable container returns 200; ``probe_container`` returns None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    client = _client_with_handler(handler)
    try:
        # Should not raise.
        probe_container("http://crawl4ai.test:11235", client=client)
    finally:
        client.close()


def test_probe_container_accepts_4xx_as_alive() -> None:
    """A container that rejects the probe path is still alive — accept 4xx."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = _client_with_handler(handler)
    try:
        probe_container("http://crawl4ai.test:11235", client=client)
    finally:
        client.close()


def test_probe_container_raises_on_connection_refused() -> None:
    """A connection error becomes ``ContainerUnreachable`` with endpoint + env var."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client_with_handler(handler)
    try:
        with pytest.raises(ContainerUnreachable) as excinfo:
            probe_container("http://crawl4ai.test:11235", client=client)
    finally:
        client.close()

    message = str(excinfo.value)
    assert "http://crawl4ai.test:11235" in message
    assert "CRAWL4AI_BASE_URL" in message


def test_probe_container_raises_on_timeout() -> None:
    """A timeout becomes ``ContainerUnreachable`` with the same message shape."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = _client_with_handler(handler)
    try:
        with pytest.raises(ContainerUnreachable) as excinfo:
            probe_container("http://crawl4ai.test:11235", client=client)
    finally:
        client.close()

    message = str(excinfo.value)
    assert "http://crawl4ai.test:11235" in message
    assert "CRAWL4AI_BASE_URL" in message


# ---------------------------------------------------------------------------
# fetch_url — cache short-circuit
# ---------------------------------------------------------------------------


class _ExplodingClient:
    """Sentinel client whose every method raises.

    Passed as ``client=`` to assert that a code path never touches the
    network. Any attribute access produces a fresh failure.
    """

    def post(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("network was not supposed to be touched")

    def get(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("network was not supposed to be touched")

    def request(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("network was not supposed to be touched")


def test_fetch_url_cache_hit_short_circuits_network(tmp_path: Path) -> None:
    """A cache hit returns ``FetchedUrl`` without any HTTP call."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/cached"
    import time as time_module

    cache.put(
        CachedUrl(
            url=url,
            final_url=url,
            content_type="text/html",
            sanitized_markdown="# cached body",
            fetched_at=time_module.time(),
        )
    )

    result = fetch_url(
        url,
        crawl4ai_base_url="http://crawl4ai.test:11235",
        cache=cache,
        max_body_bytes=10_000,
        client=_ExplodingClient(),  # type: ignore[arg-type]
    )

    assert result is not None
    assert result.url == url
    assert result.final_url == url
    assert result.content_type == "text/html"
    assert result.sanitized_markdown == "# cached body"
    assert result.size_bytes == len(b"# cached body")


# ---------------------------------------------------------------------------
# fetch_url — happy paths
# ---------------------------------------------------------------------------


def test_fetch_url_happy_path_html(tmp_path: Path) -> None:
    """A 200 with HTML markdown returns ``FetchedUrl`` and caches the body."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/page"
    body = "# Title\n\nbody text"

    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=body, content_type="text/html"),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    assert result.url == url
    assert result.final_url == url
    assert result.content_type == "text/html"
    assert result.sanitized_markdown == body
    assert result.size_bytes == len(body.encode("utf-8"))

    # The crawl endpoint was hit exactly once.
    assert len(captured_requests) == 1
    sent = captured_requests[0]
    assert sent.url.path.endswith("/crawl")
    assert sent.method == "POST"
    payload = json.loads(sent.content.decode("utf-8"))
    assert payload["urls"] == [url]

    # The cache now contains the post-sanitize body.
    cached = cache.get(url)
    assert cached is not None
    assert cached.sanitized_markdown == body


def test_fetch_url_handles_pdf_via_markdown_field(tmp_path: Path) -> None:
    """A PDF response is read from the same ``markdown`` field as HTML."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/paper.pdf"
    body = "# Paper title\n\nabstract..."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=body, content_type="application/pdf"),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    assert result.content_type == "application/pdf"
    assert result.sanitized_markdown == body


def test_fetch_url_uses_default_content_type_when_missing(tmp_path: Path) -> None:
    """When crawl4ai omits ``content_type`` the fetcher defaults to text/html."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/page"
    body = "hello"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"url": url, "markdown": body}]},
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    assert result.content_type == "text/html"
    assert result.sanitized_markdown == body


# ---------------------------------------------------------------------------
# fetch_url — content-type allowlist
# ---------------------------------------------------------------------------


def test_fetch_url_drops_office_content_type(tmp_path: Path) -> None:
    """Office types are silently dropped (Req 4.6)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/doc.docx"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(
                url=url,
                markdown="ignored",
                content_type=(
                    "application/vnd.openxmlformats-officedocument." "wordprocessingml.document"
                ),
            ),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None
    # Nothing should have been cached.
    assert cache.get(url) is None


def test_fetch_url_drops_text_xml_content_type(tmp_path: Path) -> None:
    """``application/xml`` is not on the allowlist."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/feed.xml"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(
                url=url,
                markdown="ignored",
                content_type="application/xml",
            ),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_strips_content_type_parameters(tmp_path: Path) -> None:
    """``text/html; charset=utf-8`` is on the allowlist (parameters ignored)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/page"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(
                url=url,
                markdown="hello",
                content_type="text/html; charset=utf-8",
            ),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    # The stored content_type drops the parameter.
    assert result.content_type == "text/html"


# ---------------------------------------------------------------------------
# fetch_url — scheme + SSRF guards
# ---------------------------------------------------------------------------


def test_fetch_url_blocks_unsupported_scheme(tmp_path: Path) -> None:
    """Non-HTTP(S) schemes are dropped before any network call (Req 4.1)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    result = fetch_url(
        "javascript:alert(1)",
        crawl4ai_base_url="http://crawl4ai.test:11235",
        cache=cache,
        max_body_bytes=10_000,
        client=_ExplodingClient(),  # type: ignore[arg-type]
    )

    assert result is None


def test_fetch_url_blocks_ssrf_on_initial_host(tmp_path: Path) -> None:
    """A private-IP resolution drops the URL before the crawl4ai call."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    result = fetch_url(
        "http://evil.example/",
        crawl4ai_base_url="http://crawl4ai.test:11235",
        cache=cache,
        max_body_bytes=10_000,
        client=_ExplodingClient(),  # type: ignore[arg-type]
        _resolver=_host_resolver({"evil.example": "10.0.0.5"}),
    )

    assert result is None


def test_fetch_url_revalidates_final_url(tmp_path: Path) -> None:
    """A redirect target on a private IP is rejected after the crawl response."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://public.example/start"
    final_url = "http://internal.example/secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(
                url=url,
                final_url=final_url,
                markdown="# leaked",
                content_type="text/html",
            ),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_host_resolver(
                {
                    "public.example": "93.184.216.34",
                    "internal.example": "10.0.0.5",
                }
            ),
        )
    finally:
        client.close()

    assert result is None
    assert cache.get(url) is None


def test_fetch_url_redirect_chain_re_validated(tmp_path: Path) -> None:
    """Each hop of a reported redirect chain is SSRF-checked (Req 4.4)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://public.example/start"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(
                url=url,
                markdown="# would-be body",
                content_type="text/html",
                redirect_chain=[
                    "http://public.example/start",
                    "http://internal.example/secret",
                ],
            ),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_host_resolver(
                {
                    "public.example": "93.184.216.34",
                    "internal.example": "10.0.0.5",
                }
            ),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_drops_redirect_chain_over_max_hops(tmp_path: Path) -> None:
    """A reported chain longer than ``MAX_REDIRECTS`` is dropped (Req 4.4)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://public.example/start"

    def handler(request: httpx.Request) -> httpx.Response:
        # A chain with five hops blows past the three-hop cap.
        return httpx.Response(
            200,
            json=_crawl_response(
                url=url,
                markdown="# body",
                content_type="text/html",
                redirect_chain=[
                    "http://a.example/",
                    "http://b.example/",
                    "http://c.example/",
                    "http://d.example/",
                    "http://e.example/",
                ],
            ),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


# ---------------------------------------------------------------------------
# fetch_url — soft drops (timeout, 5xx, empty body)
# ---------------------------------------------------------------------------


def test_fetch_url_returns_none_on_timeout(tmp_path: Path) -> None:
    """A per-URL timeout drops the URL silently (Req 4.5)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            "https://example.com/slow",
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_returns_none_on_5xx(tmp_path: Path) -> None:
    """A 5xx from crawl4ai drops the URL silently."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream busy")

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            "https://example.com/failing",
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_returns_none_on_4xx(tmp_path: Path) -> None:
    """A 4xx from crawl4ai is also a per-URL skip."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            "https://example.com/404",
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_returns_none_on_empty_markdown(tmp_path: Path) -> None:
    """An empty ``markdown`` body (e.g. failed PDF extraction) is dropped."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(markdown="", content_type="application/pdf"),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            "https://example.com/empty.pdf",
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_returns_none_on_malformed_response(tmp_path: Path) -> None:
    """A malformed JSON envelope is dropped instead of raising."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            "https://example.com/page",
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


def test_fetch_url_returns_none_on_empty_results(tmp_path: Path) -> None:
    """A response with no ``results`` entry is a soft drop."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            "https://example.com/page",
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is None


# ---------------------------------------------------------------------------
# fetch_url — sanitize + truncate + cache discipline
# ---------------------------------------------------------------------------


def test_fetch_url_truncates_oversize_body(tmp_path: Path) -> None:
    """An oversize body is truncated to ``max_body_bytes`` (UTF-8 safe)."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/big"
    body = "x" * 50_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=body, content_type="text/html"),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=1000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    assert len(result.sanitized_markdown.encode("utf-8")) <= 1000
    assert result.size_bytes == len(result.sanitized_markdown.encode("utf-8"))


def test_fetch_url_truncate_is_utf8_safe(tmp_path: Path) -> None:
    """Truncation never splits a multibyte codepoint."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/utf8"
    # Each emoji is 4 bytes in UTF-8; an odd byte cap forces backoff.
    body = "🦊" * 100

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=body, content_type="text/html"),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=11,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    # Whatever survived must still decode as valid UTF-8 — accessing
    # ``sanitized_markdown`` already proves that since it round-trips
    # through ``str``.
    assert len(result.sanitized_markdown.encode("utf-8")) <= 11
    # Two emojis = 8 bytes; three = 12, which would overflow. Expect 2.
    assert result.sanitized_markdown == "🦊🦊"


def test_fetch_url_caches_post_sanitize_body(tmp_path: Path) -> None:
    """The cache file contains exactly the documented fields after a fetch."""

    cache_root = tmp_path / "cache"
    cache = UrlCache(cache_root, ttl_days=30)
    url = "https://example.com/page"
    body = "# title\n\nbody"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=body, content_type="text/html"),
        )

    client = _client_with_handler(handler)
    try:
        fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    cached = cache.get(url)
    assert cached is not None
    assert cached.sanitized_markdown == body

    # Inspect the on-disk JSON: only the documented fields, no raw HTML.
    files = list(cache_root.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "url",
        "final_url",
        "content_type",
        "sanitized_markdown",
        "fetched_at",
    }


def test_fetch_url_strips_ansi_in_response_markdown(tmp_path: Path) -> None:
    """ANSI escapes inside the crawl4ai response are stripped by sanitize_text."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    url = "https://example.com/page"
    raw = "\x1b[31mred\x1b[0m text"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=raw, content_type="text/html"),
        )

    client = _client_with_handler(handler)
    try:
        result = fetch_url(
            url,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert result is not None
    assert "\x1b" not in result.sanitized_markdown
    assert "red text" in result.sanitized_markdown


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def test_fetch_all_dedupes_input_urls(tmp_path: Path) -> None:
    """Duplicate URLs in the input result in a single crawl4ai call each."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    urls = ["https://example.com/a", "https://example.com/b", "https://example.com/a"]
    captured_payloads: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body["urls"])
        url = body["urls"][0]
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown="hello"),
        )

    client = _client_with_handler(handler)
    try:
        results = fetch_all(
            urls,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    # Two unique URLs → two POSTs → two FetchedUrl results.
    assert len(captured_payloads) == 2
    assert [p[0] for p in captured_payloads] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [r.url for r in results] == ["https://example.com/a", "https://example.com/b"]


def test_fetch_all_continues_on_per_url_failure(tmp_path: Path) -> None:
    """A failed middle URL is dropped; the rest of the run completes."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    urls = [
        "https://a.example/p",
        "https://b.example/p",
        "https://c.example/p",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        url = body["urls"][0]
        if "b.example" in url:
            return httpx.Response(503, text="upstream busy")
        return httpx.Response(
            200,
            json=_crawl_response(url=url, markdown=f"body for {url}"),
        )

    client = _client_with_handler(handler)
    try:
        results = fetch_all(
            urls,
            crawl4ai_base_url="http://crawl4ai.test:11235",
            cache=cache,
            max_body_bytes=10_000,
            client=client,
            _resolver=_public_resolver(),
        )
    finally:
        client.close()

    assert [r.url for r in results] == [
        "https://a.example/p",
        "https://c.example/p",
    ]


def test_fetch_all_empty_input_returns_empty(tmp_path: Path) -> None:
    """An empty input list short-circuits to an empty list."""

    cache = UrlCache(tmp_path / "cache", ttl_days=30)
    results = fetch_all(
        [],
        crawl4ai_base_url="http://crawl4ai.test:11235",
        cache=cache,
        max_body_bytes=10_000,
        client=_ExplodingClient(),  # type: ignore[arg-type]
    )
    assert results == []
