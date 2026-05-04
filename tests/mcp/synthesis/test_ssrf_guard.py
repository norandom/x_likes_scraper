"""Tests for the synthesis-report SSRF guard (task 2.1).

Covers Requirements 4.1, 4.2, 4.3, 4.4: HTTP(S)-only scheme guard,
the unconditional block tier (loopback / cloud-metadata / broadcast /
multicast / IANA-reserved), the private-range tier with operator CIDR
allowlist override (RFC1918 + IPv6 ULA + IPv6 link-local), and the
resolve-once / pin-first-non-blocked-IP behaviour that
:mod:`x_likes_mcp.synthesis.fetcher` will rely on for redirect
re-validation.

Real DNS is never consulted: every ``resolve_and_check`` test injects a
fake ``resolver`` so the suite stays offline. The ``socket`` module is
only used to source the ``SOCK_STREAM`` / ``AF_INET`` constants the
fake resolver returns.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Sequence
from typing import Any

import pytest

from x_likes_mcp.synthesis import ssrf_guard
from x_likes_mcp.synthesis.ssrf_guard import (
    CLOUD_METADATA_ADDRESSES,
    ResolvedHost,
    SsrfBlocked,
    is_blocked_address,
    is_unconditional_blocked,
    parse_allowlist,
    resolve_and_check,
)


def _fake_resolver(
    candidates: Sequence[tuple[int, str]],
) -> Any:
    """Build a fake ``socket.getaddrinfo`` that returns the given candidates.

    Each ``(family, ip)`` pair is converted into the 5-tuple shape that
    :func:`socket.getaddrinfo` produces so the SSRF guard can walk the
    list with the real ``ai[4][0]`` extraction logic.
    """

    def _resolver(
        host: str,
        port: int | None,
        *,
        type: int = 0,
        **_kwargs: object,
    ) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        out: list[tuple[int, int, int, str, tuple[Any, ...]]] = []
        for family, ip in candidates:
            sockaddr: tuple[Any, ...]
            if family == socket.AF_INET:
                sockaddr = (ip, port or 0)
            else:
                sockaddr = (ip, port or 0, 0, 0)
            out.append((family, type or socket.SOCK_STREAM, 0, "", sockaddr))
        return out

    return _resolver


# ---------------------------------------------------------------------------
# parse_allowlist
# ---------------------------------------------------------------------------


def test_parse_allowlist_empty() -> None:
    assert parse_allowlist("") == []


def test_parse_allowlist_ipv4_and_ipv6() -> None:
    parsed = parse_allowlist("10.100.0.0/16, 2001:db8::/32")
    assert parsed == [
        ipaddress.ip_network("10.100.0.0/16"),
        ipaddress.ip_network("2001:db8::/32"),
    ]


def test_parse_allowlist_tolerates_blank_entries() -> None:
    parsed = parse_allowlist("10.100.0.0/16,,  ,192.168.0.0/24")
    assert parsed == [
        ipaddress.ip_network("10.100.0.0/16"),
        ipaddress.ip_network("192.168.0.0/24"),
    ]


def test_parse_allowlist_accepts_host_bits() -> None:
    # ``ipaddress.ip_network(..., strict=False)`` is what Config uses;
    # the SSRF guard's parser must mirror that to keep the two paths
    # in lockstep (operators copy a sample IP, not the network address).
    parsed = parse_allowlist("10.100.5.7/16")
    assert parsed == [ipaddress.ip_network("10.100.0.0/16")]


def test_parse_allowlist_raises_on_malformed_entry() -> None:
    with pytest.raises(ValueError, match="not-a-cidr"):
        parse_allowlist("10.0.0.0/8, not-a-cidr")


# ---------------------------------------------------------------------------
# is_unconditional_blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "169.254.169.254",  # AWS / GCP / Azure / DO metadata
        "192.0.0.192",  # Oracle Cloud metadata
        "fd00:ec2::254",  # AWS IPv6 metadata
        "127.0.0.1",  # IPv4 loopback
        "::1",  # IPv6 loopback
        "224.0.0.1",  # IPv4 multicast
        "ff02::1",  # IPv6 multicast
        "0.0.0.0",  # unspecified / any
        "::",  # IPv6 unspecified
        "255.255.255.255",  # IPv4 broadcast
        "240.0.0.1",  # IPv4 reserved (class E)
    ],
)
def test_is_unconditional_blocked_true(ip: str) -> None:
    assert is_unconditional_blocked(ip) is True


@pytest.mark.parametrize(
    "ip",
    [
        "8.8.8.8",  # public IPv4
        "1.1.1.1",  # public IPv4
        "2606:4700:4700::1111",  # public IPv6
        "10.0.0.1",  # RFC1918 — NOT unconditional
        "192.168.1.1",  # RFC1918 — NOT unconditional
        "fe80::1",  # IPv6 link-local — NOT unconditional
        "fd00::1",  # IPv6 ULA — NOT unconditional (different from fd00:ec2::254)
    ],
)
def test_is_unconditional_blocked_false(ip: str) -> None:
    assert is_unconditional_blocked(ip) is False


def test_cloud_metadata_addresses_set_contents() -> None:
    """The frozenset is part of the public design contract."""

    assert "169.254.169.254" in CLOUD_METADATA_ADDRESSES
    assert "192.0.0.192" in CLOUD_METADATA_ADDRESSES
    assert "fd00:ec2::254" in CLOUD_METADATA_ADDRESSES


# ---------------------------------------------------------------------------
# is_blocked_address — empty allowlist (the strict default)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "10.0.0.1",
        "10.255.255.254",
        "172.16.0.5",
        "172.31.0.5",
        "192.168.1.1",
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 ULA
        "fd12:3456:789a::1",  # IPv6 ULA
    ],
)
def test_is_blocked_address_blocks_private_by_default(ip: str) -> None:
    assert is_blocked_address(ip) is True


@pytest.mark.parametrize(
    "ip",
    [
        "8.8.8.8",
        "1.1.1.1",
        "2606:4700:4700::1111",
    ],
)
def test_is_blocked_address_allows_public(ip: str) -> None:
    assert is_blocked_address(ip) is False


# ---------------------------------------------------------------------------
# is_blocked_address — operator allowlist punches holes in the private tier
# ---------------------------------------------------------------------------


def test_is_blocked_address_allowlist_punches_hole() -> None:
    allowlist = parse_allowlist("10.100.0.0/16")
    assert is_blocked_address("10.100.0.5", private_allowlist=allowlist) is False


def test_is_blocked_address_allowlist_does_not_cover_other_private_ranges() -> None:
    allowlist = parse_allowlist("10.100.0.0/16")
    # Different RFC1918 sub-range is still blocked.
    assert is_blocked_address("10.0.0.1", private_allowlist=allowlist) is True
    assert is_blocked_address("192.168.1.1", private_allowlist=allowlist) is True


def test_allowlist_cannot_override_loopback() -> None:
    allowlist = parse_allowlist("127.0.0.0/8")
    assert is_blocked_address("127.0.0.1", private_allowlist=allowlist) is True


def test_allowlist_cannot_override_metadata_ip() -> None:
    allowlist = parse_allowlist("169.254.0.0/16")
    # Even with all of link-local on the allowlist, the metadata IP is
    # in the unconditional tier and stays blocked.
    assert is_blocked_address("169.254.169.254", private_allowlist=allowlist) is True


def test_allowlist_cannot_override_oracle_metadata() -> None:
    allowlist = parse_allowlist("192.0.0.0/24")
    assert is_blocked_address("192.0.0.192", private_allowlist=allowlist) is True


def test_allowlist_cannot_override_multicast() -> None:
    allowlist = parse_allowlist("224.0.0.0/4")
    assert is_blocked_address("224.0.0.1", private_allowlist=allowlist) is True


def test_ipv6_allowlist_punches_hole_for_ula() -> None:
    allowlist = parse_allowlist("fd12:3456::/32")
    assert is_blocked_address("fd12:3456::1", private_allowlist=allowlist) is False
    # An adjacent ULA address outside that /32 is still blocked.
    assert is_blocked_address("fdab::1", private_allowlist=allowlist) is True


# ---------------------------------------------------------------------------
# resolve_and_check — scheme + host validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/plain,foo",
        "file:///etc/passwd",
        "ftp://example.com/foo",
        "",
    ],
)
def test_resolve_and_check_unsupported_scheme(url: str) -> None:
    with pytest.raises(SsrfBlocked) as excinfo:
        resolve_and_check(url, resolver=_fake_resolver([(socket.AF_INET, "8.8.8.8")]))
    assert excinfo.value.reason == "unsupported_scheme"


def test_resolve_and_check_missing_host() -> None:
    with pytest.raises(SsrfBlocked) as excinfo:
        resolve_and_check(
            "http:///foo",
            resolver=_fake_resolver([(socket.AF_INET, "8.8.8.8")]),
        )
    assert excinfo.value.reason == "missing_host"


# ---------------------------------------------------------------------------
# resolve_and_check — happy paths
# ---------------------------------------------------------------------------


def test_resolve_and_check_public_ipv4() -> None:
    fake = _fake_resolver([(socket.AF_INET, "8.8.8.8")])
    host = resolve_and_check("https://example.com/foo", resolver=fake)
    assert host == ResolvedHost(
        hostname="example.com",
        ip="8.8.8.8",
        port=443,
        scheme="https",
    )


def test_resolve_and_check_public_ipv6() -> None:
    fake = _fake_resolver([(socket.AF_INET6, "2606:4700:4700::1111")])
    host = resolve_and_check("https://example.com/", resolver=fake)
    assert host.ip == "2606:4700:4700::1111"
    assert host.scheme == "https"
    assert host.port == 443


def test_resolve_and_check_default_port_http() -> None:
    fake = _fake_resolver([(socket.AF_INET, "8.8.8.8")])
    host = resolve_and_check("http://example.com/", resolver=fake)
    assert host.port == 80
    assert host.scheme == "http"


def test_resolve_and_check_explicit_port() -> None:
    fake = _fake_resolver([(socket.AF_INET, "8.8.8.8")])
    host = resolve_and_check("http://example.com:8080/", resolver=fake)
    assert host.port == 8080


def test_resolve_and_check_skips_blocked_candidate_and_pins_next() -> None:
    """Round-robin DNS pins the first non-blocked address."""

    fake = _fake_resolver(
        [
            (socket.AF_INET, "10.0.0.1"),  # RFC1918, blocked by default
            (socket.AF_INET, "8.8.8.8"),  # public, pin this one
        ]
    )
    host = resolve_and_check("https://example.com/", resolver=fake)
    assert host.ip == "8.8.8.8"


def test_resolve_and_check_blocked_when_all_candidates_private() -> None:
    fake = _fake_resolver(
        [
            (socket.AF_INET, "10.0.0.1"),
            (socket.AF_INET, "192.168.1.1"),
        ]
    )
    with pytest.raises(SsrfBlocked) as excinfo:
        resolve_and_check("https://internal.example/", resolver=fake)
    assert excinfo.value.reason.startswith("blocked_address")
    # Reason includes the hostname and resolved IPs for debugging.
    assert "internal.example" in excinfo.value.reason
    assert "10.0.0.1" in excinfo.value.reason


def test_resolve_and_check_blocked_for_metadata_ip() -> None:
    """The cloud-metadata IP must always block, regardless of allowlist."""

    fake = _fake_resolver([(socket.AF_INET, "169.254.169.254")])
    with pytest.raises(SsrfBlocked):
        resolve_and_check(
            "http://metadata.example/",
            resolver=fake,
            private_allowlist=parse_allowlist("169.254.0.0/16"),
        )


def test_resolve_and_check_honors_private_allowlist() -> None:
    """RFC1918 candidate is allowed when its CIDR is on the allowlist."""

    fake = _fake_resolver([(socket.AF_INET, "10.100.0.5")])
    host = resolve_and_check(
        "https://internal.mesh.example/",
        resolver=fake,
        private_allowlist=parse_allowlist("10.100.0.0/16"),
    )
    assert host.ip == "10.100.0.5"
    assert host.hostname == "internal.mesh.example"


def test_resolve_and_check_no_candidates() -> None:
    """An empty resolver result is treated as blocked, not a crash."""

    fake = _fake_resolver([])
    with pytest.raises(SsrfBlocked):
        resolve_and_check("https://example.com/", resolver=fake)


def test_resolve_and_check_default_resolver_is_socket_getaddrinfo() -> None:
    """The default resolver is ``socket.getaddrinfo`` — no real DNS in tests."""

    assert ssrf_guard._DEFAULT_RESOLVER is socket.getaddrinfo
