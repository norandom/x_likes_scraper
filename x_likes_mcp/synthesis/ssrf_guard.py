"""SSRF guard for the synthesis-report feature.

Resolves a URL's hostname once, walks the address candidates returned
by :func:`socket.getaddrinfo`, and pins the first non-blocked IP so the
caller's connection cannot be DNS-rebound mid-fetch. Two block tiers
back the decision:

* **Unconditional** (cannot be overridden): loopback, the documented
  cloud-metadata addresses (AWS / GCP / Azure / DigitalOcean / Oracle
  Cloud), broadcast / multicast, IANA-reserved, and the unspecified /
  any address.
* **Private** (operator allowlist may punch holes): RFC1918, the IPv4
  link-local 169.254.0.0/16 minus the metadata IP that lives in the
  unconditional tier, IPv6 link-local ``fe80::/10``, and IPv6 ULA
  ``fc00::/7``.

The contract is intentionally narrow — this module never opens a
socket. It tells the caller which IP to connect to. The caller (the
``fetcher`` in this spec) is expected to:

1. Use the returned :class:`ResolvedHost.ip` as the connect target.
2. Send the original ``Host: <hostname>`` header so the remote server
   sees the name the user typed.
3. Re-call :func:`resolve_and_check` for every redirect target so a
   307/308 chain that smuggles in an internal hostname is caught
   *before* the redirect is followed.

Stdlib only: :mod:`ipaddress` does the address classification,
:mod:`socket` provides the resolver, and :mod:`urllib.parse` parses
the URL. No third-party dependency.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from x_likes_mcp.sanitize import safe_http_url

# Type alias: the parser produces a list of network objects of either
# IP version. Keeping the union (rather than the abstract ``_BaseNetwork``)
# matches what :func:`ipaddress.ip_network` returns and keeps mypy strict
# happy.
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


# Cloud-metadata addresses that must always be blocked.
#
# * 169.254.169.254 — AWS / EC2 IMDS, GCP (metadata.google.internal
#   resolves here), Azure IMDS, DigitalOcean metadata.
# * 192.0.0.192      — Oracle Cloud metadata.
# * fd00:ec2::254    — AWS IPv6 metadata variant.
#
# These are matched as plain strings *and* as ``ipaddress`` objects so
# the IPv6 metadata IP is recognized regardless of how the resolver
# normalizes it (e.g. ``fd00:ec2::254`` vs ``fd00:ec2:0:0:0:0:0:254``).
CLOUD_METADATA_ADDRESSES: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "192.0.0.192",
        "fd00:ec2::254",
    }
)


# Pre-computed ``ipaddress`` objects for the metadata set so the
# membership test handles every textual form (compressed, expanded,
# mixed-case) the resolver might return.
_CLOUD_METADATA_OBJECTS: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] = frozenset(
    ipaddress.ip_address(s) for s in CLOUD_METADATA_ADDRESSES
)


# IPv6 ULA range. ``ipaddress.IPv6Address.is_private`` *does* include
# this range, but we keep an explicit network here so the intent is
# obvious to the next maintainer (and so a stdlib semantics shift would
# fail loudly in tests rather than silently relax the guard).
_IPV6_ULA = ipaddress.ip_network("fc00::/7")


@dataclass(frozen=True)
class ResolvedHost:
    """A pinned target for a single fetch.

    ``ip`` is the IP the caller must connect to; ``hostname`` is the
    name to send in the ``Host:`` header. The split is what makes the
    DNS-rebinding mitigation work.
    """

    hostname: str
    ip: str
    port: int
    scheme: str


class SsrfBlocked(Exception):
    """Raised when a URL's resolved address is not safe to connect to.

    The :attr:`reason` attribute carries a short machine-readable code
    (``"unsupported_scheme"``, ``"missing_host"``, ``"blocked_address"``)
    that the caller can log without leaking the full URL into structured
    logs.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# Duck-typed shape of :func:`socket.getaddrinfo`. The synthesis-report
# fetcher only needs ``(host, port, *, type=...)`` — a permissive
# ``Callable[..., Sequence[tuple[Any, ...]]]`` matches the stdlib
# function and any in-test fake without forcing the fake to mirror
# every keyword argument the real ``getaddrinfo`` accepts.
_Resolver = Callable[..., Sequence[tuple[Any, ...]]]


# Default resolver. Exposed at module level so tests can assert the
# default points at the real ``socket.getaddrinfo`` and so callers can
# inject a fake without monkeypatching :mod:`socket`.
_DEFAULT_RESOLVER: _Resolver = socket.getaddrinfo


def parse_allowlist(raw: str) -> list[IPNetwork]:
    """Parse a comma-separated CIDR allowlist into IP-network objects.

    Mirrors :func:`x_likes_mcp.config._load_cidr_allowlist` so the
    on-disk env value and the in-process API agree on:

    * Empty / blank → empty list.
    * Whitespace around each entry is stripped.
    * Trailing commas / blank entries are tolerated (operator intent
      is unambiguous; punishing a typo'd CIDR is the only useful loud
      failure).
    * ``ipaddress.ip_network(s, strict=False)`` so an operator who
      copies a sample IP (``10.100.5.7/16``) gets the network.

    Raises:
        ValueError: when an entry fails to parse. The offending string
        is included in the exception message so the operator sees which
        token tripped the parser.
    """

    if not raw:
        return []

    parsed: list[IPNetwork] = []
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        try:
            parsed.append(ipaddress.ip_network(candidate, strict=False))
        except ValueError as exc:
            raise ValueError(f"invalid CIDR in allowlist: {candidate!r}") from exc
    return parsed


def is_unconditional_blocked(ip: str) -> bool:
    """Return True for addresses that no allowlist may unblock.

    The unconditional tier covers the categories whose blast radius is
    universal: cloud-metadata exfiltration (169.254.169.254 et al.),
    loopback (a TLS-offload sidecar that punches a hole on
    ``10.100.0.0/16`` should never let a fetch hit ``127.0.0.1``),
    multicast / broadcast (no remote URL legitimately resolves there),
    and the IANA-reserved / unspecified ranges.

    Unparseable input is treated as blocked — the caller passed us a
    string that does not look like an IP, and we refuse to second-guess
    what to do with it.
    """

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True

    if ip in CLOUD_METADATA_ADDRESSES or addr in _CLOUD_METADATA_OBJECTS:
        return True
    if addr.is_loopback:
        return True
    if addr.is_multicast:
        return True
    if addr.is_reserved:
        return True
    if addr.is_unspecified:
        return True
    # ``IPv4Address`` exposes ``is_global`` but not ``is_broadcast``
    # before 3.13; check the broadcast literal explicitly so the
    # behaviour is identical across supported Pythons.
    return isinstance(addr, ipaddress.IPv4Address) and int(addr) == 0xFFFFFFFF


def is_blocked_address(
    ip: str,
    *,
    private_allowlist: Sequence[IPNetwork] = (),
) -> bool:
    """Decide whether ``ip`` is unsafe to connect to.

    Algorithm:

    1. If :func:`is_unconditional_blocked` is True, return True. The
       allowlist cannot punch holes in this tier.
    2. Otherwise check the private tier:

       * IPv4: ``is_private`` covers RFC1918 + link-local + the rest
         of the private aggregate. Loopback would also match
         ``is_private`` but the unconditional check already returned
         True for it, so we never reach here for ``127.0.0.0/8``.
       * IPv6: ``is_link_local`` plus the explicit ULA network
         ``fc00::/7``.

    3. If the private tier matched but the address falls inside any
       allowlisted CIDR, return False (allowed). Otherwise return the
       private-tier verdict.

    Unparseable input is blocked — same contract as
    :func:`is_unconditional_blocked`.
    """

    if is_unconditional_blocked(ip):
        return True

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True

    private_match: bool
    if isinstance(addr, ipaddress.IPv4Address):
        # IPv4 ``is_private`` on stdlib is the right umbrella here; the
        # unconditional tier above has already filtered out loopback
        # and the metadata IP, so the only ranges still surfacing are
        # RFC1918 + the rest of 169.254.0.0/16.
        private_match = addr.is_private
    else:
        private_match = addr.is_link_local or addr in _IPV6_ULA

    if not private_match:
        return False

    for network in private_allowlist:
        # ``ip_address in ip_network`` is well-defined when the
        # versions match; mismatched versions raise ``TypeError`` in
        # CPython's ``_BaseNetwork.__contains__``. We swallow that and
        # treat it as "not in this network".
        if addr.version != network.version:
            continue
        if addr in network:
            return False

    return True


def resolve_and_check(
    url: str,
    *,
    private_allowlist: Sequence[IPNetwork] = (),
    resolver: _Resolver = _DEFAULT_RESOLVER,
) -> ResolvedHost:
    """Resolve ``url`` and pin the first non-blocked IP.

    Pipeline:

    1. :func:`x_likes_mcp.sanitize.safe_http_url` — strips ANSI / BiDi,
       enforces ``http://`` / ``https://``. Anything else raises
       :class:`SsrfBlocked` with reason ``"unsupported_scheme"``.
    2. :func:`urllib.parse.urlsplit` — extract hostname, port, scheme.
       Missing host raises :class:`SsrfBlocked("missing_host")`.
    3. Determine the port: the explicit port if present, else the
       scheme default (80 / 443).
    4. Call ``resolver(host, port, type=SOCK_STREAM)`` once and walk
       the candidates in resolver order.
    5. Pin the first candidate for which
       :func:`is_blocked_address` is False.
    6. If none survives, raise :class:`SsrfBlocked` with reason
       ``"blocked_address: <hostname> ips=<...>"`` — the hostname and
       the resolved IPs are included for debugging, but the reason
       prefix lets the caller key on ``"blocked_address"`` without
       parsing the rest.

    The DNS lookup happens exactly once per call; the caller is
    expected to use the returned :attr:`ResolvedHost.ip` as the
    connect target and send the original hostname in the ``Host:``
    header. Redirect re-validation is the caller's job — pass each
    new redirect target back through this function.
    """

    cleaned = safe_http_url(url)
    if cleaned is None:
        raise SsrfBlocked("unsupported_scheme")

    parts = urlsplit(cleaned)
    hostname = parts.hostname
    if not hostname:
        raise SsrfBlocked("missing_host")

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        # ``safe_http_url`` already enforces this; the second check is
        # belt-and-braces in case a future change relaxes the upstream.
        raise SsrfBlocked("unsupported_scheme")

    explicit_port = parts.port
    port = explicit_port if explicit_port is not None else (443 if scheme == "https" else 80)

    candidates = resolver(hostname, port, type=socket.SOCK_STREAM)

    resolved_ips: list[str] = []
    for entry in candidates:
        # ``getaddrinfo`` returns 5-tuples whose final element is a
        # socket-address tuple. Index ``[0]`` is always the textual
        # IP for both AF_INET and AF_INET6.
        sockaddr = entry[4]
        if not sockaddr:
            continue
        candidate_ip = sockaddr[0]
        if not isinstance(candidate_ip, str):
            continue
        resolved_ips.append(candidate_ip)
        if not is_blocked_address(candidate_ip, private_allowlist=private_allowlist):
            return ResolvedHost(
                hostname=hostname,
                ip=candidate_ip,
                port=port,
                scheme=scheme,
            )

    raise SsrfBlocked(f"blocked_address: host={hostname!r} ips={resolved_ips!r}")
