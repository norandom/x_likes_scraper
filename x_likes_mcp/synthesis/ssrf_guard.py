"""SSRF guard for the synthesis-report feature (placeholder for task 2.1).

Resolves the URL host once, walks the address candidates, and pins the
first non-blocked IP so the connection cannot be rebound mid-fetch.
Enforces the unconditional blocklist (loopback, cloud metadata,
broadcast / multicast, IANA reserved) and the operator-supplied
private-range CIDR allowlist.
"""

from __future__ import annotations
