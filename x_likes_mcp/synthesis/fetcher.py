"""crawl4ai HTTP fetcher for the synthesis-report feature (placeholder for task 4.2).

Implements the per-URL pipeline: SSRF-validate → POST to crawl4ai with
manual redirect handling → content-type allowlist → sanitize → cache.
Soft-drops URLs that fail any guard so the rest of the run continues.
"""

from __future__ import annotations
