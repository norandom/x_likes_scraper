"""sha256-keyed URL cache for the synthesis-report feature (placeholder for task 2.2).

Provides ``get`` / ``put`` / ``expire`` operations keyed on
``sha256(url)`` under the configured cache root, with atomic writes via
temp-file + rename. Persists only post-sanitize markdown plus the
documented metadata fields; raw HTML / PDF bytes never touch disk.
"""

from __future__ import annotations
