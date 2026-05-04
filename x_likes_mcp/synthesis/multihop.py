"""Round-2 entity fan-out for the synthesis-report feature (placeholder for task 4.3).

Implements the multihop search expansion: pick top-K entities from the
round-1 KG, run K parallel round-2 searches with the same date filters,
and fuse the results deterministically by ``tweet_id``.
"""

from __future__ import annotations
