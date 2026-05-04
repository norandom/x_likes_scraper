"""Fenced synthesis-context assembly for the synthesis-report feature (placeholder for task 4.1).

Builds the fenced blob by sanitizing and fencing each tweet body,
fetched URL link, fetched URL body, entity string, KG node label, and
KG edge caption with the matching marker family; enforces per-source
caps and a total-budget enforcer that drops the lowest-rank URL bodies
first, then the lowest-rank tweets.
"""

from __future__ import annotations
