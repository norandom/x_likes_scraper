"""In-memory mini knowledge graph for the synthesis-report feature (placeholder for task 2.3).

Provides namespaced node IDs (``tweet:<id>``, ``handle:<screen_name>``,
``hashtag:<tag>``, ``domain:<host>``, ``concept:<lower-snake-case>``),
edge kinds, and ``top_entities`` / ``neighbors`` accessors the multihop
fan-out and the mindmap renderer share.
"""

from __future__ import annotations
