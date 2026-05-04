"""Markdown report renderer for the synthesis-report feature (placeholder for task 4.4).

Renders ``brief`` to a ~300-word concept brief, ``synthesis`` to a
longer narrative with a mermaid mindmap and per-cluster tweet list, and
``trend`` to a month-bucketed timeline plus mindmap. Uses each tweet's
``created_at`` for the trend buckets and runs the assembled markdown
through the shared sanitize pass once before returning.
"""

from __future__ import annotations
