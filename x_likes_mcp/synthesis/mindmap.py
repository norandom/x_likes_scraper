"""Mermaid mindmap renderer for the synthesis-report feature (placeholder for task 2.5).

Generates a depth-capped mermaid ``mindmap`` block whose root is the
user query and whose level-1 children are the entity categories present
in the KG (Authors, Sources, Themes, Hashtags). Honors
``MAX_MINDMAP_DEPTH`` from :mod:`x_likes_mcp.synthesis.shapes` and
filters node labels to a safe character subset before emission.
"""

from __future__ import annotations
