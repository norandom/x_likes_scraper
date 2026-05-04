"""Public surface for the synthesis-report feature.

The synthesis subpackage owns the orchestrator, leaf modules, and the
shared types the CLI / MCP boundary consume. Only the symbols re-exported
here are part of the stable public surface; everything else is an
implementation detail of the leaf modules.
"""

from __future__ import annotations

from .shapes import (
    MAX_MINDMAP_DEPTH,
    SHAPE_CONFIGS,
    ReportShape,
    ShapeConfig,
    parse_report_shape,
)
from .types import (
    Claim,
    Entity,
    EntityKind,
    FetchedUrl,
    MonthSummary,
    ReportOptions,
    ReportResult,
    Section,
)

__all__ = [
    "MAX_MINDMAP_DEPTH",
    "SHAPE_CONFIGS",
    "Claim",
    "Entity",
    "EntityKind",
    "FetchedUrl",
    "MonthSummary",
    "ReportOptions",
    "ReportResult",
    "ReportShape",
    "Section",
    "ShapeConfig",
    "parse_report_shape",
]
