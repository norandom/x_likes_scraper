"""Per-shape configuration for the synthesis-report feature.

This module is the single source of truth for the three report shapes
(`brief`, `synthesis`, `trend`) and the cross-cutting numeric directives
(target word count, mindmap depth cap) the renderer and synthesizer share.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "MAX_MINDMAP_DEPTH",
    "SHAPE_CONFIGS",
    "ReportShape",
    "ShapeConfig",
    "parse_report_shape",
]


# Mermaid mindmap rendering depth cap. Anything deeper trips GitHub /
# Obsidian / VS Code preview parsers, so the renderer and the KG walker
# both clamp to this single source-of-truth value (Req 8.2).
MAX_MINDMAP_DEPTH: int = 4


class ReportShape(StrEnum):
    """The three report shapes the synthesizer and renderer support."""

    BRIEF = "brief"
    SYNTHESIS = "synthesis"
    TREND = "trend"


@dataclass(frozen=True)
class ShapeConfig:
    """Per-shape directives shared by the renderer and the synthesizer.

    Attributes
    ----------
    target_word_count:
        Soft target the synthesizer aims for. ``None`` means "no fixed
        target" (the long-form shapes).
    include_mindmap:
        Whether the rendered markdown should include a mermaid mindmap
        block.
    include_per_cluster_tweet_list:
        Whether the rendered markdown should include a per-cluster anchor
        tweet list (only meaningful for the long-form synthesis shape).
    month_bucketed:
        Whether the synthesizer must group anchor tweets into
        chronologically-ordered month buckets keyed by the tweet's
        ``created_at`` (the trend shape).
    """

    target_word_count: int | None
    include_mindmap: bool
    include_per_cluster_tweet_list: bool
    month_bucketed: bool


SHAPE_CONFIGS: dict[ReportShape, ShapeConfig] = {
    ReportShape.BRIEF: ShapeConfig(
        target_word_count=300,
        include_mindmap=False,
        include_per_cluster_tweet_list=False,
        month_bucketed=False,
    ),
    ReportShape.SYNTHESIS: ShapeConfig(
        target_word_count=None,
        include_mindmap=True,
        include_per_cluster_tweet_list=True,
        month_bucketed=False,
    ),
    ReportShape.TREND: ShapeConfig(
        target_word_count=None,
        include_mindmap=True,
        include_per_cluster_tweet_list=False,
        month_bucketed=True,
    ),
}


def parse_report_shape(value: str | ReportShape) -> ReportShape:
    """Coerce a CLI / MCP boundary input into a :class:`ReportShape`.

    The synthesis-report feature rejects unknown shapes immediately
    (Req 1.3); this function is the single chokepoint that enforces
    that contract.

    Parameters
    ----------
    value:
        Either an existing :class:`ReportShape` (returned unchanged) or
        the raw string the caller supplied.

    Returns
    -------
    ReportShape
        The validated enum member.

    Raises
    ------
    ValueError
        If ``value`` is neither a :class:`ReportShape` nor one of the
        three accepted strings. The error message names the offending
        value and the full set of allowed values so the boundary layer
        can surface it directly to the user.
    """
    if isinstance(value, ReportShape):
        return value
    try:
        return ReportShape(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(member.value for member in ReportShape))
        raise ValueError(f"Unknown report shape {value!r}; allowed values: {allowed}") from exc
