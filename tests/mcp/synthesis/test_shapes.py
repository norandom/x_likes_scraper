"""Tests for the synthesis-report shape enum and per-shape config."""

from __future__ import annotations

import dataclasses

import pytest

from x_likes_mcp.synthesis import (
    MAX_MINDMAP_DEPTH,
    SHAPE_CONFIGS,
    ReportShape,
    ShapeConfig,
    parse_report_shape,
)


class TestReportShape:
    def test_brief_value(self) -> None:
        assert ReportShape("brief") is ReportShape.BRIEF

    def test_synthesis_value(self) -> None:
        assert ReportShape("synthesis") is ReportShape.SYNTHESIS

    def test_trend_value(self) -> None:
        assert ReportShape("trend") is ReportShape.TREND

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ReportShape("foo")

    def test_is_str_enum(self) -> None:
        # StrEnum members compare equal to their string value.
        assert ReportShape.BRIEF == "brief"
        assert ReportShape.SYNTHESIS == "synthesis"
        assert ReportShape.TREND == "trend"


class TestParseReportShape:
    def test_parse_brief_string(self) -> None:
        assert parse_report_shape("brief") is ReportShape.BRIEF

    def test_parse_synthesis_string(self) -> None:
        assert parse_report_shape("synthesis") is ReportShape.SYNTHESIS

    def test_parse_trend_string(self) -> None:
        assert parse_report_shape("trend") is ReportShape.TREND

    def test_idempotent_on_enum_input(self) -> None:
        assert parse_report_shape(ReportShape.TREND) is ReportShape.TREND
        assert parse_report_shape(ReportShape.BRIEF) is ReportShape.BRIEF
        assert parse_report_shape(ReportShape.SYNTHESIS) is ReportShape.SYNTHESIS

    def test_unknown_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc:
            parse_report_shape("FOO")
        message = str(exc.value)
        # Offending value must be named.
        assert "FOO" in message
        # All three allowed values must be enumerated.
        assert "brief" in message
        assert "synthesis" in message
        assert "trend" in message

    def test_unknown_value_does_not_normalize_case(self) -> None:
        # "BRIEF" is not "brief". We are deliberately strict per Req 1.3.
        with pytest.raises(ValueError):
            parse_report_shape("BRIEF")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_report_shape("")


class TestShapeConfigs:
    def test_max_mindmap_depth_is_four(self) -> None:
        assert MAX_MINDMAP_DEPTH == 4

    def test_max_mindmap_depth_importable_from_package(self) -> None:
        from x_likes_mcp.synthesis import MAX_MINDMAP_DEPTH as imported_depth

        assert imported_depth == 4

    def test_shape_configs_has_exactly_three_shapes(self) -> None:
        assert set(SHAPE_CONFIGS.keys()) == {
            ReportShape.BRIEF,
            ReportShape.SYNTHESIS,
            ReportShape.TREND,
        }

    def test_brief_config(self) -> None:
        cfg = SHAPE_CONFIGS[ReportShape.BRIEF]
        assert cfg.target_word_count == 300
        assert cfg.include_mindmap is False
        assert cfg.include_per_cluster_tweet_list is False
        assert cfg.month_bucketed is False

    def test_synthesis_config(self) -> None:
        cfg = SHAPE_CONFIGS[ReportShape.SYNTHESIS]
        assert cfg.target_word_count is None
        assert cfg.include_mindmap is True
        assert cfg.include_per_cluster_tweet_list is True
        assert cfg.month_bucketed is False

    def test_trend_config(self) -> None:
        cfg = SHAPE_CONFIGS[ReportShape.TREND]
        assert cfg.target_word_count is None
        assert cfg.include_mindmap is True
        assert cfg.include_per_cluster_tweet_list is False
        assert cfg.month_bucketed is True

    def test_shape_config_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ShapeConfig)
        cfg = SHAPE_CONFIGS[ReportShape.BRIEF]
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.target_word_count = 999  # type: ignore[misc]
