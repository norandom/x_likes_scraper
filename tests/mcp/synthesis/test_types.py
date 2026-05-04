"""Tests for the synthesis-report shared type definitions."""

from __future__ import annotations

import dataclasses

import pytest

from x_likes_mcp.synthesis import (
    Claim,
    Entity,
    EntityKind,
    FetchedUrl,
    MonthSummary,
    ReportOptions,
    ReportResult,
    ReportShape,
    Section,
)


class TestPublicSurface:
    def test_acceptance_criterion_imports_succeed(self) -> None:
        # Literal acceptance criterion from task 1.4.
        from x_likes_mcp.synthesis import (  # noqa: F401
            ReportOptions,
            ReportResult,
            ReportShape,
        )


class TestReportOptions:
    def test_defaults(self) -> None:
        opts = ReportOptions(query="cats", shape=ReportShape.BRIEF)
        assert opts.query == "cats"
        assert opts.shape is ReportShape.BRIEF
        assert opts.fetch_urls is False
        assert opts.hops == 1
        assert opts.year is None
        assert opts.month_start is None
        assert opts.month_end is None
        assert opts.limit == 50

    def test_explicit_values(self) -> None:
        opts = ReportOptions(
            query="trend test",
            shape=ReportShape.TREND,
            fetch_urls=True,
            hops=2,
            year=2025,
            month_start="2025-01",
            month_end="2025-03",
            limit=20,
        )
        assert opts.fetch_urls is True
        assert opts.hops == 2
        assert opts.year == 2025
        assert opts.month_start == "2025-01"
        assert opts.month_end == "2025-03"
        assert opts.limit == 20

    def test_is_frozen(self) -> None:
        opts = ReportOptions(query="x", shape=ReportShape.BRIEF)
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.hops = 2  # type: ignore[misc]


class TestReportResult:
    def test_fields(self) -> None:
        result = ReportResult(
            markdown="# hello",
            shape=ReportShape.SYNTHESIS,
            used_hops=1,
            fetched_url_count=0,
            synthesis_token_count=42,
        )
        assert result.markdown == "# hello"
        assert result.shape is ReportShape.SYNTHESIS
        assert result.used_hops == 1
        assert result.fetched_url_count == 0
        assert result.synthesis_token_count == 42

    def test_is_frozen(self) -> None:
        result = ReportResult(
            markdown="",
            shape=ReportShape.BRIEF,
            used_hops=1,
            fetched_url_count=0,
            synthesis_token_count=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.markdown = "mutated"  # type: ignore[misc]


class TestFetchedUrl:
    def test_fields(self) -> None:
        fu = FetchedUrl(
            url="https://example.com/a",
            final_url="https://example.com/a",
            content_type="text/html",
            sanitized_markdown="hello",
            size_bytes=5,
        )
        assert fu.url == "https://example.com/a"
        assert fu.final_url == "https://example.com/a"
        assert fu.content_type == "text/html"
        assert fu.sanitized_markdown == "hello"
        assert fu.size_bytes == 5

    def test_is_frozen(self) -> None:
        fu = FetchedUrl(
            url="https://example.com/a",
            final_url="https://example.com/a",
            content_type="text/html",
            sanitized_markdown="",
            size_bytes=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            fu.size_bytes = 9  # type: ignore[misc]


class TestEntity:
    def test_entity_kind_values(self) -> None:
        assert EntityKind("handle") is EntityKind.HANDLE
        assert EntityKind("hashtag") is EntityKind.HASHTAG
        assert EntityKind("domain") is EntityKind.DOMAIN
        assert EntityKind("concept") is EntityKind.CONCEPT

    def test_entity_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            EntityKind("nope")

    def test_entity_fields(self) -> None:
        ent = Entity(kind=EntityKind.HANDLE, value="@windsheep_", weight=1.5)
        assert ent.kind is EntityKind.HANDLE
        assert ent.value == "@windsheep_"
        assert ent.weight == 1.5

    def test_entity_is_frozen(self) -> None:
        ent = Entity(kind=EntityKind.DOMAIN, value="example.com", weight=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ent.weight = 2.0  # type: ignore[misc]


class TestPydanticModels:
    def test_claim_round_trip(self) -> None:
        c = Claim(text="The sky is blue.", sources=["tweet:1", "url:https://example.com/a"])
        dump = c.model_dump()
        assert dump == {
            "text": "The sky is blue.",
            "sources": ["tweet:1", "url:https://example.com/a"],
        }
        rebuilt = Claim.model_validate(dump)
        assert rebuilt == c

    def test_section_round_trip(self) -> None:
        s = Section(
            heading="Introduction",
            claims=[Claim(text="A", sources=["tweet:1"])],
        )
        dump = s.model_dump()
        assert dump == {
            "heading": "Introduction",
            "claims": [{"text": "A", "sources": ["tweet:1"]}],
        }
        rebuilt = Section.model_validate(dump)
        assert rebuilt == s

    def test_month_summary_round_trip(self) -> None:
        ms = MonthSummary(
            year_month="2025-01",
            summary="January.",
            anchor_tweets=["tweet:1", "tweet:2"],
        )
        dump = ms.model_dump()
        assert dump == {
            "year_month": "2025-01",
            "summary": "January.",
            "anchor_tweets": ["tweet:1", "tweet:2"],
        }
        rebuilt = MonthSummary.model_validate(dump)
        assert rebuilt == ms

    def test_claim_rejects_missing_field(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Claim.model_validate({"text": "missing sources"})
