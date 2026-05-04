"""Shared dataclass and Pydantic-model types for the synthesis-report feature.

This module owns the cross-cutting types every synthesis leaf module
exchanges. Importing it must not pull in DSPy, httpx, or any other heavy
runtime dependency: the goal is a cheap, stable type surface the public
package re-exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from .shapes import ReportShape

__all__ = [
    "Claim",
    "Entity",
    "EntityKind",
    "FetchedUrl",
    "MonthSummary",
    "ReportOptions",
    "ReportResult",
    "Section",
]


# ---------------------------------------------------------------------------
# Orchestrator contract: ReportOptions / ReportResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportOptions:
    """Inputs to a single ``run_report`` call.

    The defaults match the design's orchestrator service interface:
    ``fetch_urls`` is off by default (Req 4.7), ``hops`` is 1 (Req 2.4),
    and ``limit`` matches the existing search default.
    """

    query: str
    shape: ReportShape
    fetch_urls: bool = False
    hops: int = 1
    year: int | None = None
    month_start: str | None = None
    month_end: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class ReportResult:
    """Outputs from a single ``run_report`` call.

    The orchestrator never writes to disk itself; the CLI / MCP boundary
    consumes ``markdown`` and the structured metadata fields.
    """

    markdown: str
    shape: ReportShape
    used_hops: int
    fetched_url_count: int
    synthesis_token_count: int


# ---------------------------------------------------------------------------
# Fetcher contract: FetchedUrl
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchedUrl:
    """A single successfully fetched and sanitized URL body.

    The fetcher only returns instances of this type after sanitizing the
    body, applying the per-URL byte cap, and persisting the result through
    the URL cache. Raw HTML / PDF bytes never reach this dataclass.
    """

    url: str
    final_url: str
    content_type: str
    sanitized_markdown: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Entity-extraction contract: Entity / EntityKind
# ---------------------------------------------------------------------------


class EntityKind(StrEnum):
    """The four entity kinds the regex extractor and KG support."""

    HANDLE = "handle"
    HASHTAG = "hashtag"
    DOMAIN = "domain"
    CONCEPT = "concept"


@dataclass(frozen=True)
class Entity:
    """A weighted entity mined from hit text or a fetched URL body."""

    kind: EntityKind
    value: str
    weight: float


# ---------------------------------------------------------------------------
# DSPy-signature output models
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """A single claim emitted by the synthesizer.

    ``sources`` MUST be a subset of the orchestrator's known-source-ID
    set; the DSPy ``Assert`` enforces that contract at synthesis time.
    Source IDs are namespaced strings: ``tweet:<id>`` or
    ``url:<final_url>``.
    """

    text: str
    sources: list[str]


class Section(BaseModel):
    """A heading plus its claims (used by the long-form synthesis shape)."""

    heading: str
    claims: list[Claim]


class MonthSummary(BaseModel):
    """A single month bucket emitted by the trend synthesizer."""

    year_month: str
    summary: str
    anchor_tweets: list[str]
