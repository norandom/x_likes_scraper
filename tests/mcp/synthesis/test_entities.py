"""Tests for the regex entity extractor and the DSPy-fallback hook.

These tests pin the cheap-first contract documented in design.md and
Requirements 5.1, 5.2: a regex/counter pass over hit text and URL bodies
runs before any LM call, and the fallback hook fires only for hits where
the regex pass returned nothing.
"""

from __future__ import annotations

from x_likes_mcp.synthesis.entities import (
    extract_regex,
    extract_with_dspy_fallback,
)
from x_likes_mcp.synthesis.types import Entity, EntityKind

# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------


def test_handle_extraction() -> None:
    entities = extract_regex("hello @alice and @bob", [])
    handles = {e.value for e in entities if e.kind is EntityKind.HANDLE}
    assert handles == {"alice", "bob"}


def test_hashtag_extraction() -> None:
    entities = extract_regex("#AI is #great", [])
    tags = {e.value for e in entities if e.kind is EntityKind.HASHTAG}
    assert tags == {"ai", "great"}


def test_domain_extraction_dedupes_www() -> None:
    text = "see https://www.example.com/x and https://example.com/y"
    entities = extract_regex(text, [])
    domains = [e for e in entities if e.kind is EntityKind.DOMAIN]
    assert len(domains) == 1
    assert domains[0].value == "example.com"
    assert domains[0].weight == 2.0


def test_concept_extraction_recurring() -> None:
    text = (
        "AI Pentesting is a hot topic in security circles this year. "
        "Many practitioners now consider AI Pentesting a core capability. "
        "Random Phrase showed up here only once and should not surface."
    )
    entities = extract_regex(text, [])
    concepts = {e.value: e.weight for e in entities if e.kind is EntityKind.CONCEPT}
    assert "ai_pentesting" in concepts
    assert concepts["ai_pentesting"] == 2.0
    assert "random_phrase" not in concepts


def test_concept_extraction_short_text() -> None:
    entities = extract_regex("Quick AI Pentesting note", [])
    concepts = {e.value for e in entities if e.kind is EntityKind.CONCEPT}
    assert "ai_pentesting" in concepts


def test_extract_regex_includes_url_bodies() -> None:
    entities = extract_regex(
        "no entities in tweet body itself.",
        ["a body mentioning @carol and #devsecops"],
    )
    handles = {e.value for e in entities if e.kind is EntityKind.HANDLE}
    tags = {e.value for e in entities if e.kind is EntityKind.HASHTAG}
    assert handles == {"carol"}
    assert tags == {"devsecops"}


def test_extract_regex_returns_empty_for_stopwords() -> None:
    assert extract_regex("the and is or but", []) == []


# ---------------------------------------------------------------------------
# DSPy fallback hook
# ---------------------------------------------------------------------------


def test_fallback_not_called_when_regex_finds_entities() -> None:
    counter = {"calls": 0}

    def fallback(_text: str) -> list[Entity]:
        counter["calls"] += 1
        return [Entity(EntityKind.CONCEPT, "should_not_appear", 1.0)]

    result = extract_with_dspy_fallback("hello @alice", fallback=fallback)

    assert counter["calls"] == 0
    assert any(e.kind is EntityKind.HANDLE and e.value == "alice" for e in result)
    assert all(e.value != "should_not_appear" for e in result)


def test_fallback_called_when_regex_empty() -> None:
    counter = {"calls": 0}

    def fallback(_text: str) -> list[Entity]:
        counter["calls"] += 1
        return [Entity(EntityKind.CONCEPT, "fallback", 1.0)]

    result = extract_with_dspy_fallback(
        "the and is or but",
        fallback=fallback,
    )

    assert counter["calls"] == 1
    assert result == [Entity(EntityKind.CONCEPT, "fallback", 1.0)]


def test_fallback_called_at_most_once_per_hit() -> None:
    counter = {"calls": 0}

    def fallback(_text: str) -> list[Entity]:
        counter["calls"] += 1
        return [Entity(EntityKind.CONCEPT, "fallback", 1.0)]

    extract_with_dspy_fallback("the and is or but", fallback=fallback)

    assert counter["calls"] == 1


def test_no_fallback_returns_empty_list_when_regex_empty() -> None:
    assert extract_with_dspy_fallback("the and is or but") == []


# ---------------------------------------------------------------------------
# Concept stopword filter
# ---------------------------------------------------------------------------


def test_concept_stopword_only_phrase_dropped() -> None:
    """Bare interjections / articles never surface as concepts even when
    they recur enough to clear the count threshold."""

    text = "Yeah, well, yeah! " "Yeah, of course. The quick brown fox. " "Yeah okay yeah."
    entities = extract_regex(text, [])
    concept_values = {e.value for e in entities if e.kind is EntityKind.CONCEPT}
    assert "yeah" not in concept_values
    assert "the" not in concept_values
    assert "ok" not in concept_values
    assert "okay" not in concept_values


def test_concept_partial_stopword_phrase_survives() -> None:
    """A multi-word phrase with at least one informative token survives.

    Only phrases whose tokens are *all* stopwords get dropped. A phrase
    where the stopword sits between informative tokens is kept.
    """

    text = (
        "Quantum The Algorithm. Quantum The Algorithm. " "Some other text. Quantum The Algorithm."
    )
    entities = extract_regex(text, [])
    concept_values = {e.value for e in entities if e.kind is EntityKind.CONCEPT}
    # Phrase containing "the" between two informative tokens survives.
    assert "quantum_the_algorithm" in concept_values
