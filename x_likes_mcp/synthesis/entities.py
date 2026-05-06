"""Entity extraction for the synthesis-report feature.

Cheap regex / counter passes mine handles, hashtags, URL domains, and
recurring capitalized noun phrases out of round-1 hit text and any
fetched URL bodies before any LM call. The DSPy fallback hook is a
seam the orchestrator wires to ``dspy_modules.extract_entities`` (task
3.1) and fires only for hits where the regex pass returned nothing
(Requirements 5.1, 5.2).

The four regex families are:

* ``HANDLE`` — ``@[A-Za-z0-9_]{1,15}``; value is the handle without the
  leading ``@``, lowercased so two casings of the same screen name
  collapse into one entity.
* ``HASHTAG`` — ``#[\\w]+``; value is the tag without the leading ``#``
  and lowercased for the same reason.
* ``DOMAIN`` — ``https?://([^/\\s]+)``; the host is lowercased and a
  leading ``www.`` is stripped so ``www.example.com`` and
  ``example.com`` dedupe to a single entity.
* ``CONCEPT`` — capitalized noun phrases of one to three words. A
  phrase only surfaces once it appears at least twice across the
  combined text, with one exception: short hit text (under ~50
  characters) lets a single capitalized phrase through so a brief
  tweet's only proper noun still ends up in the KG.

The weight is the raw occurrence count, so a handle mentioned three
times across the hit body and the URL bodies has weight 3.0. Same
``value`` across different ``EntityKind`` values is allowed (handle
``openai`` and domain ``openai.com`` are distinct entities).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from .types import Entity, EntityKind

__all__ = [
    "extract_regex",
    "extract_with_dspy_fallback",
]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_HANDLE_RE = re.compile(r"@([A-Za-z0-9_]{1,15})")
_HASHTAG_RE = re.compile(r"#(\w+)")
_DOMAIN_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)
# Capitalized noun phrase: a run of consecutive Capitalized words. We
# accept both Title-cased words ("Pentesting") and all-caps acronyms
# two letters or longer ("AI", "URL") so a phrase like "AI Pentesting"
# surfaces as one concept. The regex captures the longest run and the
# caller emits every contiguous 1-3-word sub-window, so a sentence like
# "Quick AI Pentesting" surfaces both "AI Pentesting" and the longer
# "Quick AI Pentesting"; the short-text exception then lets at least
# one of them through even when the run only appeared once.
_CAPITALIZED_TOKEN = r"(?:[A-Z][a-z]+|[A-Z]{2,})"
_CONCEPT_RUN_RE = re.compile(rf"\b{_CAPITALIZED_TOKEN}(?:\s+{_CAPITALIZED_TOKEN})*\b")
_MAX_CONCEPT_WORDS = 3

# Threshold below which a short hit text is allowed to surface a
# capitalized phrase that appears only once. Anything longer than this
# requires at least two occurrences to suppress one-off proper nouns
# that would otherwise crowd the KG.
_SHORT_TEXT_LEN = 50


# Concept-extraction stopword set. A capitalized phrase whose tokens
# are *all* in this set is dropped from the concept stream; phrases
# where at least one token is outside the set survive ("Deutschland
# Bahn" would still pass even though "Deutschland" alone would not).
# This is the LM-free first line of defense against "Yeah", "The",
# bare interjections, and a few common bare demonyms / week-day-style
# proper nouns that recurrently appear at sentence boundaries in
# tweets but carry no topical signal.
#
# Strict English-only by design. Multilingual corpora should layer the
# DSPy ``FilterEntitiesByRelevance`` pass on top.
_CONCEPT_STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles + auxiliaries
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "if",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "must",
        "can",
        # Pronouns
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "you",
        "your",
        "we",
        "us",
        "our",
        "they",
        "them",
        "their",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "its",
        # Wh-words + quantifiers
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        # Interjections / filler
        "ok",
        "okay",
        "yes",
        "yeah",
        "yep",
        "nope",
        "nah",
        "huh",
        "wow",
        "lol",
        "haha",
        "uh",
        "um",
        "hey",
        "oh",
        "ah",
        "well",
        "hmm",
        # Common verbs
        "got",
        "get",
        "gets",
        "go",
        "goes",
        "going",
        "gone",
        "make",
        "makes",
        "made",
        "see",
        "sees",
        "saw",
        "seen",
        "say",
        "says",
        "said",
        "tell",
        "tells",
        "told",
        "think",
        "thinks",
        "thought",
        "know",
        "knows",
        "knew",
        "known",
        "want",
        "wants",
        "wanted",
        "use",
        "uses",
        "used",
        # Vague qualifiers
        "good",
        "great",
        "bad",
        "best",
        "better",
        "worse",
        "worst",
        "really",
        "actually",
        "basically",
        "literally",
        "totally",
        # Position
        "first",
        "second",
        "third",
        "last",
        "next",
        "previous",
        # Quantity
        "many",
        "much",
        "less",
        "lots",
    }
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def extract_regex(hit_text: str, url_bodies: list[str]) -> list[Entity]:
    """Run the four regex families over ``hit_text`` and ``url_bodies``.

    The url bodies are concatenated with the hit text into a single
    combined corpus before pattern matching; weights therefore reflect
    occurrences across both sources, which is what the orchestrator
    wants when it later turns top entities into round-2 search queries.

    The CONCEPT pass uses ``hit_text`` alone for the short-text
    exception so that a brief tweet body still surfaces its lone proper
    noun even when the URL bodies are long.
    """

    combined = hit_text
    for body in url_bodies:
        combined = f"{combined}\n{body}"

    entities: list[Entity] = []
    entities.extend(_extract_handles(combined))
    entities.extend(_extract_hashtags(combined))
    entities.extend(_extract_domains(combined))
    entities.extend(_extract_concepts(hit_text, combined))
    return entities


def extract_with_dspy_fallback(
    hit_text: str,
    *,
    fallback: Callable[[str], list[Entity]] | None = None,
) -> list[Entity]:
    """Return regex entities; only invoke ``fallback`` when regex is empty.

    The orchestrator wires ``fallback`` to the DSPy ``ExtractEntities``
    predictor (task 3.1). Per Requirement 5.2 the fallback fires at
    most once per hit and never for hits the regex pass already
    covered.
    """

    regex_result = extract_regex(hit_text, [])
    if regex_result:
        return regex_result
    if fallback is None:
        return []
    return fallback(hit_text)


# ---------------------------------------------------------------------------
# Per-kind helpers
# ---------------------------------------------------------------------------


def _extract_handles(text: str) -> list[Entity]:
    counts: Counter[str] = Counter(match.group(1).lower() for match in _HANDLE_RE.finditer(text))
    return [Entity(EntityKind.HANDLE, value, float(count)) for value, count in counts.items()]


def _extract_hashtags(text: str) -> list[Entity]:
    counts: Counter[str] = Counter(match.group(1).lower() for match in _HASHTAG_RE.finditer(text))
    return [Entity(EntityKind.HASHTAG, value, float(count)) for value, count in counts.items()]


def _extract_domains(text: str) -> list[Entity]:
    counts: Counter[str] = Counter()
    for match in _DOMAIN_RE.finditer(text):
        host = match.group(1).lower()
        if host.startswith("www."):
            host = host[4:]
        counts[host] += 1
    return [Entity(EntityKind.DOMAIN, value, float(count)) for value, count in counts.items()]


def _extract_concepts(hit_text: str, combined: str) -> list[Entity]:
    counts: Counter[str] = Counter()
    for match in _CONCEPT_RUN_RE.finditer(combined):
        words = match.group(0).split()
        # Emit every contiguous 1-3-word sub-window of the run. This is
        # how a long run like "Quick AI Pentesting" still produces the
        # "AI Pentesting" sub-phrase the KG and round-2 fan-out actually
        # want, without forcing the regex to enumerate every offset.
        for window in range(1, _MAX_CONCEPT_WORDS + 1):
            for start in range(len(words) - window + 1):
                phrase_words = [w.lower() for w in words[start : start + window]]
                # Drop the candidate when *every* token is a stopword.
                # Phrases with at least one informative token survive
                # ("Deutschland Bahn" still passes; bare "Deutschland"
                # would fall through if it were in the stopword set,
                # bare "Yeah" / "The" / "Hmm" do not).
                if all(token in _CONCEPT_STOPWORDS for token in phrase_words):
                    continue
                # Normalize to lower-snake-case so the value is a stable
                # KG ID ("AI Pentesting" -> "ai_pentesting"). The KG
                # namespace doc pins this shape.
                key = "_".join(phrase_words)
                counts[key] += 1

    is_short = len(hit_text) < _SHORT_TEXT_LEN
    threshold = 1 if is_short else 2

    return [
        Entity(EntityKind.CONCEPT, value, float(count))
        for value, count in counts.items()
        if count >= threshold
    ]
