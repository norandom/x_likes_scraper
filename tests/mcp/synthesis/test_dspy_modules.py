"""Tests for ``x_likes_mcp.synthesis.dspy_modules``.

These tests cover task 3.1 of the synthesis-report spec:

- The three ``Synthesize*`` signatures carry the system-prompt rules in
  their ``__doc__`` (Req 7.3).
- ``configure_lm`` raises a clear error when ``OPENAI_BASE_URL`` or
  ``OPENAI_MODEL`` is missing (Req 6.2) and otherwise installs a
  ``dspy.LM`` that points at the configured endpoint (Req 12.1).
- ``synthesize`` runs the per-shape ChainOfThought program, validates
  every emitted claim's ``sources`` against ``known_source_ids``,
  retries once on a hallucinated source, and raises
  ``SynthesisValidationError`` if the second attempt is also bad
  (Req 6.6).
- ``synthesize`` validates the trend shape via ``per_month[i].anchor_tweets``.
- ``extract_entities`` runs the ``ExtractEntities`` predict signature
  (Req 5.2).

The tests stub ``synthesize``'s ``program`` argument with a tiny
duck-typed fake that returns a ``dspy.Prediction`` so we exercise the
validator and result-shape mapping without going through DSPy's
structured-output parser. The autouse ``_stub_dspy_lm`` fixture from
``conftest.py`` already keeps DSPy offline; these tests layer on top of
that.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import dspy
import pytest

from x_likes_mcp.config import Config, ConfigError, RankerWeights
from x_likes_mcp.synthesis.dspy_modules import (
    _SYSTEM_PROMPT_RULES,
    ExtractEntities,
    FilterEntitiesByRelevance,
    SynthesisError,
    SynthesisResult,
    SynthesisValidationError,
    SynthesizeBrief,
    SynthesizeNarrative,
    SynthesizeTrend,
    configure_lm,
    extract_entities,
    filter_entities_by_relevance,
    make_synthesizer,
    synthesize,
)
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import Claim, Entity, EntityKind, MonthSummary, Section

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides: Any) -> Config:
    """Build a minimal :class:`Config` for the LM-configuration tests."""

    base = Config(
        output_dir=Path("output"),
        by_month_dir=Path("output/by_month"),
        likes_json=Path("output/likes.json"),
        cache_path=Path("output/tweet_tree_cache.pkl"),
        ranker_weights=RankerWeights(),
        openai_base_url="http://127.0.0.1:8080/v1",
        openai_model="local/llama-3.1-8b-instruct",
        openai_api_key="EMPTY",
    )
    if overrides:
        return replace(base, **overrides)
    return base


class _FakeProgram:
    """Duck-typed stand-in for a ``dspy.ChainOfThought`` instance.

    Returns the next queued :class:`dspy.Prediction` on each call. Tests
    queue one prediction per expected pass through the program (typically
    one for the first pass, optionally a second for the retry).
    """

    def __init__(self, predictions: list[dspy.Prediction]) -> None:
        self._predictions = list(predictions)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        self.calls.append(dict(kwargs))
        if not self._predictions:
            raise AssertionError("FakeProgram exhausted; queued no further predictions")
        return self._predictions.pop(0)


# ---------------------------------------------------------------------------
# Signature docstring rules (Req 7.3)
# ---------------------------------------------------------------------------


def test_system_prompt_rules_constant_has_three_rules() -> None:
    """``_SYSTEM_PROMPT_RULES`` carries the three fence-discipline rules."""

    text = _SYSTEM_PROMPT_RULES
    assert "user-supplied data" in text
    assert "user query" in text
    assert "Do not echo" in text


def test_synthesize_brief_docstring_carries_rules() -> None:
    assert _SYSTEM_PROMPT_RULES.strip() in (SynthesizeBrief.__doc__ or "")


def test_synthesize_narrative_docstring_carries_rules() -> None:
    assert _SYSTEM_PROMPT_RULES.strip() in (SynthesizeNarrative.__doc__ or "")


def test_synthesize_trend_docstring_carries_rules_and_month_order() -> None:
    doc = SynthesizeTrend.__doc__ or ""
    assert _SYSTEM_PROMPT_RULES.strip() in doc
    assert "month_buckets" in doc


# ---------------------------------------------------------------------------
# configure_lm (Req 6.2 / 12.1)
# ---------------------------------------------------------------------------


def test_configure_lm_raises_when_model_missing() -> None:
    config = _base_config(openai_model=None)
    with pytest.raises(ConfigError) as excinfo:
        configure_lm(config)
    assert "OPENAI_MODEL" in str(excinfo.value)


def test_configure_lm_raises_when_base_url_missing() -> None:
    config = _base_config(openai_base_url=None)
    with pytest.raises(ConfigError) as excinfo:
        configure_lm(config)
    assert "OPENAI_BASE_URL" in str(excinfo.value)


def test_configure_lm_installs_dspy_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the LM must install it via ``dspy.configure(lm=...)``."""

    captured: dict[str, Any] = {}

    class _RecordingLM:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        def __call__(self, *args: Any, **kwargs: Any) -> list[str]:
            return [""]

    monkeypatch.setattr(dspy, "LM", _RecordingLM)

    config = _base_config(
        openai_base_url="http://127.0.0.1:8080/v1",
        openai_model="local/llama-3.1-8b-instruct",
        openai_api_key="",
    )
    configure_lm(config)

    assert isinstance(dspy.settings.lm, _RecordingLM)
    kwargs = captured["kwargs"]
    assert kwargs.get("model") == "local/llama-3.1-8b-instruct"
    assert kwargs.get("api_base") == "http://127.0.0.1:8080/v1"
    # Empty api_key in config maps to a placeholder for litellm.
    assert kwargs.get("api_key")  # truthy
    # restore the autouse fake LM so subsequent tests aren't affected.
    dspy.configure(lm=None)


# ---------------------------------------------------------------------------
# make_synthesizer
# ---------------------------------------------------------------------------


def _signature_field_names(program: dspy.ChainOfThought) -> set[str]:
    """Extract the set of field names from a ChainOfThought's bound signature.

    DSPy 3.x stores the bound signature on ``program.predict.signature``;
    the resulting object is a ``StringSignature`` whose
    ``output_fields`` mapping carries the structured-output field names.
    """

    return set(program.predict.signature.output_fields.keys())


def test_make_synthesizer_brief_returns_chain_of_thought_for_brief_signature() -> None:
    program = make_synthesizer(ReportShape.BRIEF)
    assert isinstance(program, dspy.ChainOfThought)
    fields = _signature_field_names(program)
    # SynthesizeBrief contributes claims + top_entities (plus
    # ChainOfThought adds a reasoning field).
    assert "claims" in fields
    assert "top_entities" in fields
    assert "sections" not in fields
    assert "per_month" not in fields


def test_make_synthesizer_synthesis_returns_narrative_signature() -> None:
    program = make_synthesizer(ReportShape.SYNTHESIS)
    assert isinstance(program, dspy.ChainOfThought)
    fields = _signature_field_names(program)
    assert "sections" in fields
    assert "cluster_assignments" in fields
    assert "claims" not in fields


def test_make_synthesizer_trend_returns_trend_signature() -> None:
    program = make_synthesizer(ReportShape.TREND)
    assert isinstance(program, dspy.ChainOfThought)
    fields = _signature_field_names(program)
    assert "per_month" in fields
    assert "top_entities" in fields
    assert "claims" not in fields


# ---------------------------------------------------------------------------
# synthesize: brief shape
# ---------------------------------------------------------------------------


def test_synthesize_brief_returns_claims_when_sources_valid() -> None:
    claim = Claim(text="Local-first is preferred.", sources=["tweet:1"])
    fake = _FakeProgram(
        [dspy.Prediction(claims=[claim], top_entities=["@windsheep_"])],
    )

    result = synthesize(
        ReportShape.BRIEF,
        query="local-first AI",
        fenced_context="<<<TWEET_BODY>>>...<<<END_TWEET_BODY>>>",
        known_source_ids={"tweet:1"},
        program=fake,
    )

    assert isinstance(result, SynthesisResult)
    assert result.claims == [claim]
    assert result.top_entities == ["@windsheep_"]
    assert result.sections is None
    assert result.per_month is None
    # First pass succeeded; no retry.
    assert len(fake.calls) == 1
    # Inputs forwarded verbatim.
    assert fake.calls[0]["query"] == "local-first AI"
    assert fake.calls[0]["fenced_context"].startswith("<<<TWEET_BODY>>>")


def test_synthesize_brief_retries_on_unknown_source() -> None:
    bad = Claim(text="Bad cite", sources=["tweet:999"])
    good = Claim(text="Good cite", sources=["tweet:1"])
    fake = _FakeProgram(
        [
            dspy.Prediction(claims=[bad], top_entities=[]),
            dspy.Prediction(claims=[good], top_entities=[]),
        ],
    )

    result = synthesize(
        ReportShape.BRIEF,
        query="q",
        fenced_context="ctx",
        known_source_ids={"tweet:1"},
        program=fake,
    )

    assert result.claims == [good]
    assert len(fake.calls) == 2
    # The retry must include corrective feedback that names the bad ID.
    second_query = fake.calls[1]["query"]
    assert "tweet:999" in second_query
    assert "q" in second_query


def test_synthesize_brief_raises_on_repeated_unknown_source() -> None:
    bad = Claim(text="Bad", sources=["tweet:999"])
    fake = _FakeProgram(
        [
            dspy.Prediction(claims=[bad], top_entities=[]),
            dspy.Prediction(claims=[bad], top_entities=[]),
        ],
    )

    with pytest.raises(SynthesisValidationError) as excinfo:
        synthesize(
            ReportShape.BRIEF,
            query="q",
            fenced_context="ctx",
            known_source_ids={"tweet:1"},
            program=fake,
        )

    assert "tweet:999" in str(excinfo.value)
    # Validator must have called the program exactly twice.
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# synthesize: synthesis (narrative) shape
# ---------------------------------------------------------------------------


def test_synthesize_narrative_validates_section_claims() -> None:
    bad_section = Section(
        heading="Bad",
        claims=[Claim(text="x", sources=["tweet:999"])],
    )
    good_section = Section(
        heading="Good",
        claims=[Claim(text="y", sources=["tweet:1"])],
    )
    fake = _FakeProgram(
        [
            dspy.Prediction(
                sections=[bad_section],
                top_entities=[],
                cluster_assignments={},
            ),
            dspy.Prediction(
                sections=[good_section],
                top_entities=["@a"],
                cluster_assignments={"Themes": ["@a"]},
            ),
        ],
    )

    result = synthesize(
        ReportShape.SYNTHESIS,
        query="q",
        fenced_context="ctx",
        known_source_ids={"tweet:1"},
        program=fake,
    )

    assert result.sections == [good_section]
    assert result.top_entities == ["@a"]
    assert result.cluster_assignments == {"Themes": ["@a"]}
    assert result.claims is None
    assert len(fake.calls) == 2


def test_synthesize_narrative_raises_on_repeated_unknown_section_source() -> None:
    bad_section = Section(
        heading="Bad",
        claims=[Claim(text="x", sources=["tweet:999"])],
    )
    fake = _FakeProgram(
        [
            dspy.Prediction(
                sections=[bad_section],
                top_entities=[],
                cluster_assignments={},
            ),
            dspy.Prediction(
                sections=[bad_section],
                top_entities=[],
                cluster_assignments={},
            ),
        ],
    )

    with pytest.raises(SynthesisValidationError):
        synthesize(
            ReportShape.SYNTHESIS,
            query="q",
            fenced_context="ctx",
            known_source_ids={"tweet:1"},
            program=fake,
        )


# ---------------------------------------------------------------------------
# synthesize: trend shape
# ---------------------------------------------------------------------------


def test_synthesize_trend_validates_anchor_tweets() -> None:
    bad_month = MonthSummary(
        year_month="2026-01",
        summary="Bad anchor month",
        anchor_tweets=["tweet:999"],
    )
    good_month = MonthSummary(
        year_month="2026-01",
        summary="Good anchor month",
        anchor_tweets=["tweet:1"],
    )
    fake = _FakeProgram(
        [
            dspy.Prediction(per_month=[bad_month], top_entities=[]),
            dspy.Prediction(per_month=[good_month], top_entities=["#x"]),
        ],
    )

    result = synthesize(
        ReportShape.TREND,
        query="q",
        fenced_context="ctx",
        known_source_ids={"tweet:1"},
        month_buckets=["2026-01"],
        program=fake,
    )

    assert result.per_month == [good_month]
    assert result.top_entities == ["#x"]
    assert len(fake.calls) == 2
    # The trend program is called with the month_buckets input field.
    assert fake.calls[0]["month_buckets"] == ["2026-01"]


def test_synthesize_trend_raises_on_repeated_unknown_anchor() -> None:
    bad_month = MonthSummary(
        year_month="2026-01",
        summary="bad",
        anchor_tweets=["tweet:999"],
    )
    fake = _FakeProgram(
        [
            dspy.Prediction(per_month=[bad_month], top_entities=[]),
            dspy.Prediction(per_month=[bad_month], top_entities=[]),
        ],
    )

    with pytest.raises(SynthesisValidationError) as excinfo:
        synthesize(
            ReportShape.TREND,
            query="q",
            fenced_context="ctx",
            known_source_ids={"tweet:1"},
            month_buckets=["2026-01"],
            program=fake,
        )

    assert "tweet:999" in str(excinfo.value)


# ---------------------------------------------------------------------------
# synthesize: error hierarchy
# ---------------------------------------------------------------------------


def test_synthesis_validation_error_is_synthesis_error() -> None:
    """``SynthesisValidationError`` derives from ``SynthesisError``."""

    assert issubclass(SynthesisValidationError, SynthesisError)


# ---------------------------------------------------------------------------
# extract_entities (Req 5.2)
# ---------------------------------------------------------------------------


def test_extract_entities_uses_predict_signature() -> None:
    entity = Entity(kind=EntityKind.HANDLE, value="@windsheep_", weight=1.0)
    fake = _FakeProgram([dspy.Prediction(entities=[entity])])

    result = extract_entities("hello @windsheep_", hints=["focus"], program=fake)

    assert result == [entity]
    assert len(fake.calls) == 1
    assert fake.calls[0]["text"] == "hello @windsheep_"
    assert fake.calls[0]["hints"] == ["focus"]


def test_extract_entities_signature_class_present() -> None:
    """The ``ExtractEntities`` signature must be a DSPy signature subclass."""

    assert issubclass(ExtractEntities, dspy.Signature)


# ---------------------------------------------------------------------------
# filter_entities_by_relevance (LM-backed KG noise filter)
# ---------------------------------------------------------------------------


def test_filter_entities_signature_class_present() -> None:
    assert issubclass(FilterEntitiesByRelevance, dspy.Signature)


def test_filter_entities_signature_carries_system_prompt_rules() -> None:
    """The three fence-discipline rules must ride with every filter call
    via the signature ``__doc__``."""

    doc = FilterEntitiesByRelevance.__doc__ or ""
    assert _SYSTEM_PROMPT_RULES in doc


def test_filter_entities_empty_short_circuits_without_lm() -> None:
    """Empty candidate list returns ``[]`` without invoking the program."""

    program = _FakeProgram([])  # would IndexError if called
    assert filter_entities_by_relevance("query", [], program=program) == []
    assert program.calls == []


def test_filter_entities_single_candidate_short_circuits() -> None:
    """A single-candidate input is its own pass-through."""

    program = _FakeProgram([])  # would IndexError if called
    assert filter_entities_by_relevance("query", ["only_handle"], program=program) == [
        "only_handle"
    ]
    assert program.calls == []


def test_filter_entities_drops_candidates_lm_omits() -> None:
    """Candidates the LM omits from the relevant list are dropped."""

    program = _FakeProgram([dspy.Prediction(relevant=["alpha", "beta"])])

    out = filter_entities_by_relevance(
        "query",
        ["alpha", "beta", "gamma"],
        program=program,
    )

    assert out == ["alpha", "beta"]


def test_filter_entities_drops_hallucinated_labels() -> None:
    """Items the LM returns that were not in the original candidate set
    are silently dropped — a hallucinated entity must not slip in."""

    program = _FakeProgram([dspy.Prediction(relevant=["alpha", "fabricated"])])

    out = filter_entities_by_relevance(
        "query",
        ["alpha", "beta"],
        program=program,
    )

    assert out == ["alpha"]


def test_filter_entities_dedupes_repeated_lm_output() -> None:
    program = _FakeProgram([dspy.Prediction(relevant=["alpha", "alpha", "beta"])])
    out = filter_entities_by_relevance(
        "query",
        ["alpha", "beta", "gamma"],
        program=program,
    )
    assert out == ["alpha", "beta"]


def test_filter_entities_preserves_lm_order() -> None:
    program = _FakeProgram([dspy.Prediction(relevant=["gamma", "alpha"])])
    out = filter_entities_by_relevance(
        "query",
        ["alpha", "beta", "gamma"],
        program=program,
    )
    assert out == ["gamma", "alpha"]
