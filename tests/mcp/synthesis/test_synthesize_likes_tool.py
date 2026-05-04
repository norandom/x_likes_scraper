"""Tests for the ``synthesize_likes`` MCP tool boundary (task 5.2).

The tool sits one layer above the synthesis orchestrator: it validates
inputs at the MCP boundary, refuses to contact the crawl4ai container
when the caller did not opt in via ``fetch_urls=True`` (Req 4.7 / 10.3),
builds a :class:`ReportOptions` envelope, calls
:func:`x_likes_mcp.synthesis.orchestrator.run_report`, and translates
:class:`OrchestratorError` envelopes into the structured ``ToolError``
categories the MCP server boundary surfaces to the calling LLM
(``invalid_input`` / ``upstream_failure`` / ``not_found``).

Boundary discipline: these tests stub the orchestrator's
``run_report`` entry point through the seam ``tools.run_report`` so the
test never touches DSPy, the URL fetcher, or any heavyweight stage. The
autouse guards in the synthesis conftest still raise on any leaked DSPy
or HTTP call, so a regression that bypasses the seam fails loudly.

Pinned envelope: every successful call returns exactly the four
documented keys ``{"markdown", "shape", "used_hops",
"fetched_url_count"}`` (Req 10.1 / 10.2 / design ``MCP API contract``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from x_likes_mcp import tools
from x_likes_mcp.errors import ToolError
from x_likes_mcp.synthesis.orchestrator import OrchestratorError
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import ReportOptions, ReportResult

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeIndex:
    """Stub TweetIndex carrying just the ``config`` attribute the tool reads.

    The orchestrator is monkeypatched in every test, so the index does
    not need a working ``search`` or ``tweets_by_id`` map. We pin
    ``config`` to a sentinel so the assertion ``run_report`` was called
    with ``config=index.config`` survives.
    """

    def __init__(self) -> None:
        self.config = object()


def _stub_run_report(
    monkeypatch: pytest.MonkeyPatch,
    *,
    impl: Callable[..., ReportResult] | None = None,
) -> list[dict[str, Any]]:
    """Patch ``tools.run_report`` with a recording stub.

    Returns the call recorder so individual tests can assert on the
    options the orchestrator received.
    """

    calls: list[dict[str, Any]] = []

    def _stub(
        index: Any,
        options: ReportOptions,
        *,
        config: Any,
    ) -> ReportResult:
        calls.append({"index": index, "options": options, "config": config})
        if impl is not None:
            return impl(index, options, config=config)
        return ReportResult(
            markdown="# stub report\n",
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(tools, "run_report", _stub)
    return calls


# ---------------------------------------------------------------------------
# Happy path / response envelope
# ---------------------------------------------------------------------------


def test_synthesize_likes_returns_documented_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The success response carries exactly the four documented keys."""

    _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    result = tools.synthesize_likes(idx, query="ai security", report_shape="brief")

    assert set(result.keys()) == {
        "markdown",
        "shape",
        "used_hops",
        "fetched_url_count",
    }
    assert result["markdown"] == "# stub report\n"
    assert result["shape"] == "brief"
    assert result["used_hops"] == 1
    assert result["fetched_url_count"] == 0


def test_synthesize_likes_passes_options_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All caller inputs land on the :class:`ReportOptions` instance."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    tools.synthesize_likes(
        idx,
        query="  ai security  ",
        report_shape="synthesis",
        fetch_urls=True,
        hops=2,
        year=2025,
        month_start="01",
        month_end="03",
        limit=20,
    )

    assert len(calls) == 1
    options = calls[0]["options"]
    assert isinstance(options, ReportOptions)
    # Sanitized + stripped query is forwarded.
    assert options.query == "ai security"
    assert options.shape is ReportShape.SYNTHESIS
    assert options.fetch_urls is True
    assert options.hops == 2
    assert options.year == 2025
    assert options.month_start == "01"
    assert options.month_end == "03"
    assert options.limit == 20
    # The caller-supplied ``index.config`` is forwarded as the kw.
    assert calls[0]["config"] is idx.config


def test_synthesize_likes_passes_filters_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Year / month range filters reach the orchestrator unchanged."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    tools.synthesize_likes(
        idx,
        query="x",
        report_shape="brief",
        year=2024,
        month_start="06",
    )

    options = calls[0]["options"]
    assert options.year == 2024
    assert options.month_start == "06"
    assert options.month_end is None


def test_synthesize_likes_passes_limit_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller-supplied limit reaches the orchestrator."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    tools.synthesize_likes(idx, query="x", report_shape="brief", limit=20)

    assert calls[0]["options"].limit == 20


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_synthesize_likes_default_fetch_urls_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller omits ``fetch_urls``, MCP defaults to False (Req 10.3 / 4.7)."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert calls[0]["options"].fetch_urls is False


def test_synthesize_likes_default_hops_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller omits ``hops``, the orchestrator sees ``hops=1``."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert calls[0]["options"].hops == 1


def test_synthesize_likes_default_limit_matches_options_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted ``limit`` falls back to :class:`ReportOptions`'s default (50)."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert calls[0]["options"].limit == 50


# ---------------------------------------------------------------------------
# Input validation — rejected before orchestrator runs
# ---------------------------------------------------------------------------


def test_synthesize_likes_invalid_shape_raises_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown shape rejected before any orchestrator / LM call (Req 10.4 / 1.3)."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="x", report_shape="bogus")

    assert excinfo.value.category == "invalid_input"
    assert "report_shape" in excinfo.value.message
    assert calls == []  # orchestrator never invoked


def test_synthesize_likes_empty_query_raises_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-string query rejected at the boundary."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="", report_shape="brief")

    assert excinfo.value.category == "invalid_input"
    assert "query" in excinfo.value.message
    assert calls == []


def test_synthesize_likes_whitespace_query_raises_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only query rejected at the boundary."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="   ", report_shape="brief")

    assert excinfo.value.category == "invalid_input"
    assert "query" in excinfo.value.message
    assert calls == []


def test_synthesize_likes_non_string_query_raises_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-string query is rejected at the boundary, not the orchestrator."""

    calls = _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query=123, report_shape="brief")  # type: ignore[arg-type]

    assert excinfo.value.category == "invalid_input"
    assert "query" in excinfo.value.message
    assert calls == []


# ---------------------------------------------------------------------------
# Error translation from the orchestrator
# ---------------------------------------------------------------------------


def test_synthesize_likes_translates_invalid_input_from_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OrchestratorError("invalid_input", ...)`` becomes ``ToolError`` invalid_input."""

    def _raise(*_: Any, **__: Any) -> ReportResult:
        raise OrchestratorError("invalid_input", "hops=99 is out of range")

    _stub_run_report(monkeypatch, impl=_raise)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="x", report_shape="brief", hops=99)

    assert excinfo.value.category == "invalid_input"
    assert "hops" in excinfo.value.message


def test_synthesize_likes_translates_config_error_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OrchestratorError("config", ...)`` surfaces as ``upstream_failure``."""

    def _raise(*_: Any, **__: Any) -> ReportResult:
        raise OrchestratorError("config", "OPENAI_API_KEY missing")

    _stub_run_report(monkeypatch, impl=_raise)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert excinfo.value.category == "upstream_failure"
    assert "OPENAI_API_KEY" in excinfo.value.message


def test_synthesize_likes_translates_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OrchestratorError("upstream", ...)`` surfaces as ``upstream_failure``."""

    def _raise(*_: Any, **__: Any) -> ReportResult:
        raise OrchestratorError("upstream", "LM endpoint unreachable")

    _stub_run_report(monkeypatch, impl=_raise)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert excinfo.value.category == "upstream_failure"
    assert "LM endpoint unreachable" in excinfo.value.message


def test_synthesize_likes_translates_validation_error_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OrchestratorError("validation", ...)`` surfaces as ``upstream_failure``.

    The synthesizer hallucinated unknown source IDs after a corrective
    retry; the design routes this to ``upstream_failure`` so the caller
    can retry transparently.
    """

    def _raise(*_: Any, **__: Any) -> ReportResult:
        raise OrchestratorError("validation", "unknown source ids: tweet:42")

    _stub_run_report(monkeypatch, impl=_raise)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert excinfo.value.category == "upstream_failure"
    assert "synthesis validation" in excinfo.value.message
    assert "tweet:42" in excinfo.value.message


def test_synthesize_likes_translates_unknown_orchestrator_category_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized orchestrator category is conservatively mapped to upstream_failure."""

    def _raise(*_: Any, **__: Any) -> ReportResult:
        raise OrchestratorError("mystery", "unexpected category")

    _stub_run_report(monkeypatch, impl=_raise)
    idx = _FakeIndex()

    with pytest.raises(ToolError) as excinfo:
        tools.synthesize_likes(idx, query="x", report_shape="brief")

    assert excinfo.value.category == "upstream_failure"
    assert "unexpected category" in excinfo.value.message


# ---------------------------------------------------------------------------
# Empty corpus
# ---------------------------------------------------------------------------


def test_synthesize_likes_empty_corpus_returns_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-corpus reports return the documented success envelope (Req 9.4).

    The orchestrator already produces an empty-report markdown without an
    LM call. The tool boundary returns it as a successful envelope so the
    response shape stays pinned at ``{markdown, shape, used_hops,
    fetched_url_count}``; only the markdown body announces "no matching
    tweets".
    """

    empty_markdown = (
        '# Likes report — query "obscure topic"\n\n'
        "No matching tweets were found for this query.\n"
    )

    def _empty_report(_index: Any, options: ReportOptions, *, config: Any) -> ReportResult:
        return ReportResult(
            markdown=empty_markdown,
            shape=options.shape,
            used_hops=1,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    _stub_run_report(monkeypatch, impl=_empty_report)
    idx = _FakeIndex()

    result = tools.synthesize_likes(
        idx,
        query="obscure topic",
        report_shape="brief",
    )

    assert set(result.keys()) == {
        "markdown",
        "shape",
        "used_hops",
        "fetched_url_count",
    }
    assert result["markdown"] == empty_markdown
    assert result["shape"] == "brief"
    assert result["used_hops"] == 1
    assert result["fetched_url_count"] == 0


# ---------------------------------------------------------------------------
# Result shape pin
# ---------------------------------------------------------------------------


def test_synthesize_likes_returns_used_hops_and_fetched_count_from_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``used_hops`` / ``fetched_url_count`` reflect the orchestrator's result."""

    def _impl(_index: Any, options: ReportOptions, *, config: Any) -> ReportResult:
        return ReportResult(
            markdown="# r\n",
            shape=options.shape,
            used_hops=2,
            fetched_url_count=3,
            synthesis_token_count=42,
        )

    _stub_run_report(monkeypatch, impl=_impl)
    idx = _FakeIndex()

    result = tools.synthesize_likes(
        idx,
        query="x",
        report_shape="synthesis",
        hops=2,
    )

    assert result["used_hops"] == 2
    assert result["fetched_url_count"] == 3
    # ``synthesis_token_count`` is internal — must not leak through.
    assert "synthesis_token_count" not in result


def test_synthesize_likes_shape_string_not_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response carries the shape as a plain string, not the enum."""

    _stub_run_report(monkeypatch)
    idx = _FakeIndex()

    result = tools.synthesize_likes(idx, query="x", report_shape="trend")

    assert result["shape"] == "trend"
    assert isinstance(result["shape"], str)
    # Defensive: not a ReportShape instance even though StrEnum subclasses str.
    assert not isinstance(result["shape"], ReportShape)
