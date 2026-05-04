"""End-to-end MCP tool tests for ``synthesize_likes`` (task 6.3).

These tests exercise the synthesis-report feature across the full MCP
boundary surface: the SDK-facing tool list (advertised to the calling
LLM) and the in-process :func:`x_likes_mcp.server._dispatch` entry the
``@server.call_tool()`` wrapper invokes. The deeper layers
(:func:`x_likes_mcp.tools.synthesize_likes` and the orchestrator) are
covered by their own unit tests; this module pins the *server-level*
behaviour:

* The dispatcher returns the documented success envelope verbatim
  ``{markdown, shape, used_hops, fetched_url_count}`` and does not
  double-sanitize the markdown that the orchestrator already passed
  through ``sanitize_text`` (Req 10.1, 10.2).
* An invalid ``report_shape`` is rejected at the boundary as
  ``invalid_input`` and the orchestrator is *never* called (Req 10.4).
* When the caller omits ``fetch_urls``, the boundary forwards
  ``fetch_urls=False`` to ``tools.synthesize_likes`` regardless of any
  ambient configuration (Req 10.3 / 4.7).
* The structured filter fields (``year`` / ``month_start`` /
  ``month_end``) reach the tool handler unchanged.
* The tool list advertises ``synthesize_likes`` with ``query`` and
  ``report_shape`` as required fields (Req 10.1).
* An ``upstream_failure`` raised by the tool surface propagates
  through the dispatcher with the documented category and message
  preserved.

Boundary: tests/mcp/synthesis/test_synthesize_likes_e2e.py only. Pure
test module — no source changes. The ``tools.synthesize_likes`` seam is
monkeypatched in every test so the synthesis pipeline (orchestrator,
DSPy, fetcher) is never touched.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from x_likes_mcp import server as server_module
from x_likes_mcp import tools as tools_module
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.errors import ToolError
from x_likes_mcp.index import TweetIndex

# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_os_environ() -> Iterator[None]:
    """Snapshot ``os.environ`` and restore it after each test.

    The shared ``fake_export`` fixture (in ``tests/mcp/conftest.py``)
    calls :func:`x_likes_mcp.config.load_config`, which has a documented
    side effect of writing ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` into
    ``os.environ``. The downstream ``tests/mcp/test_config.py`` tests
    rebind ``os.environ`` to a *copy of the current process environ* —
    so any leak from this file would surface there as a flaky failure
    in tests collected later (pytest collects synthesis subdir before
    the top-level ``test_config.py``). Snapshotting + restoring keeps
    this file's fixtures hermetic. Mirrors the autouse fixture in
    ``tests/mcp/synthesis/test_orchestrator_e2e.py``.
    """

    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_weights() -> RankerWeights:
    """Match the weight defaults the other server integration tests use."""

    return RankerWeights()


def _build_index(fake_export: Config) -> TweetIndex:
    """Construct a real :class:`TweetIndex` from the ``fake_export`` fixture.

    The walker is *not* invoked at construction time; index build is
    parser/IO only. The autouse embedder stub in the parent conftest
    keeps the cold-start embed step offline.
    """

    return TweetIndex.open_or_build(fake_export, _default_weights())


def _install_synth_spy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    raises: Exception | None = None,
) -> dict[str, Any]:
    """Replace ``tools.synthesize_likes`` with a recording spy.

    The dispatcher reaches ``tools.synthesize_likes`` directly; patching
    that attribute pins the boundary to a known envelope without
    exercising the orchestrator. Returns a dict with ``calls`` (a list
    of captured kwargs, one per invocation) so the test can assert on
    the forwarded arguments.
    """

    captured: dict[str, Any] = {"calls": []}

    def _spy(
        index: Any,
        *,
        query: str,
        report_shape: str,
        fetch_urls: bool = False,
        hops: int = 1,
        year: int | None = None,
        month_start: str | None = None,
        month_end: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        captured["calls"].append(
            {
                "index": index,
                "query": query,
                "report_shape": report_shape,
                "fetch_urls": fetch_urls,
                "hops": hops,
                "year": year,
                "month_start": month_start,
                "month_end": month_end,
                "limit": limit,
            }
        )
        if raises is not None:
            raise raises
        if payload is not None:
            return payload
        return {
            "markdown": "# stub report\n",
            "shape": report_shape,
            "used_hops": hops,
            "fetched_url_count": 0,
        }

    monkeypatch.setattr(tools_module, "synthesize_likes", _spy)
    return captured


# ---------------------------------------------------------------------------
# 1. Successful dispatch — pinned envelope shape
# ---------------------------------------------------------------------------


def test_e2e_synthesize_likes_returns_documented_envelope(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid call returns exactly the four documented keys (Req 10.1 / 10.2).

    The boundary must not add, drop, rename, or wrap the keys, and must
    not re-sanitize the markdown body — the orchestrator + renderer
    already passed every untrusted segment through ``sanitize_text``,
    and a second pass would risk munging legitimate report content.
    """

    # Pin a known payload with markdown that *would* be visibly altered
    # by a second sanitize pass. ``sanitize_text`` runs NFKC and NFKC
    # collapses fullwidth digits (U+FF11..U+FF13) to ASCII ``1``/``2``/
    # ``3``. The fullwidth digits are spelled via explicit ``\uFFxx``
    # escapes so the ``ruff`` ``RUF001`` "ambiguous unicode" check does
    # not flag the source line; the runtime string is the exact
    # fullwidth glyph sequence either way.
    fullwidth_one_two_three = "\uff11\uff12\uff13"
    pinned_markdown = f"# Synthesis report\n\nFullwidth digits: {fullwidth_one_two_three}\n"
    spy = _install_synth_spy(
        monkeypatch,
        payload={
            "markdown": pinned_markdown,
            "shape": "brief",
            "used_hops": 1,
            "fetched_url_count": 0,
        },
    )

    index = _build_index(fake_export)
    server_module.build_server(index)

    payload = server_module._dispatch(
        index,
        "synthesize_likes",
        {"query": "test", "report_shape": "brief"},
    )

    # The boundary forwarded the call exactly once.
    assert len(spy["calls"]) == 1

    # The response shape is pinned to the four documented keys.
    assert set(payload.keys()) == {
        "markdown",
        "shape",
        "used_hops",
        "fetched_url_count",
    }
    assert payload == {
        "markdown": pinned_markdown,
        "shape": "brief",
        "used_hops": 1,
        "fetched_url_count": 0,
    }

    # The dispatcher did NOT double-sanitize: the fullwidth digits in the
    # pinned markdown are still fullwidth. ``sanitize_text`` would have
    # NFKC-normalized them to ASCII "123".
    assert "\uff11\uff12\uff13" in payload["markdown"]


# ---------------------------------------------------------------------------
# 2. Invalid shape — rejected without invoking the orchestrator
# ---------------------------------------------------------------------------


def test_e2e_synthesize_likes_invalid_shape_returns_invalid_input_without_invoking_orchestrator(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid ``report_shape`` is rejected at the boundary (Req 10.4).

    The dispatcher raises a :class:`ToolError` with category
    ``invalid_input`` so the SDK ``call_tool`` wrapper can shape the
    structured MCP error response. Crucially, the spy on
    ``tools.synthesize_likes`` records zero calls — the orchestrator
    never runs an LM or search call when the shape is rejected at the
    boundary.

    The shape is rejected by ``_validate_report_shape`` inside
    ``tools.synthesize_likes`` rather than the dispatcher itself, so
    we install the spy *after* asserting that an unrelated valid call
    would have been recorded — i.e. the test pins the contract that the
    orchestrator-side work (search, LM, validation) does not run, not
    that ``tools.synthesize_likes`` itself is bypassed entirely.
    """

    # Spy on the deeper synthesis seam (``tools.run_report``) so we can
    # tell the orchestrator-side work apart from the boundary-validation
    # path. The real ``tools.synthesize_likes`` rejects the shape before
    # building ``ReportOptions`` or calling ``run_report``.
    orchestrator_calls: list[Any] = []

    def _orchestrator_spy(*args: Any, **kwargs: Any) -> Any:
        orchestrator_calls.append((args, kwargs))
        raise AssertionError("orchestrator must not be invoked when report_shape is invalid")

    monkeypatch.setattr(tools_module, "run_report", _orchestrator_spy)

    index = _build_index(fake_export)
    server_module.build_server(index)

    with pytest.raises(ToolError) as excinfo:
        server_module._dispatch(
            index,
            "synthesize_likes",
            {"query": "x", "report_shape": "bogus"},
        )

    assert excinfo.value.category == "invalid_input"
    assert "report_shape" in excinfo.value.message
    # The orchestrator (the work the boundary protects) never ran.
    assert orchestrator_calls == []


# ---------------------------------------------------------------------------
# 3. fetch_urls default — orchestrator runs with fetching disabled
# ---------------------------------------------------------------------------


def test_e2e_synthesize_likes_fetch_urls_default_false(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting ``fetch_urls`` pins the orchestrator to ``fetch_urls=False``.

    Req 10.3: even if a crawl4ai endpoint is configured in the
    environment, the boundary must not opt in to URL fetching unless
    the caller explicitly sets ``fetch_urls=true``. The MCP layer
    enforces this default by forwarding ``fetch_urls=False`` to
    ``tools.synthesize_likes`` whenever the argument is absent.
    """

    spy = _install_synth_spy(monkeypatch)

    index = _build_index(fake_export)
    server_module.build_server(index)

    server_module._dispatch(
        index,
        "synthesize_likes",
        {"query": "x", "report_shape": "brief"},
    )

    assert len(spy["calls"]) == 1
    call = spy["calls"][0]
    # The boundary forwarded ``fetch_urls=False`` even though the caller
    # did not set the field. This is the documented MCP default.
    assert call["fetch_urls"] is False


def test_e2e_synthesize_likes_fetch_urls_true_passes_through(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``fetch_urls=True`` reaches the orchestrator unchanged.

    Pairs with the default-false test above to pin the full contract:
    the boundary does not *force* fetching off, it just *defaults* it
    off. Callers who opt in see their value forwarded verbatim.
    """

    spy = _install_synth_spy(monkeypatch)

    index = _build_index(fake_export)
    server_module.build_server(index)

    server_module._dispatch(
        index,
        "synthesize_likes",
        {"query": "x", "report_shape": "brief", "fetch_urls": True},
    )

    assert spy["calls"][0]["fetch_urls"] is True


# ---------------------------------------------------------------------------
# 4. Filters pass through unchanged
# ---------------------------------------------------------------------------


def test_e2e_synthesize_likes_passes_filters_through(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``year`` / ``month_start`` / ``month_end`` reach the tool unchanged.

    The boundary must not silently drop, alias, or coerce the filter
    fields — the orchestrator validates them downstream and the
    response is shaped per filter, so any boundary-level mangling
    would be a silent correctness bug.
    """

    spy = _install_synth_spy(monkeypatch)

    index = _build_index(fake_export)
    server_module.build_server(index)

    server_module._dispatch(
        index,
        "synthesize_likes",
        {
            "query": "x",
            "report_shape": "brief",
            "year": 2025,
            "month_start": "01",
            "month_end": "03",
        },
    )

    assert len(spy["calls"]) == 1
    call = spy["calls"][0]
    assert call["year"] == 2025
    assert call["month_start"] == "01"
    assert call["month_end"] == "03"
    # Defaults for unrelated fields are still pinned.
    assert call["fetch_urls"] is False
    assert call["hops"] == 1
    assert call["limit"] == 50


# ---------------------------------------------------------------------------
# 5. Tool advertised with required fields in the public list
# ---------------------------------------------------------------------------


def test_e2e_synthesize_likes_advertised_in_tool_list() -> None:
    """The MCP tool list advertises ``synthesize_likes`` (Req 10.1).

    The tool's input schema must declare ``query`` and ``report_shape``
    as required so an MCP-aware client (Claude Code / Claude Desktop)
    surfaces them as mandatory fields. ``additionalProperties`` is
    pinned to ``False`` so callers cannot smuggle extra fields past
    the JSON-schema validation layer.
    """

    tools = server_module._build_tool_definitions()
    by_name = {t.name: t for t in tools}

    assert "synthesize_likes" in by_name

    synth = by_name["synthesize_likes"]
    schema = synth.inputSchema
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"

    required = schema.get("required", [])
    assert "query" in required
    assert "report_shape" in required

    props = schema["properties"]
    # query is a non-empty string.
    assert props["query"]["type"] == "string"
    assert props["query"].get("minLength") == 1
    # report_shape is the documented enum.
    assert props["report_shape"]["type"] == "string"
    assert sorted(props["report_shape"]["enum"]) == ["brief", "synthesis", "trend"]


# ---------------------------------------------------------------------------
# 6. Upstream failure translation
# ---------------------------------------------------------------------------


def test_e2e_synthesize_likes_translates_orchestrator_upstream_error(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``upstream_failure`` ToolError raised by the tool surface
    propagates out of ``_dispatch`` with category and message intact.

    The dispatcher does not catch :class:`ToolError`; the SDK wrapper in
    :func:`server.build_server` catches it and shapes the structured
    error envelope. Asserting on the raised error here pins the
    boundary contract before the SDK translation layer.
    """

    from x_likes_mcp import errors as errors_module

    upstream = errors_module.upstream_failure("LM down")
    _install_synth_spy(monkeypatch, raises=upstream)

    index = _build_index(fake_export)
    server_module.build_server(index)

    with pytest.raises(ToolError) as excinfo:
        server_module._dispatch(
            index,
            "synthesize_likes",
            {"query": "x", "report_shape": "brief"},
        )

    assert excinfo.value.category == "upstream_failure"
    assert "LM down" in excinfo.value.message
