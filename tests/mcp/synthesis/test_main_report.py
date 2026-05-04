"""Tests for the ``--report`` and ``--report-optimize`` CLI modes (task 5.4).

These tests cover the new boundary helpers ``_run_report`` and
``_run_report_optimize`` in :mod:`x_likes_mcp.__main__`. The tests stub
:func:`x_likes_mcp.synthesis.orchestrator.run_report` and
:func:`x_likes_mcp.synthesis.compiled.run_optimizer` (through the seam
those names are imported into ``__main__``) so the suite stays offline:
no real LM, no crawl4ai container, no DSPy optimizer call.

Boundary discipline (Req 9.1, 9.2, 9.3, 9.5, 6.4):
* ``--report SHAPE --query Q --out PATH`` writes the rendered markdown.
* ``--out`` omitted → markdown is printed to stdout.
* Any orchestrator failure (``OrchestratorError``) translates to exit
  code ``2`` with a stderr message naming the category.
* ``--report-optimize SHAPE`` invokes the optimizer with sanitized +
  fenced demos through :func:`compiled.run_optimizer`.

The tests pin the ``TweetIndex.open_or_build`` seam to a stub so the
real index build (which would need a populated export tree) never runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from x_likes_mcp import __main__ as cli
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.synthesis.orchestrator import OrchestratorError
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import ReportOptions, ReportResult

# ---------------------------------------------------------------------------
# Stub TweetIndex + open_or_build seam
# ---------------------------------------------------------------------------


class _FakeIndex:
    """Stub :class:`TweetIndex` that exposes only ``.config``.

    The orchestrator and the optimizer are monkeypatched in every test,
    so neither inspects the index further. Pinning ``config`` lets the
    ``run_report(..., config=index.config)`` assertion survive.
    """

    def __init__(self, config: Config) -> None:
        self.config = config


def _build_stub_config(tmp_path: Path) -> Config:
    """Build a minimal :class:`Config` pointed at ``tmp_path/output``.

    Bypasses :func:`load_config` so the loader's documented
    ``os.environ`` side effects (writing ``OPENAI_BASE_URL`` /
    ``OPENAI_API_KEY``) do not leak between tests in the suite.
    """

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        output_dir=output_dir,
        by_month_dir=output_dir / "by_month",
        likes_json=output_dir / "likes.json",
        cache_path=output_dir / "tweet_tree_cache.pkl",
        ranker_weights=RankerWeights(),
        openai_base_url="http://fake/v1",
        openai_api_key="",
        openai_model="fake-model",
        url_cache_dir=output_dir / "url_cache",
    )


@pytest.fixture
def stub_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeIndex:
    """Replace ``load_config`` and ``TweetIndex.open_or_build`` with stubs.

    The CLI ``main`` calls ``load_config()`` first (which reads the
    repo's ``.env`` and writes ``OPENAI_BASE_URL`` into ``os.environ``)
    and then ``TweetIndex.open_or_build``. Both are patched so the test
    does not depend on the on-disk export tree and does not pollute
    ``os.environ`` for sibling tests.
    """

    config = _build_stub_config(tmp_path)
    fake = _FakeIndex(config)

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli.TweetIndex, "open_or_build", classmethod(lambda cls, c, w: fake))
    return fake


# ---------------------------------------------------------------------------
# --report happy paths
# ---------------------------------------------------------------------------


def test_report_brief_writes_to_out_and_exits_0(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    tmp_path: Path,
) -> None:
    """``--report brief --query Q --out PATH`` writes markdown and exits 0."""

    captured: dict[str, Any] = {}

    def _stub(index: object, options: ReportOptions, *, config: object) -> ReportResult:
        captured["index"] = index
        captured["options"] = options
        captured["config"] = config
        return ReportResult(
            markdown="# hi\n",
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    out_path = tmp_path / "report.md"
    rc = cli.main(["--report", "brief", "--query", "hello", "--out", str(out_path)])

    assert rc == 0
    assert out_path.read_text(encoding="utf-8") == "# hi\n"
    assert captured["options"].query == "hello"
    assert captured["options"].shape is ReportShape.BRIEF
    assert captured["options"].fetch_urls is False
    assert captured["options"].hops == 1
    # ``config`` is forwarded from ``index.config``.
    assert captured["config"] is stub_index.config


def test_report_brief_to_stdout_when_no_out(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--out``, the rendered markdown is printed to stdout."""

    def _stub(index: object, options: ReportOptions, *, config: object) -> ReportResult:
        return ReportResult(
            markdown="# stdout body\n",
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(["--report", "synthesis", "--query", "topic"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# stdout body" in out


def test_report_forwards_filters_and_hops_and_fetch_urls(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--limit/--year/--month-*/--hops/--fetch-urls`` reach ``ReportOptions``."""

    captured: dict[str, ReportOptions] = {}

    def _stub(index: object, options: ReportOptions, *, config: object) -> ReportResult:
        captured["options"] = options
        return ReportResult(
            markdown="# t\n",
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(
        [
            "--report",
            "trend",
            "--query",
            "ai",
            "--fetch-urls",
            "--hops",
            "2",
            "--limit",
            "20",
            "--year",
            "2025",
            "--month-start",
            "01",
            "--month-end",
            "03",
        ]
    )

    capsys.readouterr()  # drain stdout
    assert rc == 0
    opts = captured["options"]
    assert opts.shape is ReportShape.TREND
    assert opts.fetch_urls is True
    assert opts.hops == 2
    assert opts.limit == 20
    assert opts.year == 2025
    assert opts.month_start == "01"
    assert opts.month_end == "03"


# ---------------------------------------------------------------------------
# --report failure modes (Req 9.5)
# ---------------------------------------------------------------------------


def test_report_invalid_shape_exits_via_argparse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """argparse rejects an unknown shape choice with ``SystemExit(2)``."""

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--report", "bogus", "--query", "x"])

    # argparse exits with code 2 for invalid choice.
    assert excinfo.value.code == 2


def test_report_missing_query_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--report`` without ``--query`` returns exit code 2 with a stderr message."""

    def _fail_if_called(*_args: object, **_kwargs: object) -> ReportResult:
        raise AssertionError("orchestrator.run_report must not run when --query is missing")

    monkeypatch.setattr(cli.orchestrator, "run_report", _fail_if_called)

    rc = cli.main(["--report", "brief"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "query" in err.lower()


def test_report_translates_orchestrator_upstream_to_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``OrchestratorError("upstream", ...)`` → exit 2 with category in stderr."""

    def _stub(*_args: object, **_kwargs: object) -> ReportResult:
        raise OrchestratorError("upstream", "LM endpoint unreachable: connection refused")

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(["--report", "brief", "--query", "x"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "upstream" in err
    assert "LM endpoint unreachable" in err


def test_report_translates_validation_error_to_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``OrchestratorError("validation", ...)`` → exit 2."""

    def _stub(*_args: object, **_kwargs: object) -> ReportResult:
        raise OrchestratorError("validation", "synthesizer cited unknown source ids")

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(["--report", "synthesis", "--query", "topic"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "validation" in err
    assert "unknown source" in err


def test_report_translates_invalid_input_error_to_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``OrchestratorError("invalid_input", "hops out of range")`` → exit 2."""

    def _stub(*_args: object, **_kwargs: object) -> ReportResult:
        raise OrchestratorError("invalid_input", "hops out of range: 3")

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(["--report", "brief", "--query", "x", "--hops", "2"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "hops" in err.lower()


def test_report_fetch_urls_translates_container_unreachable_to_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--fetch-urls`` + crawl4ai down → exit 2 with endpoint guidance."""

    msg = (
        "crawl4ai container unreachable at http://127.0.0.1:11235; "
        "set CRAWL4AI_BASE_URL to override"
    )

    def _stub(*_args: object, **_kwargs: object) -> ReportResult:
        raise OrchestratorError("upstream", msg)

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(["--report", "brief", "--query", "x", "--fetch-urls"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "crawl4ai" in err
    assert "CRAWL4AI_BASE_URL" in err


# ---------------------------------------------------------------------------
# --report-optimize
# ---------------------------------------------------------------------------


def _write_labeled_file(config: Config, shape: str, examples: list[dict[str, Any]]) -> Path:
    """Materialise ``output/synthesis_labeled/<shape>.json`` for the optimizer test."""

    labeled_dir = config.output_dir / "synthesis_labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    path = labeled_dir / f"{shape}.json"
    path.write_text(json.dumps(examples), encoding="utf-8")
    return path


def test_report_optimize_brief_invokes_run_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--report-optimize brief`` loads the labeled file and runs the optimizer."""

    examples = [
        {
            "query": "ai security",
            "fenced_context_raw": "tweet:1 body about prompt injection",
            "expected_outputs": {"claims": ["claim 1"], "top_entities": ["@alice"]},
        },
        {
            "query": "ml ops",
            "fenced_context_raw": "tweet:2 body about deployments",
            "expected_outputs": {"claims": ["claim 2"], "top_entities": ["#mlops"]},
        },
    ]
    _write_labeled_file(stub_index.config, "brief", examples)

    captured: dict[str, Any] = {}

    def _stub(
        shape: ReportShape,
        labeled_examples: list[Any],
        *,
        root: Path,
        optimizer: str = "BootstrapFewShot",
    ) -> object:
        captured["shape"] = shape
        captured["labeled_examples"] = labeled_examples
        captured["root"] = root
        captured["optimizer"] = optimizer
        return object()

    monkeypatch.setattr(cli.compiled, "run_optimizer", _stub)

    rc = cli.main(["--report-optimize", "brief"])

    assert rc == 0
    assert captured["shape"] is ReportShape.BRIEF
    assert len(captured["labeled_examples"]) == 2
    # Each example must be a LabeledExample-shaped object exposing the
    # documented attributes — the optimizer entry point will sanitize +
    # fence them later via ``prepare_demo``.
    first = captured["labeled_examples"][0]
    assert first.query == "ai security"
    assert "prompt injection" in first.fenced_context_raw
    assert first.expected_outputs == {
        "claims": ["claim 1"],
        "top_entities": ["@alice"],
    }
    assert captured["root"] == stub_index.config.output_dir / "synthesis_compiled"


def test_report_optimize_missing_labeled_file_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No labeled-examples file → exit 2 with a clear error naming the path."""

    def _stub(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("run_optimizer must not be invoked when the labeled file is missing")

    monkeypatch.setattr(cli.compiled, "run_optimizer", _stub)

    rc = cli.main(["--report-optimize", "synthesis"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "synthesis_labeled" in err
    assert "synthesis.json" in err


def test_report_optimize_propagates_errors_to_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Any ``run_optimizer`` exception translates to exit 2."""

    examples = [
        {
            "query": "x",
            "fenced_context_raw": "body",
            "expected_outputs": {"claims": []},
        }
    ]
    _write_labeled_file(stub_index.config, "brief", examples)

    def _stub(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("optimizer blew up")

    monkeypatch.setattr(cli.compiled, "run_optimizer", _stub)

    rc = cli.main(["--report-optimize", "brief"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "optimizer" in err.lower()


# ---------------------------------------------------------------------------
# Existing modes still work (regression guard)
# ---------------------------------------------------------------------------


def test_report_and_search_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """argparse mutual-exclusion blocks ``--report`` + ``--search`` together."""

    with pytest.raises(SystemExit):
        cli.main(["--report", "brief", "--search", "x", "--query", "y"])
