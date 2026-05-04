"""End-to-end CLI tests for the ``--report`` mode (task 6.2).

These tests drive :func:`x_likes_mcp.__main__.main` end-to-end at the
CLI boundary: argparse → ``_run_report`` → file write / stdout / exit
code. The orchestrator is stubbed so the test stays offline (no real
LM, no crawl4ai container, no on-disk export tree); the focus is the
boundary contract, not the inner pipeline.

Coverage matrix (Req 9.1, 9.5, 3.3):

* ``--report synthesis --query Q --out PATH`` against a fixture index
  writes ``PATH``, the file contains a ``mermaid mindmap`` fenced
  block, and ``main`` returns ``0``.
* ``--report brief --query Q --out PATH`` writes ``PATH`` and returns
  ``0`` for the smaller markdown body.
* ``--report brief --query Q --fetch-urls`` while the crawl4ai
  container is unreachable returns exit code ``2`` and the stderr
  message names both the endpoint (``127.0.0.1:11235``) and the
  override env var (``CRAWL4AI_BASE_URL``).
* ``--report bogus --query x`` is rejected by argparse with
  ``SystemExit(2)``.
* ``--report brief`` with no ``--query`` returns exit code ``2``.
* ``--report brief --query x`` without ``--out`` writes the rendered
  markdown to stdout.

The :func:`load_config` and :meth:`TweetIndex.open_or_build` seams are
monkeypatched (mirroring ``test_main_report.py``) so the suite never
needs the on-disk export tree and never pollutes ``os.environ`` for
sibling tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from x_likes_mcp import __main__ as cli
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.synthesis.orchestrator import OrchestratorError
from x_likes_mcp.synthesis.shapes import ReportShape
from x_likes_mcp.synthesis.types import ReportOptions, ReportResult

# ---------------------------------------------------------------------------
# Stub TweetIndex + open_or_build seam (mirrors test_main_report.py)
# ---------------------------------------------------------------------------


class _FakeIndex:
    """Stub :class:`TweetIndex` exposing only ``.config``.

    The orchestrator is monkeypatched in every test, so nothing else is
    inspected on the index. Pinning ``config`` keeps the
    ``run_report(..., config=index.config)`` assertion intact.
    """

    def __init__(self, config: Config) -> None:
        self.config = config


def _build_stub_config(tmp_path: Path) -> Config:
    """Build a minimal :class:`Config` pointed at ``tmp_path/output``.

    Bypasses :func:`load_config` so the loader's documented
    ``os.environ`` side effects (writing ``OPENAI_BASE_URL`` /
    ``OPENAI_API_KEY``) do not leak into sibling tests.
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
def stub_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[_FakeIndex]:
    """Replace ``load_config`` and ``TweetIndex.open_or_build`` with stubs."""

    config = _build_stub_config(tmp_path)
    fake = _FakeIndex(config)

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli.TweetIndex, "open_or_build", classmethod(lambda cls, c, w: fake))
    yield fake


# ---------------------------------------------------------------------------
# Pre-baked markdown bodies for the orchestrator stub
# ---------------------------------------------------------------------------


_SYNTHESIS_MARKDOWN = """# Synthesis Report

## Findings

- Some claim about the topic [tweet:1].

```mermaid
mindmap
  root((query))
    Topic
      tweet:1
    Entities
      "@alice"
```

## Sources

- tweet:1 — body excerpt
"""


_BRIEF_MARKDOWN = """# Brief

- Single bullet [tweet:1].

## Sources

- tweet:1 — body excerpt
"""


# ---------------------------------------------------------------------------
# Happy paths: --report writes file and exits 0
# ---------------------------------------------------------------------------


def test_e2e_report_synthesis_writes_file_with_mindmap(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    tmp_path: Path,
) -> None:
    """``--report synthesis --query Q --out PATH`` writes a file containing
    a ``mermaid mindmap`` fenced block and exits ``0`` (Req 9.1).
    """

    captured: dict[str, Any] = {}

    def _stub(index: object, options: ReportOptions, *, config: object) -> ReportResult:
        captured["index"] = index
        captured["options"] = options
        captured["config"] = config
        return ReportResult(
            markdown=_SYNTHESIS_MARKDOWN,
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    out_path = tmp_path / "report.md"
    rc = cli.main(
        [
            "--report",
            "synthesis",
            "--query",
            "ai security",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    # The mindmap block must be present and use the mermaid fence.
    assert "```mermaid" in body
    assert "mindmap" in body
    # Sanity: the orchestrator received the right shape + query.
    assert captured["options"].shape is ReportShape.SYNTHESIS
    assert captured["options"].query == "ai security"
    assert captured["config"] is stub_index.config


def test_e2e_report_brief_writes_file_and_exits_0(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    tmp_path: Path,
) -> None:
    """``--report brief --query Q --out PATH`` writes the smaller body
    and exits ``0`` (Req 9.1).
    """

    def _stub(index: object, options: ReportOptions, *, config: object) -> ReportResult:
        return ReportResult(
            markdown=_BRIEF_MARKDOWN,
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    out_path = tmp_path / "brief.md"
    rc = cli.main(
        [
            "--report",
            "brief",
            "--query",
            "ml ops",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "# Brief" in body
    assert "tweet:1" in body


# ---------------------------------------------------------------------------
# Failure modes (Req 9.5, 3.3)
# ---------------------------------------------------------------------------


def test_e2e_report_fetch_urls_with_unreachable_container_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--fetch-urls`` while crawl4ai is unreachable → exit ``2`` with
    a stderr message naming the endpoint and the override env var
    (Req 3.3, 9.5).

    The orchestrator is the layer that translates the deeper
    :class:`ContainerUnreachable` into ``OrchestratorError("upstream",
    msg)``; the CLI boundary then surfaces ``msg`` on stderr verbatim.
    Stubbing at the orchestrator seam keeps the test focused on the
    CLI's error-translation contract while still exercising the same
    final user-visible message.
    """

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
    # The stderr line must surface the endpoint and the override env var
    # so the operator can act on the failure without source-diving.
    assert "127.0.0.1:11235" in err
    assert "CRAWL4AI_BASE_URL" in err
    # The CLI brackets the category for grep-friendly logs.
    assert "upstream" in err


def test_e2e_report_invalid_shape_exits_2_argparse() -> None:
    """``--report bogus`` is rejected by argparse with ``SystemExit(2)``."""

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--report", "bogus", "--query", "x"])

    assert excinfo.value.code == 2


def test_e2e_report_no_query_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--report brief`` with no ``--query`` returns exit ``2`` and
    explains the missing flag on stderr.
    """

    def _fail_if_called(*_args: object, **_kwargs: object) -> ReportResult:
        raise AssertionError("orchestrator.run_report must not run when --query is missing")

    monkeypatch.setattr(cli.orchestrator, "run_report", _fail_if_called)

    rc = cli.main(["--report", "brief"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "query" in err.lower()


# ---------------------------------------------------------------------------
# Stdout fallback when --out is omitted
# ---------------------------------------------------------------------------


def test_e2e_report_writes_to_stdout_when_out_omitted(
    monkeypatch: pytest.MonkeyPatch,
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--out``, the rendered markdown lands on stdout."""

    def _stub(index: object, options: ReportOptions, *, config: object) -> ReportResult:
        return ReportResult(
            markdown="# stdout body\n",
            shape=options.shape,
            used_hops=options.hops,
            fetched_url_count=0,
            synthesis_token_count=0,
        )

    monkeypatch.setattr(cli.orchestrator, "run_report", _stub)

    rc = cli.main(["--report", "brief", "--query", "x"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# stdout body" in out
