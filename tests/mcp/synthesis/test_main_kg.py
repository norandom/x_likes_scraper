"""Tests for the ``--kg`` CLI mode.

The ``--kg`` mode runs round-1 search (and round-2 fan-out when
``--hops 2``), builds the in-memory KG via ``orchestrator.build_kg``
with the DSPy fallback disabled, and renders it as either a mermaid
mindmap (default) or a JSON dump (``--json``). No LM call, no URL
fetch.

These tests stub the multihop seams (``run_round_one`` / ``run_round_two``)
imported into ``__main__`` so the suite stays offline. The KG build
itself runs against the real ``orchestrator.build_kg`` because that is
what the mode is supposed to exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from x_likes_mcp import __main__ as cli
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.ranker import ScoredHit


class _FakeUser:
    def __init__(self, screen_name: str) -> None:
        self.screen_name = screen_name
        self.verified = False


class _FakeTweet:
    def __init__(self, tweet_id: str, text: str, screen_name: str) -> None:
        self.id = tweet_id
        self.text = text
        self.user = _FakeUser(screen_name)
        self.urls: list[str] = []


class _FakeIndex:
    """Stub TweetIndex that satisfies what `_run_kg` reads."""

    def __init__(self, config: Config, tweets: list[_FakeTweet]) -> None:
        self.config = config
        self.tweets_by_id = {t.id: t for t in tweets}


def _build_stub_config(tmp_path: Path) -> Config:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        output_dir=output_dir,
        by_month_dir=output_dir / "by_month",
        likes_json=output_dir / "likes.json",
        cache_path=output_dir / "tweet_tree_cache.pkl",
        ranker_weights=RankerWeights(),
        openai_base_url=None,  # --kg must run without an LM endpoint
        openai_api_key="",
        openai_model=None,
        url_cache_dir=output_dir / "url_cache",
    )


def _make_hit(tweet_id: str) -> ScoredHit:
    return ScoredHit(
        tweet_id=tweet_id,
        score=1.0,
        walker_relevance=0.5,
        why="",
        feature_breakdown={},
    )


@pytest.fixture
def stub_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeIndex:
    config = _build_stub_config(tmp_path)
    tweets = [
        _FakeTweet("1", "AI factors and portfolio analysis from @alpha", "alice"),
        _FakeTweet("2", "macro factor models #portfolio https://example.com/x", "bob"),
    ]
    fake = _FakeIndex(config, tweets)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli.TweetIndex, "open_or_build", classmethod(lambda cls, c, w: fake))
    return fake


def test_kg_mode_renders_mermaid_mindmap_to_stdout(
    stub_index: _FakeIndex,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--kg QUERY`` (no ``--out``) prints a mermaid mindmap to stdout."""

    hits = [_make_hit("1"), _make_hit("2")]
    monkeypatch.setattr(cli, "run_round_one", lambda index, options: hits)

    rc = cli.main(["--kg", "AI factors for portfolio"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "```mermaid" in out
    assert "mindmap" in out
    assert "root((AI factors for portfolio))" in out


def test_kg_mode_writes_json_to_out_when_flag_set(
    stub_index: _FakeIndex,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--kg --json --out PATH`` writes a JSON KG dump to ``PATH``."""

    hits = [_make_hit("1"), _make_hit("2")]
    monkeypatch.setattr(cli, "run_round_one", lambda index, options: hits)

    out_path = tmp_path / "kg.json"
    rc = cli.main(["--kg", "AI portfolio", "--json", "--out", str(out_path)])

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "nodes" in payload
    assert "edges" in payload
    node_ids = [n["id"] for n in payload["nodes"]]
    assert "query:root" in node_ids
    assert "tweet:1" in node_ids
    assert "tweet:2" in node_ids


def test_kg_mode_does_not_call_lm_when_regex_returns_empty(
    stub_index: _FakeIndex,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A hit whose text is empty (regex returns nothing) does NOT trigger
    the DSPy fallback. The mode must run without an LM endpoint."""

    # Tweet whose text/urls produce no entities at all (no @, #, https://,
    # or capitalised noun phrase). The regex pass returns []; the orchestrator
    # would normally call extract_entities → DSPy → no-LM error. With
    # dspy_fallback=False the path is skipped and the run succeeds.
    stub_index.tweets_by_id["empty"] = _FakeTweet("empty", "", "")
    hits = [_make_hit("empty")]
    monkeypatch.setattr(cli, "run_round_one", lambda index, options: hits)

    spy = MagicMock(side_effect=AssertionError("LM must not be called in --kg mode"))
    monkeypatch.setattr(
        "x_likes_mcp.synthesis.orchestrator.extract_entities",
        spy,
    )

    rc = cli.main(["--kg", "anything"])
    assert rc == 0
    assert spy.call_count == 0


def test_kg_mode_empty_query_exits_2(
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--kg ""`` (or whitespace) returns exit 2 with stderr."""

    rc = cli.main(["--kg", "   "])
    assert rc == 2
    err = capsys.readouterr().err
    assert "non-empty" in err.lower() or "kg" in err.lower()


def test_kg_mode_no_hits_emits_root_only_mindmap(
    stub_index: _FakeIndex,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Empty hit list still produces a valid mindmap (root, no children)."""

    monkeypatch.setattr(cli, "run_round_one", lambda index, options: [])

    rc = cli.main(["--kg", "no matches anywhere"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "```mermaid" in out
    assert "root((no matches anywhere))" in out
    err = capsys.readouterr().err
    # Soft-warning on stderr is fine, but mode still exits 0.
    _ = err


def test_kg_mode_hops_2_calls_round_two(
    stub_index: _FakeIndex,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--kg --hops 2`` invokes ``run_round_two`` with the round-1 KG."""

    hits_one = [_make_hit("1")]
    hits_two = [_make_hit("99")]  # round-2-only id; merged via existing=kg
    stub_index.tweets_by_id["99"] = _FakeTweet("99", "AI factors @new_handle", "carol")

    captured: dict[str, Any] = {}

    def fake_round_two(index, options, kg, **kwargs):  # type: ignore[no-untyped-def]
        captured["kg"] = kg
        captured["hops"] = options.hops
        return hits_two

    monkeypatch.setattr(cli, "run_round_one", lambda index, options: hits_one)
    monkeypatch.setattr(cli, "run_round_two", fake_round_two)

    rc = cli.main(["--kg", "AI", "--hops", "2"])

    assert rc == 0
    assert captured["hops"] == 2
    out = capsys.readouterr().out
    # Round-2 entity should appear in the rendered mindmap.
    assert "new_handle" in out


def test_kg_mode_hops_3_rejected(
    stub_index: _FakeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """argparse rejects ``--hops 3`` (choices=[1,2]) with SystemExit(2)."""

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--kg", "x", "--hops", "3"])
    assert excinfo.value.code == 2


def test_kg_mode_mutually_exclusive_with_search(
    stub_index: _FakeIndex,
) -> None:
    """``--kg`` and ``--search`` cannot be combined."""

    with pytest.raises(SystemExit):
        cli.main(["--kg", "x", "--search", "y"])
