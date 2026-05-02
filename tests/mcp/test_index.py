"""Tests for :mod:`x_likes_mcp.index`.

Covers :class:`TweetIndex.open_or_build`, the cache freshness logic,
``_resolve_filter`` rule matrix, ``search`` orchestration (resolve →
walk → rank), and the per-tweet / per-month read paths
(``lookup_tweet``, ``list_months``, ``get_month_markdown``).

The walker is the only LLM call site in the package; every test below
that exercises ``search`` replaces ``x_likes_mcp.walker.walk`` with a
stub via ``monkeypatch.setattr`` so no real HTTP is made. The autouse
``_block_real_llm`` fixture in ``tests/mcp/conftest.py`` is an
additional safety net at the chat-completions seam.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from x_likes_mcp import index as index_module
from x_likes_mcp import ranker as ranker_module
from x_likes_mcp import tree as tree_module
from x_likes_mcp.bm25 import BM25Index
from x_likes_mcp.config import Config, RankerWeights
from x_likes_mcp.embeddings import CorpusEmbeddings, Embedder
from x_likes_mcp.index import IndexError as MCPIndexError
from x_likes_mcp.index import MonthInfo, TweetIndex
from x_likes_mcp.ranker import ScoredHit
from x_likes_mcp.walker import WalkerHit


# ---------------------------------------------------------------------------
# Embedder seam helpers (task 3.1)
#
# The dense-retrieval path in ``TweetIndex.open_or_build`` constructs a real
# :class:`Embedder` and calls ``open_or_build_corpus``, which on a cache miss
# invokes ``Embedder._call_embeddings_api``. Tests below patch that seam so
# no real OpenRouter HTTP call escapes; the patch returns a deterministic
# canned vector per input text.


_FAKE_EMBED_DIM = 4


def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch ``Embedder._call_embeddings_api`` with a deterministic stub.

    Returns a dict carrying a ``"calls"`` counter so tests can assert that a
    second ``open_or_build`` does not re-embed the corpus.
    """

    counter: dict[str, int] = {"calls": 0}

    def _fake(self: Embedder, texts: list[str]) -> list[list[float]]:
        counter["calls"] += 1
        # One canned 4-d vector per text. The exact content does not matter
        # for shape / cache assertions; we just need a stable, non-zero row.
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "x_likes_mcp.embeddings.Embedder._call_embeddings_api",
        _fake,
    )
    return counter


# ---------------------------------------------------------------------------
# Helpers


def _default_weights() -> RankerWeights:
    """Return a default :class:`RankerWeights` for tests."""

    return RankerWeights()


# ---------------------------------------------------------------------------
# open_or_build: happy path


def test_open_or_build_populates_tweets_and_paths(fake_export: Config) -> None:
    """``open_or_build`` against the fixture export populates the four tweets
    and the three month paths.
    """

    weights = _default_weights()
    idx = TweetIndex.open_or_build(fake_export, weights)

    # Four fixture tweets, keyed by id.
    assert set(idx.tweets_by_id.keys()) == {"1001", "1002", "2001", "3001"}
    # Three months mapped to per-month .md paths.
    assert set(idx.paths_by_month.keys()) == {"2025-01", "2025-02", "2025-03"}
    # Path values are real on-disk files.
    for ym, path in idx.paths_by_month.items():
        assert path.is_file()
        assert path.name == f"likes_{ym}.md"
    # Config and weights round-trip through the dataclass.
    assert idx.config is fake_export
    assert idx.weights is weights


def test_open_or_build_computes_author_affinity(fake_export: Config) -> None:
    """Author affinity is precomputed at build time as ``log1p(count)``.

    The fixture export has alice (2 tweets), bob (1), carol (1); the
    map values are ``log1p`` of those counts.
    """

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    assert idx.author_affinity["alice"] == pytest.approx(math.log1p(2))
    assert idx.author_affinity["bob"] == pytest.approx(math.log1p(1))
    assert idx.author_affinity["carol"] == pytest.approx(math.log1p(1))


# ---------------------------------------------------------------------------
# open_or_build: cache freshness


def test_open_or_build_writes_cache_on_first_call(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First open_or_build call invokes ``tree.build_tree`` and writes the
    cache file; the second call (no fixture changes) hits the cache and
    does not call the builder again.
    """

    real_build = tree_module.build_tree
    call_counter = {"n": 0}

    def counting_build(by_month_dir: Path):
        call_counter["n"] += 1
        return real_build(by_month_dir)

    monkeypatch.setattr(
        "x_likes_mcp.index.tree_module.build_tree", counting_build
    )

    # Cache absent before the first call.
    assert not fake_export.cache_path.exists()

    TweetIndex.open_or_build(fake_export, _default_weights())

    assert call_counter["n"] == 1
    assert fake_export.cache_path.exists()

    # Second call, no fixture changes → cache hit, no rebuild.
    TweetIndex.open_or_build(fake_export, _default_weights())

    assert call_counter["n"] == 1, (
        "second open_or_build with a fresh cache should not rebuild "
        "the tree"
    )


def test_open_or_build_rebuilds_when_md_newer_than_cache(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Touching a .md file so its mtime is newer than the cache forces a
    rebuild on the next open_or_build call.
    """

    real_build = tree_module.build_tree
    call_counter = {"n": 0}

    def counting_build(by_month_dir: Path):
        call_counter["n"] += 1
        return real_build(by_month_dir)

    monkeypatch.setattr(
        "x_likes_mcp.index.tree_module.build_tree", counting_build
    )

    # First build populates the cache.
    TweetIndex.open_or_build(fake_export, _default_weights())
    assert call_counter["n"] == 1
    assert fake_export.cache_path.exists()

    # Touch one of the .md files so it is strictly newer than the cache.
    cache_mtime = fake_export.cache_path.stat().st_mtime
    md_path = fake_export.by_month_dir / "likes_2025-01.md"
    new_mtime = cache_mtime + 10.0
    os.utime(md_path, (new_mtime, new_mtime))
    assert md_path.stat().st_mtime > cache_mtime

    # Second open_or_build sees a stale cache and rebuilds.
    TweetIndex.open_or_build(fake_export, _default_weights())
    assert call_counter["n"] == 2, (
        "open_or_build should rebuild when an .md file is newer than "
        "the cache"
    )


# ---------------------------------------------------------------------------
# open_or_build: error cases


def test_open_or_build_empty_by_month_raises_index_error(tmp_path: Path) -> None:
    """An empty ``by_month/`` directory raises :class:`IndexError`."""

    output_dir = tmp_path / "output"
    by_month_dir = output_dir / "by_month"
    by_month_dir.mkdir(parents=True)
    # Empty likes.json so the load_export call would not raise if reached.
    (output_dir / "likes.json").write_text("[]", encoding="utf-8")

    config = Config(
        output_dir=output_dir,
        by_month_dir=by_month_dir,
        likes_json=output_dir / "likes.json",
        cache_path=output_dir / "tweet_tree_cache.pkl",
        openai_base_url="http://fake/v1",
        openai_api_key="",
        openai_model="fake-model",
        ranker_weights=RankerWeights(),
    )

    with pytest.raises(MCPIndexError):
        TweetIndex.open_or_build(config, _default_weights())


def test_open_or_build_missing_by_month_raises_index_error(tmp_path: Path) -> None:
    """A missing ``by_month/`` directory raises :class:`IndexError`."""

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # by_month/ deliberately not created.
    (output_dir / "likes.json").write_text("[]", encoding="utf-8")

    config = Config(
        output_dir=output_dir,
        by_month_dir=output_dir / "by_month",
        likes_json=output_dir / "likes.json",
        cache_path=output_dir / "tweet_tree_cache.pkl",
        openai_base_url="http://fake/v1",
        openai_api_key="",
        openai_model="fake-model",
        ranker_weights=RankerWeights(),
    )

    with pytest.raises(MCPIndexError):
        TweetIndex.open_or_build(config, _default_weights())


# ---------------------------------------------------------------------------
# _resolve_filter rule matrix


def _resolve(idx: TweetIndex, year, month_start, month_end):
    """Shorthand for invoking the private resolver under test."""
    return idx._resolve_filter(year, month_start, month_end)


def test_resolve_filter_all_none_returns_none(fake_export: Config) -> None:
    """All three filter fields ``None`` → ``None`` (every month)."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    assert _resolve(idx, None, None, None) is None


def test_resolve_filter_year_only_returns_full_year(fake_export: Config) -> None:
    """``year`` only → the 12 months of that year."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    expected = [f"2025-{m:02d}" for m in range(1, 13)]
    assert _resolve(idx, 2025, None, None) == expected


def test_resolve_filter_year_and_month_start_returns_single_month(
    fake_export: Config,
) -> None:
    """``year`` + ``month_start`` only → the one named month."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    assert _resolve(idx, 2025, "03", None) == ["2025-03"]


def test_resolve_filter_year_and_range_returns_inclusive_range(
    fake_export: Config,
) -> None:
    """``year`` + ``month_start`` + ``month_end`` → inclusive range."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    assert _resolve(idx, 2025, "01", "03") == ["2025-01", "2025-02", "2025-03"]


def test_resolve_filter_month_start_without_year_raises(
    fake_export: Config,
) -> None:
    """``month_start`` set without ``year`` → ``ValueError``."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    with pytest.raises(ValueError) as excinfo:
        _resolve(idx, None, "01", None)
    assert "month_start requires year" in str(excinfo.value)


def test_resolve_filter_month_end_without_month_start_raises(
    fake_export: Config,
) -> None:
    """``month_end`` set without ``month_start`` → ``ValueError``."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    with pytest.raises(ValueError) as excinfo:
        _resolve(idx, 2025, None, "03")
    assert "month_end requires month_start" in str(excinfo.value)


def test_resolve_filter_inverted_range_raises(fake_export: Config) -> None:
    """``month_start > month_end`` → ``ValueError``."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    with pytest.raises(ValueError):
        _resolve(idx, 2025, "05", "03")


def test_resolve_filter_bad_month_format_raises(fake_export: Config) -> None:
    """Non-numeric / out-of-range month strings → ``ValueError``."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    # Non-numeric.
    with pytest.raises(ValueError):
        _resolve(idx, 2025, "ab", None)
    # Out of range high.
    with pytest.raises(ValueError):
        _resolve(idx, 2025, "13", None)
    # Out of range low.
    with pytest.raises(ValueError):
        _resolve(idx, 2025, "00", None)


# ---------------------------------------------------------------------------
# lookup_tweet / list_months / get_month_markdown


def test_lookup_tweet_returns_tweet_for_known_id(fake_export: Config) -> None:
    """``lookup_tweet`` returns the matching :class:`Tweet` for a fixture id."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    tweet = idx.lookup_tweet("1001")
    assert tweet is not None
    assert tweet.id == "1001"
    assert tweet.user.screen_name == "alice"


def test_lookup_tweet_returns_none_for_missing_id(fake_export: Config) -> None:
    """``lookup_tweet`` returns ``None`` for an unknown id."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    assert idx.lookup_tweet("missing") is None


def test_list_months_returns_reverse_chrono_with_counts(
    fake_export: Config,
) -> None:
    """``list_months`` lists the three fixture months newest-first with the
    correct per-month tweet counts.
    """

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    months = idx.list_months()

    assert len(months) == 3
    assert all(isinstance(m, MonthInfo) for m in months)
    # Reverse-chronological order.
    assert [m.year_month for m in months] == ["2025-03", "2025-02", "2025-01"]
    # Counts: jan=2 (1001+1002), feb=1 (2001), mar=1 (3001).
    counts = {m.year_month: m.tweet_count for m in months}
    assert counts == {"2025-01": 2, "2025-02": 1, "2025-03": 1}


def test_get_month_markdown_returns_file_content(fake_export: Config) -> None:
    """``get_month_markdown`` returns the file content for a known month."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    content = idx.get_month_markdown("2025-01")
    assert content is not None
    # Content carries the canonical month heading and at least one tweet
    # heading from the fixture.
    assert "## 2025-01" in content
    assert "@alice" in content


def test_get_month_markdown_returns_none_for_missing_month(
    fake_export: Config,
) -> None:
    """``get_month_markdown`` returns ``None`` for a month not on disk."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())
    assert idx.get_month_markdown("2099-12") is None


# ---------------------------------------------------------------------------
# search orchestration


def test_search_no_filter_passes_none_scope_to_walker(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``search`` without filter calls walker with ``months_in_scope=None``."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    captured: dict[str, Any] = {}

    def fake_walk(tree, query, months_in_scope, config, chunk_size=30):
        captured["tree"] = tree
        captured["query"] = query
        captured["months_in_scope"] = months_in_scope
        captured["config"] = config
        return []

    monkeypatch.setattr("x_likes_mcp.walker.walk", fake_walk)

    idx.search("anything")

    assert captured["months_in_scope"] is None
    assert captured["query"] == "anything"
    assert captured["config"] is fake_export


def test_search_filter_passes_resolved_months_to_walker(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``search`` with year+range filter calls walker with the resolved list."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    captured: dict[str, Any] = {}

    def fake_walk(tree, query, months_in_scope, config, chunk_size=30):
        captured["months_in_scope"] = months_in_scope
        return []

    monkeypatch.setattr("x_likes_mcp.walker.walk", fake_walk)

    idx.search("anything", year=2025, month_start="01", month_end="02")

    assert captured["months_in_scope"] == ["2025-01", "2025-02"]


def test_search_year_only_resolves_full_year(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``search`` with year-only filter sends the 12-month list to the walker."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    captured: dict[str, Any] = {}

    def fake_walk(tree, query, months_in_scope, config, chunk_size=30):
        captured["months_in_scope"] = months_in_scope
        return []

    monkeypatch.setattr("x_likes_mcp.walker.walk", fake_walk)

    idx.search("anything", year=2025)

    assert captured["months_in_scope"] == [f"2025-{m:02d}" for m in range(1, 13)]


def test_search_returns_results_sorted_descending_by_score(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``search`` returns the ranker output, sorted descending by score."""

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    # Three fixture ids; the walker reports them in arbitrary order with
    # different relevance scores so the ranker can produce a meaningful
    # sort.
    walker_hits = [
        WalkerHit(tweet_id="1001", relevance=0.4, why="low"),
        WalkerHit(tweet_id="2001", relevance=0.9, why="high"),
        WalkerHit(tweet_id="1002", relevance=0.6, why="mid"),
    ]

    def fake_walk(tree, query, months_in_scope, config, chunk_size=30):
        return list(walker_hits)

    monkeypatch.setattr("x_likes_mcp.walker.walk", fake_walk)

    results = idx.search("anything")

    assert len(results) == 3
    assert all(isinstance(r, ScoredHit) for r in results)
    # Descending by score.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_passes_recency_anchor_at_end_of_month_end(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``search(year=2025, month_start='01', month_end='02')`` passes a recency
    anchor at the end of February 2025 to the ranker.
    """

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    monkeypatch.setattr(
        "x_likes_mcp.walker.walk",
        lambda *args, **kwargs: [],
    )

    captured: dict[str, Any] = {}

    real_rank = ranker_module.rank

    def fake_rank(walker_hits, tweets_by_id, author_affinity, weights, anchor=None):
        captured["anchor"] = anchor
        return real_rank(
            walker_hits, tweets_by_id, author_affinity, weights, anchor=anchor
        )

    monkeypatch.setattr("x_likes_mcp.ranker.rank", fake_rank)

    idx.search("anything", year=2025, month_start="01", month_end="02")

    anchor = captured["anchor"]
    assert isinstance(anchor, datetime)
    assert anchor.year == 2025
    assert anchor.month == 2
    # End-of-month: the implementation lands at 2025-02-28 23:59:59.999999
    # via "first of next month minus a microsecond". We assert the day is
    # the last day of February rather than pinning the exact microsecond,
    # which would couple this test to the internal arithmetic.
    assert anchor.day == 28
    assert anchor.hour == 23
    assert anchor.minute == 59
    assert anchor.tzinfo is not None
    assert anchor.utcoffset() == timezone.utc.utcoffset(anchor)


# ---------------------------------------------------------------------------
# open_or_build: dense + lexical wiring (task 3.1)


def test_open_or_build_constructs_embedder(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``open_or_build`` constructs an :class:`Embedder` from the config's
    OpenRouter fields and exposes it on the resulting :class:`TweetIndex`.
    """

    _patch_embedder(monkeypatch)

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    assert isinstance(idx.embedder, Embedder)
    assert idx.embedder.model_name == fake_export.embedding_model
    assert idx.embedder.base_url == fake_export.openrouter_base_url


def test_open_or_build_builds_corpus_with_correct_shape(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus matrix has shape ``(N, D)`` where ``N == len(tweets_by_id)``
    and its ``ordered_ids`` covers every tweet id exactly once.
    """

    _patch_embedder(monkeypatch)

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    assert isinstance(idx.corpus, CorpusEmbeddings)
    n = len(idx.tweets_by_id)
    assert idx.corpus.matrix.shape == (n, _FAKE_EMBED_DIM)
    assert set(idx.corpus.ordered_ids) == set(idx.tweets_by_id.keys())
    # Cache layout requires sorted ids (open_or_build_corpus contract).
    assert idx.corpus.ordered_ids == sorted(idx.tweets_by_id.keys())


def test_open_or_build_builds_bm25_index(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The BM25 index is built over the same tweet ids and answers a query
    without raising.
    """

    _patch_embedder(monkeypatch)

    idx = TweetIndex.open_or_build(fake_export, _default_weights())

    assert isinstance(idx.bm25, BM25Index)
    assert idx.bm25.ordered_ids == sorted(idx.tweets_by_id.keys())
    # Should answer a query without raising; the result list may be empty if
    # the query has no overlap with the fixture text, but the call must not
    # raise.
    results = idx.bm25.top_k("any reasonable query", k=5)
    assert isinstance(results, list)


def test_open_or_build_reuses_embedding_cache_on_second_call(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the index a second time hits the on-disk embedding cache and
    does NOT re-invoke the embeddings API. (BM25 is rebuilt in-memory each
    time; that is by design.)
    """

    counter = _patch_embedder(monkeypatch)

    TweetIndex.open_or_build(fake_export, _default_weights())
    first_call_count = counter["calls"]
    assert first_call_count >= 1, "first build should have called the embedder"

    # Second build with the same config (same output_dir) reuses the cache.
    TweetIndex.open_or_build(fake_export, _default_weights())

    assert counter["calls"] == first_call_count, (
        "second open_or_build with a valid embedding cache should not "
        "re-invoke the embeddings API"
    )


def test_open_or_build_writes_cache_files(
    fake_export: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a fresh build the corpus matrix and metadata sidecar exist on
    disk under ``config.output_dir``.
    """

    _patch_embedder(monkeypatch)

    TweetIndex.open_or_build(fake_export, _default_weights())

    assert (fake_export.output_dir / "corpus_embeddings.npy").is_file()
    assert (fake_export.output_dir / "corpus_embeddings.meta.json").is_file()
