"""Tests for :mod:`x_likes_mcp.config`.

Covers :func:`load_config` plus the embedded :class:`RankerWeights` parsing.
The loader writes ``OPENAI_BASE_URL`` (and ``OPENAI_API_KEY`` when set) into
``os.environ`` as a documented side effect; tests that exercise that side
effect use ``monkeypatch`` so the change is rolled back at teardown and does
not leak between tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from x_likes_mcp.config import (
    Config,
    ConfigError,
    RankerWeights,
    load_config,
)


def test_load_config_minimal_env_returns_populated_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both required vars set → fully populated ``Config`` with defaults.

    The minimal environment exercises the happy path: the loader picks up
    ``OPENAI_BASE_URL`` and ``OPENAI_MODEL`` from the in-memory ``env``
    dict, defaults the output directory to ``"output"``, derives the three
    output-relative paths, and embeds a default ``RankerWeights``.
    """

    # Snapshot env so any side-effect writes are rolled back at teardown.
    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"OPENAI_BASE_URL": "x", "OPENAI_MODEL": "m"})

    assert isinstance(config, Config)
    assert config.openai_base_url == "x"
    assert config.openai_model == "m"
    assert config.openai_api_key == ""
    assert config.output_dir == Path("output")
    assert config.by_month_dir == Path("output") / "by_month"
    assert config.likes_json == Path("output") / "likes.json"
    assert config.cache_path == Path("output") / "tweet_tree_cache.pkl"
    # Default RankerWeights embedded; spot-check the documented defaults.
    assert config.ranker_weights == RankerWeights()
    assert config.ranker_weights.relevance == 10.0
    assert config.ranker_weights.affinity == 3.0
    assert config.ranker_weights.recency_halflife_days == 180.0


def test_load_config_missing_openai_base_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_BASE_URL`` absent → ``ConfigError`` naming the variable."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    with pytest.raises(ConfigError) as excinfo:
        load_config(env={"OPENAI_MODEL": "m"})
    assert "OPENAI_BASE_URL" in str(excinfo.value)


def test_load_config_missing_openai_model_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_MODEL`` absent → ``ConfigError`` naming the variable."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    with pytest.raises(ConfigError) as excinfo:
        load_config(env={"OPENAI_BASE_URL": "x"})
    assert "OPENAI_MODEL" in str(excinfo.value)


def test_load_config_writes_openai_base_url_into_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful load mirrors ``OPENAI_BASE_URL`` into ``os.environ``.

    The OpenAI SDK that ``walker.py`` constructs reads
    ``OPENAI_BASE_URL`` from the process environment; the loader writes
    the configured value there as a documented side effect. The
    ``monkeypatch.setattr`` snapshot rolls the write back at teardown so
    it does not leak into other tests in the session.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    load_config(env={"OPENAI_BASE_URL": "http://x.example/v1", "OPENAI_MODEL": "m"})
    assert os.environ["OPENAI_BASE_URL"] == "http://x.example/v1"


def test_load_config_ranker_w_affinity_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RANKER_W_AFFINITY=5.0`` overrides only that field.

    Other ranker weights remain at their defaults so a single tweak in
    ``.env`` is locally scoped.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OPENAI_BASE_URL": "x",
            "OPENAI_MODEL": "m",
            "RANKER_W_AFFINITY": "5.0",
        }
    )
    weights = config.ranker_weights
    defaults = RankerWeights()
    assert weights.affinity == 5.0
    # Every other field stays at its default.
    assert weights.relevance == defaults.relevance
    assert weights.favorite == defaults.favorite
    assert weights.retweet == defaults.retweet
    assert weights.reply == defaults.reply
    assert weights.view == defaults.view
    assert weights.recency == defaults.recency
    assert weights.verified == defaults.verified
    assert weights.media == defaults.media
    assert weights.recency_halflife_days == defaults.recency_halflife_days


def test_load_config_non_numeric_ranker_weight_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric ``RANKER_W_*`` value is loud, not silent.

    The loader rejects ``"abc"`` rather than falling back to the default,
    so a typo in ``.env`` surfaces immediately.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    with pytest.raises(ConfigError) as excinfo:
        load_config(
            env={
                "OPENAI_BASE_URL": "x",
                "OPENAI_MODEL": "m",
                "RANKER_W_RELEVANCE": "abc",
            }
        )
    assert "RANKER_W_RELEVANCE" in str(excinfo.value)


def test_load_config_default_output_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OUTPUT_DIR`` absent → ``"output"`` and the derived cache path."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"OPENAI_BASE_URL": "x", "OPENAI_MODEL": "m"})
    assert config.output_dir == Path("output")
    assert config.cache_path == Path("output") / "tweet_tree_cache.pkl"
