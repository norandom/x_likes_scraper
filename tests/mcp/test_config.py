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
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
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
    # Defaults are tuned for search (relevance dominates engagement).
    assert config.ranker_weights == RankerWeights()
    assert config.ranker_weights.relevance == 80.0
    assert config.ranker_weights.affinity == 1.0
    assert config.ranker_weights.recency_halflife_days == 180.0


def test_load_config_missing_openai_vars_resolves_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walker config is opt-in: missing ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``
    resolves to ``None`` rather than raising. The walker itself surfaces a
    ``WalkerError`` if invoked without these set."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={})
    assert config.openai_base_url is None
    assert config.openai_model is None
    assert config.openai_api_key == ""


def test_load_config_writes_openai_base_url_into_environ_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the walker variables are set, the loader mirrors
    ``OPENAI_BASE_URL`` into ``os.environ`` as a documented side effect
    so the OpenAI SDK that ``walker.py`` constructs picks it up at
    client-construction time. ``monkeypatch.setattr`` rolls the write
    back at teardown."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    load_config(env={"OPENAI_BASE_URL": "http://x.example/v1", "OPENAI_MODEL": "m"})
    assert os.environ["OPENAI_BASE_URL"] == "http://x.example/v1"


def test_load_config_skips_environ_write_when_walker_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No walker config → no ``OPENAI_BASE_URL`` write into ``os.environ``."""

    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("OPENAI_BASE_URL", None)

    load_config(env={})
    assert "OPENAI_BASE_URL" not in os.environ


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


# ---------------------------------------------------------------------------
# OpenRouter + embedding-model fields (mcp-fast-search spec, Requirement 1)
# ---------------------------------------------------------------------------


def test_load_config_openrouter_fields_populated_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three new env vars set → ``Config`` fields mirror the env values.

    Covers Requirement 1.1, 1.2, 1.3: ``OPENROUTER_BASE_URL``,
    ``EMBEDDING_MODEL``, and ``OPENROUTER_API_KEY`` are read from the
    resolved env dict and stored on ``Config``.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OPENAI_BASE_URL": "x",
            "OPENAI_MODEL": "m",
            "OPENROUTER_API_KEY": "sk-or-test-123",
            "OPENROUTER_BASE_URL": "https://openrouter.example/api/v1",
            "EMBEDDING_MODEL": "some-org/some-embedding-model:free",
        }
    )

    assert config.openrouter_api_key == "sk-or-test-123"
    assert config.openrouter_base_url == "https://openrouter.example/api/v1"
    assert config.embedding_model == "some-org/some-embedding-model:free"


def test_load_config_openrouter_base_url_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENROUTER_BASE_URL`` unset → falls back to the documented default.

    Covers Requirement 1.1: the default is
    ``https://openrouter.ai/api/v1``.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"OPENAI_BASE_URL": "x", "OPENAI_MODEL": "m"})
    assert config.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert config.openrouter_base_url == DEFAULT_OPENROUTER_BASE_URL


def test_load_config_openrouter_base_url_defaults_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``OPENROUTER_BASE_URL`` is treated the same as unset.

    Covers Requirement 1.1's "unset or empty" wording.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OPENAI_BASE_URL": "x",
            "OPENAI_MODEL": "m",
            "OPENROUTER_BASE_URL": "",
        }
    )
    assert config.openrouter_base_url == DEFAULT_OPENROUTER_BASE_URL


def test_load_config_embedding_model_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EMBEDDING_MODEL`` unset → falls back to the documented default.

    Covers Requirement 1.2: the default is the free-tier vision-language
    embedding model on OpenRouter.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"OPENAI_BASE_URL": "x", "OPENAI_MODEL": "m"})
    assert config.embedding_model == "openai/text-embedding-3-small"
    assert config.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_load_config_embedding_model_defaults_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``EMBEDDING_MODEL`` is treated the same as unset.

    Covers Requirement 1.2's "unset or empty" wording.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OPENAI_BASE_URL": "x",
            "OPENAI_MODEL": "m",
            "EMBEDDING_MODEL": "",
        }
    )
    assert config.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_load_config_openrouter_api_key_is_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset ``OPENROUTER_API_KEY`` → ``None`` (not ``""``, not a default).

    Covers Requirement 1.3: missing key resolves to ``None`` at config-
    load time. The "missing key" failure is surfaced later, at index-
    build time, so config-only tests do not need to provide a key.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"OPENAI_BASE_URL": "x", "OPENAI_MODEL": "m"})
    assert config.openrouter_api_key is None


def test_load_config_openrouter_api_key_is_none_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``OPENROUTER_API_KEY`` resolves to ``None``, not ``""``.

    Empty strings in ``.env`` files are common (``OPENROUTER_API_KEY=``);
    they must be normalized to ``None`` so downstream code can do a single
    ``if api_key is None`` check.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OPENAI_BASE_URL": "x",
            "OPENAI_MODEL": "m",
            "OPENROUTER_API_KEY": "",
        }
    )
    assert config.openrouter_api_key is None


def test_load_config_default_constants_match_field_defaults() -> None:
    """The exported default constants match the ``Config`` field defaults.

    A future ``embeddings.py`` will import these constants as the single
    source of truth; this test guards against drift between the two
    representations.
    """

    assert DEFAULT_OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert DEFAULT_EMBEDDING_MODEL == "openai/text-embedding-3-small"


def test_resolve_env_shell_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shell ``os.environ`` keys win over ``.env`` file values.

    Regression: the previous loader returned ``.env`` early when the file
    existed, so commands like ``RANKER_W_RELEVANCE=80 uv run x-likes-mcp``
    were silent no-ops on any project that shipped a ``.env``. The merge
    is now ``file_values <- os.environ``.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENAI_BASE_URL=from_file\n"
        "OPENAI_MODEL=from_file\n"
        "RANKER_W_RELEVANCE=10\n",
        encoding="utf-8",
    )
    os.environ["RANKER_W_RELEVANCE"] = "42"

    config = load_config(env_path=dotenv)

    # File-only key still propagates.
    assert config.openai_base_url == "from_file"
    # Shell-supplied override wins over the file value.
    assert config.ranker_weights.relevance == 42.0


def test_resolve_env_falls_back_to_dotenv_when_shell_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the shell has no override, ``.env`` values still reach the loader."""

    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("RANKER_W_RELEVANCE", None)

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENAI_BASE_URL=x\nOPENAI_MODEL=m\nRANKER_W_RELEVANCE=7\n",
        encoding="utf-8",
    )

    config = load_config(env_path=dotenv)
    assert config.ranker_weights.relevance == 7.0
