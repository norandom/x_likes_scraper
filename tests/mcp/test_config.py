"""Tests for :mod:`x_likes_mcp.config`.

Covers :func:`load_config` plus the embedded :class:`RankerWeights` parsing.
The loader writes ``OPENAI_BASE_URL`` (and ``OPENAI_API_KEY`` when set) into
``os.environ`` as a documented side effect; tests that exercise that side
effect use ``monkeypatch`` so the change is rolled back at teardown and does
not leak between tests.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

import pytest

from x_likes_mcp.config import (
    DEFAULT_CRAWL4AI_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_SYNTHESIS_MAX_HOPS,
    DEFAULT_SYNTHESIS_PER_SOURCE_BYTES,
    DEFAULT_SYNTHESIS_ROUND_TWO_K,
    DEFAULT_SYNTHESIS_TOTAL_CONTEXT_BYTES,
    DEFAULT_URL_CACHE_TTL_DAYS,
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
        "OPENAI_BASE_URL=from_file\nOPENAI_MODEL=from_file\nRANKER_W_RELEVANCE=10\n",
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


# ---------------------------------------------------------------------------
# Synthesis-report fields (synthesis-report spec, Requirement 4.3 + 12.1-12.3)
# ---------------------------------------------------------------------------


def test_load_config_synthesis_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``.env`` that omits every synthesis-report var still loads.

    Covers Requirement 12.3: each new env var has a documented default
    that works without any extra configuration. Loading from an empty
    env produces a working ``Config`` with the design-documented defaults.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={})

    assert config.crawl4ai_base_url == "http://127.0.0.1:11235"
    assert config.crawl4ai_base_url == DEFAULT_CRAWL4AI_BASE_URL
    # url_cache_dir tracks output_dir by default — not an absolute path.
    assert config.url_cache_dir == Path("output") / "url_cache"
    assert config.url_cache_ttl_days == 30
    assert config.url_cache_ttl_days == DEFAULT_URL_CACHE_TTL_DAYS
    assert config.synthesis_max_hops == 2
    assert config.synthesis_max_hops == DEFAULT_SYNTHESIS_MAX_HOPS
    assert config.synthesis_per_source_bytes == 4096
    assert config.synthesis_per_source_bytes == DEFAULT_SYNTHESIS_PER_SOURCE_BYTES
    assert config.synthesis_total_context_bytes == 32768
    assert config.synthesis_total_context_bytes == DEFAULT_SYNTHESIS_TOTAL_CONTEXT_BYTES
    assert config.synthesis_round_two_k == 5
    assert config.synthesis_round_two_k == DEFAULT_SYNTHESIS_ROUND_TWO_K
    # Strict default: no private CIDR is allowed unless the operator opts in.
    assert config.url_fetch_allowed_private_cidrs == []


def test_load_config_synthesis_explicit_values_flow_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit env values override every synthesis default.

    Covers Requirement 12.2 (CRAWL4AI_BASE_URL override) and 12.3 (the
    other synthesis vars are read from documented env names).
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "CRAWL4AI_BASE_URL": "http://crawl4ai.internal:8080",
            "URL_CACHE_DIR": "/tmp/x_likes/url_cache",
            "URL_CACHE_TTL_DAYS": "7",
            "SYNTHESIS_MAX_HOPS": "3",
            "SYNTHESIS_PER_SOURCE_BYTES": "2048",
            "SYNTHESIS_TOTAL_CONTEXT_BYTES": "16384",
            "SYNTHESIS_ROUND_TWO_K": "8",
        }
    )

    assert config.crawl4ai_base_url == "http://crawl4ai.internal:8080"
    # Absolute URL_CACHE_DIR is honored verbatim (not joined under output_dir).
    assert config.url_cache_dir == Path("/tmp/x_likes/url_cache")
    assert config.url_cache_ttl_days == 7
    assert config.synthesis_max_hops == 3
    assert config.synthesis_per_source_bytes == 2048
    assert config.synthesis_total_context_bytes == 16384
    assert config.synthesis_round_two_k == 8


def test_load_config_url_cache_dir_tracks_output_dir_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``URL_CACHE_DIR`` unset → cache dir tracks ``OUTPUT_DIR``.

    The url-cache directory follows the project's ``OUTPUT_DIR`` so
    operators only have to override one variable to relocate state.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"OUTPUT_DIR": "/var/lib/xlikes"})

    assert config.output_dir == Path("/var/lib/xlikes")
    assert config.url_cache_dir == Path("/var/lib/xlikes") / "url_cache"


def test_load_config_url_cache_dir_explicit_overrides_output_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``URL_CACHE_DIR`` is honored verbatim, not joined."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OUTPUT_DIR": "/var/lib/xlikes",
            "URL_CACHE_DIR": "/srv/cache/url",
        }
    )

    assert config.output_dir == Path("/var/lib/xlikes")
    # Absolute override wins; not joined under output_dir.
    assert config.url_cache_dir == Path("/srv/cache/url")


def test_load_config_cidr_allowlist_parses_comma_separated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``URL_FETCH_ALLOWED_PRIVATE_CIDRS`` parses into IP-network objects.

    Covers Requirement 4.3 / 12.3: the comma-separated allowlist is
    parsed at load time so malformed values fail loudly before any
    network fetch happens.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "URL_FETCH_ALLOWED_PRIVATE_CIDRS": "10.100.0.0/16,192.168.10.0/24",
        }
    )

    assert config.url_fetch_allowed_private_cidrs == [
        ipaddress.ip_network("10.100.0.0/16", strict=False),
        ipaddress.ip_network("192.168.10.0/24", strict=False),
    ]


def test_load_config_cidr_allowlist_handles_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace around comma-separated CIDR entries is tolerated."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "URL_FETCH_ALLOWED_PRIVATE_CIDRS": "  10.100.0.0/16 , 192.168.10.0/24  ",
        }
    )

    assert config.url_fetch_allowed_private_cidrs == [
        ipaddress.ip_network("10.100.0.0/16", strict=False),
        ipaddress.ip_network("192.168.10.0/24", strict=False),
    ]


def test_load_config_cidr_allowlist_empty_value_is_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ``URL_FETCH_ALLOWED_PRIVATE_CIDRS`` yields ``[]`` (strict default)."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"URL_FETCH_ALLOWED_PRIVATE_CIDRS": ""})
    assert config.url_fetch_allowed_private_cidrs == []


def test_load_config_cidr_allowlist_supports_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv6 CIDRs parse via ``ipaddress.ip_network`` like IPv4."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={"URL_FETCH_ALLOWED_PRIVATE_CIDRS": "fd00::/8,10.0.0.0/8"})
    assert config.url_fetch_allowed_private_cidrs == [
        ipaddress.ip_network("fd00::/8", strict=False),
        ipaddress.ip_network("10.0.0.0/8", strict=False),
    ]


def test_load_config_malformed_cidr_raises_with_offending_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed CIDR raises ``ConfigError`` immediately, naming the bad value.

    Covers Requirement 4.3: the allowlist parses at load time so a typo
    surfaces before any fetch happens — not as a silent skip during
    egress checks.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    with pytest.raises(ConfigError) as excinfo:
        load_config(
            env={
                "URL_FETCH_ALLOWED_PRIVATE_CIDRS": "10.100.0.0/16,not-a-cidr",
            }
        )
    # The message must name both the env var and the offending value so
    # an operator can find the typo without guessing.
    msg = str(excinfo.value)
    assert "URL_FETCH_ALLOWED_PRIVATE_CIDRS" in msg
    assert "not-a-cidr" in msg


def test_load_config_malformed_int_synthesis_var_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-numeric ``URL_CACHE_TTL_DAYS`` is loud, not silent."""

    monkeypatch.setattr(os, "environ", dict(os.environ))

    with pytest.raises(ConfigError) as excinfo:
        load_config(env={"URL_CACHE_TTL_DAYS": "forever"})
    msg = str(excinfo.value)
    assert "URL_CACHE_TTL_DAYS" in msg
    assert "forever" in msg


def test_load_config_synthesis_does_not_disturb_openai_or_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding synthesis env vars does not change OpenAI / OpenRouter behavior.

    Regression guard for Requirement 12.1: the synthesizer LM reuses
    ``OPENAI_BASE_URL`` / ``OPENAI_MODEL`` — there is no separate
    synthesis endpoint pair. Setting the synthesis vars must not
    perturb the existing walker / OpenRouter fields.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(
        env={
            "OPENAI_BASE_URL": "http://walker.example/v1",
            "OPENAI_MODEL": "walker-model",
            "OPENAI_API_KEY": "sk-walker",
            "OPENROUTER_API_KEY": "sk-or-test",
            "OPENROUTER_BASE_URL": "https://openrouter.example/api/v1",
            "EMBEDDING_MODEL": "some-org/some-embedding-model",
            "CRAWL4AI_BASE_URL": "http://crawl4ai.internal:8080",
            "SYNTHESIS_MAX_HOPS": "4",
            "URL_FETCH_ALLOWED_PRIVATE_CIDRS": "10.0.0.0/8",
        }
    )

    # Existing fields untouched.
    assert config.openai_base_url == "http://walker.example/v1"
    assert config.openai_model == "walker-model"
    assert config.openai_api_key == "sk-walker"
    assert config.openrouter_api_key == "sk-or-test"
    assert config.openrouter_base_url == "https://openrouter.example/api/v1"
    assert config.embedding_model == "some-org/some-embedding-model"
    # New fields populated from env.
    assert config.crawl4ai_base_url == "http://crawl4ai.internal:8080"
    assert config.synthesis_max_hops == 4


def test_config_remains_frozen_dataclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Config`` stays frozen after the synthesis-report fields are added.

    Mutation must continue to raise ``FrozenInstanceError`` so callers
    cannot accidentally rewrite the resolved configuration at runtime.
    """

    monkeypatch.setattr(os, "environ", dict(os.environ))

    config = load_config(env={})

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        config.crawl4ai_base_url = "http://other"  # type: ignore[misc]


def test_load_config_default_synthesis_constants_match_field_defaults() -> None:
    """Synthesis default constants match the documented values from design.md.

    Single source of truth for downstream modules (`fetcher`, `cache`,
    `orchestrator`) that import these constants directly.
    """

    assert DEFAULT_CRAWL4AI_BASE_URL == "http://127.0.0.1:11235"
    assert DEFAULT_URL_CACHE_TTL_DAYS == 30
    assert DEFAULT_SYNTHESIS_MAX_HOPS == 2
    assert DEFAULT_SYNTHESIS_PER_SOURCE_BYTES == 4096
    assert DEFAULT_SYNTHESIS_TOTAL_CONTEXT_BYTES == 32768
    assert DEFAULT_SYNTHESIS_ROUND_TWO_K == 5
