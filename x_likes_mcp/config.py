"""Configuration loader for the X Likes MCP server.

Reads the OpenAI/walker variables (``OPENAI_BASE_URL``, ``OPENAI_API_KEY``,
``OPENAI_MODEL``), the OpenRouter/embedding variables
(``OPENROUTER_API_KEY``, ``OPENROUTER_BASE_URL``, ``EMBEDDING_MODEL``), and
an optional ``OUTPUT_DIR`` from a ``.env`` file in the current working
directory or, failing that, from ``os.environ``. The loader returns a
frozen :class:`Config` dataclass and writes ``OPENAI_BASE_URL`` (and
``OPENAI_API_KEY`` when set) into ``os.environ`` so the OpenAI SDK that
PageIndex constructs internally picks the values up at client-construction
time.

The OpenRouter API key is intentionally allowed to be missing at config-
load time: the dense-retrieval path surfaces that failure later, at
index-build time, so the rest of the server (including the walker and the
config tests) does not require an OpenRouter key to function.

Stdlib only; no ``python-dotenv`` dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Module-level defaults for the OpenRouter / embeddings configuration.
# These constants are the single source of truth: ``Config`` field defaults
# read from them, and the future ``embeddings.py`` module is expected to
# import them so a model-name change propagates to one place only.
DEFAULT_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class RankerWeights:
    """Per-feature weights for the heavy-ranker-style scoring formula.

    Defaults are intentionally engagement-and-affinity-heavy: tweets the
    user already returned to (high author affinity) and tweets that struck
    a chord with the rest of the network (high engagement) outrank
    one-off topical matches.
    """

    relevance: float = 10.0
    favorite: float = 2.0
    retweet: float = 2.5
    reply: float = 1.0
    view: float = 0.5
    affinity: float = 3.0
    recency: float = 1.5
    verified: float = 0.5
    media: float = 0.3
    recency_halflife_days: float = 180.0


@dataclass(frozen=True)
class Config:
    """Resolved server configuration."""

    output_dir: Path
    by_month_dir: Path
    likes_json: Path
    cache_path: Path
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    ranker_weights: RankerWeights
    openrouter_api_key: str | None = None
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a ``dict``.

    Each non-blank line that does not start (after leading whitespace is
    stripped) with ``#`` is split on the first ``=``. Surrounding whitespace
    around the key and value is stripped. Inline ``#`` comments after a value
    are not supported. Empty values (``KEY=``) become empty strings.
    """

    result: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        result[key] = value
    return result


def _resolve_env(
    env_path: Path | None,
    env: dict[str, str] | None,
) -> dict[str, str]:
    """Pick the source of environment values for :func:`load_config`."""

    if env is not None:
        return dict(env)
    if env_path is not None:
        return _read_env_file(env_path)
    default_dotenv = Path.cwd() / ".env"
    if default_dotenv.exists():
        return _read_env_file(default_dotenv)
    return dict(os.environ)


def _require(env: dict[str, str], name: str) -> str:
    """Return ``env[name]`` after asserting it is set and non-empty."""

    value = env.get(name, "")
    if value is None or value == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def load_config(
    env_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Load and validate server configuration.

    Parameters:
        env_path: Optional explicit path to a ``.env`` file. When provided,
            the file at this path is read instead of the CWD ``.env`` and
            instead of ``os.environ``.
        env: Optional in-memory mapping that fully replaces the file and
            ``os.environ`` lookup. Intended for tests.

    Returns:
        A populated :class:`Config`.

    Raises:
        ConfigError: If ``OPENAI_BASE_URL`` or ``OPENAI_MODEL`` is missing or
            empty.
    """

    resolved = _resolve_env(env_path, env)

    openai_base_url = _require(resolved, "OPENAI_BASE_URL")
    openai_model = _require(resolved, "OPENAI_MODEL")
    openai_api_key = resolved.get("OPENAI_API_KEY", "") or ""

    # OpenRouter / embedding configuration. The URL and model fall back to
    # documented defaults; the API key is allowed to be ``None`` here and is
    # surfaced as an error later, at index-build time, so a developer can
    # run the walker (and the config tests) without an OpenRouter key.
    openrouter_base_url = (
        resolved.get("OPENROUTER_BASE_URL", "") or DEFAULT_OPENROUTER_BASE_URL
    )
    embedding_model = (
        resolved.get("EMBEDDING_MODEL", "") or DEFAULT_EMBEDDING_MODEL
    )
    openrouter_api_key_raw = resolved.get("OPENROUTER_API_KEY", "") or ""
    openrouter_api_key: str | None = openrouter_api_key_raw or None

    output_dir_raw = resolved.get("OUTPUT_DIR", "") or "output"
    output_dir = Path(output_dir_raw)
    by_month_dir = output_dir / "by_month"
    likes_json = output_dir / "likes.json"
    cache_path = output_dir / "tweet_tree_cache.pkl"

    # Side effect: hand off the OpenAI base URL (and API key, when set) to
    # ``os.environ`` so the OpenAI SDK in walker.py picks them up at
    # client-construction time.
    os.environ["OPENAI_BASE_URL"] = openai_base_url
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return Config(
        output_dir=output_dir,
        by_month_dir=by_month_dir,
        likes_json=likes_json,
        cache_path=cache_path,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ranker_weights=_load_ranker_weights(resolved),
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        embedding_model=embedding_model,
    )


def _load_ranker_weights(env: dict[str, str]) -> RankerWeights:
    """Build ``RankerWeights`` from optional ``RANKER_*`` env entries.

    Any unset variable falls back to the dataclass default. Non-numeric
    values raise ``ConfigError`` rather than silently using the default,
    so a typo in ``.env`` is loud.
    """

    def _f(name: str, default: float) -> float:
        raw = env.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} is not a valid float: {raw!r}") from exc

    defaults = RankerWeights()
    return RankerWeights(
        relevance=_f("RANKER_W_RELEVANCE", defaults.relevance),
        favorite=_f("RANKER_W_FAVORITE", defaults.favorite),
        retweet=_f("RANKER_W_RETWEET", defaults.retweet),
        reply=_f("RANKER_W_REPLY", defaults.reply),
        view=_f("RANKER_W_VIEW", defaults.view),
        affinity=_f("RANKER_W_AFFINITY", defaults.affinity),
        recency=_f("RANKER_W_RECENCY", defaults.recency),
        verified=_f("RANKER_W_VERIFIED", defaults.verified),
        media=_f("RANKER_W_MEDIA", defaults.media),
        recency_halflife_days=_f("RANKER_RECENCY_HALFLIFE_DAYS", defaults.recency_halflife_days),
    )
