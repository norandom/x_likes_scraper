"""Configuration loader for the X Likes MCP server.

Reads three environment variables (``OPENAI_BASE_URL``, ``OPENAI_API_KEY``,
``OPENAI_MODEL``) plus an optional ``OUTPUT_DIR`` from a ``.env`` file in the
current working directory or, failing that, from ``os.environ``. The loader
returns a frozen :class:`Config` dataclass and writes ``OPENAI_BASE_URL``
(and ``OPENAI_API_KEY`` when set) into ``os.environ`` so the OpenAI SDK that
PageIndex constructs internally picks the values up at client-construction
time.

Stdlib only; no ``python-dotenv`` dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


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

    output_dir_raw = resolved.get("OUTPUT_DIR", "") or "output"
    output_dir = Path(output_dir_raw)
    by_month_dir = output_dir / "by_month"
    likes_json = output_dir / "likes.json"
    cache_path = output_dir / "pageindex_cache.pkl"

    # Side effect: hand off the OpenAI base URL (and API key, when set) to
    # ``os.environ`` so the OpenAI SDK that PageIndex instantiates picks them
    # up at client-construction time.
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
    )
