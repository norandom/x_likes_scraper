"""Configuration loader for the X Likes MCP server.

Reads optional ``OUTPUT_DIR``, the OpenRouter/embedding variables
(``OPENROUTER_API_KEY``, ``OPENROUTER_BASE_URL``, ``EMBEDDING_MODEL``),
the optional walker/chat variables (``OPENAI_BASE_URL``,
``OPENAI_API_KEY``, ``OPENAI_MODEL``), and the optional
synthesis-report variables (``CRAWL4AI_BASE_URL``, ``URL_CACHE_DIR``,
``URL_CACHE_TTL_DAYS``, ``SYNTHESIS_MAX_HOPS``,
``SYNTHESIS_PER_SOURCE_BYTES``, ``SYNTHESIS_TOTAL_CONTEXT_BYTES``,
``SYNTHESIS_ROUND_TWO_K``, ``URL_FETCH_ALLOWED_PRIVATE_CIDRS``) from a
``.env`` file in the current working directory or, failing that, from
``os.environ``. The loader returns a frozen :class:`Config` dataclass
and, when the walker variables are present, writes ``OPENAI_BASE_URL``
(and ``OPENAI_API_KEY`` when set) into ``os.environ`` so the OpenAI SDK
that the walker constructs internally picks the values up at
client-construction time.

Both the OpenRouter API key and the walker variables are allowed to be
missing at config-load time. The dense-retrieval path surfaces a missing
``OPENROUTER_API_KEY`` later, at index-build time. The walker surfaces
missing ``OPENAI_BASE_URL`` / ``OPENAI_MODEL`` only when ``walker.walk``
is actually invoked (search_likes called with ``with_why=true``). The
synthesis-report orchestrator reuses ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``
as the synthesizer LM endpoint (no separate synthesis endpoint exists).
This keeps the default-path server starting cleanly with just an
OpenRouter key set.

The ``URL_FETCH_ALLOWED_PRIVATE_CIDRS`` value is parsed at load time via
``ipaddress.ip_network`` so a malformed CIDR fails loudly *before* any
fetch is attempted; the same fail-loud discipline applies to the
numeric synthesis vars.

Stdlib only; no ``python-dotenv`` dependency.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path

# Module-level defaults for the OpenRouter / embeddings configuration.
# These constants are the single source of truth: ``Config`` field defaults
# read from them, and the future ``embeddings.py`` module is expected to
# import them so a model-name change propagates to one place only.
DEFAULT_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

# Synthesis-report defaults. Same single-source-of-truth pattern: the
# ``fetcher``, ``cache``, and ``orchestrator`` modules import these
# constants so a default change propagates without grepping for magic
# numbers. Values are documented in design.md (synthesis-report spec).
DEFAULT_CRAWL4AI_BASE_URL: str = "http://127.0.0.1:11235"
DEFAULT_URL_CACHE_TTL_DAYS: int = 30
DEFAULT_SYNTHESIS_MAX_HOPS: int = 2
DEFAULT_SYNTHESIS_PER_SOURCE_BYTES: int = 4096
DEFAULT_SYNTHESIS_TOTAL_CONTEXT_BYTES: int = 32768
DEFAULT_SYNTHESIS_ROUND_TWO_K: int = 5

# Type alias for the parsed CIDR allowlist. Keeping ``IPv4Network |
# IPv6Network`` rather than the abstract ``_BaseNetwork`` matches what
# ``ipaddress.ip_network`` returns and keeps mypy strict-mode happy.
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class RankerWeights:
    """Per-feature weights for the heavy-ranker-style scoring formula.

    Defaults are tuned for *search* (relevance dominates), not feed-style
    recommendation. With cosine ``relevance`` in ``[0, 1]`` the maximum
    relevance contribution is ``80``, while ``log1p`` of typical engagement
    counts contributes 1-3 each. This keeps niche queries (e.g. "AI
    pentesting") from being buried by popular adjacent tweets that share
    only the broad topic.

    Override any field via ``RANKER_W_<UPPER>`` env variables when a
    different balance is wanted.
    """

    relevance: float = 80.0
    favorite: float = 0.5
    retweet: float = 0.5
    reply: float = 0.3
    view: float = 0.1
    affinity: float = 1.0
    recency: float = 1.5
    verified: float = 0.5
    media: float = 0.3
    recency_halflife_days: float = 180.0


@dataclass(frozen=True)
class Config:
    """Resolved server configuration.

    Synthesis-report fields (``crawl4ai_base_url``, ``url_cache_dir``,
    ``url_cache_ttl_days``, ``synthesis_max_hops``,
    ``synthesis_per_source_bytes``, ``synthesis_total_context_bytes``,
    ``synthesis_round_two_k``, ``url_fetch_allowed_private_cidrs``) are
    documented in design.md (synthesis-report spec). The synthesizer LM
    endpoint reuses ``openai_base_url`` / ``openai_model``; there is no
    separate synthesis endpoint.

    ``url_cache_dir`` defaults to ``output_dir / "url_cache"`` so a
    relocated ``OUTPUT_DIR`` carries the URL cache with it; an explicit
    ``URL_CACHE_DIR`` env value is honored verbatim (absolute or
    relative). ``url_fetch_allowed_private_cidrs`` is parsed into IP-
    network objects at load time so a malformed CIDR raises before any
    fetch is attempted.
    """

    output_dir: Path
    by_month_dir: Path
    likes_json: Path
    cache_path: Path
    ranker_weights: RankerWeights
    openai_base_url: str | None = None
    openai_api_key: str = ""
    openai_model: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # ``url_cache_dir`` defaults to ``Path("output/url_cache")`` to keep
    # ``Config(...)`` callable in tests without a long argument list. The
    # public ``load_config`` always re-derives this from ``output_dir``
    # so the on-disk layout follows ``OUTPUT_DIR``.
    url_cache_dir: Path = field(default_factory=lambda: Path("output") / "url_cache")
    crawl4ai_base_url: str = DEFAULT_CRAWL4AI_BASE_URL
    url_cache_ttl_days: int = DEFAULT_URL_CACHE_TTL_DAYS
    synthesis_max_hops: int = DEFAULT_SYNTHESIS_MAX_HOPS
    synthesis_per_source_bytes: int = DEFAULT_SYNTHESIS_PER_SOURCE_BYTES
    synthesis_total_context_bytes: int = DEFAULT_SYNTHESIS_TOTAL_CONTEXT_BYTES
    synthesis_round_two_k: int = DEFAULT_SYNTHESIS_ROUND_TWO_K
    url_fetch_allowed_private_cidrs: list[IPNetwork] = field(default_factory=list)


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
    """Pick the source of environment values for :func:`load_config`.

    Precedence (highest first):
      1. Explicit ``env`` dict — used as-is, intended for tests.
      2. ``os.environ`` keys — shell exports always win over file values
         (12-factor convention; lets users override ``.env`` per command).
      3. ``env_path`` if provided, else CWD ``.env`` if it exists.

    The previous behavior returned the file early when ``.env`` existed,
    silently shadowing shell exports. That made flags like
    ``RANKER_W_RELEVANCE=80 uv run x-likes-mcp ...`` no-ops.
    """

    if env is not None:
        return dict(env)

    if env_path is not None:
        file_values = _read_env_file(env_path)
    else:
        default_dotenv = Path.cwd() / ".env"
        file_values = _read_env_file(default_dotenv) if default_dotenv.exists() else {}

    merged = dict(file_values)
    merged.update(os.environ)
    return merged


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
        ConfigError: If a ``RANKER_W_*`` variable is set to a non-numeric
            value, if a ``URL_CACHE_TTL_DAYS`` / ``SYNTHESIS_*`` numeric
            variable is non-numeric, or if ``URL_FETCH_ALLOWED_PRIVATE_CIDRS``
            contains a malformed CIDR.
    """

    resolved = _resolve_env(env_path, env)

    openai_base_url = resolved.get("OPENAI_BASE_URL", "") or None
    openai_model = resolved.get("OPENAI_MODEL", "") or None
    openai_api_key = resolved.get("OPENAI_API_KEY", "") or ""

    # OpenRouter / embedding configuration. The URL and model fall back to
    # documented defaults; the API key is allowed to be ``None`` here and is
    # surfaced as an error later, at index-build time, so a developer can
    # run the walker (and the config tests) without an OpenRouter key.
    openrouter_base_url = resolved.get("OPENROUTER_BASE_URL", "") or DEFAULT_OPENROUTER_BASE_URL
    embedding_model = resolved.get("EMBEDDING_MODEL", "") or DEFAULT_EMBEDDING_MODEL
    openrouter_api_key_raw = resolved.get("OPENROUTER_API_KEY", "") or ""
    openrouter_api_key: str | None = openrouter_api_key_raw or None

    output_dir_raw = resolved.get("OUTPUT_DIR", "") or "output"
    output_dir = Path(output_dir_raw)
    by_month_dir = output_dir / "by_month"
    likes_json = output_dir / "likes.json"
    cache_path = output_dir / "tweet_tree_cache.pkl"

    # Synthesis-report fields. The CIDR allowlist parses at load time so
    # a malformed entry raises ``ConfigError`` before the orchestrator
    # ever opens a socket. Numeric fields use the same fail-loud
    # discipline as ``_load_ranker_weights``.
    crawl4ai_base_url = resolved.get("CRAWL4AI_BASE_URL", "") or DEFAULT_CRAWL4AI_BASE_URL
    url_cache_dir_raw = resolved.get("URL_CACHE_DIR", "") or ""
    # Empty / unset → track ``output_dir`` so relocating ``OUTPUT_DIR``
    # carries the URL cache with it. Explicit value (absolute or relative)
    # is honored verbatim.
    url_cache_dir = Path(url_cache_dir_raw) if url_cache_dir_raw else output_dir / "url_cache"
    url_cache_ttl_days = _load_int(resolved, "URL_CACHE_TTL_DAYS", DEFAULT_URL_CACHE_TTL_DAYS)
    synthesis_max_hops = _load_int(resolved, "SYNTHESIS_MAX_HOPS", DEFAULT_SYNTHESIS_MAX_HOPS)
    synthesis_per_source_bytes = _load_int(
        resolved, "SYNTHESIS_PER_SOURCE_BYTES", DEFAULT_SYNTHESIS_PER_SOURCE_BYTES
    )
    synthesis_total_context_bytes = _load_int(
        resolved, "SYNTHESIS_TOTAL_CONTEXT_BYTES", DEFAULT_SYNTHESIS_TOTAL_CONTEXT_BYTES
    )
    synthesis_round_two_k = _load_int(
        resolved, "SYNTHESIS_ROUND_TWO_K", DEFAULT_SYNTHESIS_ROUND_TWO_K
    )
    url_fetch_allowed_private_cidrs = _load_cidr_allowlist(
        resolved, "URL_FETCH_ALLOWED_PRIVATE_CIDRS"
    )

    # Side effect: hand off the OpenAI base URL (and API key, when set) to
    # ``os.environ`` so the OpenAI SDK in walker.py picks them up at
    # client-construction time. Skip when the walker is not configured —
    # the walker surfaces its own error if invoked without these.
    if openai_base_url:
        os.environ["OPENAI_BASE_URL"] = openai_base_url
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return Config(
        output_dir=output_dir,
        by_month_dir=by_month_dir,
        likes_json=likes_json,
        cache_path=cache_path,
        url_cache_dir=url_cache_dir,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ranker_weights=_load_ranker_weights(resolved),
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        embedding_model=embedding_model,
        crawl4ai_base_url=crawl4ai_base_url,
        url_cache_ttl_days=url_cache_ttl_days,
        synthesis_max_hops=synthesis_max_hops,
        synthesis_per_source_bytes=synthesis_per_source_bytes,
        synthesis_total_context_bytes=synthesis_total_context_bytes,
        synthesis_round_two_k=synthesis_round_two_k,
        url_fetch_allowed_private_cidrs=url_fetch_allowed_private_cidrs,
    )


def _load_int(env: dict[str, str], name: str, default: int) -> int:
    """Read a non-negative integer env value with a documented default.

    Mirrors ``_load_ranker_weights``' fail-loud discipline: a non-numeric
    value raises ``ConfigError`` with the offending input quoted, rather
    than silently falling back to the default. Empty / unset → default.
    """

    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} is not a valid integer: {raw!r}") from exc


def _load_cidr_allowlist(env: dict[str, str], name: str) -> list[IPNetwork]:
    """Parse a comma-separated CIDR allowlist into IP-network objects.

    Empty / unset → ``[]`` (the strict default: no private range is
    permitted unless the operator explicitly opts in). Whitespace around
    each comma-separated entry is tolerated. A malformed CIDR raises
    ``ConfigError`` immediately, naming both the env var and the
    offending value, so a typo surfaces *before* any fetch is attempted
    rather than as a silent skip during egress checks.

    ``ipaddress.ip_network(s, strict=False)`` is used so a host-bit-set
    input like ``10.100.5.7/16`` is accepted (operators commonly copy a
    sample IP rather than the network address).
    """

    raw = env.get(name)
    if raw is None or raw == "":
        return []

    parsed: list[IPNetwork] = []
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            # Tolerate trailing commas / blank entries; the operator's
            # intent is unambiguous here (vs. a typo'd CIDR).
            continue
        try:
            parsed.append(ipaddress.ip_network(candidate, strict=False))
        except ValueError as exc:
            raise ConfigError(f"{name} contains an invalid CIDR: {candidate!r}") from exc
    return parsed


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
