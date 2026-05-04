"""Per-month LLM walk over a :class:`TweetTree`.

This is the only place the package talks to a real model. For each
in-scope month the walker chunks the tweets, hands one chunk per call
to chat-completions, and asks "which of these are plausibly relevant
to the user's query?". The model returns JSON; we parse, drop entries
that fail validation (id not in chunk, relevance outside 0..1, etc.),
trim ``why`` to 240 chars, and emit :class:`WalkerHit` instances.

Note on the OpenAI client: we build ``OpenAI()`` with no ``base_url``
argument. The SDK reads ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``
from the process environment when the client is constructed.
``config.load_config`` writes both into ``os.environ`` first, which
is how the user's local proxy URL gets picked up.

Tests stub either :func:`walk` or the :func:`_call_chat_completions`
seam. The autouse guard in ``tests/mcp/conftest.py`` raises if any
test forgets to stub it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import Config
from .tree import TweetTree


# ---------------------------------------------------------------------------
# Public dataclasses

@dataclass(frozen=True)
class WalkerHit:
    """One plausibly-relevant tweet, as judged by the walker LLM call."""

    tweet_id: str
    relevance: float   # in [0, 1]
    why: str           # short snippet from the model, truncated to ~240 chars


class WalkerError(RuntimeError):
    """Raised when an LLM call fails or returns an unsalvageable response.

    ``tools.search_likes`` translates this into an ``upstream_failure``
    tool error.
    """


# ---------------------------------------------------------------------------
# Constants

# Truncate tweet text in the prompt to keep chunks small. The model only needs
# enough context to judge relevance, not the full thread.
_PROMPT_TEXT_MAX_CHARS = 280

# Per spec: truncate the model's `why` field to a reasonable length.
_WHY_MAX_CHARS = 240

_SYSTEM_PROMPT = (
    "You are a tweet relevance judge. The user will give you a query and a "
    "list of tweets, each tagged with a numeric id. Your job is to return "
    "ONLY the tweets that are plausibly relevant to the query, including "
    "INDIRECT and THEMATIC relevance (not just literal keyword overlap). "
    "Skip tweets that are clearly off-topic; do not include them with "
    "relevance 0.\n\n"
    "Respond with a JSON object of the shape:\n"
    '  {"hits": [{"id": "<tweet_id>", "relevance": <float in [0,1]>, '
    '"why": "<short reason>"}, ...]}\n\n'
    "If no tweets are plausibly relevant, return {\"hits\": []}. "
    "Do not include any prose outside the JSON object."
)


# ---------------------------------------------------------------------------
# Internal helpers

def _truncate(text: str, max_chars: int) -> str:
    """Return ``text`` collapsed onto one line and truncated to ``max_chars``."""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars]


def _build_user_prompt(query: str, chunk: list[Any]) -> str:
    """Build the user-message body for one chunk of tweets.

    Each tweet renders as ``[id={tweet_id}] @{handle}: {text}`` on its own
    line. The text is truncated to keep the prompt small.
    """
    lines = [f"Query: {query}", "", "Tweets:"]
    for node in chunk:
        text = _truncate(node.text or "", _PROMPT_TEXT_MAX_CHARS)
        lines.append(f"[id={node.tweet_id}] @{node.handle}: {text}")
    lines.append("")
    lines.append(
        "Return only the plausibly-relevant tweets as JSON per the system "
        "prompt's contract."
    )
    return "\n".join(lines)


def _call_chat_completions(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    """Invoke the OpenAI chat-completions endpoint and return the raw content.

    Tries JSON mode first (``response_format={"type": "json_object"}``). If
    the local proxy/model rejects ``response_format`` with a
    ``BadRequestError``, retries without it. Other exceptions propagate so
    :func:`walk` can wrap them in :class:`WalkerError`.

    This helper is the test mock seam.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Only retry without response_format if the failure looks like
        # "this endpoint does not support that param".
        msg = str(exc).lower()
        if "response_format" in msg or "json_object" in msg:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        else:
            raise

    content = response.choices[0].message.content
    if content is None:
        return ""
    return content


# A leading ```json ... ``` or ``` ... ``` markdown fence around JSON output.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _parse_response(raw: str) -> Any:
    """Tolerantly extract a JSON value from ``raw``.

    Handles three shapes:
      1. Pure JSON object or array.
      2. Markdown-fenced JSON (```` ```json ... ``` ````).
      3. Free-form text with an embedded JSON object/array; we pick from the
         first ``{`` or ``[`` to its matching closer (greedy to end).

    Raises :class:`ValueError` if no JSON value can be parsed.
    """
    if raw is None:
        raise ValueError("empty response")

    candidate = raw.strip()
    if not candidate:
        raise ValueError("empty response")

    # 1) Try as-is.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2) Strip markdown fences if present.
    fence_match = _FENCE_RE.match(candidate)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # 3) Locate the first JSON-y opener and try parsing from there to the
    #    last matching closer of the same kind.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        end = candidate.rfind(closer)
        if end <= start:
            continue
        slice_ = candidate[start : end + 1]
        try:
            return json.loads(slice_)
        except json.JSONDecodeError:
            continue

    raise ValueError("no parseable JSON in response")


def _coerce_hits(parsed: Any) -> list[dict[str, Any]]:
    """Pull the hits list out of various lenient wrapper shapes.

    Accepts:
      - a list of hit objects directly
      - ``{"hits": [...]}``
      - ``{"results": [...]}``
      - ``{"tweets": [...]}``
      - ``{"data": [...]}``

    Returns ``[]`` for any other shape (caller will then drop the chunk).
    """
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("hits", "results", "tweets", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _entry_to_hit(
    entry: dict[str, Any],
    valid_ids: set[str],
) -> WalkerHit | None:
    """Validate one parsed entry. Return ``None`` if it must be dropped."""
    raw_id = entry.get("id")
    if raw_id is None:
        return None
    tweet_id = str(raw_id)
    if tweet_id not in valid_ids:
        return None

    raw_relevance = entry.get("relevance")
    if isinstance(raw_relevance, bool):
        # bool is a subclass of int; treat it as not-a-number for relevance.
        return None
    if not isinstance(raw_relevance, (int, float)):
        return None
    relevance = float(raw_relevance)
    if not math.isfinite(relevance):
        return None
    if relevance < 0.0 or relevance > 1.0:
        return None

    raw_why = entry.get("why", "")
    why = str(raw_why) if raw_why is not None else ""
    if len(why) > _WHY_MAX_CHARS:
        why = why[:_WHY_MAX_CHARS]

    return WalkerHit(tweet_id=tweet_id, relevance=relevance, why=why)


# ---------------------------------------------------------------------------
# Public entry point

def _resolve_months(
    tree: TweetTree, months_in_scope: list[str] | None
) -> list[str]:
    """Return the months to walk, in caller-determined order.

    ``None`` means every month, ascending. A non-``None`` list keeps the
    caller's order (TweetIndex already builds it chronologically) and
    drops months that the tree has no nodes for.
    """

    if months_in_scope is None:
        return sorted(tree.nodes_by_month.keys())
    return [m for m in months_in_scope if m in tree.nodes_by_month]


def _assert_walker_configured(config: Config) -> None:
    """Raise :class:`WalkerError` when the opt-in walker config is missing.

    The default search path does not call the walker, so the config can
    legitimately be unset. We surface a clear error here instead of
    letting the OpenAI SDK fail later with a less specific message.
    """

    if not config.openai_base_url or not config.openai_model:
        raise WalkerError(
            "Walker invoked but OPENAI_BASE_URL and/or OPENAI_MODEL are not "
            "set. The walker explainer is opt-in via search_likes(with_why=true); "
            "set both variables in .env to enable it."
        )


def _walk_chunk(
    client: OpenAI,
    model: str,
    query: str,
    chunk: list[Any],
    month: str,
    chunk_index: int,
) -> list[WalkerHit]:
    """Process one chunk: call the LLM, parse, return validated hits."""

    valid_ids = {node.tweet_id for node in chunk}
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(query, chunk)},
    ]

    try:
        raw = _call_chat_completions(client, model, messages)
        parsed = _parse_response(raw)
    except WalkerError:
        raise
    except Exception as exc:
        raise WalkerError(
            f"LLM call failed for month {month} chunk {chunk_index}: {exc}"
        ) from exc

    hits: list[WalkerHit] = []
    for entry in _coerce_hits(parsed):
        hit = _entry_to_hit(entry, valid_ids)
        if hit is not None:
            hits.append(hit)
    return hits


def walk(
    tree: TweetTree,
    query: str,
    months_in_scope: list[str] | None,
    config: Config,
    chunk_size: int = 30,
) -> list[WalkerHit]:
    """Walk in-scope months and return plausibly-relevant tweets.

    Parameters:
        tree: Parsed :class:`TweetTree` from :mod:`x_likes_mcp.tree`.
        query: Natural-language search query.
        months_in_scope: Either ``None`` (walk every month, ascending) or a
            list of ``"YYYY-MM"`` strings; only the listed months that exist
            in the tree are walked.
        config: Resolved :class:`x_likes_mcp.config.Config` (for the model
            name; the SDK reads ``OPENAI_BASE_URL`` from ``os.environ``).
        chunk_size: Number of tweets per LLM call. Default 30.

    Returns:
        Accumulated list of :class:`WalkerHit` across all chunks, in the
        order chunks were processed (months ascending, within-month chunks
        in order).

    Raises:
        WalkerError: On any per-chunk LLM failure (HTTP error, malformed
            JSON that cannot be salvaged).
    """

    months = _resolve_months(tree, months_in_scope)
    if not months:
        return []

    _assert_walker_configured(config)

    # The SDK reads OPENAI_BASE_URL from os.environ; config.load_config wrote
    # it there. We pass api_key explicitly because local OpenAI-compatible
    # proxies often don't require auth (the config allows an empty key), but
    # the SDK constructor still demands *some* string.
    client = OpenAI(api_key=config.openai_api_key or "not-required")
    hits: list[WalkerHit] = []

    for month in months:
        nodes = tree.nodes_by_month[month]
        for chunk_start in range(0, len(nodes), chunk_size):
            chunk = nodes[chunk_start : chunk_start + chunk_size]
            if not chunk:
                continue
            hits.extend(
                _walk_chunk(
                    client,
                    config.openai_model,
                    query,
                    chunk,
                    month,
                    chunk_start // chunk_size,
                )
            )

    return hits
