"""OpenRouter-backed embedder for the X Likes MCP server.

This module owns the dense-retrieval seam. It is the only place in the
package that issues HTTP requests for embeddings: ``Embedder`` wraps the
``openai`` SDK pointed at OpenRouter's ``/v1/embeddings`` endpoint and
exposes one batched HTTP method (``_call_embeddings_api``) plus higher-
level convenience methods (``embed_query``, ``embed_corpus``).

Design notes:

* The ``openai.OpenAI`` client is constructed lazily on the first
  ``_call_embeddings_api`` invocation. ``__init__`` only records
  configuration. Tests that patch ``_call_embeddings_api`` (the
  documented seam, mirroring ``walker._call_chat_completions``) never
  trigger a real client construction.
* Empty / ``None`` ``api_key`` raises :class:`EmbeddingError` naming
  ``OPENROUTER_API_KEY`` *before* any client construction. This keeps the
  failure message the user sees about the env var, not about the SDK.
* ``_call_embeddings_api`` retries up to ``max_retries`` times on
  transient errors (``429`` rate limits, ``5xx`` server errors,
  connection errors). Auth errors (``401``/``403``) propagate immediately
  as :class:`EmbeddingError` without retrying.
* The cosine top-k helper, the on-disk cache (``corpus_embeddings.npy``
  + ``corpus_embeddings.meta.json``), and the ``open_or_build_corpus``
  orchestrator land in subsequent tasks (2.2, 2.3, 2.4). This file
  currently provides only the embedding seam from task 2.1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import openai

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OPENROUTER_BASE_URL


# ---------------------------------------------------------------------------
# Module-level constants
#
# ``DEFAULT_BASE_URL`` and ``DEFAULT_EMBEDDING_MODEL`` are re-exported from
# :mod:`x_likes_mcp.config` so callers that import either symbol from this
# module see the same literal as ``Config`` defaults. ``config.py`` remains
# the single source of truth for the literal values.

CACHE_SCHEMA_VERSION: int = 1
DEFAULT_BASE_URL: str = DEFAULT_OPENROUTER_BASE_URL
DEFAULT_TOP_K: int = 200
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_MAX_RETRIES: int = 3

# Re-exported so callers can ``from x_likes_mcp.embeddings import
# DEFAULT_EMBEDDING_MODEL`` (the spec lists it in the embeddings.py
# constants block).
__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_TOP_K",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_RETRIES",
    "EmbeddingError",
    "CorpusEmbeddings",
    "Embedder",
]


# ---------------------------------------------------------------------------
# Errors and value objects


class EmbeddingError(RuntimeError):
    """Raised when corpus embedding fails fatally.

    Triggers include: missing ``OPENROUTER_API_KEY``, persistent rate
    limits or server errors after the retry budget is exhausted, auth
    failures, and (in later tasks) cache write errors.
    """


@dataclass
class CorpusEmbeddings:
    """Container for the embedded corpus and the id alignment metadata.

    Attributes:
        matrix: Shape ``(N, D)``, ``float32``, L2-normalized rows.
        ordered_ids: Length-``N`` list mapping row index ``i`` to tweet id.
        model_name: The embedding model that produced ``matrix``. Used for
            cache invalidation (a model change forces a rebuild).
    """

    matrix: np.ndarray
    ordered_ids: list[str]
    model_name: str


# ---------------------------------------------------------------------------
# Internal helpers


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` divided by its L2 norm, or ``vec`` itself if the norm
    is zero (so we never divide by zero).

    The model should not produce a zero vector for any non-empty input,
    but empty strings or pathological inputs might land here. Returning
    the original (zero) vector keeps downstream cosine math finite.
    """

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Embedder


class Embedder:
    """OpenRouter-backed embedder for queries and the corpus.

    The ``_call_embeddings_api`` method is the test mock seam (mirrors
    :func:`x_likes_mcp.walker._call_chat_completions`). Tests that need
    canned vectors patch it directly with ``monkeypatch.setattr``.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Record configuration. The OpenAI client is constructed lazily.

        Parameters:
            api_key: The OpenRouter API key. ``None`` or empty string is
                accepted at construction time and surfaces as an
                :class:`EmbeddingError` only when ``_call_embeddings_api``
                is actually invoked. This lets the rest of the server
                (the walker, the config tests) start without an
                OpenRouter key.
            base_url: The OpenRouter ``/v1`` base URL.
            model_name: Embedding model identifier.
            batch_size: Maximum number of inputs per HTTP call. Used by
                :meth:`embed_corpus` to chunk the corpus.
            max_retries: How many times to retry a transient error before
                giving up. ``max_retries=3`` means up to 4 total attempts
                per call (one initial + three retries) with exponential
                backoff (1s, 2s, 4s) between them.
        """

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Lazy: built on first _call_embeddings_api invocation.
        self._client: openai.OpenAI | None = None

    # ------------------------------------------------------------------
    # Test seam

    def _call_embeddings_api(self, texts: list[str]) -> list[list[float]]:
        """Issue one ``client.embeddings.create`` call and return vectors.

        This is the documented mock seam. Tests patch this method to
        return canned vectors so no real HTTP request escapes the test
        process.

        Behavior:
            1. If ``self.api_key`` is falsy, raise :class:`EmbeddingError`
               naming ``OPENROUTER_API_KEY`` before constructing the
               client.
            2. Lazily construct ``openai.OpenAI(api_key=..., base_url=...)``
               on first invocation; cache it on the instance.
            3. Call ``client.embeddings.create(model=..., input=texts)``,
               sort the response data by ``.index``, and return the list
               of embedding vectors.
            4. Retry up to ``self.max_retries`` times on
               ``openai.RateLimitError`` and on ``openai.APIConnectionError``;
               also retry on ``openai.APIStatusError`` whose ``status_code``
               is in the 5xx range. Exponential backoff between retries
               (1s, 2s, 4s).
            5. ``openai.AuthenticationError`` propagates immediately as
               :class:`EmbeddingError`.
            6. After ``max_retries`` failures, raise :class:`EmbeddingError`
               naming the underlying cause.
        """

        if not self.api_key:
            raise EmbeddingError(
                "OPENROUTER_API_KEY is not set. Add it to .env (or the "
                "environment) before starting the MCP server; the dense "
                "retrieval path requires it."
            )

        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url
            )

        max_attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
            except openai.AuthenticationError as exc:
                # Auth errors are terminal; do not retry, surface immediately.
                raise EmbeddingError(
                    f"OpenRouter authentication failed (check "
                    f"OPENROUTER_API_KEY): {exc}"
                ) from exc
            except openai.RateLimitError as exc:
                last_error = exc
            except openai.APIConnectionError as exc:
                last_error = exc
            except openai.APIStatusError as exc:
                # Only retry transient 5xx; 4xx other than 401/403 are
                # caller errors and should fail fast.
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and 500 <= status < 600:
                    last_error = exc
                else:
                    raise EmbeddingError(
                        f"OpenRouter embeddings request failed "
                        f"(status={status}): {exc}"
                    ) from exc
            else:
                # Success: sort by .index, return the embedding lists.
                sorted_data = sorted(
                    response.data, key=lambda d: d.index
                )
                return [list(d.embedding) for d in sorted_data]

            # Sleep between attempts. Skip the final sleep when there will
            # not be another attempt.
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)

        # All retries exhausted on a transient error.
        raise EmbeddingError(
            f"OpenRouter embeddings request failed after {max_attempts} "
            f"attempts (max_retries={self.max_retries}); last error: "
            f"{last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Higher-level methods

    def embed_query(self, query: str) -> np.ndarray:
        """Encode one query string. Returns ``(D,)`` ``float32``, L2-normalized."""

        rows = self._call_embeddings_api([query])
        vec = np.asarray(rows[0], dtype=np.float32)
        return _l2_normalize(vec).astype(np.float32, copy=False)

    def embed_corpus(
        self, ordered_ids: list[str], texts: list[str]
    ) -> np.ndarray:
        """Encode ``texts`` in id order. Returns ``(N, D)`` ``float32``, row-normalized.

        Chunks ``texts`` into batches of ``self.batch_size``; calls
        :meth:`_call_embeddings_api` once per batch; concatenates the
        resulting vectors; L2-normalizes each row; returns the matrix
        aligned with ``ordered_ids``.

        A small ``time.sleep(0.05)`` is inserted between batches to
        avoid burst-rate edges on the OpenRouter free tier; the per-call
        retry loop in :meth:`_call_embeddings_api` handles the rest. The
        sleep is skipped after the final batch.
        """

        if len(ordered_ids) != len(texts):
            raise ValueError(
                f"ordered_ids length ({len(ordered_ids)}) does not match "
                f"texts length ({len(texts)})"
            )

        if not texts:
            # Empty corpus: shape (0, 0) is fine; downstream cosine/cache
            # paths handle the empty case.
            return np.zeros((0, 0), dtype=np.float32)

        rows: list[list[float]] = []
        n = len(texts)
        for batch_start in range(0, n, self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_rows = self._call_embeddings_api(batch)
            rows.extend(batch_rows)
            # Pace between batches; the final batch does not sleep.
            if batch_start + self.batch_size < n:
                time.sleep(0.05)

        matrix = np.asarray(rows, dtype=np.float32)

        # L2-normalize per row. ``np.linalg.norm(matrix, axis=1)`` returns
        # the per-row L2 norms; we broadcast-divide. Zero rows are masked
        # to 1.0 to avoid division by zero (kept as-is, so they remain
        # all-zero post-normalize and contribute nothing to cosine).
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = (matrix / norms).astype(np.float32, copy=False)

        return normalized
