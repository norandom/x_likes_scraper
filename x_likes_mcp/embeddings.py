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

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openai

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OPENROUTER_BASE_URL

if TYPE_CHECKING:
    from x_likes_exporter import Tweet


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

# Canonical filenames for the on-disk corpus cache. Callers (notably
# ``open_or_build_corpus`` in task 2.4 and the index integration in task
# 3.1) reference these constants instead of string-literaling the names so
# a future rename only touches one site.
CACHE_NPY_NAME: str = "corpus_embeddings.npy"
CACHE_META_NAME: str = "corpus_embeddings.meta.json"

# Used in :meth:`Embedder.embed_corpus` to substitute for empty/None
# tweet text. Some embedding endpoints reject (or silently empty-out)
# batches containing empty strings; this placeholder keeps every input
# non-empty without breaking row alignment.
_EMPTY_TEXT_PLACEHOLDER: str = "[empty tweet]"

# Re-exported so callers can ``from x_likes_mcp.embeddings import
# DEFAULT_EMBEDDING_MODEL`` (the spec lists it in the embeddings.py
# constants block).
__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CACHE_NPY_NAME",
    "CACHE_META_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_TOP_K",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_RETRIES",
    "EmbeddingError",
    "CorpusEmbeddings",
    "Embedder",
    "open_or_build_corpus",
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
            except ValueError as exc:
                # The OpenAI SDK's response parser raises ValueError when
                # the upstream returns 200 with empty `data`. This is a
                # provider-side payload-shape problem (common with flaky
                # free-tier endpoints) and is not retryable. Surface it
                # with a clear, actionable message.
                raise EmbeddingError(
                    f"OpenRouter returned an empty embeddings payload for "
                    f"model={self.model_name!r} (batch size {len(texts)}): "
                    f"{exc}. Try a different model (e.g. "
                    f"openai/text-embedding-3-small)."
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

        # Some embedding endpoints reject (or silently empty-out) batches
        # that contain empty strings. Substitute a benign placeholder so
        # every input is non-empty while keeping row alignment intact.
        # Tweets that genuinely have no text still get a stable vector;
        # the cosine score is just lower than for tweets with real text.
        safe_texts = [t if t else _EMPTY_TEXT_PLACEHOLDER for t in texts]

        rows: list[list[float]] = []
        n = len(safe_texts)
        for batch_start in range(0, n, self.batch_size):
            batch = safe_texts[batch_start : batch_start + self.batch_size]
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

    # ------------------------------------------------------------------
    # Cosine top-K

    def cosine_top_k(
        self,
        query_vec: np.ndarray,
        corpus: CorpusEmbeddings,
        k: int = DEFAULT_TOP_K,
        restrict_to_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(tweet_id, cosine_similarity)`` pairs, descending.

        Both ``query_vec`` and ``corpus.matrix`` are expected to be
        L2-normalized (which :meth:`embed_query` and :meth:`embed_corpus`
        produce). Cosine similarity reduces to a single matrix-vector dot
        product on normalized vectors:

            scores = corpus.matrix @ query_vec  # shape (N,)

        When ``restrict_to_ids`` is ``None`` the top-k is taken over the
        whole matrix. When it is provided, the candidates are first
        gathered to the rows whose ``ordered_ids[i]`` is in the set; if
        the restricted scope is smaller than ``k``, every restricted
        candidate is returned (sorted by score, descending).

        Edge cases:
            * Empty corpus (``matrix.shape == (0, 0)`` or ``ordered_ids ==
              []``): return ``[]``.
            * ``query_vec`` shape mismatch with ``corpus.matrix.shape[1]``:
              raise :class:`ValueError` naming both shapes.
            * ``restrict_to_ids`` is the empty set: return ``[]`` (empty
              restriction means "no candidates", distinct from ``None``).
            * ``restrict_to_ids`` has no overlap with the corpus: return
              ``[]``.
            * ``k <= 0``: raise :class:`ValueError`.
        """

        if k <= 0:
            raise ValueError(f"k must be positive (got k={k})")

        # Empty restriction is distinct from None: it means "no candidates".
        if restrict_to_ids is not None and not restrict_to_ids:
            return []

        # Empty corpus short-circuits before the dim-mismatch check so that
        # callers building a placeholder (0, 0) corpus never see a spurious
        # ValueError on a query that would otherwise be valid.
        if corpus.matrix.size == 0 or len(corpus.ordered_ids) == 0:
            return []

        expected_dim = corpus.matrix.shape[1]
        if query_vec.shape != (expected_dim,):
            raise ValueError(
                f"query_vec shape {tuple(query_vec.shape)} does not match "
                f"corpus.matrix.shape[1] ({expected_dim},); "
                f"corpus.matrix.shape={tuple(corpus.matrix.shape)}"
            )

        # Cosine on L2-normalized inputs is a dot product.
        scores = corpus.matrix @ query_vec  # shape (N,)

        if restrict_to_ids is None:
            # Whole-corpus top-k.
            n = scores.shape[0]
            top_count = min(k, n)
            if top_count == n:
                # Take everything; argpartition with kth = n-1 is wasted work.
                candidate_indices = np.arange(n)
            else:
                # argpartition gives an unsorted top_count; sort that slice.
                candidate_indices = np.argpartition(-scores, kth=top_count - 1)[
                    :top_count
                ]
            # Sort the selected indices by score descending.
            sorted_local = np.argsort(-scores[candidate_indices])
            selected = candidate_indices[sorted_local]
        else:
            # Restricted-scope top-k. Gather the row indices whose id is in
            # the restriction set first; then top-k within that scope.
            restricted_indices = np.array(
                [
                    i
                    for i, tid in enumerate(corpus.ordered_ids)
                    if tid in restrict_to_ids
                ],
                dtype=np.int64,
            )
            if restricted_indices.size == 0:
                return []

            restricted_scores = scores[restricted_indices]

            if restricted_indices.size <= k:
                # Smaller-than-k restricted scope: return all of it, sorted.
                sorted_local = np.argsort(-restricted_scores)
                selected = restricted_indices[sorted_local]
            else:
                top_count = k
                local_top = np.argpartition(
                    -restricted_scores, kth=top_count - 1
                )[:top_count]
                local_sorted = local_top[np.argsort(-restricted_scores[local_top])]
                selected = restricted_indices[local_sorted]

        return [
            (corpus.ordered_ids[int(i)], float(scores[int(i)])) for i in selected
        ]


# ---------------------------------------------------------------------------
# On-disk cache helpers (task 2.3)
#
# These are module-level functions, not methods on ``Embedder``. The
# ``open_or_build_corpus`` orchestrator (task 2.4) calls them; ``Embedder``
# itself stays focused on the HTTP seam and on the cosine math.
#
# Format:
#   * ``corpus_embeddings.npy`` — numpy float32 matrix of shape ``(N, D)``,
#     L2-normalized rows, written via ``np.lib.format.write_array`` so the
#     ``.tmp`` filename is honored exactly (``np.save`` would otherwise
#     append an extra ``.npy`` suffix to a non-``.npy`` path).
#   * ``corpus_embeddings.meta.json`` — JSON sidecar with the documented
#     schema fields (``schema_version``, ``model_name``, ``n_tweets``,
#     ``embedding_dim``, ``tweet_ids_in_order``).
#
# Atomicity: each file is first written to ``<path>.tmp``, then promoted
# with ``os.replace``. ``os.replace`` is atomic on POSIX and on Windows
# (when both paths sit on the same filesystem), so a crash mid-write
# leaves either the previous cache file intact or no file at all — never a
# half-written one. We do NOT replace until both ``.tmp`` files have been
# written successfully, but the two replacements themselves are sequential
# (a crash between them can leave one file new and one file old; the
# loader's invalidation checks (id-set + schema) catch that case and force
# a rebuild).


def _save_cache(
    cache_npy: Path,
    cache_meta: Path,
    matrix: np.ndarray,
    ordered_ids: list[str],
    model_name: str,
) -> None:
    """Write the corpus matrix and metadata sidecar atomically.

    The matrix lands at ``cache_npy`` and the metadata JSON at
    ``cache_meta``. Both writes go through ``<path>.tmp`` companions and
    are then promoted with :func:`os.replace`. On any I/O failure the
    function raises :class:`EmbeddingError` naming the parent directory,
    and tries to clean up any ``.tmp`` artefacts it managed to create.

    The metadata schema:

    .. code-block:: json

        {
          "schema_version": <CACHE_SCHEMA_VERSION>,
          "model_name": "<str>",
          "n_tweets": <int>,
          "embedding_dim": <int = matrix.shape[1]>,
          "tweet_ids_in_order": [<id>, ...]
        }

    ``embedding_dim`` is read from ``matrix.shape[1]``; an empty corpus
    (``matrix.shape == (0, 0)``) records ``embedding_dim=0``. Callers that
    care about an empty corpus should short-circuit before calling this
    helper (the orchestrator in task 2.4 does that).
    """

    tmp_npy = cache_npy.with_suffix(cache_npy.suffix + ".tmp")
    tmp_meta = cache_meta.with_suffix(cache_meta.suffix + ".tmp")

    # ``embedding_dim`` is matrix.shape[1] for any 2-D matrix (including a
    # ``(0, D)`` corpus). For a defensively-shaped non-2-D input (which
    # callers should not hand us, but which the orchestrator's empty case
    # ``np.zeros((0, 0))`` already covers) we record 0 so the meta file
    # remains internally consistent.
    embedding_dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "n_tweets": len(ordered_ids),
        "embedding_dim": embedding_dim,
        "tweet_ids_in_order": list(ordered_ids),
    }

    try:
        # Write the matrix to <cache_npy>.tmp. We use the lower-level
        # ``np.lib.format.write_array`` against an open file handle so the
        # filename is honored exactly. ``np.save(path, ...)`` would
        # otherwise append ".npy" to any non-".npy" path, leaving a
        # surprise ``<...>.tmp.npy`` file behind.
        with open(tmp_npy, "wb") as fh:
            np.lib.format.write_array(fh, matrix, allow_pickle=False)

        # Write metadata as pretty JSON so the sidecar is human-inspectable.
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)

        # Promote both files atomically. If the first replace succeeds and
        # the second fails, the loader's invalidation checks (id-set +
        # schema) will reject the half-promoted cache and trigger a
        # rebuild on the next start.
        os.replace(tmp_npy, cache_npy)
        os.replace(tmp_meta, cache_meta)
    except OSError as exc:
        # Clean up any tmp files we managed to create. ``missing_ok=True``
        # skips files that were never written (or were already promoted).
        for tmp in (tmp_npy, tmp_meta):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                # Cleanup is best-effort; we are already in the error path.
                pass
        raise EmbeddingError(
            f"Failed to write embedding cache to {cache_npy.parent}: {exc}"
        ) from exc


def _load_cache(
    cache_npy: Path,
    cache_meta: Path,
    expected_model: str,
    expected_ids: set[str],
    expected_schema_version: int = CACHE_SCHEMA_VERSION,
) -> CorpusEmbeddings | None:
    """Load the cached corpus if every invalidation check passes.

    Returns a :class:`CorpusEmbeddings` only when:

    * Both files exist and parse cleanly.
    * ``meta["schema_version"] == expected_schema_version``.
    * ``meta["model_name"] == expected_model``.
    * ``set(meta["tweet_ids_in_order"]) == expected_ids``.
    * ``matrix.shape == (len(meta["tweet_ids_in_order"]), meta["embedding_dim"])``.

    Returns ``None`` when ANY of those checks fails. Missing files,
    malformed JSON, corrupt npy data, schema mismatch, model mismatch,
    id-set mismatch, and shape mismatch all map to ``None`` — they all
    force a rebuild via the same code path.

    This helper does not raise on the "rebuild is needed" cases. The only
    exceptions that propagate out are unexpected I/O errors that are not
    "file is missing" (e.g. a permission denied on a file that does
    exist), which surface as :class:`EmbeddingError` so real I/O problems
    are not silently masked as cache misses.
    """

    # Existence check up front. This avoids relying on FileNotFoundError
    # round-trips deeper in the pipeline.
    if not cache_npy.exists() or not cache_meta.exists():
        return None

    # Parse metadata. JSON / encoding errors -> rebuild.
    try:
        with open(cache_meta, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        # Race between the existence check and the open: treat as missing.
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    except OSError as exc:
        # Real I/O problem (e.g. permission denied). Surface it.
        raise EmbeddingError(
            f"Failed to read embedding cache metadata at {cache_meta}: {exc}"
        ) from exc

    # Schema version. Missing field or non-int counts as "absent" -> rebuild.
    schema_version = meta.get("schema_version")
    if not isinstance(schema_version, int) or schema_version != expected_schema_version:
        return None

    # Model name. Mismatch (or missing) -> rebuild.
    cached_model = meta.get("model_name")
    if not isinstance(cached_model, str) or cached_model != expected_model:
        return None

    # Tweet ids in order. Validate type, then compare as sets.
    tweet_ids = meta.get("tweet_ids_in_order")
    if not isinstance(tweet_ids, list) or not all(isinstance(t, str) for t in tweet_ids):
        return None
    if set(tweet_ids) != expected_ids:
        return None

    # Embedding dim must be present and an int for the shape check.
    embedding_dim = meta.get("embedding_dim")
    if not isinstance(embedding_dim, int):
        return None

    # Load the matrix. Corrupt files raise ValueError or OSError; both -> None.
    try:
        matrix = np.load(cache_npy, allow_pickle=False)
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        # ``np.load`` raises ValueError on a malformed npy header and
        # OSError on truncated reads. Both are recoverable via rebuild.
        return None

    # Shape sanity. The metadata records (n_tweets, embedding_dim); the
    # matrix must agree exactly.
    expected_shape = (len(tweet_ids), embedding_dim)
    if matrix.shape != expected_shape:
        return None

    return CorpusEmbeddings(
        matrix=matrix,
        ordered_ids=list(tweet_ids),
        model_name=expected_model,
    )


# ---------------------------------------------------------------------------
# Orchestrator (task 2.4)
#
# ``open_or_build_corpus`` is the single entry point the index layer calls
# at startup. It tries the on-disk cache first; on any miss (model change,
# id-set change, schema bump, missing/corrupt files) it rebuilds the matrix
# via ``embedder.embed_corpus`` and persists the result. The function is
# module-level (not a method on ``Embedder``) so the embedder stays focused
# on the HTTP seam and the cosine math; the orchestration knot lives here.


def open_or_build_corpus(
    embedder: Embedder,
    tweets_by_id: dict[str, "Tweet"],
    cache_dir: Path,
) -> CorpusEmbeddings:
    """Load the corpus cache when valid; otherwise rebuild and save.

    The function is duck-typed on the tweet objects: only ``.text`` is read
    at runtime, so any object with that attribute is acceptable. The
    ``Tweet`` annotation is a ``TYPE_CHECKING`` import to avoid pulling
    ``x_likes_exporter`` into the embeddings module's import graph.

    Algorithm:
        1. Locate the on-disk cache files under ``cache_dir`` using the
           module-level constants (``CACHE_NPY_NAME``, ``CACHE_META_NAME``).
        2. Compute ``expected_ids = set(tweets_by_id.keys())``.
        3. Try ``_load_cache`` with the embedder's model name + the
           expected id set + the current schema version. If it returns a
           :class:`CorpusEmbeddings`, return it (cache hit).
        4. Otherwise build fresh: ``ordered_ids = sorted(tweets_by_id)``,
           ``texts = [tweets_by_id[i].text or "" for i in ordered_ids]``,
           call ``embedder.embed_corpus(ordered_ids, texts)``, persist via
           ``_save_cache``, and return a new :class:`CorpusEmbeddings`.
        5. Empty ``tweets_by_id`` short-circuits: returns an empty
           ``CorpusEmbeddings`` with a ``(0, 0)`` matrix and does NOT write
           the cache. The index layer is responsible for not invoking the
           dense path on an empty corpus.

    Args:
        embedder: An :class:`Embedder` configured with the target model
            and (eventually) a real API key. Only ``model_name`` and
            ``embed_corpus`` are used here.
        tweets_by_id: Mapping ``tweet_id -> Tweet``. Tweets are
            duck-typed: each value must have a ``.text`` attribute (which
            may be ``None``; a ``None`` text is encoded as ``""``).
        cache_dir: Directory under which the two cache files live. The
            caller is responsible for ensuring the directory exists; we
            do not create it (a missing directory will surface as
            :class:`EmbeddingError` from ``_save_cache``).

    Returns:
        A :class:`CorpusEmbeddings` whose ``ordered_ids`` is
        ``sorted(tweets_by_id.keys())`` (or ``[]`` on the empty path) and
        whose ``model_name`` matches ``embedder.model_name``.

    Raises:
        EmbeddingError: Propagated from ``embedder.embed_corpus`` (e.g.
            missing API key, persistent rate limits) or from
            ``_save_cache`` (e.g. unwritable cache directory).
    """

    cache_npy = cache_dir / CACHE_NPY_NAME
    cache_meta = cache_dir / CACHE_META_NAME

    # Empty corpus: skip cache I/O entirely. Returning a (0, 0) matrix
    # gives downstream code a sentinel without a real model dimension.
    if not tweets_by_id:
        return CorpusEmbeddings(
            matrix=np.zeros((0, 0), dtype=np.float32),
            ordered_ids=[],
            model_name=embedder.model_name,
        )

    expected_ids = set(tweets_by_id.keys())

    cached = _load_cache(
        cache_npy,
        cache_meta,
        expected_model=embedder.model_name,
        expected_ids=expected_ids,
        expected_schema_version=CACHE_SCHEMA_VERSION,
    )
    if cached is not None:
        return cached

    # Cache miss: rebuild. ``sorted`` gives the stable ordering the design
    # requires so cache hits across runs do not flake on dict ordering.
    ordered_ids = sorted(tweets_by_id.keys())
    texts = [tweets_by_id[i].text or "" for i in ordered_ids]

    matrix = embedder.embed_corpus(ordered_ids, texts)

    _save_cache(cache_npy, cache_meta, matrix, ordered_ids, embedder.model_name)

    return CorpusEmbeddings(
        matrix=matrix,
        ordered_ids=ordered_ids,
        model_name=embedder.model_name,
    )
