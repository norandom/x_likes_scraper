"""Tests for :mod:`x_likes_mcp.embeddings` (task 2.1 RED phase).

Task 2.1 introduces the ``Embedder`` class with the OpenRouter HTTP seam,
the ``EmbeddingError`` exception, the ``CorpusEmbeddings`` dataclass, and
the module-level constants. This file covers ONLY task 2.1 surface area:

* Module-level constants exist with the documented values and re-exports.
* ``EmbeddingError`` is a ``RuntimeError`` subclass.
* ``CorpusEmbeddings`` is a dataclass with the documented fields.
* ``Embedder.__init__`` records configuration without constructing a client
  (lazy client construction).
* ``embed_query`` returns the documented shape and is L2-normalized.
* ``embed_corpus`` chunks per ``batch_size`` and concatenates correctly.
* ``embed_corpus`` raises on length-mismatched id/text inputs.
* Empty / ``None`` ``api_key`` raises ``EmbeddingError`` from the seam
  before any HTTP work.
* ``_call_embeddings_api`` retries transient failures up to
  ``DEFAULT_MAX_RETRIES`` times and then raises ``EmbeddingError``.
* ``AuthenticationError`` from the SDK propagates immediately as
  ``EmbeddingError`` with no retry.

``cosine_top_k`` (task 2.2), the cache I/O helpers (task 2.3), and
``open_or_build_corpus`` (task 2.4) are explicitly out of scope here.
"""

from __future__ import annotations

import dataclasses
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from x_likes_mcp import config as config_module
from x_likes_mcp import embeddings as emb


# ---------------------------------------------------------------------------
# Helpers


class _FakeRateLimitError(Exception):
    """Stand-in for ``openai.RateLimitError``.

    The real openai SDK exception requires an ``httpx.Response`` in its
    constructor, which is awkward to fake. We use issubclass detection in
    the production retry loop, so a subclass of the real exception is the
    cleanest path. We attach the subclass at test-time by patching the
    ``openai.RateLimitError`` symbol that ``embeddings.py`` imports.
    """


class _FakeAuthError(Exception):
    """Stand-in for ``openai.AuthenticationError``."""


def _patched_openai_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the openai exception classes the embedder catches.

    We rebind ``embeddings.openai.RateLimitError`` /
    ``embeddings.openai.AuthenticationError`` so the test's own throwaway
    exception classes participate in ``embeddings.py``'s try/except.
    The actual ``openai.OpenAI`` client is patched per-test as well.
    """

    monkeypatch.setattr(emb.openai, "RateLimitError", _FakeRateLimitError, raising=True)
    monkeypatch.setattr(emb.openai, "AuthenticationError", _FakeAuthError, raising=True)


def _make_embedder(api_key: str | None = "test-key", **overrides: object) -> emb.Embedder:
    """Default-shaped ``Embedder`` for tests."""

    return emb.Embedder(api_key=api_key, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module-level constants


def test_constants_exposed_with_documented_values() -> None:
    # Schema version was bumped to 2 when corpus text started including
    # tweet.urls (resolved short-link expansions) alongside tweet.text.
    assert emb.CACHE_SCHEMA_VERSION == 2
    assert emb.DEFAULT_TOP_K == 200
    assert emb.DEFAULT_BATCH_SIZE == 32
    assert emb.DEFAULT_MAX_RETRIES == 3


def test_default_base_url_re_exports_config_value() -> None:
    # The spec says embeddings.py exposes DEFAULT_BASE_URL; config.py owns the
    # literal string. They must agree.
    assert emb.DEFAULT_BASE_URL == config_module.DEFAULT_OPENROUTER_BASE_URL


def test_default_embedding_model_re_exports_config_value() -> None:
    assert emb.DEFAULT_EMBEDDING_MODEL == config_module.DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# EmbeddingError


def test_embedding_error_is_runtime_error_subclass() -> None:
    assert issubclass(emb.EmbeddingError, RuntimeError)
    err = emb.EmbeddingError("oops")
    assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# CorpusEmbeddings


def test_corpus_embeddings_is_dataclass_with_documented_fields() -> None:
    assert dataclasses.is_dataclass(emb.CorpusEmbeddings)
    fields = {f.name for f in dataclasses.fields(emb.CorpusEmbeddings)}
    assert fields == {"matrix", "ordered_ids", "model_name"}

    # Round-trip: build one and confirm fields stick.
    matrix = np.zeros((2, 3), dtype=np.float32)
    ce = emb.CorpusEmbeddings(matrix=matrix, ordered_ids=["a", "b"], model_name="x")
    assert ce.matrix is matrix
    assert ce.ordered_ids == ["a", "b"]
    assert ce.model_name == "x"


# ---------------------------------------------------------------------------
# Embedder __init__: lazy client construction


def test_init_records_config_without_constructing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch openai.OpenAI on the embeddings module's import-bound reference
    # and assert it is never called during __init__.
    fake_ctor = MagicMock()
    monkeypatch.setattr(emb.openai, "OpenAI", fake_ctor, raising=True)

    embedder = emb.Embedder(
        api_key="abc",
        base_url="http://fake/v1",
        model_name="fake-model",
        batch_size=8,
        max_retries=2,
    )

    assert fake_ctor.call_count == 0
    # Configuration is recorded for use by _call_embeddings_api.
    assert embedder.api_key == "abc"
    assert embedder.base_url == "http://fake/v1"
    assert embedder.model_name == "fake-model"
    assert embedder.batch_size == 8
    assert embedder.max_retries == 2


def test_init_uses_documented_defaults() -> None:
    embedder = emb.Embedder(api_key="abc")
    assert embedder.base_url == emb.DEFAULT_BASE_URL
    assert embedder.model_name == emb.DEFAULT_EMBEDDING_MODEL
    assert embedder.batch_size == emb.DEFAULT_BATCH_SIZE
    assert embedder.max_retries == emb.DEFAULT_MAX_RETRIES


# ---------------------------------------------------------------------------
# embed_query


def test_embed_query_returns_l2_normalized_1d_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _make_embedder()
    monkeypatch.setattr(
        embedder,
        "_call_embeddings_api",
        lambda texts: [[3.0, 0.0, 4.0]],  # length 5 vector pre-normalize
    )

    out = embedder.embed_query("anything")

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (3,)
    # 3-4-5 triangle: pre-norm length is 5, post-norm length is 1.
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-5)
    np.testing.assert_allclose(out, np.array([0.6, 0.0, 0.8], dtype=np.float32), rtol=1e-5)


def test_embed_query_passes_single_text_to_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _make_embedder()
    received: list[list[str]] = []

    def stub(texts: list[str]) -> list[list[float]]:
        received.append(list(texts))
        return [[1.0, 0.0]]

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    embedder.embed_query("hello world")

    assert received == [["hello world"]]


# ---------------------------------------------------------------------------
# embed_corpus shape and batching


def test_embed_corpus_chunks_into_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """100 inputs at batch_size=32 must hit the seam exactly 4 times."""

    embedder = _make_embedder(batch_size=32)
    call_sizes: list[int] = []

    def stub(texts: list[str]) -> list[list[float]]:
        call_sizes.append(len(texts))
        # Each row is [float(global_index), 0.0] — encode the input position
        # so we can later assert correct concatenation/order.
        # We can't know the global offset from inside the stub, so we encode
        # only the within-batch position; the test below asserts on row count.
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    # Avoid the inter-batch sleep slowing tests down.
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    ids = [f"id-{i:03d}" for i in range(100)]
    texts = [f"text-{i:03d}" for i in range(100)]

    out = embedder.embed_corpus(ids, texts)

    assert call_sizes == [32, 32, 32, 4]
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (100, 2)


def test_embed_corpus_concatenates_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row N of the matrix must correspond to id position N."""

    embedder = _make_embedder(batch_size=2)
    call_index = {"n": 0}

    def stub(texts: list[str]) -> list[list[float]]:
        # Each call returns vectors that encode the call number and
        # within-batch position so the resulting matrix has predictable rows.
        n = call_index["n"]
        call_index["n"] += 1
        return [[float(n), float(j)] for j, _ in enumerate(texts)]

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    ids = ["a", "b", "c", "d", "e"]
    texts = ["A", "B", "C", "D", "E"]

    out = embedder.embed_corpus(ids, texts)

    assert out.shape == (5, 2)
    # Each row is L2-normalized. The pre-norm raw rows would have been:
    # call 0: [(0,0), (0,1)]
    # call 1: [(1,0), (1,1)]
    # call 2: [(2,0)]
    # Verify row 0 (norm of (0,0) is zero -> stays zero, no NaN).
    # Verify row 4 normalizes (2,0) -> (1,0).
    np.testing.assert_allclose(out[4], np.array([1.0, 0.0], dtype=np.float32), rtol=1e-5)
    # Row 3 normalizes (1,1) -> (1/sqrt(2), 1/sqrt(2))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(
        out[3], np.array([inv_sqrt2, inv_sqrt2], dtype=np.float32), rtol=1e-5
    )


def test_embed_corpus_raises_on_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _make_embedder()
    monkeypatch.setattr(embedder, "_call_embeddings_api", lambda _t: [[1.0]])

    with pytest.raises(ValueError):
        embedder.embed_corpus(["a", "b"], ["only-one-text"])


# ---------------------------------------------------------------------------
# Empty api_key handling


def test_call_embeddings_api_raises_on_empty_api_key_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``None`` api_key must surface as ``EmbeddingError`` before any HTTP call.

    Importantly, this happens without ever constructing the openai client —
    so we do NOT patch ``openai.OpenAI`` here; constructing it should be
    impossible.
    """

    fake_ctor = MagicMock()
    monkeypatch.setattr(emb.openai, "OpenAI", fake_ctor, raising=True)

    embedder = emb.Embedder(api_key=None)
    with pytest.raises(emb.EmbeddingError, match="OPENROUTER_API_KEY"):
        embedder.embed_query("anything")

    assert fake_ctor.call_count == 0


def test_call_embeddings_api_raises_on_empty_api_key_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ctor = MagicMock()
    monkeypatch.setattr(emb.openai, "OpenAI", fake_ctor, raising=True)

    embedder = emb.Embedder(api_key="")
    with pytest.raises(emb.EmbeddingError, match="OPENROUTER_API_KEY"):
        embedder.embed_query("anything")

    assert fake_ctor.call_count == 0


# ---------------------------------------------------------------------------
# Retry behavior on _call_embeddings_api


def _make_response_object(rows: list[list[float]]) -> object:
    """Fake the openai-SDK response object shape: ``response.data[i].embedding``,
    ``response.data[i].index``."""

    response = MagicMock()
    response.data = [MagicMock(embedding=row, index=i) for i, row in enumerate(rows)]
    return response


def test_call_embeddings_api_retries_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_openai_module(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    # Build a fake openai.OpenAI client whose .embeddings.create raises
    # _FakeRateLimitError twice and then succeeds.
    create_calls = {"n": 0}

    def fake_create(**kwargs: object) -> object:
        create_calls["n"] += 1
        if create_calls["n"] < 3:
            raise _FakeRateLimitError("rate limited")
        return _make_response_object([[1.0, 0.0, 0.0]])

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = fake_create

    fake_ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(emb.openai, "OpenAI", fake_ctor, raising=True)

    embedder = emb.Embedder(api_key="abc", max_retries=3)
    out = embedder._call_embeddings_api(["only-one"])

    assert create_calls["n"] == 3
    assert out == [[1.0, 0.0, 0.0]]


def test_call_embeddings_api_persistent_transient_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_openai_module(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    create_calls = {"n": 0}

    def always_fail(**kwargs: object) -> object:
        create_calls["n"] += 1
        raise _FakeRateLimitError("rate limited forever")

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = always_fail
    monkeypatch.setattr(
        emb.openai, "OpenAI", MagicMock(return_value=fake_client), raising=True
    )

    embedder = emb.Embedder(api_key="abc", max_retries=3)
    with pytest.raises(emb.EmbeddingError, match=r"(?i)retr"):
        embedder._call_embeddings_api(["x"])

    # max_retries=3 means up to 4 total attempts (initial + 3 retries).
    assert create_calls["n"] == 4


def test_call_embeddings_api_auth_error_propagates_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched_openai_module(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    create_calls = {"n": 0}

    def auth_fail(**kwargs: object) -> object:
        create_calls["n"] += 1
        raise _FakeAuthError("bad key")

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = auth_fail
    monkeypatch.setattr(
        emb.openai, "OpenAI", MagicMock(return_value=fake_client), raising=True
    )

    embedder = emb.Embedder(api_key="abc", max_retries=3)
    with pytest.raises(emb.EmbeddingError):
        embedder._call_embeddings_api(["x"])

    # Auth errors must not retry. Exactly one underlying create call.
    assert create_calls["n"] == 1


def test_call_embeddings_api_sorts_response_data_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI SDK occasionally returns out-of-order ``data`` entries.
    The embedder must sort by ``index`` before returning."""

    _patched_openai_module(monkeypatch)

    response = MagicMock()
    # Construct entries deliberately out of order.
    response.data = [
        MagicMock(embedding=[2.0, 2.0], index=2),
        MagicMock(embedding=[0.0, 0.0], index=0),
        MagicMock(embedding=[1.0, 1.0], index=1),
    ]
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = response
    monkeypatch.setattr(
        emb.openai, "OpenAI", MagicMock(return_value=fake_client), raising=True
    )

    embedder = emb.Embedder(api_key="abc")
    out = embedder._call_embeddings_api(["a", "b", "c"])
    assert out == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


# ---------------------------------------------------------------------------
# cosine_top_k (task 2.2)
#
# These tests cover the new method added in task 2.2. The dense retrieval
# path computes cosine similarity as a single matrix-vector dot product on
# already L2-normalized inputs, then takes the top-k indices (optionally
# masked to a restricted id set). The test fixtures below build small,
# hand-normalized corpora so the expected ranking is mechanical.


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows are left as zero (no NaN)."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


def _make_corpus(
    raw_vectors: list[list[float]],
    ids: list[str],
    *,
    model_name: str = "test-model",
) -> emb.CorpusEmbeddings:
    """Build a ``CorpusEmbeddings`` from raw vectors, normalizing per row.

    Tests pass arbitrary integer-valued rows for clarity; this helper does
    the L2 normalization so the corpus matches the production invariant
    that ``embed_corpus`` rows are unit length.
    """

    matrix = np.asarray(raw_vectors, dtype=np.float32)
    matrix = _normalize_rows(matrix)
    return emb.CorpusEmbeddings(matrix=matrix, ordered_ids=list(ids), model_name=model_name)


def _normalize_vec(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return (arr / norm).astype(np.float32, copy=False)


def test_cosine_top_k_descending_order() -> None:
    # 5 rows in 3-dim space:
    #   row 0: aligned with query (1,0,0)        -> highest score
    #   row 1: perpendicular (0,1,0)             -> 0.0
    #   row 2: perpendicular (0,0,1)             -> 0.0
    #   row 3: anti-aligned (-1,0,0)             -> -1.0
    #   row 4: anti-aligned variant (-1,-1,0)    -> < 0
    embedder = _make_embedder()
    corpus = _make_corpus(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
        ],
        ["a", "b", "c", "d", "e"],
    )
    query = _normalize_vec([1.0, 0.0, 0.0])

    result = embedder.cosine_top_k(query, corpus, k=5)

    assert isinstance(result, list)
    assert len(result) == 5
    # Top entry is the aligned row.
    assert result[0][0] == "a"
    np.testing.assert_allclose(result[0][1], 1.0, atol=1e-5)
    # Scores are descending.
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_cosine_top_k_returns_at_most_k() -> None:
    embedder = _make_embedder()
    # 10 rows; use varied alignments with the query (1,0).
    raw = [[float(10 - i), 0.0] for i in range(10)]  # decreasing alignment
    ids = [f"id_{i}" for i in range(10)]
    corpus = _make_corpus(raw, ids)
    query = _normalize_vec([1.0, 0.0])

    result = embedder.cosine_top_k(query, corpus, k=3)

    assert len(result) == 3
    # The top three by alignment are id_0, id_1, id_2 (largest x components).
    returned_ids = [tid for tid, _ in result]
    assert returned_ids == ["id_0", "id_1", "id_2"]


def test_cosine_top_k_default_k_is_200() -> None:
    """When ``k`` is unspecified, the default top-K (200) is used.

    With a 5-row corpus and the default k=200, every row is returned
    (count is min(k, N)), and DEFAULT_TOP_K must be 200.
    """

    assert emb.DEFAULT_TOP_K == 200

    embedder = _make_embedder()
    corpus = _make_corpus(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-1.0, 0.0]],
        ["a", "b", "c", "d", "e"],
    )
    query = _normalize_vec([1.0, 0.0])

    result = embedder.cosine_top_k(query, corpus)  # k omitted -> default 200

    # Default k=200 with N=5 returns all 5 rows.
    assert len(result) == 5


def test_cosine_top_k_with_restrict_returns_only_restricted_ids() -> None:
    embedder = _make_embedder()
    # 10 rows where id_5 has the strongest alignment with query (1,0).
    raw = [[1.0, 0.0]] * 10
    # Make id_5 stand out before normalization is irrelevant — they all
    # normalize to (1,0). To break ties deterministically, vary x slightly.
    raw = [
        [1.0, 0.1],  # id_0
        [1.0, 0.2],  # id_1
        [1.0, 0.3],  # id_2
        [1.0, 0.4],  # id_3
        [1.0, 0.5],  # id_4
        [10.0, 0.0],  # id_5  -> after normalize (1,0), best alignment with (1,0)
        [1.0, 0.6],  # id_6
        [1.0, 0.7],  # id_7
        [1.0, 0.8],  # id_8
        [1.0, 0.9],  # id_9
    ]
    ids = [f"id_{i}" for i in range(10)]
    corpus = _make_corpus(raw, ids)
    query = _normalize_vec([1.0, 0.0])

    result = embedder.cosine_top_k(
        query, corpus, k=200, restrict_to_ids={"id_2", "id_8"}
    )

    assert len(result) == 2
    returned_ids = {tid for tid, _ in result}
    assert returned_ids == {"id_2", "id_8"}
    # Scores are descending.
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_cosine_top_k_with_restrict_smaller_than_k_returns_all() -> None:
    embedder = _make_embedder()
    raw = [[float(i + 1), 0.0] for i in range(10)]
    ids = [f"id_{i}" for i in range(10)]
    corpus = _make_corpus(raw, ids)
    query = _normalize_vec([1.0, 0.0])

    result = embedder.cosine_top_k(
        query, corpus, k=10, restrict_to_ids={"id_3", "id_7"}
    )

    # Restricted scope (size 2) is smaller than k (10): return all of it.
    assert len(result) == 2
    returned_ids = {tid for tid, _ in result}
    assert returned_ids == {"id_3", "id_7"}


def test_cosine_top_k_with_empty_restrict_returns_empty() -> None:
    embedder = _make_embedder()
    corpus = _make_corpus([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    query = _normalize_vec([1.0, 0.0])

    result = embedder.cosine_top_k(query, corpus, k=5, restrict_to_ids=set())

    assert result == []


def test_cosine_top_k_with_restrict_no_matches_returns_empty() -> None:
    embedder = _make_embedder()
    corpus = _make_corpus([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    query = _normalize_vec([1.0, 0.0])

    result = embedder.cosine_top_k(
        query, corpus, k=5, restrict_to_ids={"id_does_not_exist"}
    )

    assert result == []


def test_cosine_top_k_empty_corpus_returns_empty() -> None:
    embedder = _make_embedder()
    corpus = emb.CorpusEmbeddings(
        matrix=np.zeros((0, 0), dtype=np.float32),
        ordered_ids=[],
        model_name="x",
    )
    # Any query shape is fine; empty corpus short-circuits before the
    # dim-mismatch check.
    query = np.zeros((4,), dtype=np.float32)

    result = embedder.cosine_top_k(query, corpus, k=5)
    assert result == []


def test_cosine_top_k_query_dim_mismatch_raises() -> None:
    embedder = _make_embedder()
    corpus = _make_corpus(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ["a", "b"],
    )
    bad_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # shape (3,)

    with pytest.raises(ValueError) as excinfo:
        embedder.cosine_top_k(bad_query, corpus, k=2)

    msg = str(excinfo.value)
    # Error message should name both shapes for actionable diagnostics.
    assert "(3,)" in msg or "3" in msg
    assert "4" in msg


def test_cosine_top_k_invalid_k_raises() -> None:
    embedder = _make_embedder()
    corpus = _make_corpus([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    query = _normalize_vec([1.0, 0.0])

    with pytest.raises(ValueError):
        embedder.cosine_top_k(query, corpus, k=0)

    with pytest.raises(ValueError):
        embedder.cosine_top_k(query, corpus, k=-1)


# ---------------------------------------------------------------------------
# Task 2.3: on-disk cache helpers (_save_cache, _load_cache)
#
# Covers:
#   * Atomic writes via temp + rename (req 2.7).
#   * Build fails on unwritable output dir (req 2.8 — write side).
#   * .npy + .meta.json shape (req 3.1, 3.2).
#   * Schema-version mismatch forces rebuild via None return (req 3.3).
#   * Missing/unreadable file forces rebuild via None return (req 3.4).
#
# The helpers are module-level (not Embedder methods); tests reach them via
# the ``emb`` import directly.

import json  # noqa: E402  (deliberately co-located with the cache tests)
from pathlib import Path  # noqa: E402


def test_cache_filename_constants_exposed() -> None:
    # Callers (and tests) should not have to string-literal the filenames;
    # they live as module-level constants.
    assert emb.CACHE_NPY_NAME == "corpus_embeddings.npy"
    assert emb.CACHE_META_NAME == "corpus_embeddings.meta.json"


def _sample_matrix(n: int = 3, d: int = 5) -> np.ndarray:
    """Stable matrix for round-trip tests; values are easy to eyeball."""

    return np.array(
        [[float(i * d + j) for j in range(d)] for i in range(n)],
        dtype=np.float32,
    )


def test_save_cache_writes_npy_and_meta_atomically(tmp_path: Path) -> None:
    matrix = _sample_matrix(n=3, d=5)
    ordered_ids = ["a", "b", "c"]
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    emb._save_cache(cache_npy, cache_meta, matrix, ordered_ids, "model-X")

    assert cache_npy.exists()
    assert cache_meta.exists()

    # The npy file loads back to the same matrix.
    loaded_matrix = np.load(cache_npy, allow_pickle=False)
    assert np.array_equal(loaded_matrix, matrix)
    assert loaded_matrix.dtype == matrix.dtype

    # Metadata has all five required fields with the right values.
    meta = json.loads(cache_meta.read_text())
    assert meta["schema_version"] == emb.CACHE_SCHEMA_VERSION
    assert meta["model_name"] == "model-X"
    assert meta["n_tweets"] == 3
    assert meta["embedding_dim"] == 5
    assert meta["tweet_ids_in_order"] == ["a", "b", "c"]


def test_save_cache_uses_tmp_then_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _sample_matrix(n=2, d=3)
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    captured: list[tuple[Path, Path]] = []
    real_replace = emb.os.replace

    def spy_replace(src, dst):  # type: ignore[no-untyped-def]
        captured.append((Path(str(src)), Path(str(dst))))
        return real_replace(src, dst)

    monkeypatch.setattr(emb.os, "replace", spy_replace)

    emb._save_cache(cache_npy, cache_meta, matrix, ["a", "b"], "model-X")

    # Both canonical paths were the destination of an os.replace call.
    destinations = {dst for _, dst in captured}
    assert cache_npy in destinations
    assert cache_meta in destinations

    # Sources end in .tmp so atomic semantics held.
    sources = [src for src, _ in captured]
    for src in sources:
        assert str(src).endswith(".tmp"), f"non-tmp source: {src}"

    # No .tmp leftovers in the cache directory.
    leftover_tmps = list(tmp_path.glob("*.tmp"))
    assert leftover_tmps == []


def test_save_cache_unwritable_dir_raises(tmp_path: Path) -> None:
    # Path under a directory that does not exist; the .tmp write fails.
    bad_dir = tmp_path / "does_not_exist"
    cache_npy = bad_dir / emb.CACHE_NPY_NAME
    cache_meta = bad_dir / emb.CACHE_META_NAME

    matrix = _sample_matrix(n=2, d=3)

    with pytest.raises(emb.EmbeddingError) as excinfo:
        emb._save_cache(cache_npy, cache_meta, matrix, ["a", "b"], "model-X")

    msg = str(excinfo.value)
    # Error names the directory so the operator can find it.
    assert str(bad_dir) in msg or "does_not_exist" in msg


def test_load_cache_round_trip(tmp_path: Path) -> None:
    matrix = _sample_matrix(n=4, d=7)
    ordered_ids = ["a", "b", "c", "d"]
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    emb._save_cache(cache_npy, cache_meta, matrix, ordered_ids, "model-X")

    loaded = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids=set(ordered_ids),
    )

    assert loaded is not None
    assert isinstance(loaded, emb.CorpusEmbeddings)
    assert np.array_equal(loaded.matrix, matrix)
    assert loaded.ordered_ids == ordered_ids
    assert loaded.model_name == "model-X"


def test_load_cache_missing_npy_returns_none(tmp_path: Path) -> None:
    # Only the meta file exists.
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME
    cache_meta.write_text(
        json.dumps(
            {
                "schema_version": emb.CACHE_SCHEMA_VERSION,
                "model_name": "model-X",
                "n_tweets": 2,
                "embedding_dim": 3,
                "tweet_ids_in_order": ["a", "b"],
            }
        )
    )

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b"},
    )
    assert result is None


def test_load_cache_missing_meta_returns_none(tmp_path: Path) -> None:
    # Only the npy file exists.
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME
    np.save(cache_npy, _sample_matrix(n=2, d=3))

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b"},
    )
    assert result is None


def test_load_cache_corrupt_meta_returns_none(tmp_path: Path) -> None:
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME
    np.save(cache_npy, _sample_matrix(n=2, d=3))
    cache_meta.write_text("not json {{{")

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b"},
    )
    assert result is None


def test_load_cache_corrupt_npy_returns_none(tmp_path: Path) -> None:
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME
    cache_npy.write_bytes(b"not a real npy file")
    cache_meta.write_text(
        json.dumps(
            {
                "schema_version": emb.CACHE_SCHEMA_VERSION,
                "model_name": "model-X",
                "n_tweets": 2,
                "embedding_dim": 3,
                "tweet_ids_in_order": ["a", "b"],
            }
        )
    )

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b"},
    )
    assert result is None


def test_load_cache_schema_version_mismatch_returns_none(tmp_path: Path) -> None:
    matrix = _sample_matrix(n=2, d=3)
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    emb._save_cache(cache_npy, cache_meta, matrix, ["a", "b"], "model-X")

    # Patch the schema_version field to something the loader does not expect.
    meta = json.loads(cache_meta.read_text())
    meta["schema_version"] = 999
    cache_meta.write_text(json.dumps(meta))

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b"},
        expected_schema_version=emb.CACHE_SCHEMA_VERSION,
    )
    assert result is None


def test_load_cache_model_name_mismatch_returns_none(tmp_path: Path) -> None:
    matrix = _sample_matrix(n=2, d=3)
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    emb._save_cache(cache_npy, cache_meta, matrix, ["a", "b"], "model-A")

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-B",
        expected_ids={"a", "b"},
    )
    assert result is None


def test_load_cache_id_set_mismatch_returns_none(tmp_path: Path) -> None:
    matrix = _sample_matrix(n=3, d=4)
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    emb._save_cache(cache_npy, cache_meta, matrix, ["a", "b", "c"], "model-X")

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b", "d"},  # "d" not in cache, "c" missing from expected
    )
    assert result is None


def test_load_cache_shape_mismatch_returns_none(tmp_path: Path) -> None:
    # Hand-craft a cache where the meta says (5, 4) but the matrix is (5, 3).
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    matrix = np.zeros((5, 3), dtype=np.float32)
    np.save(cache_npy, matrix)

    cache_meta.write_text(
        json.dumps(
            {
                "schema_version": emb.CACHE_SCHEMA_VERSION,
                "model_name": "model-X",
                "n_tweets": 5,
                "embedding_dim": 4,
                "tweet_ids_in_order": ["a", "b", "c", "d", "e"],
            }
        )
    )

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b", "c", "d", "e"},
    )
    assert result is None


def test_load_cache_extra_id_in_meta_returns_none(tmp_path: Path) -> None:
    # Save a (2, D) cache, then patch the meta to claim three tweet ids.
    # The matrix shape no longer matches n_tweets, so the loader returns None.
    matrix = _sample_matrix(n=2, d=3)
    cache_npy = tmp_path / emb.CACHE_NPY_NAME
    cache_meta = tmp_path / emb.CACHE_META_NAME

    emb._save_cache(cache_npy, cache_meta, matrix, ["a", "b"], "model-X")

    meta = json.loads(cache_meta.read_text())
    meta["tweet_ids_in_order"] = ["a", "b", "c"]
    meta["n_tweets"] = 3
    cache_meta.write_text(json.dumps(meta))

    result = emb._load_cache(
        cache_npy,
        cache_meta,
        expected_model="model-X",
        expected_ids={"a", "b", "c"},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Task 2.4: open_or_build_corpus orchestrator
#
# Covers:
#   * Cold call (no cache present) builds via embed_corpus and persists
#     the cache files.
#   * Warm call (cache present and matching) skips _call_embeddings_api.
#   * Model-name change forces a rebuild.
#   * Tweet-id-set change forces a rebuild.
#   * Empty tweets_by_id returns an empty (0, 0) corpus and does not
#     write the cache.
#   * Tweets with .text == None pass "" to embed_corpus.
#   * After a build, _load_cache directly recovers a CorpusEmbeddings.
#
# Synthetic tweet objects are duck-typed: any object with a ``.text``
# attribute works (the orchestrator only reads ``tweets_by_id[i].text``).

import types  # noqa: E402  (co-located with the orchestrator tests)


def _fake_tweet(text: str | None) -> types.SimpleNamespace:
    """Minimal stand-in for a Tweet: only the ``.text`` attribute is used."""

    return types.SimpleNamespace(text=text)


def _fake_tweets(n: int, prefix: str = "id") -> dict[str, types.SimpleNamespace]:
    """Build a dict of synthetic tweets with predictable ids and texts."""

    return {f"{prefix}_{i}": _fake_tweet(f"text {i}") for i in range(n)}


def _canned_vectors_factory(dim: int = 4):
    """Return a stub for ``_call_embeddings_api`` that emits unit-length rows.

    Each row is dimension-``dim``; the value is derived from the input
    text's hash so distinct inputs land on distinct vectors. Exact values
    do not matter for orchestrator tests — what matters is shape + that
    the matrix can round-trip through the cache.
    """

    def stub(texts: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for text in texts:
            # Deterministic per-text vector: place 1.0 at one of D positions.
            # Empty strings are allowed (they map to position 0).
            idx = (hash(text) % dim) if text else 0
            row = [0.0] * dim
            row[idx] = 1.0
            rows.append(row)
        return rows

    return stub


def test_open_or_build_corpus_cold_calls_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _make_embedder(model_name="test-model")
    tweets_by_id = _fake_tweets(5)

    counter = {"n": 0}
    base_stub = _canned_vectors_factory(dim=4)

    def stub(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        return base_stub(texts)

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    result = emb.open_or_build_corpus(embedder, tweets_by_id, tmp_path)

    assert isinstance(result, emb.CorpusEmbeddings)
    assert result.matrix.shape == (5, 4)
    assert result.ordered_ids == sorted(tweets_by_id.keys())
    assert result.model_name == "test-model"
    assert counter["n"] >= 1

    # Cache files now exist on disk.
    assert (tmp_path / emb.CACHE_NPY_NAME).exists()
    assert (tmp_path / emb.CACHE_META_NAME).exists()


def test_open_or_build_corpus_warm_skips_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _make_embedder(model_name="test-model")
    tweets_by_id = _fake_tweets(5)

    counter = {"n": 0}
    base_stub = _canned_vectors_factory(dim=4)

    def stub(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        return base_stub(texts)

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    # First (cold) call builds + persists.
    first = emb.open_or_build_corpus(embedder, tweets_by_id, tmp_path)
    cold_calls = counter["n"]
    assert cold_calls >= 1

    # Reset counter and re-open against the same cache_dir + tweets.
    counter["n"] = 0
    second = emb.open_or_build_corpus(embedder, tweets_by_id, tmp_path)

    assert counter["n"] == 0  # cache hit; no encoding work
    assert second.ordered_ids == first.ordered_ids
    assert second.model_name == first.model_name
    assert np.array_equal(second.matrix, first.matrix)


def test_open_or_build_corpus_model_change_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tweets_by_id = _fake_tweets(5)

    counter = {"n": 0}
    base_stub = _canned_vectors_factory(dim=4)

    def stub(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        return base_stub(texts)

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    embedder1 = emb.Embedder(api_key="x", model_name="model-a")
    monkeypatch.setattr(embedder1, "_call_embeddings_api", stub)
    emb.open_or_build_corpus(embedder1, tweets_by_id, tmp_path)
    initial_calls = counter["n"]
    assert initial_calls >= 1

    # Now switch model name; the cache should invalidate and rebuild.
    counter["n"] = 0
    embedder2 = emb.Embedder(api_key="x", model_name="model-b")
    monkeypatch.setattr(embedder2, "_call_embeddings_api", stub)
    result = emb.open_or_build_corpus(embedder2, tweets_by_id, tmp_path)

    assert counter["n"] >= 1, "model change must force a rebuild"
    assert result.model_name == "model-b"

    # Meta file on disk now records the new model name.
    meta = json.loads((tmp_path / emb.CACHE_META_NAME).read_text())
    assert meta["model_name"] == "model-b"


def test_open_or_build_corpus_id_change_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _make_embedder(model_name="test-model")

    counter = {"n": 0}
    base_stub = _canned_vectors_factory(dim=4)

    def stub(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        return base_stub(texts)

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    # Initial build with 5 tweets.
    initial_tweets = _fake_tweets(5, prefix="id")
    emb.open_or_build_corpus(embedder, initial_tweets, tmp_path)
    assert counter["n"] >= 1

    # Reset and rebuild with one extra tweet (id_5). The id-set differs, so
    # the cache invalidates.
    counter["n"] = 0
    expanded_tweets = _fake_tweets(6, prefix="id")  # id_0 .. id_5
    result = emb.open_or_build_corpus(embedder, expanded_tweets, tmp_path)

    assert counter["n"] >= 1, "id-set change must force a rebuild"
    assert result.ordered_ids == sorted(expanded_tweets.keys())
    assert result.matrix.shape[0] == 6


def test_open_or_build_corpus_empty_tweets_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _make_embedder(model_name="test-model")

    counter = {"n": 0}

    def stub(texts: list[str]) -> list[list[float]]:
        counter["n"] += 1
        return [[0.0]] * len(texts)

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)

    result = emb.open_or_build_corpus(embedder, {}, tmp_path)

    assert counter["n"] == 0, "empty corpus must not call the embeddings API"
    assert isinstance(result, emb.CorpusEmbeddings)
    assert result.matrix.shape == (0, 0)
    assert result.ordered_ids == []
    assert result.model_name == "test-model"

    # No cache files were written for the empty case.
    assert not (tmp_path / emb.CACHE_NPY_NAME).exists()
    assert not (tmp_path / emb.CACHE_META_NAME).exists()


def test_open_or_build_corpus_uses_text_or_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tweets with ``text=None`` must pass ``""`` to ``embed_corpus``.

    The orchestrator builds ``texts = [tweets_by_id[i].text or "" for i ...]``.
    Verifying the substitution is happening means asserting the stub
    received ``""`` for the None entries (and never raised on a None).
    """

    embedder = _make_embedder(model_name="test-model", batch_size=8)

    received_texts: list[str] = []

    def stub(texts: list[str]) -> list[list[float]]:
        received_texts.extend(texts)
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    tweets_by_id = {
        "a": _fake_tweet("hello"),
        "b": _fake_tweet(None),
        "c": _fake_tweet("world"),
        "d": _fake_tweet(None),
    }

    result = emb.open_or_build_corpus(embedder, tweets_by_id, tmp_path)

    # Ordered by sorted ids: a, b, c, d. None -> "" by the orchestrator,
    # then "" -> "[empty tweet]" by embed_corpus' empty-string coercion
    # (some embedding endpoints reject empty strings).
    assert received_texts == ["hello", "[empty tweet]", "world", "[empty tweet]"]
    assert result.matrix.shape == (4, 4)
    assert result.ordered_ids == ["a", "b", "c", "d"]


def test_open_or_build_corpus_persists_after_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a cold build, ``_load_cache`` directly recovers the corpus."""

    embedder = _make_embedder(model_name="persist-model")
    tweets_by_id = _fake_tweets(3)

    base_stub = _canned_vectors_factory(dim=4)
    monkeypatch.setattr(embedder, "_call_embeddings_api", base_stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    built = emb.open_or_build_corpus(embedder, tweets_by_id, tmp_path)

    loaded = emb._load_cache(
        tmp_path / emb.CACHE_NPY_NAME,
        tmp_path / emb.CACHE_META_NAME,
        expected_model="persist-model",
        expected_ids=set(tweets_by_id.keys()),
    )

    assert loaded is not None
    assert loaded.ordered_ids == built.ordered_ids
    assert loaded.model_name == "persist-model"
    assert np.array_equal(loaded.matrix, built.matrix)


def test_embed_corpus_substitutes_placeholder_for_empty_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input strings must reach the seam as ``[empty tweet]``.

    Some embedding endpoints reject (or silently empty-out) batches
    containing empty strings; the coercion in ``embed_corpus`` keeps
    every input non-empty without breaking row alignment. Tweets with
    no text still get a stable vector; downstream cosine just scores
    them lower.
    """

    embedder = _make_embedder()

    received: list[list[str]] = []

    def stub(texts: list[str]) -> list[list[float]]:
        received.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(embedder, "_call_embeddings_api", stub)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    embedder.embed_corpus(
        ordered_ids=["a", "b", "c"],
        texts=["real text", "", "another"],
    )

    assert received == [["real text", emb._EMPTY_TEXT_PLACEHOLDER, "another"]]


def test_call_embeddings_api_wraps_value_error_as_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI SDK raises ``ValueError("No embedding data received")``
    when the upstream returns 200 with empty ``data``. Surface this as
    an ``EmbeddingError`` naming the model rather than letting the raw
    SDK error escape."""

    embedder = _make_embedder(model_name="some/model")

    class _StubClient:
        class embeddings:
            @staticmethod
            def create(model: str, input: list[str]):
                raise ValueError("No embedding data received")

    embedder._client = _StubClient()
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    with pytest.raises(emb.EmbeddingError) as excinfo:
        embedder._call_embeddings_api(["one", "two"])

    msg = str(excinfo.value)
    assert "some/model" in msg
    assert "empty" in msg.lower()
    assert "openai/text-embedding-3-small" in msg
