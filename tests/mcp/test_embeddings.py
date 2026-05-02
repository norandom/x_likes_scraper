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
    assert emb.CACHE_SCHEMA_VERSION == 1
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
