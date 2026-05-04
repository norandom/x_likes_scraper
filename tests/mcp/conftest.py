"""Test infrastructure for the ``x_likes_mcp`` package.

This conftest layers MCP-specific guards on top of the session-level guards
declared in ``tests/conftest.py``:

1. :class:`RealLLMCallAttempted` + the autouse ``_block_real_llm`` fixture
   monkeypatches :func:`x_likes_mcp.walker._call_chat_completions` so any
   walker code path that has not been explicitly mocked fails loudly. The
   parent conftest already blocks raw HTTP via the ``responses`` library;
   this is an additional, more specific layer aimed at the OpenAI SDK call
   site (which would otherwise reach the network through ``httpx`` rather
   than ``requests``).

2. The autouse ``_no_cookies_access`` fixture documents and re-asserts the
   "no real cookies.json reads" invariant for tests in this subtree. The
   actual disk-read block is the parent conftest's ``_no_real_cookies``
   fixture (which patches ``CookieManager._load_cookies``); this fixture
   simply records that constraint so an incidental ``Path('cookies.json')``
   read inside an MCP test would still be visible at review time.

3. The :func:`fake_export` fixture copies ``tests/mcp/fixtures/`` into a
   fresh ``tmp_path`` directory (so individual tests can mutate the export
   layout without polluting the on-disk fixtures) and returns a populated
   :class:`x_likes_mcp.config.Config` pointing at that copy. The config
   uses ``OPENAI_BASE_URL="http://fake/v1"`` and
   ``OPENAI_MODEL="fake-model"`` so any accidental SDK construction lands
   on a non-routable host rather than the real OpenAI API.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from x_likes_mcp.config import Config, load_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class RealLLMCallAttempted(AssertionError):
    """Raised when an unmocked walker LLM call escapes test isolation.

    Tests that legitimately exercise the walker must monkeypatch
    :func:`x_likes_mcp.walker._call_chat_completions` (or
    :func:`x_likes_mcp.walker.walk` directly) with a stub. Anything that
    reaches the real chat-completions call site will surface this error
    rather than emit a real HTTP request.
    """


@pytest.fixture(autouse=True)
def _block_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the walker's chat-completions seam to fail loudly.

    Each test in ``tests/mcp/`` starts with the seam wired to raise
    :class:`RealLLMCallAttempted`. Tests that need a working stub
    re-monkeypatch the same attribute with their own fake.
    """

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise RealLLMCallAttempted(
            "x_likes_mcp.walker._call_chat_completions was invoked without "
            "a test-supplied stub; mock walker.walk or "
            "walker._call_chat_completions explicitly in the test."
        )

    monkeypatch.setattr(
        "x_likes_mcp.walker._call_chat_completions",
        _raise,
        raising=True,
    )


@pytest.fixture(autouse=True)
def _block_real_embeddings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-stub the OpenRouter embeddings seam for every MCP test.

    ``TweetIndex.open_or_build`` (task 3.1 onward) constructs an
    :class:`x_likes_mcp.embeddings.Embedder` and embeds the corpus at
    cold-start. The fixture-backed :class:`Config` does not set
    ``OPENROUTER_API_KEY``, so the real ``_call_embeddings_api`` would
    raise :class:`EmbeddingError` before any test logic runs.

    The stub returns a deterministic 4-dim canned vector per text. Tests
    that need a different shape, a counter, or an error injection
    re-patch the same attribute.

    ``test_embeddings.py`` exercises ``_call_embeddings_api`` itself
    (api-key guard, retry loop, response sorting); skipping the autouse
    stub for that module lets those tests target the real implementation.
    """

    if request.node.fspath.basename == "test_embeddings.py":
        return

    def _fake(_self: object, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "x_likes_mcp.embeddings.Embedder._call_embeddings_api",
        _fake,
        raising=True,
    )


@pytest.fixture(autouse=True)
def _no_cookies_access() -> None:
    """Document the no-cookies-access invariant for the MCP test subtree.

    The actual block lives in the session-level ``_no_real_cookies`` fixture
    declared in ``tests/conftest.py``. This fixture is intentionally a
    no-op: its presence makes the constraint visible at this level and
    serves as an attachment point if a future MCP-specific check is added
    (e.g. asserting that no test in this subtree even references the file).
    """

    yield


@pytest.fixture
def fake_export(tmp_path: Path) -> Config:
    """Materialise a self-contained export tree in ``tmp_path``.

    Copies ``tests/mcp/fixtures/by_month/`` and
    ``tests/mcp/fixtures/likes.json`` into ``tmp_path/output/`` and returns
    a :class:`Config` whose :attr:`output_dir` points at that copy. The
    OpenAI-related fields are wired to a non-routable fake host so any
    accidental SDK call inside a test cannot reach the real API.
    """

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Copy the per-month markdown tree.
    shutil.copytree(FIXTURES_DIR / "by_month", output_dir / "by_month")

    # Copy the likes.json companion.
    shutil.copy2(FIXTURES_DIR / "likes.json", output_dir / "likes.json")

    return load_config(
        env={
            "OPENAI_BASE_URL": "http://fake/v1",
            "OPENAI_API_KEY": "",
            "OPENAI_MODEL": "fake-model",
            "OUTPUT_DIR": str(output_dir),
        }
    )
