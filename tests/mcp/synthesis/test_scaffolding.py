"""Placeholder smoke tests for the synthesis test scaffolding.

These tests verify the autouse fixtures and the ``real_lm`` marker hook
defined in ``tests/mcp/synthesis/conftest.py`` actually take effect for
every test in this package.

Each test maps onto one acceptance criterion of task 1.5:

1. ``test_block_real_url_fetch_active`` — the ``_block_real_url_fetch``
   fixture monkeypatches ``httpx.Client.send`` to raise
   :class:`RealUrlFetchAttempted` so any unmocked HTTP call surfaces
   loudly rather than reaching the network.
2. ``test_stub_dspy_lm_installed`` — the ``_stub_dspy_lm`` fixture
   installs a :class:`FakeDspyLM` instance via ``dspy.configure`` so
   signature tests stay offline by default.
3. ``test_real_lm_marker_skipped_by_default`` — the ``real_lm`` marker
   hook skips any test carrying ``@pytest.mark.real_lm`` unless
   ``--run-real-lm`` is passed on the command line.
"""

from __future__ import annotations

import httpx
import pytest

from tests.mcp.synthesis.conftest import (
    FakeDspyLM,
    RealUrlFetchAttempted,
)


def test_block_real_url_fetch_active() -> None:
    """An unmocked ``httpx.Client.get`` must raise ``RealUrlFetchAttempted``.

    The autouse ``_block_real_url_fetch`` fixture patches
    ``httpx.Client.send`` (the well-known stable seam below
    ``Client.get``/``post``/``request``) to fail loudly. Tests that
    legitimately want HTTP re-monkeypatch the same attribute.
    """

    with pytest.raises(RealUrlFetchAttempted), httpx.Client() as client:
        client.get("http://example.com/")


def test_stub_dspy_lm_installed() -> None:
    """``dspy.settings.lm`` must be the ``FakeDspyLM`` installed by the fixture."""

    import dspy

    lm = dspy.settings.lm
    assert isinstance(lm, FakeDspyLM), f"expected FakeDspyLM, got {type(lm).__name__}"


@pytest.mark.real_lm
def test_real_lm_marker_skipped_by_default() -> None:
    """Without ``--run-real-lm`` this test must be skipped.

    The body deliberately fails — if the marker hook is ever broken so
    that the test is collected and run, the assertion fires and surfaces
    the regression.
    """

    raise AssertionError(
        "real_lm-marked test ran without --run-real-lm; the marker hook is broken."
    )
