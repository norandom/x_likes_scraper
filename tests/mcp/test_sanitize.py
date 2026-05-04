"""Tests for :mod:`x_likes_mcp.sanitize`.

Covers the three boundary helpers:

* :func:`sanitize_text` — strip ANSI / control / BiDi / BOM and NFKC.
* :func:`safe_http_url` — accept HTTP(S) only; drop everything else.
* :func:`fence_for_llm` — wrap with delimiters and neutralize embedded
  fence markers so a crafted tweet body cannot prematurely close the
  fence and hijack the prompt.
"""

from __future__ import annotations

import pytest

from x_likes_mcp.sanitize import (
    LLM_FENCE_CLOSE,
    LLM_FENCE_OPEN,
    URL_FENCE_CLOSE,
    URL_FENCE_OPEN,
    fence_for_llm,
    fence_url_for_llm,
    safe_http_url,
    sanitize_text,
)


# ---------------------------------------------------------------------------
# sanitize_text
# ---------------------------------------------------------------------------


def test_sanitize_text_passes_through_plain_ascii() -> None:
    assert sanitize_text("hello world") == "hello world"


def test_sanitize_text_strips_csi_escape() -> None:
    assert sanitize_text("hello\x1b[31mworld\x1b[0m") == "helloworld"


def test_sanitize_text_strips_osc_escape() -> None:
    # ESC ] ... BEL is the OSC form
    assert sanitize_text("a\x1b]0;title\x07b") == "ab"


def test_sanitize_text_keeps_newline_and_tab() -> None:
    assert sanitize_text("line1\nline2\tcol2") == "line1\nline2\tcol2"


def test_sanitize_text_strips_other_c0_controls() -> None:
    raw = "a\x00b\x07c\x0bd\x0ce\rf"
    assert sanitize_text(raw) == "abcdef"


def test_sanitize_text_strips_c1_controls() -> None:
    raw = "a\x80b\x9fc"
    assert sanitize_text(raw) == "abc"


def test_sanitize_text_strips_bidi_overrides() -> None:
    # U+202E RIGHT-TO-LEVEL OVERRIDE is the Trojan-Source classic
    raw = "a" + chr(0x202E) + "b" + chr(0x202D) + "c"
    assert sanitize_text(raw) == "abc"


def test_sanitize_text_strips_zero_width_codepoints() -> None:
    raw = "a" + chr(0x200B) + "b" + chr(0x200D) + "c" + chr(0xFEFF) + "d"
    assert sanitize_text(raw) == "abcd"


def test_sanitize_text_runs_nfkc() -> None:
    # U+FB01 (LATIN SMALL LIGATURE FI) -> "fi" under NFKC
    assert sanitize_text("ﬁnal") == "final"


def test_sanitize_text_handles_lone_surrogate() -> None:
    # Surrogates are illegal in UTF-8. The encode("utf-8", "replace")
    # substitutes ``?`` for unencodable codepoints; the round-trip never
    # crashes and the bracketing characters survive.
    raw = "a\ud83dz"
    out = sanitize_text(raw)
    assert out.startswith("a")
    assert out.endswith("z")
    assert "\ud83d" not in out


@pytest.mark.parametrize("bad", [None, 42, b"bytes", object(), [1, 2]])
def test_sanitize_text_non_string_returns_empty(bad: object) -> None:
    assert sanitize_text(bad) == ""


# ---------------------------------------------------------------------------
# safe_http_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com/path?q=1",
        "HTTPS://EXAMPLE.COM",
        "  https://example.com/x  ",
    ],
)
def test_safe_http_url_accepts_http_https(url: str) -> None:
    assert safe_http_url(url) is not None


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,<script>",
        "file:///etc/passwd",
        "ftp://example.com",
        "//no-scheme",
        "no-scheme",
        "",
        "  ",
    ],
)
def test_safe_http_url_rejects_other_schemes(url: str) -> None:
    assert safe_http_url(url) is None


@pytest.mark.parametrize("bad", [None, 42, b"http://x", object()])
def test_safe_http_url_rejects_non_string(bad: object) -> None:
    assert safe_http_url(bad) is None


def test_safe_http_url_strips_ansi_inside_url() -> None:
    # Sanitize runs before the scheme check; an ANSI sequence inside the
    # URL is removed.
    out = safe_http_url("https://example.com\x1b[31m/x")
    assert out == "https://example.com/x"


# ---------------------------------------------------------------------------
# fence_for_llm
# ---------------------------------------------------------------------------


def test_fence_for_llm_wraps_with_markers() -> None:
    out = fence_for_llm("hello")
    assert out.startswith(LLM_FENCE_OPEN + "\n")
    assert out.endswith("\n" + LLM_FENCE_CLOSE)
    assert "hello" in out


def test_fence_for_llm_neutralizes_embedded_open_marker() -> None:
    """A tweet that contains the open marker must not be able to break
    out of the fence."""

    body = f"prefix {LLM_FENCE_OPEN} payload"
    out = fence_for_llm(body)
    inner = "\n".join(out.split("\n")[1:-1])
    assert LLM_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


def test_fence_for_llm_neutralizes_embedded_close_marker() -> None:
    body = f"prefix {LLM_FENCE_CLOSE} payload"
    out = fence_for_llm(body)
    inner = "\n".join(out.split("\n")[1:-1])
    assert LLM_FENCE_CLOSE not in inner


def test_fence_for_llm_sanitizes_body() -> None:
    """ANSI / BiDi inside the body is stripped before fencing."""

    body = "tweet\x1b[31m text" + chr(0x202E) + "rtl"
    out = fence_for_llm(body)
    assert "\x1b" not in out
    assert chr(0x202E) not in out


# ---------------------------------------------------------------------------
# fence_url_for_llm
# ---------------------------------------------------------------------------


def test_fence_url_for_llm_wraps_http_url() -> None:
    out = fence_url_for_llm("https://example.com/path")
    assert out == f"{URL_FENCE_OPEN}https://example.com/path{URL_FENCE_CLOSE}"


@pytest.mark.parametrize(
    "bad_url",
    [
        "javascript:alert(1)",
        "data:text/html,",
        "file:///etc/passwd",
        "",
        None,
        42,
    ],
)
def test_fence_url_for_llm_returns_none_for_unsafe(bad_url: object) -> None:
    assert fence_url_for_llm(bad_url) is None


def test_fence_url_for_llm_neutralizes_embedded_url_fence_open() -> None:
    """A URL whose path contains the URL fence open marker cannot break
    out and reopen prompt control."""

    crafted = f"https://example.com/{URL_FENCE_OPEN}/exfil"
    out = fence_url_for_llm(crafted)
    assert out is not None
    # Strip the outer fence; the inner body must not contain a second
    # raw open marker.
    assert out.startswith(URL_FENCE_OPEN)
    assert out.endswith(URL_FENCE_CLOSE)
    inner = out[len(URL_FENCE_OPEN) : -len(URL_FENCE_CLOSE)]
    assert URL_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


def test_fence_url_for_llm_neutralizes_tweet_body_fence() -> None:
    """A URL containing the *tweet body* fence markers must also be
    neutralized; otherwise an attacker could break out of an outer
    walker prompt that interleaves URLs and tweet bodies."""

    crafted = f"https://example.com/{LLM_FENCE_OPEN}"
    out = fence_url_for_llm(crafted)
    assert out is not None
    assert LLM_FENCE_OPEN not in out[len(URL_FENCE_OPEN) : -len(URL_FENCE_CLOSE)]


def test_fence_url_for_llm_strips_ansi_inside_path() -> None:
    out = fence_url_for_llm("https://example.com\x1b[31m/x")
    assert out == f"{URL_FENCE_OPEN}https://example.com/x{URL_FENCE_CLOSE}"
