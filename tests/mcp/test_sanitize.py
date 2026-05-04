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
    ENTITY_FENCE_CLOSE,
    ENTITY_FENCE_OPEN,
    KG_EDGE_FENCE_CLOSE,
    KG_EDGE_FENCE_OPEN,
    KG_NODE_FENCE_CLOSE,
    KG_NODE_FENCE_OPEN,
    LLM_FENCE_CLOSE,
    LLM_FENCE_OPEN,
    URL_BODY_FENCE_CLOSE,
    URL_BODY_FENCE_OPEN,
    URL_FENCE_CLOSE,
    URL_FENCE_OPEN,
    fence_entity_for_llm,
    fence_for_llm,
    fence_kg_edge_for_llm,
    fence_kg_node_for_llm,
    fence_url_body_for_llm,
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


# ---------------------------------------------------------------------------
# New fence families: URL_BODY, ENTITY, KG_NODE, KG_EDGE
# ---------------------------------------------------------------------------


def test_new_fence_marker_constants_have_documented_values() -> None:
    """The eight new constants must be exposed with the exact spec-fixed
    string values; downstream prompts and tests pin the wire format."""

    assert URL_BODY_FENCE_OPEN == "<<<URL_BODY>>>"
    assert URL_BODY_FENCE_CLOSE == "<<<END_URL_BODY>>>"
    assert ENTITY_FENCE_OPEN == "<<<ENTITY>>>"
    assert ENTITY_FENCE_CLOSE == "<<<END_ENTITY>>>"
    assert KG_NODE_FENCE_OPEN == "<<<KG_NODE>>>"
    assert KG_NODE_FENCE_CLOSE == "<<<END_KG_NODE>>>"
    assert KG_EDGE_FENCE_OPEN == "<<<KG_EDGE>>>"
    assert KG_EDGE_FENCE_CLOSE == "<<<END_KG_EDGE>>>"


# ---------------------------------------------------------------------------
# fence_url_body_for_llm
# ---------------------------------------------------------------------------


def test_fence_url_body_for_llm_wraps_with_markers() -> None:
    out = fence_url_body_for_llm("readme content")
    assert out.startswith(URL_BODY_FENCE_OPEN + "\n")
    assert out.endswith("\n" + URL_BODY_FENCE_CLOSE)
    assert "readme content" in out


def test_fence_url_body_for_llm_neutralizes_own_open_marker() -> None:
    body = f"prefix {URL_BODY_FENCE_OPEN} payload"
    out = fence_url_body_for_llm(body)
    inner = "\n".join(out.split("\n")[1:-1])
    assert URL_BODY_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


def test_fence_url_body_for_llm_neutralizes_own_close_marker() -> None:
    body = f"prefix {URL_BODY_FENCE_CLOSE} payload"
    out = fence_url_body_for_llm(body)
    inner = "\n".join(out.split("\n")[1:-1])
    assert URL_BODY_FENCE_CLOSE not in inner


def test_fence_url_body_for_llm_sanitizes_body() -> None:
    body = "doc\x1b[31m text" + chr(0x202E) + "rtl"
    out = fence_url_body_for_llm(body)
    assert "\x1b" not in out
    assert chr(0x202E) not in out


# ---------------------------------------------------------------------------
# fence_entity_for_llm
# ---------------------------------------------------------------------------


def test_fence_entity_for_llm_wraps_single_line() -> None:
    out = fence_entity_for_llm("OpenAI")
    assert out == f"{ENTITY_FENCE_OPEN}OpenAI{ENTITY_FENCE_CLOSE}"


def test_fence_entity_for_llm_neutralizes_own_open_marker() -> None:
    out = fence_entity_for_llm(f"x{ENTITY_FENCE_OPEN}y")
    assert out is not None
    inner = out[len(ENTITY_FENCE_OPEN) : -len(ENTITY_FENCE_CLOSE)]
    assert ENTITY_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


@pytest.mark.parametrize("empty", ["", "   ", "\t\n", None, 42, b"bytes"])
def test_fence_entity_for_llm_returns_none_for_empty(empty: object) -> None:
    assert fence_entity_for_llm(empty) is None  # type: ignore[arg-type]


def test_fence_entity_for_llm_returns_none_when_sanitize_collapses_to_empty() -> None:
    # Whole input is bidi/control codepoints — sanitize produces empty.
    raw = "\x1b[31m" + chr(0x202E) + chr(0x200B)
    assert fence_entity_for_llm(raw) is None


def test_fence_entity_for_llm_strips_ansi() -> None:
    out = fence_entity_for_llm("Open\x1b[31mAI")
    assert out == f"{ENTITY_FENCE_OPEN}OpenAI{ENTITY_FENCE_CLOSE}"


# ---------------------------------------------------------------------------
# fence_kg_node_for_llm / fence_kg_edge_for_llm
# ---------------------------------------------------------------------------


def test_fence_kg_node_for_llm_wraps_single_line() -> None:
    out = fence_kg_node_for_llm("Person")
    assert out == f"{KG_NODE_FENCE_OPEN}Person{KG_NODE_FENCE_CLOSE}"


def test_fence_kg_node_for_llm_emits_empty_fence_for_empty_label() -> None:
    """Unlike entities, KG nodes always exist structurally; an empty label
    still produces an empty fence so the LM sees the placeholder."""

    out = fence_kg_node_for_llm("")
    assert out == f"{KG_NODE_FENCE_OPEN}{KG_NODE_FENCE_CLOSE}"


def test_fence_kg_node_for_llm_emits_empty_fence_for_whitespace_label() -> None:
    # sanitize keeps whitespace; we don't strip it here. The contract is
    # "do not return None"; the wrapped body may be the sanitized
    # whitespace itself. Just assert it is wrapped, never None.
    out = fence_kg_node_for_llm("   ")
    assert out is not None
    assert out.startswith(KG_NODE_FENCE_OPEN)
    assert out.endswith(KG_NODE_FENCE_CLOSE)


def test_fence_kg_node_for_llm_neutralizes_own_marker() -> None:
    out = fence_kg_node_for_llm(f"x{KG_NODE_FENCE_OPEN}y")
    inner = out[len(KG_NODE_FENCE_OPEN) : -len(KG_NODE_FENCE_CLOSE)]
    assert KG_NODE_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


def test_fence_kg_node_for_llm_strips_ansi() -> None:
    out = fence_kg_node_for_llm("Per\x1b[31mson")
    assert out == f"{KG_NODE_FENCE_OPEN}Person{KG_NODE_FENCE_CLOSE}"


def test_fence_kg_edge_for_llm_wraps_single_line() -> None:
    out = fence_kg_edge_for_llm("knows")
    assert out == f"{KG_EDGE_FENCE_OPEN}knows{KG_EDGE_FENCE_CLOSE}"


def test_fence_kg_edge_for_llm_emits_empty_fence_for_empty_label() -> None:
    out = fence_kg_edge_for_llm("")
    assert out == f"{KG_EDGE_FENCE_OPEN}{KG_EDGE_FENCE_CLOSE}"


def test_fence_kg_edge_for_llm_neutralizes_own_marker() -> None:
    out = fence_kg_edge_for_llm(f"x{KG_EDGE_FENCE_OPEN}y")
    inner = out[len(KG_EDGE_FENCE_OPEN) : -len(KG_EDGE_FENCE_CLOSE)]
    assert KG_EDGE_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


def test_fence_kg_edge_for_llm_strips_bidi() -> None:
    out = fence_kg_edge_for_llm("kn" + chr(0x202E) + "ows")
    assert out == f"{KG_EDGE_FENCE_OPEN}knows{KG_EDGE_FENCE_CLOSE}"


# ---------------------------------------------------------------------------
# Cross-family neutralization: every wrapper neutralizes ALL twelve markers
# ---------------------------------------------------------------------------


_ALL_TWELVE_MARKERS = (
    LLM_FENCE_OPEN,
    LLM_FENCE_CLOSE,
    URL_FENCE_OPEN,
    URL_FENCE_CLOSE,
    URL_BODY_FENCE_OPEN,
    URL_BODY_FENCE_CLOSE,
    ENTITY_FENCE_OPEN,
    ENTITY_FENCE_CLOSE,
    KG_NODE_FENCE_OPEN,
    KG_NODE_FENCE_CLOSE,
    KG_EDGE_FENCE_OPEN,
    KG_EDGE_FENCE_CLOSE,
)


def _body_with_every_marker() -> str:
    """A body that crams every fence marker into one string. Used to
    prove no marker can survive into a foreign fence's interior."""

    return "head " + " ".join(_ALL_TWELVE_MARKERS) + " tail"


def _inner_block(out: str, open_marker: str, close_marker: str) -> str:
    """Return the content between the outer fence markers regardless of
    whether the wrapper uses newline-shape or single-line shape."""

    assert out.startswith(open_marker), out
    assert out.endswith(close_marker), out
    return out[len(open_marker) : -len(close_marker)]


def test_fence_for_llm_neutralizes_all_twelve_markers() -> None:
    out = fence_for_llm(_body_with_every_marker())
    inner = _inner_block(out, LLM_FENCE_OPEN + "\n", "\n" + LLM_FENCE_CLOSE)
    for marker in _ALL_TWELVE_MARKERS:
        assert marker not in inner, marker


def test_fence_url_body_for_llm_neutralizes_all_twelve_markers() -> None:
    out = fence_url_body_for_llm(_body_with_every_marker())
    inner = _inner_block(out, URL_BODY_FENCE_OPEN + "\n", "\n" + URL_BODY_FENCE_CLOSE)
    for marker in _ALL_TWELVE_MARKERS:
        assert marker not in inner, marker


def test_fence_entity_for_llm_neutralizes_all_twelve_markers() -> None:
    body = _body_with_every_marker()
    out = fence_entity_for_llm(body)
    assert out is not None
    inner = _inner_block(out, ENTITY_FENCE_OPEN, ENTITY_FENCE_CLOSE)
    for marker in _ALL_TWELVE_MARKERS:
        assert marker not in inner, marker


def test_fence_kg_node_for_llm_neutralizes_all_twelve_markers() -> None:
    out = fence_kg_node_for_llm(_body_with_every_marker())
    inner = _inner_block(out, KG_NODE_FENCE_OPEN, KG_NODE_FENCE_CLOSE)
    for marker in _ALL_TWELVE_MARKERS:
        assert marker not in inner, marker


def test_fence_kg_edge_for_llm_neutralizes_all_twelve_markers() -> None:
    out = fence_kg_edge_for_llm(_body_with_every_marker())
    inner = _inner_block(out, KG_EDGE_FENCE_OPEN, KG_EDGE_FENCE_CLOSE)
    for marker in _ALL_TWELVE_MARKERS:
        assert marker not in inner, marker


def test_fence_url_for_llm_neutralizes_new_families() -> None:
    """``fence_url_for_llm`` already existed; now that the four new
    families joined ``_ALL_FENCES``, a URL whose path embeds e.g. the
    ``KG_NODE`` marker must also be neutralized."""

    crafted = f"https://example.com/{KG_NODE_FENCE_OPEN}/{ENTITY_FENCE_OPEN}"
    out = fence_url_for_llm(crafted)
    assert out is not None
    inner = _inner_block(out, URL_FENCE_OPEN, URL_FENCE_CLOSE)
    assert KG_NODE_FENCE_OPEN not in inner
    assert ENTITY_FENCE_OPEN not in inner
    assert "[FENCE]" in inner


def test_fence_for_llm_neutralizes_new_families_inside_tweet_body() -> None:
    body = f"a {URL_BODY_FENCE_OPEN} b {KG_EDGE_FENCE_CLOSE} c"
    out = fence_for_llm(body)
    inner = "\n".join(out.split("\n")[1:-1])
    assert URL_BODY_FENCE_OPEN not in inner
    assert KG_EDGE_FENCE_CLOSE not in inner
