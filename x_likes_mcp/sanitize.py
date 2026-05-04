"""Defensive sanitization for untrusted tweet text.

Tweets in ``likes.json`` carry content authored by arbitrary X users.
The exporter loads it as-is, the indexer happily stores it, and the
search path hands it back to:

* The CLI printer (terminal output, vulnerable to ANSI escape spoofing).
* The MCP tool response (consumed by an LLM client; vulnerable to prompt
  injection if a tweet says ``Ignore previous instructions ...``).
* The walker chat-completions call (the same LLM-prompt-injection
  surface, this time inside our own LLM call).

Two helpers cover both threats:

* :func:`sanitize_text` strips ANSI escape sequences, C0/C1 control
  characters (except ``\\n`` / ``\\t``), Unicode bidirectional-override
  codepoints, the BOM, and runs NFKC normalization. Output is plain
  UTF-8 with no terminal-control or layout-override magic.

* :func:`fence_for_llm` wraps a sanitized body in a delimited block
  with a fixed marker. Any occurrence of the marker inside the body is
  replaced with a neutral token before fencing so the model cannot be
  tricked into treating user content as a prompt boundary.

Apply :func:`sanitize_text` at every boundary where tweet text leaves
the index. Apply :func:`fence_for_llm` only inside the walker prompt
builder.

NOTE: Bidirectional / formatting codepoints below are constructed at
runtime via ``chr(...)`` so this source file itself stays free of
literal Trojan-Source-style hidden direction overrides; static
analyzers that flag bidi codepoints in source ASCII never see them
here.
"""

from __future__ import annotations

import re
import unicodedata


# Matches ANSI / VT escape sequences. Covers CSI ("ESC ["), OSC ("ESC ]"),
# and standalone ESC + final byte. The character class on the inside is
# permissive on purpose — we drop everything between the introducer and
# the terminator rather than try to enumerate valid SGR codes.
_ANSI_RE = re.compile(
    r"""
    \x1b              # ESC
    (?:
        \[ [0-?]* [ -/]* [@-~]    # CSI: ESC [ ... <terminator>
      | \] [^\x07\x1b]* (?:\x07|\x1b\\)  # OSC: ESC ] ... BEL or ESC \
      | [@-Z\\-_]                # short ESC sequences (ESC + final byte)
    )
    """,
    re.VERBOSE,
)


# C0 (0x00-0x1F) and C1 (0x7F-0x9F) controls we never want in output.
# Newline and tab are kept; everything else in those ranges is stripped.
_CONTROL_CHARS = "".join(
    chr(c)
    for c in range(0x00, 0xA0)
    if c not in (0x09, 0x0A) and not (0x20 <= c < 0x7F)
)
_CONTROL_RE = re.compile(f"[{re.escape(_CONTROL_CHARS)}]")


# Unicode formatting codepoints that can flip rendering direction or
# hide content from a casual reader. Built from numeric codepoints so
# this source file never contains a literal bidi override (defense
# against Trojan-Source on this codebase itself).
#
#   U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ
#   U+200E LRM,  U+200F RLM
#   U+202A LRE,  U+202B RLE,  U+202C PDF,  U+202D LRO,  U+202E RLO
#   U+2060 WORD JOINER
#   U+2066 LRI,  U+2067 RLI,  U+2068 FSI,  U+2069 PDI
#   U+FEFF BOM
_FORMAT_CODEPOINTS = (
    list(range(0x200B, 0x2010))     # ZWSP/ZWNJ/ZWJ/LRM/RLM
    + list(range(0x202A, 0x202F))   # LRE/RLE/PDF/LRO/RLO
    + [0x2060]                      # WORD JOINER
    + list(range(0x2066, 0x206A))   # LRI/RLI/FSI/PDI
    + [0xFEFF]                      # BOM
)
_FORMAT_CHARS = "".join(chr(cp) for cp in _FORMAT_CODEPOINTS)
_FORMAT_RE = re.compile(f"[{re.escape(_FORMAT_CHARS)}]")


# Fence used by :func:`fence_for_llm`. The marker is intentionally
# distinctive so a model can be instructed to never treat its content
# as instructions. Any occurrence of either marker inside the body is
# replaced with the neutral token below before fencing.
LLM_FENCE_OPEN = "<<<TWEET_BODY>>>"
LLM_FENCE_CLOSE = "<<<END_TWEET_BODY>>>"
_FENCE_NEUTRAL = "[FENCE]"


def sanitize_text(text: object) -> str:
    """Return ``text`` with terminal-control, BiDi, and weird codepoints stripped.

    Steps:
      1. Coerce to ``str``. ``None``, bytes, ints, etc. become ``""``.
      2. NFKC normalize so visually-identical confusables collapse to a
         canonical form before later regex stripping.
      3. Strip ANSI / VT escape sequences (CSI, OSC, short ESC).
      4. Strip C0 / C1 control characters except ``\\n`` and ``\\t``.
      5. Strip Unicode bidi overrides, the BOM, and zero-width
         space / joiner / non-joiner.
      6. Coerce to clean UTF-8 by encoding/decoding with ``errors="replace"``.
         Surrogates and other ill-formed sequences become U+FFFD instead
         of crashing downstream consumers.
    """

    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    cleaned = _ANSI_RE.sub("", normalized)
    cleaned = _CONTROL_RE.sub("", cleaned)
    cleaned = _FORMAT_RE.sub("", cleaned)
    return cleaned.encode("utf-8", "replace").decode("utf-8", "replace")


def safe_http_url(url: object) -> str | None:
    """Return ``url`` if it parses as a plain ``http://`` / ``https://`` URL.

    Returns ``None`` for anything else: schemes like ``javascript:``,
    ``data:``, ``file://``, missing scheme, non-strings, or URLs that
    contain control / BiDi codepoints (we sanitize first; if anything
    survives that and the scheme is not HTTP(S), drop the URL).

    The URL field in ``Tweet.urls`` comes from Twitter's resolved
    ``expanded_url``, which should already be HTTP(S), but we don't
    trust upstream.
    """

    if not isinstance(url, str):
        return None
    cleaned = sanitize_text(url).strip()
    if not cleaned:
        return None
    lower = cleaned.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return cleaned
    return None


def fence_for_llm(body: str) -> str:
    """Wrap ``body`` in :data:`LLM_FENCE_OPEN` / :data:`LLM_FENCE_CLOSE`.

    The body is sanitized first (so the LLM never sees ANSI / BiDi / BOM
    junk in its prompt). Any occurrence of either fence marker inside
    the resulting body is replaced with :data:`_FENCE_NEUTRAL` so a
    crafted tweet cannot prematurely close the fence and resume control
    of the prompt.

    The caller is responsible for telling the model (in the system
    prompt) that text inside the fence is data, not instructions.
    """

    sanitized = sanitize_text(body)
    sanitized = sanitized.replace(LLM_FENCE_OPEN, _FENCE_NEUTRAL)
    sanitized = sanitized.replace(LLM_FENCE_CLOSE, _FENCE_NEUTRAL)
    return f"{LLM_FENCE_OPEN}\n{sanitized}\n{LLM_FENCE_CLOSE}"
