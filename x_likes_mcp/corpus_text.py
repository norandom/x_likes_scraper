"""Single source of truth for the text used by both retrieval paths
(BM25 and dense embeddings) when indexing a tweet.

Why this exists: the export pipeline already replaces Twitter ``t.co``
shortlinks with their resolved targets in :attr:`Tweet.urls`, but the
``Tweet.text`` field still carries the opaque ``t.co/abc`` form. If we
embed/tokenize ``Tweet.text`` alone, the resolved domains and slugs
(``github.com/foo/bar``, ``arxiv.org/abs/...``) are not searchable.

:func:`tweet_index_text` returns ``text + " " + " ".join(urls)`` so the
resolved URLs participate in both BM25 lexical matching and the dense
embedding. This is a free recall win — the export already resolved the
links; we just stop discarding that signal at index time.

A change to the format produced here must bump
:data:`embeddings.CACHE_SCHEMA_VERSION` so existing on-disk corpora are
rebuilt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from x_likes_exporter.models import Tweet


def tweet_index_text(tweet: Tweet) -> str:
    """Return the text that BM25 and the embedder should both index.

    The format is the tweet's text followed by each resolved URL,
    space-separated. Empty values are skipped. Returns ``""`` when the
    tweet has neither text nor URLs.
    """

    parts: list[str] = []
    text = getattr(tweet, "text", None) or ""
    if text:
        parts.append(text)
    urls = getattr(tweet, "urls", None) or []
    for url in urls:
        if url:
            parts.append(url)
    return " ".join(parts)
