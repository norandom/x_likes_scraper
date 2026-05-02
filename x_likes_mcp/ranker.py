"""Tweet scoring formula. Pure functions only.

The score combines the walker's LLM relevance number with engagement
counts (favorite, retweet, reply, view), author affinity (log1p of how
many times this user has liked the same handle), recency decay, and
two small flags (verified, has_media). Everything is a single sum,
weights from ``RankerWeights``.

The feature shape is borrowed from ``twitter/the-algorithm``'s heavy
ranker for the bits the export actually has. It is not a port. The
infrastructure-heavy parts (real-graph, SimClusters, TwHIN) don't
apply to a single-user offline archive and aren't here.

No I/O, no LLM, no openai import.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from x_likes_exporter.dates import parse_x_datetime

if TYPE_CHECKING:  # pragma: no cover
    from x_likes_exporter.models import Tweet

    from .config import RankerWeights
    from .walker import WalkerHit


@dataclass(frozen=True)
class ScoredHit:
    """One ranked match returned by :func:`rank`."""

    tweet_id: str
    score: float
    walker_relevance: float
    why: str
    feature_breakdown: dict[str, float]


def compute_author_affinity(tweets: list[Tweet]) -> dict[str, float]:
    """Per-handle affinity score based on the user's own like history.

    Returns ``{handle: log1p(count_of_likes_from_handle)}``. Empty
    handles are excluded so anonymized historical tweets do not
    accumulate into a single bucket.
    """
    counts: Counter[str] = Counter(
        t.user.screen_name for t in tweets if t.user.screen_name
    )
    return {handle: math.log1p(count) for handle, count in counts.items()}


def recency_decay(created_at: str, anchor: datetime, halflife_days: float) -> float:
    """Exponential recency decay.

    ``days = max(0, (anchor - created) / 1 day)``; returns
    ``exp(-days / halflife_days)``. ``1.0`` when the tweet is at the
    anchor; ``~0.5`` after one halflife; asymptotes to 0 for very old
    tweets.

    Raises ``ValueError`` if ``created_at`` is unparseable; callers
    typically catch this and contribute 0 for the recency term.
    """
    created = parse_x_datetime(created_at)
    if created is None:
        raise ValueError(f"unparseable created_at: {created_at!r}")
    days = max(0.0, (anchor - created).total_seconds() / 86400.0)
    return math.exp(-days / halflife_days)


def rank(
    walker_hits: list[WalkerHit],
    tweets_by_id: dict[str, Tweet],
    author_affinity: dict[str, float],
    weights: RankerWeights,
    anchor: datetime | None = None,
) -> list[ScoredHit]:
    """Combine walker relevance with engagement / affinity / recency features.

    Walker hits whose tweet ID isn't in ``tweets_by_id`` are skipped
    (the walker may return IDs the tree saw but the tweets list does
    not, e.g. due to API drift between scrape and load).

    Sorted descending by score; ties broken by ``walker_relevance``
    descending, then ``tweet_id`` ascending, for determinism.
    """
    if anchor is None:
        anchor = datetime.now(timezone.utc)

    scored: list[ScoredHit] = []
    recency_logged = False
    for hit in walker_hits:
        tweet = tweets_by_id.get(hit.tweet_id)
        if tweet is None:
            continue

        relevance = hit.relevance * weights.relevance
        favorite = math.log1p(max(0, tweet.favorite_count)) * weights.favorite
        retweet = math.log1p(max(0, tweet.retweet_count)) * weights.retweet
        reply = math.log1p(max(0, tweet.reply_count)) * weights.reply
        view = math.log1p(max(0, tweet.view_count)) * weights.view
        affinity = author_affinity.get(tweet.user.screen_name, 0.0) * weights.affinity

        try:
            recency = (
                recency_decay(
                    tweet.created_at, anchor, weights.recency_halflife_days
                )
                * weights.recency
            )
        except ValueError:
            recency = 0.0
            if not recency_logged:
                print(
                    "ranker: unparseable created_at; recency=0 for one or more hits",
                    file=sys.stderr,
                )
                recency_logged = True

        verified = (1.0 if tweet.user.verified else 0.0) * weights.verified
        media = (1.0 if tweet.media else 0.0) * weights.media

        breakdown = {
            "relevance": relevance,
            "favorite": favorite,
            "retweet": retweet,
            "reply": reply,
            "view": view,
            "affinity": affinity,
            "recency": recency,
            "verified": verified,
            "media": media,
        }
        score = sum(breakdown.values())

        scored.append(
            ScoredHit(
                tweet_id=hit.tweet_id,
                score=score,
                walker_relevance=hit.relevance,
                why=hit.why,
                feature_breakdown=breakdown,
            )
        )

    scored.sort(key=lambda s: (-s.score, -s.walker_relevance, s.tweet_id))
    return scored
