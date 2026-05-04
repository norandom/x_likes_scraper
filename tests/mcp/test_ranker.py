"""Tests for :mod:`x_likes_mcp.ranker`.

The ranker is a pure module: no I/O, no LLM, no network. These tests exercise
:func:`compute_author_affinity`, :func:`recency_decay`, and :func:`rank`
directly with hand-built :class:`Tweet`, :class:`User`, :class:`Media`, and
:class:`WalkerHit` instances.

The X ``created_at`` format is ``"%a %b %d %H:%M:%S %z %Y"`` (see
:func:`x_likes_exporter.dates.parse_x_datetime`); fixture datetimes are
emitted in that exact shape via :func:`_fmt_x_datetime` to avoid brittle
hand-typed strings.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from x_likes_exporter.models import Media, Tweet, User
from x_likes_mcp.config import RankerWeights
from x_likes_mcp.ranker import (
    ScoredHit,
    compute_author_affinity,
    rank,
    recency_decay,
)
from x_likes_mcp.walker import WalkerHit

# ---------------------------------------------------------------------------
# Helpers


def _fmt_x_datetime(dt: datetime) -> str:
    """Render a UTC ``datetime`` in the X ``created_at`` format.

    Matches ``X_CREATED_AT_FORMAT`` in :mod:`x_likes_exporter.dates`. Using a
    helper keeps the test inputs honest: any string we pass through this
    function round-trips through :func:`parse_x_datetime` cleanly.
    """

    return dt.strftime("%a %b %d %H:%M:%S %z %Y")


def _user(handle: str = "alice", verified: bool = False) -> User:
    """Construct a minimal :class:`User` for ranking tests."""

    return User(
        id=f"uid_{handle}",
        screen_name=handle,
        name=handle.title(),
        verified=verified,
    )


def _tweet(
    tweet_id: str,
    handle: str = "alice",
    *,
    created_at: str | None = None,
    favorite_count: int = 0,
    retweet_count: int = 0,
    reply_count: int = 0,
    view_count: int = 0,
    verified: bool = False,
    media: list[Media] | None = None,
) -> Tweet:
    """Construct a :class:`Tweet` with documented engagement defaults."""

    if created_at is None:
        created_at = _fmt_x_datetime(datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC))
    return Tweet(
        id=tweet_id,
        text=f"hello from @{handle}",
        created_at=created_at,
        user=_user(handle, verified=verified),
        favorite_count=favorite_count,
        retweet_count=retweet_count,
        reply_count=reply_count,
        view_count=view_count,
        media=list(media) if media is not None else [],
    )


# ---------------------------------------------------------------------------
# compute_author_affinity


def test_compute_author_affinity_log1p_per_handle() -> None:
    """Three @alice tweets and two @bob tweets → log1p(3) and log1p(2)."""

    tweets = [
        _tweet("1", "alice"),
        _tweet("2", "alice"),
        _tweet("3", "alice"),
        _tweet("4", "bob"),
        _tweet("5", "bob"),
    ]

    affinity = compute_author_affinity(tweets)

    assert affinity == {
        "alice": math.log1p(3),
        "bob": math.log1p(2),
    }


def test_compute_author_affinity_excludes_empty_handles() -> None:
    """A tweet with ``screen_name=""`` does not accumulate into a ``""`` key.

    The implementation drops empty handles so anonymized historical tweets do
    not collapse into a single bucket. The empty-handle tweet is simply
    skipped; non-empty handles still get their normal log1p(count) score.
    """

    tweets = [
        _tweet("1", "alice"),
        _tweet("2", ""),  # empty handle - must not contribute
        _tweet("3", "alice"),
    ]

    affinity = compute_author_affinity(tweets)

    assert affinity == {"alice": math.log1p(2)}
    assert "" not in affinity


# ---------------------------------------------------------------------------
# recency_decay


def test_recency_decay_at_anchor_returns_one() -> None:
    """``created == anchor`` → ``exp(0) == 1.0`` exactly."""

    anchor = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    created = _fmt_x_datetime(anchor)

    assert recency_decay(created, anchor, halflife_days=180.0) == 1.0


def test_recency_decay_one_halflife_returns_one_over_e() -> None:
    """``days == halflife`` → ``exp(-1) ≈ 0.3679`` per the documented formula.

    Note: the ranker uses ``exp(-d / h)``, not ``2**(-d/h)``. So at one
    halflife the decay is ``1/e``, not ``0.5`` — see the formula in
    :func:`recency_decay`.
    """

    anchor = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    halflife = 30.0
    created = _fmt_x_datetime(anchor - timedelta(days=halflife))

    decay = recency_decay(created, anchor, halflife_days=halflife)

    assert decay == pytest.approx(math.exp(-1.0), rel=1e-9)
    assert decay == pytest.approx(0.36787944117, rel=1e-6)


def test_recency_decay_future_tweet_clamps_to_one() -> None:
    """``created > anchor`` → ``max(0, ...)`` clamps days to 0; decay = 1.0.

    Defends against clock skew between the user's machine and the X export
    timestamp: a tweet that appears to be in the future relative to the
    anchor must not produce ``decay > 1`` (which would inflate the score).
    """

    anchor = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    created = _fmt_x_datetime(anchor + timedelta(days=10))

    assert recency_decay(created, anchor, halflife_days=180.0) == 1.0


def test_recency_decay_unparseable_raises_value_error() -> None:
    """Garbage ``created_at`` raises :class:`ValueError`.

    :func:`rank` catches this internally and contributes 0 for the recency
    term; direct callers see the raise.
    """

    anchor = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    with pytest.raises(ValueError):
        recency_decay("not a date", anchor, halflife_days=180.0)


# ---------------------------------------------------------------------------
# rank: feature_breakdown sums to score


def test_rank_feature_breakdown_sums_to_score() -> None:
    """For a single ``WalkerHit`` the breakdown values reconstruct ``score``.

    This pins the contract that ``feature_breakdown`` is the full additive
    decomposition: there are no hidden offsets or non-summed terms.
    """

    anchor = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    tweet = _tweet(
        "100",
        "alice",
        created_at=_fmt_x_datetime(anchor - timedelta(days=10)),
        favorite_count=42,
        retweet_count=5,
        reply_count=3,
        view_count=1000,
        verified=True,
        media=[Media(type="photo", url="https://x.com/img.jpg")],
    )
    hits = [WalkerHit(tweet_id="100", relevance=0.7, why="topical")]
    affinity = {"alice": math.log1p(7)}
    weights = RankerWeights()

    ranked = rank(hits, {"100": tweet}, affinity, weights, anchor=anchor)

    assert len(ranked) == 1
    scored = ranked[0]
    assert isinstance(scored, ScoredHit)
    assert sum(scored.feature_breakdown.values()) == pytest.approx(
        scored.score, rel=1e-12, abs=1e-12
    )
    # All nine documented features are present.
    assert set(scored.feature_breakdown.keys()) == {
        "relevance",
        "favorite",
        "retweet",
        "reply",
        "view",
        "affinity",
        "recency",
        "verified",
        "media",
    }


# ---------------------------------------------------------------------------
# rank: monotonicity


def _rank_one(
    hit: WalkerHit,
    tweet: Tweet,
    affinity: dict[str, float],
    weights: RankerWeights,
    anchor: datetime,
) -> ScoredHit:
    """Score a single hit and return the resulting :class:`ScoredHit`."""

    ranked = rank([hit], {tweet.id: tweet}, affinity, weights, anchor=anchor)
    assert len(ranked) == 1
    return ranked[0]


def test_rank_monotonic_in_walker_relevance() -> None:
    """All features held constant: higher walker relevance → higher score."""

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    tweet = _tweet("1", "alice", created_at=_fmt_x_datetime(anchor))
    weights = RankerWeights()
    affinity: dict[str, float] = {}

    low = _rank_one(WalkerHit("1", 0.2, ""), tweet, affinity, weights, anchor)
    high = _rank_one(WalkerHit("1", 0.9, ""), tweet, affinity, weights, anchor)

    assert high.score > low.score


def test_rank_monotonic_in_author_affinity() -> None:
    """Higher precomputed affinity for the tweet's handle → higher score."""

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    tweet = _tweet("1", "alice", created_at=_fmt_x_datetime(anchor))
    weights = RankerWeights()
    hit = WalkerHit("1", 0.5, "")

    low = _rank_one(hit, tweet, {"alice": 0.5}, weights, anchor)
    high = _rank_one(hit, tweet, {"alice": 5.0}, weights, anchor)

    assert high.score > low.score


@pytest.mark.parametrize(
    "field",
    ["favorite_count", "retweet_count", "reply_count", "view_count"],
)
def test_rank_monotonic_in_engagement_counts(field: str) -> None:
    """Each engagement count contributes monotonically to the final score.

    Held all else equal, ``log1p(high) > log1p(low)`` and the corresponding
    weight is positive, so ``high_score > low_score`` for every count
    feature in :class:`RankerWeights`.
    """

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    base_kwargs = {
        "created_at": _fmt_x_datetime(anchor),
        "favorite_count": 1,
        "retweet_count": 1,
        "reply_count": 1,
        "view_count": 1,
    }
    low_kwargs = dict(base_kwargs)
    high_kwargs = dict(base_kwargs)
    low_kwargs[field] = 1
    high_kwargs[field] = 1000

    low_tweet = _tweet("1", "alice", **low_kwargs)
    high_tweet = _tweet("1", "alice", **high_kwargs)

    weights = RankerWeights()
    hit = WalkerHit("1", 0.5, "")

    low = _rank_one(hit, low_tweet, {}, weights, anchor)
    high = _rank_one(hit, high_tweet, {}, weights, anchor)

    assert high.score > low.score


def test_rank_monotonic_in_recency() -> None:
    """A more recent tweet (closer to the anchor) outscores an older one.

    Recency contributes ``exp(-days/halflife) * weights.recency``; a smaller
    ``days`` means a larger contribution.
    """

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    weights = RankerWeights()
    hit = WalkerHit("1", 0.5, "")

    fresh_tweet = _tweet("1", "alice", created_at=_fmt_x_datetime(anchor - timedelta(days=1)))
    stale_tweet = _tweet("1", "alice", created_at=_fmt_x_datetime(anchor - timedelta(days=400)))

    fresh = _rank_one(hit, fresh_tweet, {}, weights, anchor)
    stale = _rank_one(hit, stale_tweet, {}, weights, anchor)

    assert fresh.score > stale.score


# ---------------------------------------------------------------------------
# rank: missing tweets are dropped


def test_rank_skips_walker_hit_when_tweet_missing() -> None:
    """A ``WalkerHit`` whose tweet_id is not in ``tweets_by_id`` is dropped.

    The walker can return IDs the loaded tweets list does not contain (e.g.
    API drift between scrape and load); the ranker silently skips them
    rather than raising.
    """

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    present_tweet = _tweet("present", "alice", created_at=_fmt_x_datetime(anchor))
    hits = [
        WalkerHit("present", 0.6, "ok"),
        WalkerHit("ghost", 0.9, "nope"),
    ]
    weights = RankerWeights()

    ranked = rank(hits, {"present": present_tweet}, {}, weights, anchor=anchor)

    assert [s.tweet_id for s in ranked] == ["present"]


# ---------------------------------------------------------------------------
# rank: sort order and tie-breaking


def test_rank_sort_order_with_ties() -> None:
    """Sort: descending score; ties → walker_relevance desc, then tweet_id asc.

    Construction:
      - Three tweets, all with identical engagement, identical authors with
        zero affinity, identical timestamps.
      - Walker relevances chosen so that the resulting scores tie between
        two of the three. Since every other feature is held equal, two hits
        with the same relevance produce identical scores.
      - For the tied pair, the secondary key (walker_relevance desc) is also
        a tie, so the tertiary key (tweet_id ascending) breaks the tie.

    Expected order:
      - "high" (relevance=0.9, score highest)
      - "tie_a" (relevance=0.5, lower tweet_id)
      - "tie_b" (relevance=0.5, higher tweet_id)
    """

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    created = _fmt_x_datetime(anchor)

    def _t(tid: str) -> Tweet:
        return _tweet(
            tid,
            "alice",
            created_at=created,
            favorite_count=10,
            retweet_count=10,
            reply_count=10,
            view_count=10,
        )

    tweets_by_id = {
        "high": _t("high"),
        "tie_a": _t("tie_a"),
        "tie_b": _t("tie_b"),
    }

    # Intentionally feed the hits out of order so the sort has work to do.
    hits = [
        WalkerHit("tie_b", 0.5, "b"),
        WalkerHit("high", 0.9, "h"),
        WalkerHit("tie_a", 0.5, "a"),
    ]

    weights = RankerWeights()
    ranked = rank(hits, tweets_by_id, {}, weights, anchor=anchor)

    # All three survive; "high" is strictly larger than the tied pair.
    assert [s.tweet_id for s in ranked] == ["high", "tie_a", "tie_b"]
    assert ranked[0].score > ranked[1].score
    # The tied pair really does have equal scores (within float tolerance).
    assert ranked[1].score == pytest.approx(ranked[2].score, rel=1e-12, abs=1e-12)
    # Tertiary tie-break: tweet_id ascending.
    assert ranked[1].tweet_id < ranked[2].tweet_id


def test_rank_tie_break_by_walker_relevance_when_scores_equal() -> None:
    """Two hits whose final scores tie: walker_relevance desc breaks the tie.

    To force a score tie despite different walker relevances, we set the
    relevance weight to 0 so the walker score does not actually contribute
    to the final ``score``; the secondary key (walker_relevance desc) then
    decides the order.
    """

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    created = _fmt_x_datetime(anchor)
    tweets_by_id = {
        "x": _tweet("x", "alice", created_at=created),
        "y": _tweet("y", "alice", created_at=created),
    }

    hits = [
        WalkerHit("x", 0.3, ""),
        WalkerHit("y", 0.8, ""),
    ]

    # Relevance weight zeroed: identical tweets → identical scores.
    weights = RankerWeights(relevance=0.0)
    ranked = rank(hits, tweets_by_id, {}, weights, anchor=anchor)

    assert ranked[0].score == pytest.approx(ranked[1].score, rel=1e-12, abs=1e-12)
    # Higher walker_relevance comes first.
    assert ranked[0].tweet_id == "y"
    assert ranked[1].tweet_id == "x"


# ---------------------------------------------------------------------------
# rank: unparseable created_at degrades to recency=0 and logs once


def test_rank_unparseable_created_at_zeros_recency_and_logs_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tweet with a garbage ``created_at`` contributes 0 for recency.

    ``rank`` must not raise. The breakdown's ``recency`` entry is exactly
    0.0 for the offending hit, and a single stderr line is emitted across
    the entire ``rank()`` call regardless of how many hits hit the same
    error path.
    """

    anchor = datetime(2025, 6, 1, tzinfo=UTC)
    bad_tweet_a = _tweet("a", "alice", created_at="not a date")
    bad_tweet_b = _tweet("b", "bob", created_at="also not a date")
    tweets_by_id = {"a": bad_tweet_a, "b": bad_tweet_b}
    hits = [
        WalkerHit("a", 0.5, ""),
        WalkerHit("b", 0.5, ""),
    ]
    weights = RankerWeights()

    ranked = rank(hits, tweets_by_id, {}, weights, anchor=anchor)

    assert {s.tweet_id for s in ranked} == {"a", "b"}
    for scored in ranked:
        assert scored.feature_breakdown["recency"] == 0.0

    # Single stderr line for the whole rank() call (not per-hit).
    captured = capsys.readouterr()
    stderr_lines = [
        line
        for line in captured.err.splitlines()
        if "ranker" in line.lower() or "recency" in line.lower() or "unparseable" in line.lower()
    ]
    assert len(stderr_lines) == 1
