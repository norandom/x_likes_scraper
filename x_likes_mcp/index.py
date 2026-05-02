"""TweetIndex: orchestrator that owns the cached TweetTree, the in-memory
tweet map, the precomputed author affinity, and the four read paths the MCP
tools call.

The cache file (``output/tweet_tree_cache.pkl``) holds the parsed
:class:`TweetTree` and is invalidated by mtime: if any ``likes_YYYY-MM.md``
under ``output/by_month/`` is newer than the cache, the tree is rebuilt on
next :meth:`TweetIndex.open_or_build` call. Cache writes are atomic
(``.tmp`` + ``os.replace``) so a crash mid-write does not corrupt the cache.

Boundary: this module imports only from Spec 1's public read API
(``load_export``, ``iter_monthly_markdown``, ``Tweet``), the local
``config`` and ``tree`` modules, and the standard library.
"""

from __future__ import annotations

import math
import os
import pickle  # nosem: avoid-pickle (single-user local cache, see comment in open_or_build)
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datetime import datetime, timezone

from x_likes_exporter import iter_monthly_markdown, load_export

from . import ranker as ranker_module
from . import tree as tree_module
from . import walker as walker_module
from .config import Config, RankerWeights

if TYPE_CHECKING:  # pragma: no cover
    from x_likes_exporter.models import Tweet

    from .tree import TweetTree


# Filename pattern for per-month Markdown: matches ``likes_YYYY-MM.md``.
_MONTHLY_FILENAME_RE = re.compile(r"^likes_(\d{4}-\d{2})\.md$")


class IndexError(Exception):  # noqa: A001 (intentional shadow inside this module's namespace)
    """Raised when the index cannot be built or loaded."""


@dataclass(frozen=True)
class MonthInfo:
    """Per-month metadata returned by :meth:`TweetIndex.list_months`."""

    year_month: str
    path: Path
    tweet_count: int | None


def _compute_author_affinity(tweets: list[Tweet]) -> dict[str, float]:
    """Affinity score per handle: ``log1p(count_of_user_likes_from_handle)``.

    Lives here for now; task 3.3c moves it into ``ranker.py`` and this
    module re-imports.
    """
    counts: Counter[str] = Counter(
        t.user.screen_name for t in tweets if t.user.screen_name
    )
    return {handle: math.log1p(count) for handle, count in counts.items()}


@dataclass
class TweetIndex:
    """In-memory index over a finished export.

    Holds the parsed :class:`TweetTree`, the ``tweets_by_id`` map keyed on
    ``tweet.id``, the original ``list[Tweet]`` (used by ``list_months``
    for tweet counts), the ``paths_by_month`` map from ``YYYY-MM`` to the
    corresponding ``likes_YYYY-MM.md`` path, the precomputed
    ``author_affinity`` dict, the loaded :class:`Config`, and the
    :class:`RankerWeights`.

    Instances are read-only by convention: methods do not mutate any of
    the maps or the tree.
    """

    tree: TweetTree
    tweets_by_id: dict[str, Tweet]
    tweets: list[Tweet]
    paths_by_month: dict[str, Path]
    author_affinity: dict[str, float]
    config: Config
    weights: RankerWeights

    @classmethod
    def open_or_build(cls, config: Config, weights: RankerWeights) -> TweetIndex:
        """Build a fresh index or load the tree from cache.

        Steps:
          1. Enumerate per-month Markdown files via
             :func:`iter_monthly_markdown`. Raise :class:`IndexError` if
             none yield.
          2. Compute ``newest_md_mtime`` over the enumerated paths.
          3. Cache hit if the cache exists and its mtime is at least
             ``newest_md_mtime``. Otherwise call
             :func:`tree.build_tree` and write the cache atomically
             (``.tmp`` + ``os.replace``).
          4. Load tweets via :func:`load_export` and key them by id.
          5. Build the ``paths_by_month`` map from filenames.
          6. Compute ``author_affinity``.

        Raises:
            IndexError: when ``config.by_month_dir`` yields no
                ``likes_YYYY-MM.md`` files.
        """
        # 1. Enumerate. iter_monthly_markdown raises FileNotFoundError when
        #    the directory itself is missing or not a directory; convert
        #    to IndexError so the startup pipeline can treat both shapes
        #    the same way.
        try:
            paths: list[Path] = list(iter_monthly_markdown(config.by_month_dir))
        except FileNotFoundError as exc:
            raise IndexError("output/by_month/ is empty or missing") from exc

        if not paths:
            raise IndexError("output/by_month/ is empty or missing")

        # 2. Newest .md mtime for the cache freshness check.
        newest_md_mtime = max(p.stat().st_mtime for p in paths)

        # 3. Cache hit/miss. Pickle usage is a deliberate design choice
        #    (design.md): single-user, single-machine cache file written
        #    by this server only, never crosses a trust boundary.
        cache_path = config.cache_path
        if cache_path.exists() and cache_path.stat().st_mtime >= newest_md_mtime:
            with cache_path.open("rb") as fh:
                tree_obj = pickle.load(fh)  # nosem: avoid-pickle
        else:
            tree_obj = tree_module.build_tree(config.by_month_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: dump to a sibling .tmp file then os.replace.
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=str(cache_path.parent),
                prefix=cache_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as fh:
                pickle.dump(tree_obj, fh)  # nosem: avoid-pickle
                tmp_name = fh.name
            os.replace(tmp_name, cache_path)

        # 4. Tweet list keyed by id.
        tweets: list[Tweet] = load_export(config.likes_json)
        tweets_by_id: dict[str, Tweet] = {t.id: t for t in tweets}

        # 5. paths_by_month from filenames.
        paths_by_month: dict[str, Path] = {}
        for p in paths:
            match = _MONTHLY_FILENAME_RE.match(p.name)
            if match is not None:
                paths_by_month[match.group(1)] = p

        # 6. Precompute author affinity.
        author_affinity = _compute_author_affinity(tweets)

        return cls(
            tree=tree_obj,
            tweets_by_id=tweets_by_id,
            tweets=tweets,
            paths_by_month=paths_by_month,
            author_affinity=author_affinity,
            config=config,
            weights=weights,
        )

    # --- per-tweet / per-month read paths ---------------------------------

    def lookup_tweet(self, tweet_id: str) -> Tweet | None:
        """Return the :class:`Tweet` with the given id, or ``None``."""
        return self.tweets_by_id.get(tweet_id)

    def list_months(self) -> list[MonthInfo]:
        """List available months reverse-chronologically with tweet counts.

        Counts come from the in-memory tweet list grouped by
        ``Tweet.get_created_datetime``; tweets with unparseable
        ``created_at`` are skipped from counts but the month entry
        still appears as long as the file is on disk.
        """
        # Group counts by YYYY-MM via the tweet's parseable date.
        counts: dict[str, int] = {}
        for tweet in self.tweets:
            try:
                created = tweet.get_created_datetime()
            except (ValueError, TypeError):
                continue
            ym = created.strftime("%Y-%m")
            counts[ym] = counts.get(ym, 0) + 1

        infos: list[MonthInfo] = []
        for year_month, path in self.paths_by_month.items():
            infos.append(
                MonthInfo(
                    year_month=year_month,
                    path=path,
                    tweet_count=counts.get(year_month),
                )
            )
        infos.sort(key=lambda m: m.year_month, reverse=True)
        return infos

    def get_month_markdown(self, year_month: str) -> str | None:
        """Read the per-month Markdown file. ``None`` if missing."""
        path = self.paths_by_month.get(year_month)
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    # --- query / search seams ---------------------------------------------

    def _resolve_filter(
        self,
        year: int | None,
        month_start: str | None,
        month_end: str | None,
    ) -> list[str] | None:
        """Resolve the structured filter to a list of YYYY-MM strings.

        Returns:
            ``None`` when no filter is set (search every month).
            A list of ``YYYY-MM`` strings spanning the requested range
            otherwise.

        Raises:
            ValueError: when the filter combination is invalid.
                ``tools.search_likes`` translates that into
                ``errors.invalid_input("filter", ...)``.

        Rules:
          - ``month_start`` set requires ``year`` set.
          - ``month_end`` set requires ``month_start`` set.
          - When both ``month_start`` and ``month_end`` set,
            ``month_start <= month_end``.
          - All three ``None`` returns ``None`` (every month).
          - ``year`` only spans the whole year (12 months).
          - ``year`` + ``month_start`` only is one month.
          - ``year`` + ``month_start`` + ``month_end`` is the inclusive
            month range.
        """
        if year is None and month_start is None and month_end is None:
            return None

        if month_start is None and month_end is not None:
            raise ValueError("filter: month_end requires month_start")

        if month_start is not None and year is None:
            raise ValueError("filter: month_start requires year")

        # year alone -> all 12 months
        if year is not None and month_start is None:
            return [f"{year}-{m:02d}" for m in range(1, 13)]

        # Validate month formats and ranges.
        try:
            ms = int(month_start) if month_start is not None else None
            me = int(month_end) if month_end is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "filter: month values must be two-digit numeric strings"
            ) from exc

        if ms is None or not (1 <= ms <= 12):
            raise ValueError("filter: month_start must be in 01..12")
        if me is not None and not (1 <= me <= 12):
            raise ValueError("filter: month_end must be in 01..12")

        if me is None:
            return [f"{year}-{ms:02d}"]

        if ms > me:
            raise ValueError("filter: month_start must be <= month_end")

        return [f"{year}-{m:02d}" for m in range(ms, me + 1)]

    def search(
        self,
        query: str,
        year: int | None = None,
        month_start: str | None = None,
        month_end: str | None = None,
        top_n: int = 50,
    ) -> list[ranker_module.ScoredHit]:
        """Resolve filter, walk the in-scope subtree, rank, return top-N.

        Walker and ranker exceptions propagate; the tools layer shapes
        them into MCP error responses.
        """
        months_in_scope = self._resolve_filter(year, month_start, month_end)
        anchor = _compute_anchor(year, month_start, month_end)

        walker_hits = walker_module.walk(
            self.tree, query, months_in_scope, self.config
        )
        scored = ranker_module.rank(
            walker_hits,
            self.tweets_by_id,
            self.author_affinity,
            self.weights,
            anchor=anchor,
        )
        return scored[:top_n]


def _compute_anchor(
    year: int | None,
    month_start: str | None,
    month_end: str | None,
) -> datetime:
    """Recency anchor for the ranker:

    - end of ``month_end``'s month if both ``month_start`` and ``month_end`` set
    - end of ``month_start``'s month if ``month_start`` set without ``month_end``
    - end of ``year`` if only ``year`` set
    - ``datetime.now(timezone.utc)`` otherwise
    """
    now_utc = datetime.now(timezone.utc)

    if year is None:
        return now_utc

    if month_start is None and month_end is None:
        return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    target_month_str = month_end or month_start
    target_month = int(target_month_str)  # already validated by _resolve_filter
    if target_month == 12:
        return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    next_month_first = datetime(year, target_month + 1, 1, tzinfo=timezone.utc)
    # End of target_month = first of next month minus a microsecond.
    return datetime.fromtimestamp(
        next_month_first.timestamp() - 0.000001, tz=timezone.utc
    )
