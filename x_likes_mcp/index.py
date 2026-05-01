"""Index: build/load the PageIndex tree, hold the in-memory Tweet map.

This module owns the cache file (``output/pageindex_cache.pkl``) and the
mtime-based invalidation policy: if any ``.md`` under
``output/by_month/`` is newer than the cache, the tree is rebuilt on next
:func:`Index.open_or_build` call. Cache writes are atomic (``.tmp`` +
``os.replace``) so a crash mid-write does not corrupt the cache.

The PageIndex tree-build and query calls are isolated behind two
module-level seams (:func:`_build_tree`, :func:`_query`) so unit tests can
mock them without monkeypatching PageIndex itself. Task 3.1 leaves both
seams as ``NotImplementedError`` stubs; tasks 3.2 and 3.3 fill them in.

Boundary: this module imports only from Spec 1's public read API
(``load_export``, ``iter_monthly_markdown``, ``Tweet``), the local
``config`` module, and the standard library.
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from x_likes_exporter import iter_monthly_markdown, load_export
from x_likes_exporter.models import Tweet

from .config import Config


# Filename pattern for per-month Markdown: matches `likes_YYYY-MM.md`. The
# captured group is the `YYYY-MM` token used as a key in `paths_by_month`.
_MONTHLY_FILENAME_RE = re.compile(r"^likes_(\d{4}-\d{2})\.md$")


class IndexError(Exception):
    """Raised when the index cannot be built or loaded."""


@dataclass(frozen=True)
class SearchHit:
    """One match returned by :meth:`Index.search`."""

    tweet_id: str
    year_month: str
    handle: str
    snippet: str


@dataclass(frozen=True)
class MonthInfo:
    """One entry in the response of :meth:`Index.list_months`."""

    year_month: str
    path: Path
    tweet_count: int | None


def _build_tree(paths: list[Path], model: str) -> tuple[Any, dict]:
    """Build a PageIndex tree over the per-month Markdown files.

    Task 3.2 implements this. It will call PageIndex's tree-build entry
    point with ``paths`` and ``model`` and return a ``(tree, side_table)``
    tuple.
    """

    raise NotImplementedError("task 3.2")


def _query(tree_or_subtree: Any, query: str) -> list[Any]:
    """Run a PageIndex query against a tree (or subtree).

    Task 3.3 implements this. It will call PageIndex's query entry point
    and return a list of match objects.
    """

    raise NotImplementedError("task 3.3")


class Index:
    """In-memory index over a finished export.

    Holds the PageIndex ``tree`` plus the ``side_table`` that maps tree
    leaves to ``tweet_id`` strings, the in-memory ``dict[str, Tweet]``
    keyed on tweet id, the original ``list[Tweet]`` (used by
    ``list_months`` for month bucketing), and a ``paths_by_month`` map
    from ``YYYY-MM`` to the corresponding ``likes_YYYY-MM.md`` path under
    ``config.by_month_dir``.

    Instances are read-only by convention: methods do not mutate the
    tree, the side-table, or any of the maps.
    """

    def __init__(
        self,
        tree: Any,
        side_table: dict,
        tweets_by_id: dict[str, Tweet],
        tweets: list[Tweet],
        paths_by_month: dict[str, Path],
        config: Config,
    ) -> None:
        self.tree = tree
        self.side_table = side_table
        self.tweets_by_id = tweets_by_id
        self.tweets = tweets
        self.paths_by_month = paths_by_month
        self.config = config

    @classmethod
    def open_or_build(cls, config: Config) -> "Index":
        """Build a fresh index or load it from cache.

        Steps:
          1. Enumerate per-month Markdown files via
             :func:`iter_monthly_markdown`. Raise :class:`IndexError` if
             none yield.
          2. Compute ``newest_md_mtime`` over the enumerated paths.
          3. If the cache exists and its mtime is at least
             ``newest_md_mtime``, load ``(tree, side_table)`` from the
             pickle. Otherwise call :func:`_build_tree` and write the
             cache atomically (``.tmp`` + ``os.replace``).
          4. Load tweets via :func:`load_export` and key them by id.
          5. Build the ``paths_by_month`` map from filenames.

        Raises:
            IndexError: when ``config.by_month_dir`` yields no
                ``likes_YYYY-MM.md`` files.
        """

        # 1. Enumerate per-month Markdown files. iter_monthly_markdown
        #    raises FileNotFoundError if the directory itself is missing
        #    or not a directory; convert that to IndexError so the
        #    startup pipeline in __main__ can treat both shapes the same
        #    way.
        try:
            paths: list[Path] = list(iter_monthly_markdown(config.by_month_dir))
        except FileNotFoundError as exc:
            raise IndexError("output/by_month/ is empty or missing") from exc

        if not paths:
            raise IndexError("output/by_month/ is empty or missing")

        # 2. Compute the newest .md mtime for the cache freshness check.
        newest_md_mtime = max(p.stat().st_mtime for p in paths)

        # 3. Cache hit/miss.
        # Pickle usage is a deliberate design choice (design.md line 155-156,
        # 432, 564, 737): single-user, single-machine cache file written by
        # this server only, never crosses a trust boundary. PageIndex's
        # tree shape is opaque at this layer, so JSON is not available.
        # If PageIndex publishes a safer serialization at impl time, prefer
        # that. CWE-502 acknowledged and mitigated by the trust boundary.
        cache_path = config.cache_path
        if cache_path.exists() and cache_path.stat().st_mtime >= newest_md_mtime:
            with cache_path.open("rb") as fh:
                tree, side_table = pickle.load(fh)  # nosem: avoid-pickle
        else:
            tree, side_table = _build_tree(paths, config.openai_model)
            # Atomic cache write: dump to .tmp then os.replace. Ensures
            # a crash mid-write does not corrupt the cache file.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp_path.open("wb") as fh:
                pickle.dump((tree, side_table), fh)  # nosem: avoid-pickle
            os.replace(tmp_path, cache_path)

        # 4. Load the tweet list and key by id.
        tweets: list[Tweet] = load_export(config.likes_json)
        tweets_by_id: dict[str, Tweet] = {t.id: t for t in tweets}

        # 5. Map YYYY-MM -> Path by parsing each filename.
        paths_by_month: dict[str, Path] = {}
        for p in paths:
            m = _MONTHLY_FILENAME_RE.match(p.name)
            if m is None:
                # iter_monthly_markdown only yields likes_YYYY-MM.md
                # files, so this should not be reachable. Skip
                # defensively rather than blow up the build.
                continue
            paths_by_month[m.group(1)] = p

        return cls(
            tree=tree,
            side_table=side_table,
            tweets_by_id=tweets_by_id,
            tweets=tweets,
            paths_by_month=paths_by_month,
            config=config,
        )
