"""Entry point for ``python -m x_likes_mcp``.

Boot sequence:
  1. ``load_config()`` — parse ``.env`` / ``os.environ``, validate the
     OpenAI vars, build the :class:`Config` (including ``ranker_weights``).
  2. ``TweetIndex.open_or_build(config, config.ranker_weights)`` — build
     or load the cached :class:`TweetTree`, load the export, precompute
     author affinity.
  3. ``server.run(index)`` — drive the stdio MCP loop until the client
     disconnects (skipped under ``--init`` / ``--search``).
  4. Return ``0`` on clean shutdown.

Startup-error handling:
  * :class:`config.ConfigError` (bad ``.env``), :class:`index.IndexError`
    (empty / missing ``output/by_month/``), and :class:`FileNotFoundError`
    (no ``likes.json``) are caught at the top of :func:`main`. A single
    stderr line names the failing condition; the function returns exit
    code ``2``.
  * :class:`KeyboardInterrupt` exits ``0`` cleanly so ``Ctrl-C`` from a
    foreground shell does not look like a crash.
  * Other exceptions during startup propagate (real bugs surface as
    tracebacks).

CLI flags:
  ``--init``: build (or load) the index, print a one-shot summary on
  stderr, exit ``0`` without entering the stdio loop. Useful for warm-up
  or smoke testing without an MCP client attached.

  ``--search QUERY``: build (or load) the index, run the same hybrid
  ``search_likes`` pipeline the MCP tool uses, print results on stdout,
  exit. Mutually exclusive with ``--init``. Filter flags (``--year``,
  ``--month-start``, ``--month-end``), ``--with-why``, ``--limit``, and
  ``--json`` shape the query and the output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import server, tools
from .config import ConfigError, load_config
from .errors import ToolError
from .index import IndexError, TweetIndex
from .sanitize import safe_http_url, sanitize_text


_TCO_RE = re.compile(r"https?://t\.co/\S+")
_TWEET_ID_RE = re.compile(r"^[0-9]+$")


def _local_media_files(tweet_id: str, media_dir: Path) -> list[Path]:
    """Return downloaded media files for ``tweet_id`` under ``media_dir``.

    The exporter writes media as ``<tweet_id>_<index>.<ext>``. We glob
    instead of trusting ``Tweet.media[i].local_path`` because the export
    leaves that field unpopulated for older runs.

    ``tweet_id`` is validated against ``^[0-9]+$`` before being
    interpolated into the glob pattern. Twitter IDs are always numeric;
    rejecting anything else stops a malformed/malicious id (e.g. one
    containing ``..`` or ``/``) from escaping ``media_dir`` via
    ``Path.glob``'s literal handling of those tokens.
    """

    if not media_dir.is_dir():
        return []
    if not _TWEET_ID_RE.match(tweet_id):
        return []
    return sorted(media_dir.glob(f"{tweet_id}_*"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="x-likes-mcp")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init",
        action="store_true",
        help=(
            "Build or load the index, print a summary on stderr, exit. "
            "Skips the stdio MCP loop."
        ),
    )
    mode.add_argument(
        "--search",
        metavar="QUERY",
        help=(
            "Run the hybrid search_likes pipeline against QUERY and print "
            "the ranked hits. Skips the stdio MCP loop."
        ),
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month-start", default=None, dest="month_start")
    parser.add_argument("--month-end", default=None, dest="month_end")
    parser.add_argument(
        "--with-why",
        action="store_true",
        help="Run the optional walker explainer over the top hits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit the number of printed hits (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_out",
        help="Print results as one JSON object per line instead of pretty text.",
    )
    return parser.parse_args(argv)


def _print_init_summary(index: TweetIndex) -> None:
    config = index.config
    matrix = index.corpus.matrix
    rows, dims = matrix.shape
    print("x_likes_mcp: init complete", file=sys.stderr)
    print(f"  output_dir       : {config.output_dir}", file=sys.stderr)
    print(f"  tweets           : {len(index.tweets)}", file=sys.stderr)
    print(f"  months           : {len(index.paths_by_month)}", file=sys.stderr)
    print(f"  embedding_model  : {index.corpus.model_name}", file=sys.stderr)
    print(f"  corpus_matrix    : {rows} rows x {dims} dims", file=sys.stderr)


def _expand_snippet(snippet: str, urls: list[str]) -> str:
    """Strip ``t.co`` shortlinks from the snippet and append the resolved
    URLs the export already captured in ``Tweet.urls``.

    Both the snippet and each URL are passed through
    :func:`sanitize_text` so terminal-control / BiDi tricks in tweet
    content cannot reach the user's screen. URLs are additionally
    filtered with :func:`safe_http_url` to ``http://`` / ``https://``
    only — anything else (``javascript:``, ``data:``, ``file://``, or a
    URL that turned into garbage after sanitization) is dropped.
    """

    cleaned = _TCO_RE.sub("", sanitize_text(snippet))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    real_urls = [u for u in (safe_http_url(raw) for raw in urls) if u]
    if not real_urls:
        return cleaned
    if cleaned:
        return cleaned + "  " + " ".join(real_urls)
    return " ".join(real_urls)


def _format_meta_line(
    i: int,
    hit: dict,
    *,
    color: bool,
) -> str:
    parts = [
        f"score={hit['score']:.2f}",
        f"wr={hit['walker_relevance']:.2f}",
        hit["year_month"] or "?",
        f"@{hit['handle'] or '?'}",
        f"id={hit['tweet_id']}",
    ]
    sep = " │ "
    body = sep.join(parts)
    prefix = f"{i:>2}."
    if color:
        # Dim the metadata so the snippet on the next line stands out.
        return f"\x1b[1m{prefix}\x1b[0m \x1b[2m{body}\x1b[0m"
    return f"{prefix} {body}"


def _print_search_results(
    hits: list[dict],
    *,
    index: TweetIndex,
    json_out: bool,
) -> None:
    if json_out:
        for hit in hits:
            print(json.dumps(hit, ensure_ascii=False))
        return

    if not hits:
        print("(no hits)", file=sys.stderr)
        return

    color = sys.stdout.isatty()
    media_dir = (index.config.output_dir / "media").resolve()

    for i, hit in enumerate(hits, 1):
        tweet = index.tweets_by_id.get(hit["tweet_id"])
        urls = list(tweet.urls) if tweet is not None else []
        snippet = _expand_snippet(hit["snippet"].replace("\n", " "), urls)

        print(_format_meta_line(i, hit, color=color))
        print(f"    {snippet}")
        for path in _local_media_files(hit["tweet_id"], media_dir):
            print(f"    media: file://{path}")
        why = hit.get("why") or ""
        if why:
            print(f"    why: {why}")
        print()


def _run_search(index: TweetIndex, args: argparse.Namespace) -> int:
    try:
        results = tools.search_likes(
            index,
            args.search,
            year=args.year,
            month_start=args.month_start,
            month_end=args.month_end,
            with_why=args.with_why,
        )
    except ToolError as exc:
        print(f"x_likes_mcp: search error: {exc}", file=sys.stderr)
        return 2

    limit = max(0, args.limit)
    _print_search_results(
        results[:limit],
        index=index,
        json_out=args.json_out,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the startup pipeline and the stdio loop. Return process exit code."""
    args = _parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"x_likes_mcp: configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        index = TweetIndex.open_or_build(config, config.ranker_weights)
    except IndexError as exc:
        print(f"x_likes_mcp: index error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"x_likes_mcp: missing export file: {exc}", file=sys.stderr)
        return 2

    if args.init:
        _print_init_summary(index)
        return 0

    if args.search is not None:
        return _run_search(index, args)

    try:
        server.run(index)
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
