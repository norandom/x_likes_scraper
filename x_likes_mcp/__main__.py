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
import sys

from . import server, tools
from .config import ConfigError, load_config
from .errors import ToolError
from .index import IndexError, TweetIndex


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


def _print_search_results(
    hits: list[dict],
    *,
    json_out: bool,
) -> None:
    if json_out:
        for hit in hits:
            print(json.dumps(hit, ensure_ascii=False))
        return

    if not hits:
        print("(no hits)", file=sys.stderr)
        return

    for i, hit in enumerate(hits, 1):
        score = hit["score"]
        wr = hit["walker_relevance"]
        ym = hit["year_month"] or "?"
        handle = hit["handle"] or "?"
        snippet = hit["snippet"].replace("\n", " ")
        print(
            f"{i:>2}. [score={score:.3f} wr={wr:.2f}] {ym} @{handle} "
            f"({hit['tweet_id']})"
        )
        print(f"    {snippet}")
        why = hit.get("why") or ""
        if why:
            print(f"    why: {why}")


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
    _print_search_results(results[:limit], json_out=args.json_out)
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
