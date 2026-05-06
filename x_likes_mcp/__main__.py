"""Entry point for ``python -m x_likes_mcp``.

Boot sequence:
  1. ``load_config()`` — parse ``.env`` / ``os.environ``, validate the
     OpenAI vars, build the :class:`Config` (including ``ranker_weights``).
  2. ``TweetIndex.open_or_build(config, config.ranker_weights)`` — build
     or load the cached :class:`TweetTree`, load the export, precompute
     author affinity.
  3. ``server.run(index)`` — drive the stdio MCP loop until the client
     disconnects (skipped under ``--init`` / ``--search`` / ``--report``
     / ``--report-optimize``).
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

  ``--report {brief,synthesis,trend} --query Q``: drive the
  synthesis-report orchestrator and write the rendered markdown to
  ``--out PATH`` (or stdout when ``--out`` is omitted). ``--fetch-urls``
  enables the crawl4ai container probe + URL fetch path; ``--hops``
  picks 1- or 2-hop search. Filter flags (``--year``, ``--month-start``,
  ``--month-end``, ``--limit``) are honored. Mutually exclusive with
  ``--init`` / ``--search`` / ``--report-optimize``. Exits ``2`` on any
  orchestrator failure (LM unreachable, crawl4ai unreachable while
  ``--fetch-urls`` was set, malformed shape, synthesis-validation
  error, missing ``--query``).

  ``--report-optimize {brief,synthesis,trend}``: read labeled demos
  from ``<output_dir>/synthesis_labeled/<shape>.json`` and run the
  DSPy optimizer (``BootstrapFewShot``) for that shape. The compiled
  program lands in ``<output_dir>/synthesis_compiled/``. Mutually
  exclusive with the other modes; exits ``2`` on any optimizer or
  I/O failure.
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
from .synthesis import compiled, orchestrator
from .synthesis.compiled import LabeledExample
from .synthesis.kg import serialize_kg
from .synthesis.mindmap import DEFAULT_MIN_LEVEL3_WEIGHT, render_mindmap
from .synthesis.multihop import (
    HopsOutOfRange,
    run_round_one,
    run_round_two,
    validate_hops,
)
from .synthesis.orchestrator import OrchestratorError
from .synthesis.shapes import ReportShape, parse_report_shape
from .synthesis.types import ReportOptions

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


def _add_mode_group(parser: argparse.ArgumentParser) -> None:
    """Register the mutually-exclusive top-level mode flags on ``parser``."""

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init",
        action="store_true",
        help=(
            "Build or load the index, print a summary on stderr, exit. Skips the stdio MCP loop."
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
    mode.add_argument(
        "--report",
        choices=[shape.value for shape in ReportShape],
        default=None,
        help=(
            "Run the synthesis-report orchestrator for the given shape "
            "(brief, synthesis, trend) and print or write the rendered "
            "markdown. Requires --query."
        ),
    )
    mode.add_argument(
        "--report-optimize",
        choices=[shape.value for shape in ReportShape],
        default=None,
        dest="report_optimize",
        help=(
            "Run the DSPy optimizer for the given report shape, reading "
            "labeled demos from output/synthesis_labeled/<shape>.json and "
            "writing the compiled program to output/synthesis_compiled/."
        ),
    )
    mode.add_argument(
        "--kg",
        metavar="QUERY",
        default=None,
        help=(
            "Run round-1 search (and round-2 fan-out when --hops 2), build "
            "the in-memory knowledge graph, render it as a mermaid mindmap "
            "(or JSON via --json), print to --out or stdout, exit. No LM "
            "call, no URL fetch."
        ),
    )


def _add_report_flags(parser: argparse.ArgumentParser) -> None:
    """Register flags consumed by ``--report`` / ``--report-optimize`` / ``--kg``."""

    parser.add_argument(
        "--query",
        default=None,
        help="Synthesis query string for --report / --report-optimize modes.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write the rendered report to this path (default: stdout).",
    )
    parser.add_argument(
        "--fetch-urls",
        action="store_true",
        dest="fetch_urls",
        help=(
            "Allow the synthesis-report orchestrator to fetch URLs cited "
            "in the recalled tweets via the configured crawl4ai container."
        ),
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of search hops the synthesis-report orchestrator runs (default: 1).",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=None,
        dest="min_weight",
        help=(
            "For --kg: drop level-3 nodes (top-K entities) whose cumulative "
            "weight is below this threshold. Defaults to "
            "synthesis.mindmap.DEFAULT_MIN_LEVEL3_WEIGHT (2.0). Set to 0 to "
            "disable the filter."
        ),
    )
    parser.add_argument(
        "--filter-entities",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="filter_entities",
        help=(
            "Run the LM-backed entity-relevance filter "
            "(synthesis.dspy_modules.FilterEntitiesByRelevance) over the "
            "round-1 KG before round-2 fan-out. Defaults to ON for "
            "--report and --report-optimize, OFF for --kg (no LM call "
            "in --kg mode). Use --no-filter-entities to disable for "
            "--report; --filter-entities to enable for --kg "
            "(requires OPENAI_BASE_URL configured)."
        ),
    )


def _add_search_filter_flags(parser: argparse.ArgumentParser) -> None:
    """Register the shared filter / output flags used by --search and friends."""

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="x-likes-mcp")
    _add_mode_group(parser)
    _add_report_flags(parser)
    _add_search_filter_flags(parser)
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
        tweet_url = hit.get("tweet_url") or ""
        if tweet_url:
            print(f"    {tweet_url}")
        print(f"    {snippet}")
        for path in _local_media_files(hit["tweet_id"], media_dir):
            print(f"    media: file://{path}")
        why = hit.get("why") or ""
        if why:
            print(f"    why: {why}")
        print()


def _run_report(index: TweetIndex, args: argparse.Namespace) -> int:
    """Build :class:`ReportOptions`, drive the orchestrator, write markdown.

    Returns ``0`` on success, ``2`` on any failure (LM unreachable,
    crawl4ai unreachable while ``--fetch-urls`` was set, malformed shape,
    synthesis-validation error, missing ``--query``).
    """

    query = (args.query or "").strip()
    if not query:
        print(
            "x_likes_mcp: --report requires --query (the synthesis query string)",
            file=sys.stderr,
        )
        return 2

    try:
        shape = parse_report_shape(args.report)
    except ValueError as exc:  # pragma: no cover - argparse pre-filters this.
        print(f"x_likes_mcp: invalid report shape: {exc}", file=sys.stderr)
        return 2

    # Default for --report is filter-entities=True (LM available, noise
    # filter is the whole point of LM-backed synthesis). The user can
    # turn it off with --no-filter-entities when prompt-cost matters.
    filter_entities = True if args.filter_entities is None else bool(args.filter_entities)

    options = ReportOptions(
        query=query,
        shape=shape,
        fetch_urls=bool(args.fetch_urls),
        hops=int(args.hops),
        year=args.year,
        month_start=args.month_start,
        month_end=args.month_end,
        limit=int(args.limit),
        filter_entities=filter_entities,
    )

    try:
        result = orchestrator.run_report(index, options, config=index.config)
    except OrchestratorError as exc:
        print(
            f"x_likes_mcp: report failed [{exc.category}]: {exc.message}",
            file=sys.stderr,
        )
        return 2

    markdown = result.markdown
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    else:
        # ``print`` adds a trailing newline; ``markdown`` typically already
        # ends with one. Use ``sys.stdout.write`` so we don't double up.
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")

    return 0


def _run_report_optimize(index: TweetIndex, args: argparse.Namespace) -> int:
    """Run the DSPy optimizer for the requested report shape.

    Reads labeled demos from
    ``<output_dir>/synthesis_labeled/<shape>.json`` (a JSON list of
    ``{"query", "fenced_context_raw", "expected_outputs"}`` records) and
    writes the compiled program to ``<output_dir>/synthesis_compiled/``
    via :func:`compiled.run_optimizer`. Sanitization and fencing of
    each demo happens inside :func:`compiled.prepare_demo`, which
    ``run_optimizer`` invokes per example.

    Returns ``0`` on success, ``2`` on any failure (missing labeled
    file, malformed JSON, optimizer exception).
    """

    try:
        shape = parse_report_shape(args.report_optimize)
    except ValueError as exc:  # pragma: no cover - argparse pre-filters this.
        print(f"x_likes_mcp: invalid report shape: {exc}", file=sys.stderr)
        return 2

    output_dir = index.config.output_dir
    labeled_path = output_dir / "synthesis_labeled" / f"{shape.value}.json"
    if not labeled_path.exists():
        print(
            "x_likes_mcp: --report-optimize requires labeled examples at "
            f"{labeled_path} (a JSON list of {{query, fenced_context_raw, "
            "expected_outputs}} records)",
            file=sys.stderr,
        )
        return 2

    try:
        raw = json.loads(labeled_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"x_likes_mcp: failed to read labeled examples at {labeled_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    if not isinstance(raw, list):
        print(
            f"x_likes_mcp: labeled examples at {labeled_path} must be a JSON list",
            file=sys.stderr,
        )
        return 2

    examples: list[LabeledExample] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            print(
                f"x_likes_mcp: labeled example #{i} in {labeled_path} is not an object",
                file=sys.stderr,
            )
            return 2
        try:
            examples.append(
                LabeledExample(
                    query=str(entry["query"]),
                    fenced_context_raw=str(entry["fenced_context_raw"]),
                    expected_outputs=dict(entry.get("expected_outputs") or {}),
                )
            )
        except KeyError as exc:
            print(
                f"x_likes_mcp: labeled example #{i} in {labeled_path} is "
                f"missing required field {exc}",
                file=sys.stderr,
            )
            return 2

    compiled_root = output_dir / "synthesis_compiled"
    try:
        compiled.run_optimizer(
            shape,
            examples,
            root=compiled_root,
            optimizer="BootstrapFewShot",
        )
    except Exception as exc:
        print(
            f"x_likes_mcp: optimizer failed for shape {shape.value}: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"x_likes_mcp: optimizer wrote compiled program to "
        f"{compiled_root / f'{shape.value}.json'}",
        file=sys.stderr,
    )
    return 0


def _run_kg(index: TweetIndex, args: argparse.Namespace) -> int:
    """Build and dump the synthesis KG for ``args.kg`` (no LM, no fetch).

    Pipeline: ``run_round_one`` → optional ``run_round_two`` (when
    ``--hops 2``) → ``orchestrator.build_kg`` (regex entity pass with
    DSPy fallback only for empty-regex hits) → render as mermaid mindmap
    (default) or JSON (``--json``). Output goes to ``args.out`` or
    stdout. Exit ``0`` on success, ``2`` on validation / search failure.
    """

    query = (args.kg or "").strip()
    if not query:
        print("x_likes_mcp: --kg requires a non-empty query", file=sys.stderr)
        return 2

    options = ReportOptions(
        query=query,
        shape=ReportShape.SYNTHESIS,  # placeholder — not consumed by KG path
        fetch_urls=False,
        hops=args.hops,
        year=args.year,
        month_start=args.month_start,
        month_end=args.month_end,
        limit=args.limit,
    )

    try:
        validate_hops(options.hops, max_hops=index.config.synthesis_max_hops)
    except HopsOutOfRange as exc:
        print(f"x_likes_mcp: kg error: invalid_input: {exc}", file=sys.stderr)
        return 2

    try:
        hits_round_one = run_round_one(index, options)
    except ValueError as exc:
        print(f"x_likes_mcp: kg error: invalid_input: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"x_likes_mcp: kg error: search failed: {exc}", file=sys.stderr)
        return 2

    if not hits_round_one:
        print(
            f"x_likes_mcp: kg: no matching tweets for {query!r}",
            file=sys.stderr,
        )
        # Emit an empty mindmap / JSON shell so downstream consumers see
        # a well-formed payload rather than nothing.
        kg = orchestrator.build_kg(query, [], index, dspy_fallback=False)
    else:
        kg = orchestrator.build_kg(query, hits_round_one, index, dspy_fallback=False)
        if options.hops >= 2:
            try:
                hits_round_two = run_round_two(
                    index,
                    options,
                    kg,
                    k=index.config.synthesis_round_two_k,
                )
            except Exception as exc:
                print(f"x_likes_mcp: kg error: round-2 failed: {exc}", file=sys.stderr)
                return 2
            round_two_only = [
                hit
                for hit in hits_round_two
                if hit.tweet_id not in {h.tweet_id for h in hits_round_one}
            ]
            if round_two_only:
                orchestrator.build_kg(
                    query, round_two_only, index, existing=kg, dspy_fallback=False
                )

    if args.json_out:
        body = json.dumps(serialize_kg(kg), ensure_ascii=False, indent=2) + "\n"
    else:
        min_weight = args.min_weight if args.min_weight is not None else DEFAULT_MIN_LEVEL3_WEIGHT
        body = render_mindmap(query, kg, min_level3_weight=min_weight) + "\n"

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)

    return 0


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

    if args.report is not None:
        return _run_report(index, args)

    if args.report_optimize is not None:
        return _run_report_optimize(index, args)

    if args.kg is not None:
        return _run_kg(index, args)

    try:
        server.run(index)
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
