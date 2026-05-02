"""Entry point for ``python -m x_likes_mcp``.

Boot sequence:
  1. ``load_config()`` — parse ``.env`` / ``os.environ``, validate the
     OpenAI vars, build the :class:`Config` (including ``ranker_weights``).
  2. ``TweetIndex.open_or_build(config, config.ranker_weights)`` — build
     or load the cached :class:`TweetTree`, load the export, precompute
     author affinity.
  3. ``server.run(index)`` — drive the stdio MCP loop until the client
     disconnects.
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
"""

from __future__ import annotations

import sys

from . import server
from .config import ConfigError, load_config
from .index import IndexError, TweetIndex


def main() -> int:
    """Run the startup pipeline and the stdio loop. Return process exit code."""
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

    try:
        server.run(index)
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
