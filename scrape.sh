#!/usr/bin/env bash
# Wrapper around cli.py that reads config from .env and runs via uv.
# Usage:
#   ./scrape.sh                # full export, resuming from a checkpoint if one exists
#   ./scrape.sh --no-media     # skip media download (fast)
#   ./scrape.sh --stats        # print stats after the run
# Any arguments you pass are forwarded to cli.py.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed. See https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [ ! -f .env ]; then
    echo "Error: .env not found." >&2
    echo "Copy .env.sample to .env and fill in X_USER_ID and COOKIES_FILE." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${X_USER_ID:?X_USER_ID must be set in .env}"
: "${COOKIES_FILE:=cookies.json}"
: "${OUTPUT_DIR:=output}"

if [ ! -f "$COOKIES_FILE" ]; then
    echo "Error: cookies file not found at '$COOKIES_FILE'." >&2
    echo "Export your X cookies first (see README)." >&2
    exit 1
fi

echo "Exporting likes for @${X_USERNAME:-?} (user ID ${X_USER_ID}) to ${OUTPUT_DIR}/"
exec uv run python cli.py "$COOKIES_FILE" "$X_USER_ID" --output "$OUTPUT_DIR" --resume "$@"
