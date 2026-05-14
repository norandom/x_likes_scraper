"""Dagger pipeline: lint, type-check, test, and build the x-likes-exporter wheel.

Runs the same gates the project uses locally (ruff, mypy on ``x_likes_mcp``,
pytest) inside an ``astral-sh/uv`` container, then ``uv build`` produces the
sdist + wheel and exports ``./dist`` back to the host.

Local invocation:
    uv pip install --system dagger-io anyio   # or into a venv
    dagger run python ci/build.py

CI invocation: see ``.github/workflows/build.yml`` — the Dagger CLI is
provided by ``dagger/dagger-for-github`` and this script runs against it.
"""

from __future__ import annotations

import sys

import anyio
import dagger
from dagger import dag

UV_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"

# Keep the build context lean: ignore caches, scan outputs, generated data,
# and editor / agent state that would otherwise bust the cache on every run.
HOST_EXCLUDES = [
    ".venv",
    ".venv-dagger",
    ".git",
    ".github",
    ".claude",
    ".kiro",
    ".lean-ctx",
    ".sentrux",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "output",
    "examples",
    "dist",
    "**/__pycache__",
    "static_analysis_codeql_*",
    "static_analysis_semgrep_*",
]


async def main() -> None:
    async with dagger.connection():
        src = dag.host().directory(".", exclude=HOST_EXCLUDES)
        uv_cache = dag.cache_volume("uv-cache-x-likes-exporter")

        base = (
            dag.container()
            .from_(UV_IMAGE)
            .with_env_variable("UV_LINK_MODE", "copy")
            .with_mounted_cache("/root/.cache/uv", uv_cache)
            .with_workdir("/src")
            .with_directory("/src", src)
            .with_exec(["uv", "sync", "--frozen"])
        )

        await base.with_exec(["uv", "run", "ruff", "check", "."]).sync()
        await base.with_exec(["uv", "run", "ruff", "format", "--check", "."]).sync()

        # mypy scope is pinned in pyproject.toml ([tool.mypy].files = x_likes_mcp).
        await base.with_exec(["uv", "run", "mypy"]).sync()

        await base.with_exec(["uv", "run", "pytest", "-q"]).sync()

        built = base.with_exec(["uv", "build", "--sdist", "--wheel"])
        await built.directory("/src/dist").export("./dist")

    print("dagger pipeline ok: artifacts written to ./dist", file=sys.stderr)


if __name__ == "__main__":
    anyio.run(main)
