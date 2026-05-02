# Roadmap

## Overview

The X Likes Exporter is a working CLI and library that pulls a user's liked tweets off X and writes them out as JSON, CSV, HTML, and per-month Markdown. The user-facing flow is wrapped in `scrape.sh`. What's missing is a test suite. Every change so far has been validated by running the script against a real account and eyeballing the output. That's fine for the current owner, but it blocks confident refactoring and blocks any external code from depending on the lib.

This roadmap covers two specs in order:

1. Get the codebase to a state where the lib is testable, tested, and has a small read-only public surface that doesn't require running the scraper.
2. Build a stdio MCP server on top of that surface, using PageIndex over the per-month Markdown to give an LLM searchable access to the like history.

When both ship, the workflow is "scrape, then talk to your likes."

## Approach Decision

- **Chosen:** Two specs in sequence. The first lays the foundation (tests + an importable read API). The second sits on top.
- **Why:** The MCP server depends on a stable, tested way to load and iterate the export without re-running the X API. Building the MCP server first would mean either re-fetching every time or coupling it to internal data structures that aren't meant to be public.
- **Rejected alternatives:** A single mega-spec mixing tests, refactors, and the MCP server. Too big. Test work would block MCP work for weeks; the seam between them is clean enough to split.

## Scope

- **In:** Test suite for the lib, a small public read API, an MCP server that exposes the per-month Markdown to an LLM via PageIndex.
- **Out:** Re-fetching from X inside the MCP server (it consumes existing `likes.json` and `output/by_month/` only). New export formats. A web UI. Any kind of vector store. PageIndex is intentionally vectorless and that's the point.

## Constraints

- Python 3.12+, uv-managed, single-user local tool.
- No external services in the test suite. X API calls must be mocked.
- PageIndex's reasoning step needs an LLM. Spec 2 uses a local LLM endpoint configured in `.env`; no calls to hosted Anthropic or OpenAI by default.
- `cookies.json` and `.env` stay gitignored. Tests must not require either.

## Boundary Strategy

- **Why this split:** Spec 1's deliverable is "the lib is callable from outside and won't silently break under refactoring." Spec 2's deliverable is "you can ask questions about your likes from any MCP client." Each is a complete unit on its own.
- **Shared seams to watch:** the public read API. Spec 1 designs it; Spec 2 is the first consumer. If Spec 2 finds the API awkward, that goes back to Spec 1, not as a bolt-on inside Spec 2.

## Specs (dependency order)

- [x] codebase-foundation -- pytest suite for the lib, a public read API, refactors only where needed for testability or extension. Dependencies: none. Spec docs ready (requirements.md, design.md, tasks.md).
- [x] mcp-pageindex -- stdio MCP server using PageIndex over `output/by_month/`. Dependencies: codebase-foundation. Spec docs ready (requirements.md, design.md, tasks.md).
- [ ] mcp-fast-search -- replace the LLM-walker hot path in `search_likes` with local-embedding retrieval + the existing ranker; optional LLM explainer over top hits. Dependencies: mcp-pageindex.

## Phase 2 retrospective (2026-05-02)

mcp-pageindex shipped functionally complete (149 tests green, all four MCP tools registered with Claude Code) but the search hot path doesn't scale to a real archive. The walker calls the LLM once per chunk per month; for 4 months covering ~2,200 tweets that's 60+ sequential LLM calls, busting MCP client timeouts. The ranker formula and tool surface are sound; the retrieval layer is the part that needs replacing. mcp-fast-search owns that.
