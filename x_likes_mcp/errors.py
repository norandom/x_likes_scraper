"""Tool-error shape and category helpers for the MCP layer.

A single :class:`ToolError` carries a stable ``category`` plus a human-readable
``message``. The three module-level factory functions produce errors with the
fixed category strings the tool handlers and server boundary surface to the
calling LLM (``"invalid_input"``, ``"not_found"``, ``"upstream_failure"``).
"""

from __future__ import annotations


class ToolError(Exception):
    """Exception type for MCP tool failures.

    Attributes:
        category: One of ``"invalid_input"``, ``"not_found"``, or
            ``"upstream_failure"``. Stable string so the calling LLM can react.
        message: Human-readable description. ``str(err)`` returns this string.
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


def invalid_input(field: str, message: str) -> ToolError:
    """Build an ``invalid_input`` ToolError naming the offending field."""
    return ToolError("invalid_input", f"invalid input for {field}: {message}")


def not_found(what: str, identifier: str) -> ToolError:
    """Build a ``not_found`` ToolError for a missing resource by identifier."""
    return ToolError("not_found", f"{what} not found: {identifier}")


def upstream_failure(detail: str) -> ToolError:
    """Build an ``upstream_failure`` ToolError describing the upstream issue."""
    return ToolError("upstream_failure", f"upstream failure: {detail}")
