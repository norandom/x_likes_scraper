"""Compiled DSPy program persistence for the synthesis-report feature (placeholder for task 3.2).

Resolves the per-shape compiled-program path, loads the stored program
when present (returning ``None`` when missing or stale), saves
atomically, and exposes the optimizer entry point that sanitizes and
fences each demo's input fields before the optimizer sees them.
"""

from __future__ import annotations
