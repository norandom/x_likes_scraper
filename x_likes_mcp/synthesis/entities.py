"""Entity extraction for the synthesis-report feature (placeholder for task 2.4).

Runs cheap regex / counter passes for handles, hashtags, URL domains,
and recurring noun phrases over hit text and fetched URL bodies, with a
DSPy fallback hook the orchestrator wires for hits where the regex pass
returned nothing.
"""

from __future__ import annotations
