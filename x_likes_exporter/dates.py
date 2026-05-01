"""Single home for parsing the X-format `created_at` string.

This module exists so the four near-identical date-parse blocks scattered
across `models.py`, `exporter.py`, and `formatters.py` collapse into one
tested function. The helper never raises; callers that need to raise on
unparseable input do so locally.
"""

from datetime import datetime
from typing import Optional

X_CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_x_datetime(value: str) -> Optional[datetime]:
    """Parse an X created_at string. Returns None on any failure."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, X_CREATED_AT_FORMAT)
    except (ValueError, TypeError):
        return None
