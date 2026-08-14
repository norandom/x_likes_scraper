"""Release metadata consistency tests."""

from importlib.metadata import version

import x_likes_exporter


def test_runtime_version_matches_distribution_metadata() -> None:
    """Keep the runtime and packaged versions synchronized."""
    assert x_likes_exporter.__version__ == version("x-likes-exporter")
