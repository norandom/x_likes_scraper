"""Tests for x_likes_exporter.downloader.MediaDownloader.

Sentrux test_gaps flagged ``downloader.py`` (risk 22, cog 11, fan_in 1)
as the highest-risk untested production file. This module covers the
load-bearing paths:

* construction creates the output directory
* download_media short-circuits on existing files
* download_media handles photo URL upgrade to ``:orig``
* download_media writes downloaded bytes to disk
* download_media tolerates network errors and returns ``None``
* _get_extension dispatches via URL path, content-type, then media-type
* download_tweet_media iterates and stamps ``Media.local_path``
* download_all_media tallies totals and calls progress_callback
* get_relative_path returns relative when possible, absolute otherwise

Network calls go through the ``responses`` library which is activated
session-wide by ``tests/conftest.py`` so any unmocked HTTP fails loud.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import responses
from PIL import Image

from x_likes_exporter.downloader import MediaDownloader
from x_likes_exporter.models import Media, Tweet, User

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(handle: str = "alice") -> User:
    return User(id=f"u_{handle}", screen_name=handle, name=handle.title())


def _tweet(tweet_id: str, *, media: list[Media] | None = None) -> Tweet:
    return Tweet(
        id=tweet_id,
        text="hello",
        created_at="Wed Jan 01 12:00:00 +0000 2025",
        user=_user(),
        media=list(media or []),
    )


def _png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    """Encode a tiny in-memory PNG so PIL.Image.open can load it."""

    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(0, 255, 0)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "media_dir"
        assert not target.exists()
        MediaDownloader(str(target))
        assert target.is_dir()

    def test_accepts_existing_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "media_dir"
        target.mkdir()
        # Should not raise on a pre-existing directory.
        MediaDownloader(str(target))
        assert target.is_dir()


# ---------------------------------------------------------------------------
# download_media
# ---------------------------------------------------------------------------


class TestDownloadMedia:
    @responses.activate
    def test_returns_existing_file_without_network_call(self, tmp_path: Path) -> None:
        """An on-disk ``<tweet_id>_<index>.*`` short-circuits the download."""

        d = MediaDownloader(str(tmp_path))
        existing = tmp_path / "111_0.jpg"
        existing.write_bytes(b"already here")

        media = Media(type="photo", url="https://example.com/x.jpg")
        # No responses.add(...) — any HTTP request would raise.
        result = d.download_media(media, "111", 0)
        assert result == str(existing)

    @responses.activate
    def test_returns_none_when_no_url(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        media = Media(type="photo", url="")
        assert d.download_media(media, "222", 0) is None

    @responses.activate
    def test_writes_downloaded_bytes_to_disk(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        png = _png_bytes()
        responses.add(
            responses.GET,
            "https://cdn.example.com/foo.png",
            body=png,
            status=200,
            headers={"content-type": "image/png"},
        )
        media = Media(type="photo", url="https://cdn.example.com/foo.png")

        result = d.download_media(media, "333", 0)
        assert result is not None
        out = Path(result)
        assert out.exists()
        assert out.suffix == ".png"
        # Optimization may rewrite the bytes, but the file is non-empty
        # and PIL can still open it.
        with Image.open(out) as img:
            assert img.size == (4, 4)

    @responses.activate
    def test_returns_none_on_http_error(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        responses.add(
            responses.GET,
            "https://cdn.example.com/missing.jpg",
            status=404,
        )
        media = Media(type="photo", url="https://cdn.example.com/missing.jpg")
        assert d.download_media(media, "444", 0) is None

    @responses.activate
    def test_photo_upgrades_to_orig_quality(self, tmp_path: Path) -> None:
        """``media.type == 'photo'`` with a populated ``media_url`` rewrites
        the URL to request the original-quality variant."""

        d = MediaDownloader(str(tmp_path))
        responses.add(
            responses.GET,
            "https://pbs.twimg.com/media/abc.jpg?format=jpg&name=orig",
            body=_jpg_bytes(),
            status=200,
            headers={"content-type": "image/jpeg"},
        )
        media = Media(
            type="photo",
            url="https://pbs.twimg.com/media/abc.jpg",
            media_url="https://pbs.twimg.com/media/abc.jpg?format=jpg&name=small",
        )
        result = d.download_media(media, "555", 0)
        assert result is not None

    @responses.activate
    def test_video_does_not_get_orig_suffix(self, tmp_path: Path) -> None:
        """The ``:orig`` rewrite only fires for photos."""

        d = MediaDownloader(str(tmp_path))
        responses.add(
            responses.GET,
            "https://video.twimg.com/x.mp4",
            body=b"fake mp4",
            status=200,
            headers={"content-type": "video/mp4"},
        )
        media = Media(
            type="video",
            url="https://video.twimg.com/x.mp4",
            media_url="https://video.twimg.com/x.mp4",
        )
        result = d.download_media(media, "666", 0)
        assert result is not None
        assert result.endswith(".mp4")


# ---------------------------------------------------------------------------
# _get_extension
# ---------------------------------------------------------------------------


class TestGetExtension:
    def test_url_path_extension_wins(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        assert d._get_extension("https://x.com/foo.png", "image/jpeg", "photo") == ".png"
        assert d._get_extension("https://x.com/clip.mp4", "video/mp4", "video") == ".mp4"

    def test_content_type_fallback(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        # No path extension; dispatch on content-type.
        assert d._get_extension("https://x.com/anon", "image/jpeg", "photo") == ".jpg"
        assert d._get_extension("https://x.com/anon", "image/png", "photo") == ".png"
        assert d._get_extension("https://x.com/anon", "image/gif", "animated_gif") == ".gif"
        assert d._get_extension("https://x.com/anon", "image/webp", "photo") == ".webp"
        assert d._get_extension("https://x.com/anon", "video/mp4", "video") == ".mp4"

    def test_media_type_fallback(self, tmp_path: Path) -> None:
        """Neither path nor content-type carries the extension."""

        d = MediaDownloader(str(tmp_path))
        assert d._get_extension("https://x.com/anon", "", "photo") == ".jpg"
        assert d._get_extension("https://x.com/anon", "", "video") == ".mp4"
        assert d._get_extension("https://x.com/anon", "", "animated_gif") == ".gif"

    def test_unknown_media_type_defaults_to_jpg(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        assert d._get_extension("https://x.com/anon", "", "??") == ".jpg"


# ---------------------------------------------------------------------------
# download_tweet_media
# ---------------------------------------------------------------------------


class TestDownloadTweetMedia:
    @responses.activate
    def test_stamps_local_path_on_each_media(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        png = _png_bytes()
        responses.add(
            responses.GET,
            "https://cdn.example.com/a.png",
            body=png,
            headers={"content-type": "image/png"},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://cdn.example.com/b.png",
            body=png,
            headers={"content-type": "image/png"},
            status=200,
        )
        tweet = _tweet(
            "777",
            media=[
                Media(type="photo", url="https://cdn.example.com/a.png"),
                Media(type="photo", url="https://cdn.example.com/b.png"),
            ],
        )

        downloaded = d.download_tweet_media(tweet)
        assert len(downloaded) == 2
        assert all(m.local_path is not None for m in tweet.media)
        assert all(Path(p).exists() for p in downloaded)

    @responses.activate
    def test_skips_failed_media_continues_for_others(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        png = _png_bytes()
        responses.add(
            responses.GET,
            "https://cdn.example.com/ok.png",
            body=png,
            headers={"content-type": "image/png"},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://cdn.example.com/bad.png",
            status=500,
        )
        tweet = _tweet(
            "888",
            media=[
                Media(type="photo", url="https://cdn.example.com/ok.png"),
                Media(type="photo", url="https://cdn.example.com/bad.png"),
            ],
        )

        downloaded = d.download_tweet_media(tweet)
        assert len(downloaded) == 1


# ---------------------------------------------------------------------------
# download_all_media
# ---------------------------------------------------------------------------


class TestDownloadAllMedia:
    @responses.activate
    def test_calls_progress_callback_per_tweet(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        png = _png_bytes()
        responses.add(
            responses.GET,
            "https://cdn.example.com/p.png",
            body=png,
            headers={"content-type": "image/png"},
            status=200,
        )
        tweets = [
            _tweet("1", media=[Media(type="photo", url="https://cdn.example.com/p.png")]),
            _tweet("2", media=[Media(type="photo", url="https://cdn.example.com/p.png")]),
        ]
        events: list[tuple[int, int]] = []
        total = d.download_all_media(tweets, progress_callback=lambda c, t: events.append((c, t)))
        assert total == 2
        assert events == [(1, 2), (2, 2)]

    def test_no_progress_callback_does_not_raise(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        # No tweets → no HTTP needed.
        total = d.download_all_media([], progress_callback=None)
        assert total == 0


# ---------------------------------------------------------------------------
# get_relative_path
# ---------------------------------------------------------------------------


class TestGetRelativePath:
    def test_relative_when_inside_base(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        nested = tmp_path / "media" / "x.jpg"
        nested.parent.mkdir()
        nested.write_bytes(b"x")
        rel = d.get_relative_path(str(nested), str(tmp_path))
        # rel uses os.sep; check both components are present.
        assert "media" in rel
        assert "x.jpg" in rel

    def test_falls_back_to_absolute_when_outside_base(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        outside = "/totally/elsewhere/y.jpg"
        rel = d.get_relative_path(outside, str(tmp_path))
        # When the file isn't under ``base_dir`` the function returns the
        # original path verbatim.
        assert rel == outside


# ---------------------------------------------------------------------------
# _optimize_image (best effort, never raises into caller)
# ---------------------------------------------------------------------------


class TestOptimizeImage:
    def test_resizes_oversize_image(self, tmp_path: Path) -> None:
        d = MediaDownloader(str(tmp_path))
        # Build a 4000x4000 JPG; _optimize_image clamps to (1920, 1920).
        big = tmp_path / "big.jpg"
        Image.new("RGB", (4000, 4000), color=(0, 0, 255)).save(big, format="JPEG")
        d._optimize_image(big)
        with Image.open(big) as img:
            assert img.size[0] <= 1920 and img.size[1] <= 1920

    def test_unreadable_file_does_not_raise(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A corrupt/missing image emits a warning but never propagates."""

        d = MediaDownloader(str(tmp_path))
        bad = tmp_path / "not_an_image.jpg"
        bad.write_bytes(b"not a real image")
        d._optimize_image(bad)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
