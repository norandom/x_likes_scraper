"""
Media downloader for tweet images and videos
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
from .models import Tweet, Media


class MediaDownloader:
    """Downloads and manages media files from tweets"""

    def __init__(self, output_dir: str = "media"):
        """
        Initialize media downloader

        Args:
            output_dir: Directory to save downloaded media files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def download_tweet_media(self, tweet: Tweet) -> List[str]:
        """
        Download all media from a tweet

        Args:
            tweet: Tweet object containing media

        Returns:
            List of local file paths for downloaded media
        """
        downloaded_files = []

        for i, media in enumerate(tweet.media):
            try:
                local_path = self.download_media(media, tweet.id, i)
                if local_path:
                    media.local_path = local_path
                    downloaded_files.append(local_path)
            except Exception as e:
                print(f"Error downloading media {i} from tweet {tweet.id}: {e}")

        return downloaded_files

    def download_media(self, media: Media, tweet_id: str, index: int = 0) -> Optional[str]:
        """
        Download a single media file

        Args:
            media: Media object with URL
            tweet_id: Tweet ID for naming
            index: Media index in the tweet (for multiple media)

        Returns:
            Local file path or None if download failed
        """
        # Determine the URL to download
        download_url = media.media_url or media.url

        if not download_url:
            return None

        # For photos, get the highest quality
        if media.type == "photo" and media.media_url:
            # Remove size suffix and add :orig for original quality
            download_url = re.sub(r'\?.*$', '', download_url)
            if not download_url.endswith(':orig'):
                download_url = f"{download_url}?format=jpg&name=orig"

        try:
            # Download the file
            response = self.session.get(download_url, timeout=30)
            response.raise_for_status()

            # Determine file extension
            content_type = response.headers.get('content-type', '')
            ext = self._get_extension(download_url, content_type, media.type)

            # Generate filename
            filename = f"{tweet_id}_{index}{ext}"
            filepath = self.output_dir / filename

            # Save the file
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Optimize images
            if media.type == "photo" and ext in ['.jpg', '.jpeg', '.png']:
                self._optimize_image(filepath)

            return str(filepath)

        except Exception as e:
            print(f"Error downloading {download_url}: {e}")
            return None

    def download_all_media(self, tweets: List[Tweet], progress_callback=None) -> int:
        """
        Download media from all tweets

        Args:
            tweets: List of Tweet objects
            progress_callback: Optional callback function(current, total)

        Returns:
            Number of media files downloaded
        """
        total_downloaded = 0

        for i, tweet in enumerate(tweets):
            downloaded = self.download_tweet_media(tweet)
            total_downloaded += len(downloaded)

            if progress_callback:
                progress_callback(i + 1, len(tweets))

        return total_downloaded

    def _get_extension(self, url: str, content_type: str, media_type: str) -> str:
        """Determine file extension from URL, content-type, or media type"""
        # Try to get extension from URL
        parsed = urlparse(url)
        path = parsed.path
        if '.' in path:
            ext = os.path.splitext(path)[1]
            if ext:
                return ext.lower()

        # Try to get from content-type
        if 'image/jpeg' in content_type or 'image/jpg' in content_type:
            return '.jpg'
        elif 'image/png' in content_type:
            return '.png'
        elif 'image/gif' in content_type:
            return '.gif'
        elif 'image/webp' in content_type:
            return '.webp'
        elif 'video/mp4' in content_type:
            return '.mp4'

        # Fallback based on media type
        if media_type == 'photo':
            return '.jpg'
        elif media_type == 'video':
            return '.mp4'
        elif media_type == 'animated_gif':
            return '.gif'

        return '.jpg'  # Default

    def _optimize_image(self, filepath: Path, max_size: tuple = (1920, 1920), quality: int = 85):
        """
        Optimize image file size

        Args:
            filepath: Path to image file
            max_size: Maximum dimensions (width, height)
            quality: JPEG quality (1-100)
        """
        try:
            with Image.open(filepath) as img:
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background

                # Resize if larger than max_size
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Save with optimization
                img.save(filepath, optimize=True, quality=quality)

        except Exception as e:
            print(f"Warning: Could not optimize {filepath}: {e}")

    def get_relative_path(self, filepath: str, base_dir: str = ".") -> str:
        """
        Get relative path from base directory

        Args:
            filepath: Absolute or relative file path
            base_dir: Base directory for relative path

        Returns:
            Relative path
        """
        filepath = Path(filepath)
        base_dir = Path(base_dir)

        try:
            return str(filepath.relative_to(base_dir))
        except ValueError:
            # If files are not in the same tree, return the filepath as-is
            return str(filepath)
