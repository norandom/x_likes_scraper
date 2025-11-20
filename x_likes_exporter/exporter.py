"""
Main exporter class that orchestrates the export process
"""

from pathlib import Path
from typing import Optional, Callable, List
from .cookies import CookieManager
from .client import XAPIClient
from .downloader import MediaDownloader
from .formatters import JSONFormatter, PandasFormatter, MarkdownFormatter, HTMLFormatter
from .models import Tweet
from .checkpoint import Checkpoint


class XLikesExporter:
    """Main class for exporting X (Twitter) likes"""

    def __init__(self, cookies_file: str, output_dir: str = "output", enable_checkpoints: bool = True):
        """
        Initialize the exporter

        Args:
            cookies_file: Path to cookies.json file
            output_dir: Directory for output files
            enable_checkpoints: Enable checkpoint system for resume functionality
        """
        self.cookies_file = cookies_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.cookie_manager = CookieManager(cookies_file)
        self.api_client = XAPIClient(self.cookie_manager)
        self.media_downloader = MediaDownloader(str(self.output_dir / "media"))
        self.checkpoint = Checkpoint(str(self.output_dir)) if enable_checkpoints else None

        # Validate cookies
        if not self.cookie_manager.validate():
            raise Exception("Invalid cookies. Please ensure ct0 and auth_token are present.")

        # Storage for fetched tweets
        self.tweets: List[Tweet] = []

    def fetch_likes(
        self,
        user_id: str,
        download_media: bool = True,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
        resume: bool = False
    ) -> List[Tweet]:
        """
        Fetch all liked tweets

        Args:
            user_id: User ID to fetch likes for
            download_media: Whether to download media files
            progress_callback: Optional callback function(current, total) for progress
            stop_callback: Optional callback that returns True to stop
            resume: Resume from previous checkpoint if available

        Returns:
            List of Tweet objects
        """
        # Check for resume
        start_cursor = None
        if resume and self.checkpoint and self.checkpoint.exists():
            checkpoint_data = self.checkpoint.load()
            if checkpoint_data and checkpoint_data.get('user_id') == user_id:
                self.tweets = checkpoint_data.get('tweets', [])
                start_cursor = checkpoint_data.get('cursor')
                print(f"✓ Resuming from checkpoint: {len(self.tweets)} tweets already fetched")
                print(f"  Starting from cursor: {start_cursor[:20]}..." if start_cursor else "  No cursor found")
            else:
                print("⚠ Checkpoint found but for different user, starting fresh")
                if self.checkpoint:
                    self.checkpoint.clear()
        else:
            print(f"Fetching likes for user {user_id}...")

        # Fetch tweets with checkpoint support
        new_tweets = self.api_client.fetch_all_likes(
            user_id=user_id,
            progress_callback=progress_callback,
            stop_callback=stop_callback,
            start_cursor=start_cursor,
            checkpoint_callback=lambda new_tweets_batch, cursor: self._save_checkpoint(
                user_id, 
                self.tweets + new_tweets_batch, 
                cursor, 
                download_media
            ) if self.checkpoint else None
        )

        # Merge with existing tweets if resuming
        if resume and self.tweets:
            # Deduplicate by tweet ID
            existing_ids = {t.id for t in self.tweets}
            unique_new = [t for t in new_tweets if t.id not in existing_ids]
            self.tweets.extend(unique_new)
            print(f"✓ Added {len(unique_new)} new tweets (total: {len(self.tweets)})")
        else:
            self.tweets = new_tweets

        # Clear checkpoint on successful completion
        if self.checkpoint and not stop_callback:
            self.checkpoint.clear()

        # Download media if requested
        if download_media and self.tweets:
            print(f"\nDownloading media from {len(self.tweets)} tweets...")
            total_media = self.media_downloader.download_all_media(
                self.tweets,
                progress_callback=lambda curr, total: print(f"Downloading media: {curr}/{len(self.tweets)} tweets processed", end='\r')
            )
            print(f"\nDownloaded {total_media} media files")

        return self.tweets

    def _save_checkpoint(self, user_id: str, tweets: List[Tweet], cursor: Optional[str], download_media: bool):
        """Internal method to save checkpoint during fetch"""
        if self.checkpoint:
            self.checkpoint.save(
                user_id=user_id,
                tweets=tweets,
                cursor=cursor,
                total_fetched=len(tweets),
                download_media=download_media
            )

    def export_json(self, filename: Optional[str] = None, include_raw: bool = False):
        """
        Export tweets to JSON

        Args:
            filename: Output filename (default: likes.json)
            include_raw: Include raw API response data
        """
        if not self.tweets:
            print("No tweets to export. Run fetch_likes() first.")
            return

        filename = filename or "likes.json"
        output_file = self.output_dir / filename

        JSONFormatter.export(self.tweets, str(output_file), include_raw=include_raw)

    def export_csv(self, filename: Optional[str] = None):
        """
        Export tweets to CSV using Pandas

        Args:
            filename: Output filename (default: likes.csv)
        """
        if not self.tweets:
            print("No tweets to export. Run fetch_likes() first.")
            return

        filename = filename or "likes.csv"
        output_file = self.output_dir / filename

        PandasFormatter.export(self.tweets, str(output_file), format='csv')

    def export_excel(self, filename: Optional[str] = None):
        """
        Export tweets to Excel using Pandas

        Args:
            filename: Output filename (default: likes.xlsx)
        """
        if not self.tweets:
            print("No tweets to export. Run fetch_likes() first.")
            return

        filename = filename or "likes.xlsx"
        output_file = self.output_dir / filename

        # Requires openpyxl
        try:
            PandasFormatter.export(self.tweets, str(output_file), format='excel')
        except ImportError:
            print("Excel export requires openpyxl. Install with: pip install openpyxl")

    def export_markdown(self, filename: Optional[str] = None, include_media: bool = True, split_by_month: bool = True):
        """
        Export tweets to Markdown with embedded media

        Args:
            filename: Output filename (default: likes.md for single file, or by_month/ directory if split_by_month=True)
            include_media: Include embedded media images
            split_by_month: Split into separate files by year/month (default: True)
        """
        if not self.tweets:
            print("No tweets to export. Run fetch_likes() first.")
            return

        formatter = MarkdownFormatter(self.media_downloader)

        if split_by_month:
            # Group by year/month
            from collections import defaultdict
            from datetime import datetime

            tweets_by_month = defaultdict(list)
            for tweet in self.tweets:
                try:
                    # Parse date: "Sun Nov 09 11:05:17 +0000 2025"
                    dt = datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S %z %Y")
                    year_month = dt.strftime("%Y-%m")
                    tweets_by_month[year_month].append(tweet)
                except:
                    tweets_by_month["unknown"].append(tweet)

            # Create output directory
            output_dir = self.output_dir / "by_month"
            output_dir.mkdir(exist_ok=True)

            # Export each month
            count = 0
            for year_month in sorted(tweets_by_month.keys(), reverse=True):
                tweets = tweets_by_month[year_month]
                month_file = output_dir / f"likes_{year_month}.md"
                formatter.export(tweets, str(month_file), include_media=include_media)
                count += 1

            print(f"Exported {len(self.tweets)} tweets to {count} monthly files in {output_dir}/")
        else:
            # Export to single file
            filename = filename or "likes.md"
            output_file = self.output_dir / filename
            formatter.export(self.tweets, str(output_file), include_media=include_media)

    def export_html(self, filename: Optional[str] = None):
        """
        Export tweets to HTML

        Args:
            filename: Output filename (default: likes.html)
        """
        if not self.tweets:
            print("No tweets to export. Run fetch_likes() first.")
            return

        filename = filename or "likes.html"
        output_file = self.output_dir / filename

        HTMLFormatter().export(self.tweets, str(output_file))

    def export_all(self, base_name: str = "likes", include_raw: bool = False):
        """
        Export to all formats

        Args:
            base_name: Base filename (without extension)
            include_raw: Include raw API data in JSON export
        """
        print("\nExporting to all formats...")

        self.export_json(f"{base_name}.json", include_raw=include_raw)
        self.export_csv(f"{base_name}.csv")
        self.export_markdown(f"{base_name}.md", include_media=True)
        self.export_html(f"{base_name}.html")

        print(f"\nAll exports complete! Files saved to: {self.output_dir}")

    def get_dataframe(self):
        """
        Get tweets as Pandas DataFrame

        Returns:
            Pandas DataFrame
        """
        if not self.tweets:
            print("No tweets available. Run fetch_likes() first.")
            return None

        return PandasFormatter.to_dataframe(self.tweets)

    def get_stats(self) -> dict:
        """
        Get statistics about the exported tweets

        Returns:
            Dictionary with statistics
        """
        if not self.tweets:
            return {"total": 0}

        total_media = sum(len(tweet.media) for tweet in self.tweets)
        total_retweets = sum(tweet.retweet_count for tweet in self.tweets)
        total_likes = sum(tweet.favorite_count for tweet in self.tweets)

        return {
            "total_tweets": len(self.tweets),
            "total_media": total_media,
            "total_retweets": total_retweets,
            "total_likes": total_likes,
            "tweets_with_media": len([t for t in self.tweets if t.media]),
            "retweets": len([t for t in self.tweets if t.is_retweet]),
            "quotes": len([t for t in self.tweets if t.is_quote]),
        }
