"""
Export formatters for different output formats (JSON, Pandas, Markdown)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime
from .models import Tweet
from .downloader import MediaDownloader
from .dates import parse_x_datetime


class JSONFormatter:
    """Export tweets to JSON format"""

    @staticmethod
    def export(tweets: List[Tweet], output_file: str, include_raw: bool = False):
        """
        Export tweets to JSON file

        Args:
            tweets: List of Tweet objects
            output_file: Output JSON file path
            include_raw: Include raw API response data
        """
        data = [tweet.to_dict(include_raw=include_raw) for tweet in tweets]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(tweets)} tweets to {output_file}")


class PandasFormatter:
    """Export tweets to Pandas DataFrame"""

    @staticmethod
    def to_dataframe(tweets: List[Tweet]) -> pd.DataFrame:
        """
        Convert tweets to Pandas DataFrame

        Args:
            tweets: List of Tweet objects

        Returns:
            Pandas DataFrame
        """
        data = []

        for tweet in tweets:
            row = {
                'tweet_id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'user_id': tweet.user.id,
                'user_screen_name': tweet.user.screen_name,
                'user_name': tweet.user.name,
                'user_verified': tweet.user.verified,
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count,
                'reply_count': tweet.reply_count,
                'quote_count': tweet.quote_count,
                'view_count': tweet.view_count,
                'lang': tweet.lang,
                'is_retweet': tweet.is_retweet,
                'is_quote': tweet.is_quote,
                'has_media': len(tweet.media) > 0,
                'media_count': len(tweet.media),
                'media_types': ','.join([m.type for m in tweet.media]),
                'url_count': len(tweet.urls),
                'hashtag_count': len(tweet.hashtags),
                'hashtags': ','.join(tweet.hashtags),
                'mention_count': len(tweet.mentions),
                'tweet_url': tweet.get_url(),
            }

            data.append(row)

        df = pd.DataFrame(data)

        # Convert created_at to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])

        return df

    @staticmethod
    def export(tweets: List[Tweet], output_file: str, format: str = 'csv'):
        """
        Export tweets to file using Pandas

        Args:
            tweets: List of Tweet objects
            output_file: Output file path
            format: Output format ('csv', 'excel', 'parquet')
        """
        df = PandasFormatter.to_dataframe(tweets)

        if format == 'csv':
            df.to_csv(output_file, index=False, encoding='utf-8')
        elif format == 'excel':
            df.to_excel(output_file, index=False, engine='openpyxl')
        elif format == 'parquet':
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Exported {len(tweets)} tweets to {output_file}")


class MarkdownFormatter:
    """Export tweets to Markdown format with embedded media"""

    def __init__(self, media_downloader: MediaDownloader = None):
        """
        Initialize Markdown formatter

        Args:
            media_downloader: MediaDownloader instance for handling media paths
        """
        self.media_downloader = media_downloader

    def export(
        self,
        tweets: List[Tweet],
        output_file: str,
        include_media: bool = True,
        *,
        omit_global_header: bool = False,
    ):
        """
        Export tweets to Markdown file

        Args:
            tweets: List of Tweet objects
            output_file: Output Markdown file path
            include_media: Include embedded media images
            omit_global_header: When True, skip the file-level h1, the
                ``**Exported:**`` timestamp, the ``**Total Tweets:**`` line,
                and the trailing ``---`` divider. The per-month sections and
                per-tweet blocks are unchanged. Used by the per-month split
                branch so PageIndex sees ``## YYYY-MM`` as the effective top
                of each per-month tree.
        """
        output_path = Path(output_file)
        md_lines = []

        # Header
        if not omit_global_header:
            md_lines.append("# X (Twitter) Liked Tweets\n")
            md_lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            md_lines.append(f"**Total Tweets:** {len(tweets)}\n")
            md_lines.append("---\n")

        # Group tweets by month
        tweets_by_month = {}
        for tweet in tweets:
            created = parse_x_datetime(tweet.created_at)
            if created is None:
                month_key = 'unknown'
            else:
                month_key = created.strftime('%Y-%m')
            if month_key not in tweets_by_month:
                tweets_by_month[month_key] = []
            tweets_by_month[month_key].append(tweet)

        # Sort months in reverse chronological order
        sorted_months = sorted(tweets_by_month.keys(), reverse=True)

        # Generate markdown for each month
        for month in sorted_months:
            month_tweets = tweets_by_month[month]

            if month != 'unknown':
                md_lines.append(f"\n## {month} ({len(month_tweets)} tweets)\n")
            else:
                md_lines.append(f"\n## Unknown Date ({len(month_tweets)} tweets)\n")

            for tweet in month_tweets:
                md_lines.extend(self._format_tweet(tweet, output_path, include_media))

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        print(f"Exported {len(tweets)} tweets to {output_file}")

    def _format_tweet(self, tweet: Tweet, output_path: Path, include_media: bool) -> List[str]:
        """Format a single tweet as markdown"""
        lines = []

        # Tweet header
        lines.append(f"\n### [@{tweet.user.screen_name}](https://x.com/{tweet.user.screen_name})")
        lines.append(f"**{tweet.user.name}** {'✓' if tweet.user.verified else ''}")

        created = parse_x_datetime(tweet.created_at)
        if created is None:
            date_str = tweet.created_at
        else:
            date_str = created.strftime('%Y-%m-%d %H:%M:%S')

        lines.append(f"*{date_str}*")

        # Tweet text
        lines.append(f"\n{tweet.text}\n")

        # Media
        if include_media and tweet.media:
            lines.append("**Media:**\n")

            for media in tweet.media:
                if media.local_path and self.media_downloader:
                    # Get relative path from markdown file to media
                    rel_path = self.media_downloader.get_relative_path(
                        media.local_path,
                        output_path.parent
                    )

                    if media.type == "photo":
                        lines.append(f"![Image]({rel_path})\n")
                    elif media.type == "video":
                        lines.append(f"🎥 [Video]({rel_path})\n")
                    elif media.type == "animated_gif":
                        lines.append(f"![GIF]({rel_path})\n")
                else:
                    # Fallback to URL if no local file
                    if media.type == "photo" and media.media_url:
                        lines.append(f"![Image]({media.media_url})\n")
                    elif media.url:
                        lines.append(f"🔗 [{media.type}]({media.url})\n")

        # Stats
        stats = []
        if tweet.retweet_count > 0:
            stats.append(f"🔄 {tweet.retweet_count}")
        if tweet.favorite_count > 0:
            stats.append(f"❤️ {tweet.favorite_count}")
        if tweet.reply_count > 0:
            stats.append(f"💬 {tweet.reply_count}")
        if tweet.view_count > 0:
            stats.append(f"👁️ {tweet.view_count}")

        if stats:
            lines.append(f"\n*{' • '.join(stats)}*")

        # Tweet URL
        lines.append(f"\n🔗 [View on X]({tweet.get_url()})")

        # Hashtags
        if tweet.hashtags:
            lines.append(f"\n**Tags:** {' '.join(['#' + tag for tag in tweet.hashtags])}")

        # Separator
        lines.append("\n---")

        return lines


class HTMLFormatter:
    """Export tweets to a single HTML page."""

    def export(self, tweets: List[Tweet], output_file: str):
        """
        Export tweets to HTML file

        Args:
            tweets: List of Tweet objects
            output_file: Output HTML file path
        """
        html_lines = []

        # HTML header
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html lang='en'>")
        html_lines.append("<head>")
        html_lines.append("    <meta charset='UTF-8'>")
        html_lines.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html_lines.append("    <title>X Liked Tweets</title>")
        html_lines.append(self._get_css())
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("    <div class='container'>")
        html_lines.append(f"        <h1>X (Twitter) Liked Tweets</h1>")
        html_lines.append(f"        <p class='meta'>Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Total: {len(tweets)} tweets</p>")

        # Tweets
        for tweet in tweets:
            html_lines.append(self._format_tweet_html(tweet))

        # HTML footer
        html_lines.append("    </div>")
        html_lines.append("</body>")
        html_lines.append("</html>")

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_lines))

        print(f"Exported {len(tweets)} tweets to {output_file}")

    def _format_tweet_html(self, tweet: Tweet) -> str:
        """Format a single tweet as HTML"""
        html = []

        html.append("        <div class='tweet'>")
        html.append(f"            <div class='user'>")
        html.append(f"                <strong>{tweet.user.name}</strong> @{tweet.user.screen_name}")
        html.append(f"            </div>")
        html.append(f"            <div class='text'>{self._escape_html(tweet.text)}</div>")

        # Media
        if tweet.media:
            html.append("            <div class='media'>")
            for media in tweet.media:
                if media.local_path:
                    if media.type == "photo":
                        html.append(f"                <img src='{media.local_path}' alt='Tweet image'>")
                elif media.media_url:
                    html.append(f"                <img src='{media.media_url}' alt='Tweet image'>")
            html.append("            </div>")

        # Stats
        html.append(f"            <div class='stats'>")
        html.append(f"                <span>🔄 {tweet.retweet_count}</span>")
        html.append(f"                <span>❤️ {tweet.favorite_count}</span>")
        html.append(f"                <span>💬 {tweet.reply_count}</span>")
        html.append(f"                <a href='{tweet.get_url()}' target='_blank'>View on X</a>")
        html.append(f"            </div>")
        html.append("        </div>")

        return '\n'.join(html)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

    def _get_css(self) -> str:
        """Get CSS styles for HTML export"""
        return """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1d9bf0;
            margin-bottom: 10px;
        }
        .meta {
            color: #666;
            margin-bottom: 30px;
        }
        .tweet {
            border: 1px solid #e1e8ed;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background: #fafafa;
        }
        .user {
            font-size: 16px;
            margin-bottom: 10px;
        }
        .user strong {
            color: #000;
        }
        .text {
            font-size: 18px;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        .media img {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 10px;
        }
        .stats {
            color: #666;
            font-size: 14px;
            display: flex;
            gap: 15px;
        }
        .stats a {
            color: #1d9bf0;
            text-decoration: none;
        }
    </style>
"""
