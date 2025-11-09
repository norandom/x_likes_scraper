#!/usr/bin/env python3
"""
Command-line interface for X Likes Exporter
"""

import argparse
import sys
from pathlib import Path
from x_likes_exporter import XLikesExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export your liked tweets from X (Twitter) to multiple formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export likes to all formats (Markdown will be split by month)
  %(prog)s cookies.json 123456789

  # Export only to JSON
  %(prog)s cookies.json 123456789 --format json

  # Export markdown as single file instead of splitting by month
  %(prog)s cookies.json 123456789 --format markdown --single-file

  # Export without downloading media (faster)
  %(prog)s cookies.json 123456789 --no-media

  # Resume interrupted export
  %(prog)s cookies.json 123456789 --resume

  # Check checkpoint status
  %(prog)s cookies.json 123456789 --checkpoint-info

  # Clear checkpoint and start fresh
  %(prog)s cookies.json 123456789 --clear-checkpoint

  # Custom output directory
  %(prog)s cookies.json 123456789 --output my_likes

  # Export to CSV and Markdown only
  %(prog)s cookies.json 123456789 --format csv --format markdown
"""
    )

    parser.add_argument(
        "cookies",
        help="Path to cookies.json file exported from browser"
    )

    parser.add_argument(
        "user_id",
        help="Your X (Twitter) user ID"
    )

    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "-f", "--format",
        action="append",
        choices=["json", "csv", "excel", "markdown", "html", "all"],
        help="Export format(s). Can be specified multiple times. Use 'all' for all formats."
    )

    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Skip downloading media files"
    )

    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw API response data in JSON export"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics after export"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available"
    )

    parser.add_argument(
        "--checkpoint-info",
        action="store_true",
        help="Show checkpoint information and exit"
    )

    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint and exit"
    )

    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Export markdown as single file instead of splitting by month (default: split by month)"
    )

    args = parser.parse_args()

    # Validate inputs
    cookies_path = Path(args.cookies)
    if not cookies_path.exists():
        print(f"Error: Cookies file not found: {args.cookies}")
        sys.exit(1)

    # Determine formats
    formats = args.format or ["all"]
    if "all" in formats:
        formats = ["json", "csv", "markdown", "html"]

    try:
        # Initialize exporter
        print(f"Initializing X Likes Exporter...")
        exporter = XLikesExporter(
            cookies_file=str(cookies_path),
            output_dir=args.output
        )

        # Handle checkpoint info
        if args.checkpoint_info:
            if exporter.checkpoint and exporter.checkpoint.exists():
                info = exporter.checkpoint.get_info()
                print("\n" + "="*50)
                print("Checkpoint Information:")
                print("="*50)
                print(f"  User ID: {info.get('user_id')}")
                print(f"  Tweets fetched: {info.get('total_fetched')}")
                print(f"  Timestamp: {info.get('timestamp')}")
                print(f"  Has cursor: {'Yes' if info.get('cursor') else 'No'}")
                print(f"  Download media: {info.get('download_media')}")
                print("\nTo resume: python cli.py cookies.json USER_ID --resume")
                print("To clear: python cli.py cookies.json USER_ID --clear-checkpoint")
            else:
                print("No checkpoint found.")
            return

        # Handle clear checkpoint
        if args.clear_checkpoint:
            if exporter.checkpoint and exporter.checkpoint.exists():
                exporter.checkpoint.clear()
                print("✓ Checkpoint cleared")
            else:
                print("No checkpoint to clear.")
            return

        # Check for existing checkpoint
        if args.resume and exporter.checkpoint and exporter.checkpoint.exists():
            info = exporter.checkpoint.get_info()
            print(f"\n✓ Found checkpoint: {info.get('total_fetched')} tweets")
            print(f"  Saved at: {info.get('timestamp')}")

        # Fetch likes
        print(f"\nFetching likes for user {args.user_id}...")
        tweets = exporter.fetch_likes(
            user_id=args.user_id,
            download_media=not args.no_media,
            resume=args.resume
        )

        if not tweets:
            print("No tweets found.")
            return

        print(f"\n✓ Fetched {len(tweets)} liked tweets")

        # Export to requested formats
        print("\nExporting...")

        if "json" in formats:
            exporter.export_json(include_raw=args.include_raw)

        if "csv" in formats:
            exporter.export_csv()

        if "excel" in formats:
            try:
                exporter.export_excel()
            except ImportError:
                print("⚠ Excel export skipped (requires openpyxl)")

        if "markdown" in formats:
            exporter.export_markdown(include_media=not args.no_media, split_by_month=not args.single_file)

        if "html" in formats:
            exporter.export_html()

        # Show statistics
        if args.stats:
            print("\n" + "="*50)
            print("Statistics:")
            print("="*50)
            stats = exporter.get_stats()
            for key, value in stats.items():
                print(f"  {key.replace('_', ' ').title()}: {value:,}")

        print(f"\n✓ Export complete! Files saved to: {args.output}/")

    except KeyboardInterrupt:
        print("\n\nExport cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
