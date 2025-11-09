"""
X (Twitter) API client with rate limiting and pagination support
"""

import time
import json
import requests
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from .cookies import CookieManager
from .auth import XAuthenticator
from .models import Tweet, User, Media


@dataclass
class RateLimitInfo:
    """Rate limit information from API response"""
    limit: int
    remaining: int
    reset: int  # Unix timestamp

    def should_wait(self) -> bool:
        """Check if we should wait for rate limit reset"""
        return self.remaining <= 1

    def get_wait_time(self) -> int:
        """Get seconds to wait until rate limit resets"""
        now = int(time.time())
        wait_time = self.reset - now + 5  # Add 5 second buffer
        return max(0, wait_time)


class XAPIClient:
    """Client for X (Twitter) GraphQL API"""

    def __init__(self, cookie_manager: CookieManager):
        """
        Initialize API client

        Args:
            cookie_manager: CookieManager with loaded cookies
        """
        self.cookie_manager = cookie_manager
        self.authenticator = XAuthenticator(cookie_manager)
        self.session = requests.Session()
        self.rate_limit_info: Optional[RateLimitInfo] = None
        self._request_delay = 1.0  # Polite delay between requests (seconds)

    def fetch_likes(
        self,
        user_id: str,
        cursor: Optional[str] = None,
        count: int = 20,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None
    ) -> tuple[List[Tweet], Optional[str], RateLimitInfo]:
        """
        Fetch a page of liked tweets

        Args:
            user_id: User ID to fetch likes for
            cursor: Pagination cursor (None for first page)
            count: Number of likes to fetch per request (max 100, recommended 20)
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            Tuple of (tweets, next_cursor, rate_limit_info)

        Raises:
            Exception: If API request fails
        """
        # Get authentication tokens
        bearer_token = self.authenticator.get_bearer_token()
        query_id = self.authenticator.get_query_id("Likes")
        csrf_token = self.cookie_manager.get_csrf_token()

        # Build request parameters
        variables = {
            "userId": user_id,
            "count": count,
            "includePromotedContent": False,
            "withClientEventToken": False,
            "withBirdwatchNotes": False,
            "withVoice": True,
            "withV2Timeline": True
        }

        if cursor:
            variables["cursor"] = cursor

        # Features object (required by X API)
        features = {
            "rweb_tipjar_consumption_enabled": True,
            "responsive_web_graphql_exclude_directive_enabled": True,
            "verified_phone_label_enabled": False,
            "creator_subscriptions_tweet_preview_api_enabled": True,
            "responsive_web_graphql_timeline_navigation_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "communities_web_enable_tweet_community_results_fetch": True,
            "c9s_tweet_anatomy_moderator_badge_enabled": True,
            "articles_preview_enabled": True,
            "responsive_web_edit_tweet_api_enabled": True,
            "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
            "view_counts_everywhere_api_enabled": True,
            "longform_notetweets_consumption_enabled": True,
            "responsive_web_twitter_article_tweet_consumption_enabled": True,
            "tweet_awards_web_tipping_enabled": False,
            "creator_subscriptions_quote_tweet_preview_enabled": False,
            "freedom_of_speech_not_reach_fetch_enabled": True,
            "standardized_nudges_misinfo": True,
            "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
            "rweb_video_timestamps_enabled": True,
            "longform_notetweets_rich_text_read_enabled": True,
            "longform_notetweets_inline_media_enabled": True,
            "responsive_web_enhance_cards_enabled": False,
            # Additional required features
            "responsive_web_grok_analyze_post_followups_enabled": False,
            "responsive_web_grok_imagine_annotation_enabled": False,
            "premium_content_api_read_enabled": False,
            "responsive_web_grok_analysis_button_from_backend": False,
            "responsive_web_profile_redirect_enabled": False,
            "responsive_web_grok_share_attachment_enabled": False,
            "responsive_web_grok_show_grok_translated_post": False,
            "profile_label_improvements_pcf_label_in_post_enabled": False,
            "payments_enabled": False,
            "rweb_video_screen_enabled": False,
            "responsive_web_jetfuel_frame": False,
            "responsive_web_grok_community_note_auto_translation_is_enabled": False,
            "responsive_web_grok_image_annotation_enabled": False,
            "responsive_web_grok_analyze_button_fetch_trends_enabled": False
        }

        # Build URL
        url = f"https://x.com/i/api/graphql/{query_id}/Likes"
        params = {
            "variables": json.dumps(variables),
            "features": json.dumps(features)
        }

        # Build headers
        headers = {
            "authorization": bearer_token,
            "x-csrf-token": csrf_token,
            "x-twitter-active-user": "yes",
            "x-twitter-auth-type": "OAuth2Session",
            "x-twitter-client-language": "en",
            "content-type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://x.com/",
            "Origin": "https://x.com",
        }

        # Make request
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                cookies=self.cookie_manager.get_cookie_dict()
            )
            response.raise_for_status()

            # Extract rate limit info from headers
            rate_limit_info = RateLimitInfo(
                limit=int(response.headers.get("x-rate-limit-limit", 0)),
                remaining=int(response.headers.get("x-rate-limit-remaining", 0)),
                reset=int(response.headers.get("x-rate-limit-reset", 0))
            )
            self.rate_limit_info = rate_limit_info

            # Parse response
            data = response.json()

            # Extract tweets
            tweets = self._extract_tweets(data)

            # Extract next cursor
            next_cursor = self._extract_cursor(data)

            return tweets, next_cursor, rate_limit_info

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise Exception("Rate limit exceeded. Please wait before retrying.")
            elif e.response.status_code == 401:
                raise Exception("Authentication failed. Please check your cookies.")
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e}")
        except Exception as e:
            raise Exception(f"Error fetching likes: {e}")

    def fetch_all_likes(
        self,
        user_id: str,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
        start_cursor: Optional[str] = None,
        checkpoint_callback: Optional[Callable[[List[Tweet], Optional[str]], None]] = None,
        checkpoint_interval: int = 10
    ) -> List[Tweet]:
        """
        Fetch all liked tweets with automatic pagination and rate limit handling

        Args:
            user_id: User ID to fetch likes for
            progress_callback: Optional callback function(current, total) for progress updates
            stop_callback: Optional callback that returns True to stop fetching
            start_cursor: Optional cursor to resume from
            checkpoint_callback: Optional callback to save checkpoint (tweets, cursor)
            checkpoint_interval: Save checkpoint every N pages (default: 10)

        Returns:
            List of all liked tweets

        Raises:
            Exception: If API request fails
        """
        all_tweets = []
        cursor = start_cursor
        page_count = 0

        while True:
            # Check if we should stop
            if stop_callback and stop_callback():
                print("Stopped by user")
                # Save checkpoint before stopping
                if checkpoint_callback:
                    checkpoint_callback(all_tweets, cursor)
                break

            # Fetch page
            print(f"Fetching page {page_count + 1}...")
            tweets, next_cursor, rate_limit = self.fetch_likes(
                user_id=user_id,
                cursor=cursor,
                count=20
            )

            # Add tweets to collection
            all_tweets.extend(tweets)
            page_count += 1

            # Update progress
            if progress_callback:
                progress_callback(len(all_tweets), None)

            print(f"Fetched {len(tweets)} likes. Total: {len(all_tweets)}")
            print(f"Rate limit: {rate_limit.remaining}/{rate_limit.limit}")

            # Save checkpoint periodically
            if checkpoint_callback and page_count % checkpoint_interval == 0:
                checkpoint_callback(all_tweets, next_cursor)

            # Check if we have more pages
            if not next_cursor or len(tweets) == 0:
                print("No more pages to fetch")
                break

            cursor = next_cursor

            # Handle rate limiting
            if rate_limit.should_wait():
                wait_time = rate_limit.get_wait_time()
                if wait_time > 0:
                    print(f"Rate limit reached. Waiting {wait_time} seconds...")
                    # Save checkpoint before waiting
                    if checkpoint_callback:
                        checkpoint_callback(all_tweets, cursor)
                    time.sleep(wait_time)

            # Polite delay between requests
            time.sleep(self._request_delay)

        print(f"Fetch complete! Total likes: {len(all_tweets)}")
        return all_tweets

    def _extract_tweets(self, response: Dict[str, Any]) -> List[Tweet]:
        """Extract tweets from API response"""
        tweets = []

        try:
            instructions = (
                response.get("data", {})
                .get("user", {})
                .get("result", {})
                .get("timeline", {})
                .get("timeline", {})
                .get("instructions", [])
            )

            for instruction in instructions:
                if instruction.get("type") == "TimelineAddEntries":
                    entries = instruction.get("entries", [])

                    for entry in entries:
                        if entry.get("content", {}).get("entryType") == "TimelineTimelineItem":
                            tweet_data = (
                                entry.get("content", {})
                                .get("itemContent", {})
                                .get("tweet_results", {})
                                .get("result", {})
                            )

                            if tweet_data and "legacy" in tweet_data:
                                tweet = self._parse_tweet(tweet_data)
                                if tweet:
                                    tweets.append(tweet)

        except Exception as e:
            print(f"Error extracting tweets: {e}")

        return tweets

    def _parse_tweet(self, tweet_data: Dict[str, Any]) -> Optional[Tweet]:
        """Parse tweet data into Tweet model"""
        try:
            legacy = tweet_data.get("legacy", {})
            core = tweet_data.get("core", {})
            user_results = core.get("user_results", {}).get("result", {})
            user_legacy = user_results.get("legacy", {})

            # Parse user
            user = User(
                id=user_results.get("rest_id", ""),
                screen_name=user_legacy.get("screen_name", ""),
                name=user_legacy.get("name", ""),
                profile_image_url=user_legacy.get("profile_image_url_https", ""),
                verified=user_legacy.get("verified", False),
                followers_count=user_legacy.get("followers_count", 0),
                following_count=user_legacy.get("friends_count", 0)
            )

            # Parse media
            media_list = []
            extended_entities = legacy.get("extended_entities", {})
            for media_item in extended_entities.get("media", []):
                media = Media(
                    type=media_item.get("type", "photo"),
                    url=media_item.get("url", ""),
                    media_url=media_item.get("media_url_https", ""),
                    width=media_item.get("original_info", {}).get("width"),
                    height=media_item.get("original_info", {}).get("height")
                )
                media_list.append(media)

            # Parse entities
            entities = legacy.get("entities", {})
            urls = [url["expanded_url"] for url in entities.get("urls", [])]
            hashtags = [tag["text"] for tag in entities.get("hashtags", [])]
            mentions = [mention["screen_name"] for mention in entities.get("user_mentions", [])]

            # Create tweet
            tweet = Tweet(
                id=tweet_data.get("rest_id", legacy.get("id_str", "")),
                text=legacy.get("full_text", ""),
                created_at=legacy.get("created_at", ""),
                user=user,
                retweet_count=legacy.get("retweet_count", 0),
                favorite_count=legacy.get("favorite_count", 0),
                reply_count=legacy.get("reply_count", 0),
                quote_count=legacy.get("quote_count", 0),
                lang=legacy.get("lang", "en"),
                is_retweet="retweeted_status_result" in legacy,
                is_quote="quoted_status_result" in tweet_data,
                media=media_list,
                urls=urls,
                hashtags=hashtags,
                mentions=mentions,
                conversation_id=legacy.get("conversation_id_str"),
                in_reply_to_user_id=legacy.get("in_reply_to_user_id_str"),
                raw_data=tweet_data
            )

            # Parse view count if available
            views = tweet_data.get("views", {}).get("count")
            if views:
                try:
                    tweet.view_count = int(views)
                except:
                    pass

            return tweet

        except Exception as e:
            print(f"Error parsing tweet: {e}")
            return None

    def _extract_cursor(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract next cursor from API response"""
        try:
            instructions = (
                response.get("data", {})
                .get("user", {})
                .get("result", {})
                .get("timeline", {})
                .get("timeline", {})
                .get("instructions", [])
            )

            for instruction in instructions:
                if instruction.get("type") == "TimelineAddEntries":
                    entries = instruction.get("entries", [])

                    for entry in entries:
                        content = entry.get("content", {})
                        if (content.get("entryType") == "TimelineTimelineCursor" and
                            content.get("cursorType") == "Bottom"):
                            return content.get("value")

        except Exception as e:
            print(f"Error extracting cursor: {e}")

        return None
