"""
Data models for tweets and users
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class User:
    """Twitter/X user model"""
    id: str
    screen_name: str
    name: str
    profile_image_url: Optional[str] = None
    verified: bool = False
    followers_count: int = 0
    following_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Media:
    """Tweet media (image/video) model"""
    type: str  # photo, video, animated_gif
    url: str
    media_url: Optional[str] = None
    preview_image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    local_path: Optional[str] = None  # Path to downloaded file

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Tweet:
    """Twitter/X tweet model"""
    id: str
    text: str
    created_at: str
    user: User
    retweet_count: int = 0
    favorite_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    lang: str = "en"
    is_retweet: bool = False
    is_quote: bool = False
    quoted_tweet: Optional['Tweet'] = None
    retweeted_tweet: Optional['Tweet'] = None
    media: List[Media] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    in_reply_to_user_id: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary

        Args:
            include_raw: Include raw API response data

        Returns:
            Dictionary representation of the tweet
        """
        data = {
            'id': self.id,
            'text': self.text,
            'created_at': self.created_at,
            'user': self.user.to_dict(),
            'retweet_count': self.retweet_count,
            'favorite_count': self.favorite_count,
            'reply_count': self.reply_count,
            'quote_count': self.quote_count,
            'view_count': self.view_count,
            'lang': self.lang,
            'is_retweet': self.is_retweet,
            'is_quote': self.is_quote,
            'media': [m.to_dict() for m in self.media],
            'urls': self.urls,
            'hashtags': self.hashtags,
            'mentions': self.mentions,
            'conversation_id': self.conversation_id,
            'in_reply_to_user_id': self.in_reply_to_user_id,
        }

        if self.quoted_tweet:
            data['quoted_tweet'] = self.quoted_tweet.to_dict(include_raw=False)

        if self.retweeted_tweet:
            data['retweeted_tweet'] = self.retweeted_tweet.to_dict(include_raw=False)

        if include_raw and self.raw_data:
            data['raw_data'] = self.raw_data

        return data

    def get_url(self) -> str:
        """Get the URL to this tweet"""
        return f"https://x.com/{self.user.screen_name}/status/{self.id}"

    def get_created_datetime(self) -> datetime:
        """Parse and return created_at as datetime object"""
        from dateutil import parser
        return parser.parse(self.created_at)
