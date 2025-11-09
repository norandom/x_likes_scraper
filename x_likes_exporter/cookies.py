"""
Cookie parsing and management for X authentication
"""

import json
from typing import Dict, List
from http.cookiejar import Cookie


class CookieManager:
    """Manages cookies for X API authentication"""

    def __init__(self, cookies_file: str):
        """
        Initialize cookie manager with a cookies JSON file

        Args:
            cookies_file: Path to cookies.json file exported from browser
        """
        self.cookies_file = cookies_file
        self.cookies = self._load_cookies()

    def _load_cookies(self) -> Dict[str, str]:
        """Load cookies from JSON file and convert to dict format"""
        with open(self.cookies_file, 'r', encoding='utf-8') as f:
            cookies_list = json.load(f)

        # Convert list of cookie objects to simple dict
        cookies_dict = {}
        for cookie in cookies_list:
            cookies_dict[cookie['name']] = cookie['value']

        return cookies_dict

    def get_cookie_dict(self) -> Dict[str, str]:
        """Get cookies as a dictionary for requests library"""
        return self.cookies

    def get_cookie_header(self) -> str:
        """Get cookies as a Cookie header string"""
        return "; ".join([f"{name}={value}" for name, value in self.cookies.items()])

    def get_csrf_token(self) -> str:
        """Extract CSRF token (ct0) from cookies"""
        return self.cookies.get('ct0', '')

    def get_auth_token(self) -> str:
        """Extract auth_token from cookies"""
        return self.cookies.get('auth_token', '')

    def get_guest_id(self) -> str:
        """Extract guest_id from cookies"""
        return self.cookies.get('guest_id', '')

    def validate(self) -> bool:
        """
        Validate that required cookies are present

        Returns:
            True if all required cookies are present, False otherwise
        """
        required = ['ct0', 'auth_token']
        return all(cookie in self.cookies for cookie in required)
