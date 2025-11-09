"""
Authentication and token extraction for X API
"""

import re
import requests
from typing import Optional, Dict
from .cookies import CookieManager


class XAuthenticator:
    """Handles authentication with X (Twitter) API"""

    def __init__(self, cookie_manager: CookieManager):
        """
        Initialize authenticator

        Args:
            cookie_manager: CookieManager instance with loaded cookies
        """
        self.cookie_manager = cookie_manager
        self._bearer_token: Optional[str] = None
        self._query_ids: Dict[str, str] = {}

    def get_bearer_token(self) -> str:
        """
        Extract Bearer token from X.com main script

        Returns:
            Bearer token string

        Raises:
            Exception: If token extraction fails
        """
        if self._bearer_token:
            return self._bearer_token

        try:
            # Fetch X.com home page
            response = requests.get(
                "https://x.com/home",
                cookies=self.cookie_manager.get_cookie_dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            html = response.text

            # Extract main script URL
            script_pattern = r'<link[^>]+href="(https://abs\.twimg\.com/responsive-web/client-web/main\.[^"]+\.js)"[^>]*>'
            script_match = re.search(script_pattern, html)

            if not script_match:
                raise Exception("Failed to find main script URL in X.com homepage")

            script_url = script_match.group(1)

            # Fetch main script
            script_response = requests.get(script_url)
            script_response.raise_for_status()
            script_text = script_response.text

            # Extract Bearer token
            bearer_pattern = r'"(Bearer [\w%]+)"'
            bearer_match = re.search(bearer_pattern, script_text)

            if not bearer_match:
                raise Exception("Failed to extract Bearer token from main script")

            self._bearer_token = bearer_match.group(1)
            return self._bearer_token

        except Exception as e:
            raise Exception(f"Error getting Bearer token: {e}")

    def get_query_id(self, operation_name: str) -> str:
        """
        Extract GraphQL query ID for a specific operation

        Args:
            operation_name: Name of the GraphQL operation (e.g., "Likes")

        Returns:
            Query ID string

        Raises:
            Exception: If query ID extraction fails
        """
        if operation_name in self._query_ids:
            return self._query_ids[operation_name]

        try:
            # Fetch X.com home page
            response = requests.get(
                "https://x.com/home",
                cookies=self.cookie_manager.get_cookie_dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            html = response.text

            # Extract main script URL
            script_pattern = r'<link[^>]+href="(https://abs\.twimg\.com/responsive-web/client-web/main\.[^"]+\.js)"[^>]*>'
            script_match = re.search(script_pattern, html)

            if not script_match:
                raise Exception("Failed to find main script URL")

            script_url = script_match.group(1)

            # Fetch main script
            script_response = requests.get(script_url)
            script_response.raise_for_status()
            script_text = script_response.text

            # Extract query metadata for the operation
            query_pattern = rf'{{queryId:"([^"]+)",operationName:"{operation_name}"'
            query_match = re.search(query_pattern, script_text)

            if not query_match:
                raise Exception(f"Failed to extract query ID for {operation_name}")

            query_id = query_match.group(1)
            self._query_ids[operation_name] = query_id
            return query_id

        except Exception as e:
            raise Exception(f"Error getting query ID for {operation_name}: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for requests"""
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        }
