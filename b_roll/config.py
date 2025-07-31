#!/usr/bin/env python3
"""
Configuration module for API request management
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class APIConfig:
    """Configuration class for managing API requests"""

    def __init__(self):
        # Load environment variables
        project_root = Path(__file__).parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # API request mode - True for real API calls, False for mock
        self.api_request_enabled = self._get_api_request_mode()

    def _get_api_request_mode(self) -> bool:
        """Get API request mode from environment variable"""
        api_request = os.getenv("API_REQUEST", "true").lower()
        return api_request in ("true", "1", "yes", "on")

    @property
    def is_api_enabled(self) -> bool:
        """Check if real API requests are enabled"""
        return self.api_request_enabled

    @property
    def is_mock_enabled(self) -> bool:
        """Check if mock mode is enabled"""
        return not self.api_request_enabled

    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment variables"""
        if self.is_mock_enabled:
            return "mock_key"
        return os.getenv(key_name)


# Global configuration instance
config = APIConfig()
