"""
O*NET Web Services API helper — supplementary data source.

Use this for fields not available in the local database files,
such as detailed job descriptions and education requirement percentages.

The primary data source is onet_data.py (local CSV files).

Usage:
    from src.onet_api import ONetClient
    client = ONetClient()  # reads credentials from .env
    skills = client.get_skills("11-9021.00")
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api-v2.onetcenter.org"


class ONetClient:
    """Lightweight wrapper around the O*NET Web Services REST API v2."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ONET_API_KEY")
        if not self.api_key:
            raise ValueError(
                "O*NET API key not found. "
                "Set ONET_API_KEY in .env file."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        })

    def _get(self, endpoint: str) -> dict:
        """Make a GET request with rate limiting."""
        url = f"{BASE_URL}/{endpoint}"
        resp = self.session.get(url)
        resp.raise_for_status()
        time.sleep(0.5)
        return resp.json()

    def get_occupation(self, code: str) -> dict:
        """Get basic occupation info (title, description)."""
        return self._get(f"online/occupations/{code}")

    def get_skills(self, code: str) -> list[dict]:
        """Get skills with importance and level scores."""
        data = self._get(f"online/occupations/{code}/summary/skills")
        return data.get("element", [])

    def get_knowledge(self, code: str) -> list[dict]:
        """Get knowledge areas."""
        data = self._get(f"online/occupations/{code}/summary/knowledge")
        return data.get("element", [])

    def get_abilities(self, code: str) -> list[dict]:
        """Get abilities."""
        data = self._get(f"online/occupations/{code}/summary/abilities")
        return data.get("element", [])

    def get_education(self, code: str) -> list[dict]:
        """Get education requirements."""
        data = self._get(f"online/occupations/{code}/summary/education")
        return data.get("element", [])

    def get_related_occupations(self, code: str) -> list[dict]:
        """Get related occupations."""
        data = self._get(f"online/occupations/{code}/summary/related_occupations")
        return data.get("occupation", [])
