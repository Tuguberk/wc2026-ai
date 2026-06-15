"""Fetcher for WC2026 fixtures and results via football-data.org API.

Rate limiting is driven by response headers (not a fixed sleep):
  X-RequestsAvailable    — remaining calls before the window closes
  X-RequestCounter-Reset — seconds until the counter resets

We inspect these after every response and sleep only when the window
is exhausted, avoiding both unnecessary waits and 429s.
Competition code: WC (FIFA World Cup)
"""

from __future__ import annotations

import logging
import time

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from wc26.config import settings
from wc26.fetchers.base import BaseFetcher
from wc26.schemas import make_match_id

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
WC_CODE = "WC"
WC_SEASON = 2026

# Sleep when fewer than this many requests remain in the window
_LOW_QUOTA_THRESHOLD = 2


def _respect_quota(resp: httpx.Response) -> None:
    """Read rate-limit headers and sleep if the window is nearly exhausted."""
    try:
        available = int(resp.headers.get("X-RequestsAvailable", 999))
        reset_in = int(resp.headers.get("X-RequestCounter-Reset", 0))
    except (ValueError, TypeError):
        return

    logger.debug(f"football-data.org quota: {available} requests left, resets in {reset_in}s")

    if available < _LOW_QUOTA_THRESHOLD and reset_in > 0:
        sleep_for = reset_in + 1  # +1 s safety margin
        logger.info(
            f"football-data.org quota low ({available} left). "
            f"Sleeping {sleep_for}s until window resets."
        )
        time.sleep(sleep_for)


class FootballDataFetcher(BaseFetcher):
    """WC2026 fixtures + results from football-data.org."""

    source_name = "football_data"

    def _fetch(self) -> pd.DataFrame:
        api_key = settings.football_data_api_key
        if not api_key:
            raise ValueError("FOOTBALL_DATA_API_KEY not set")

        headers = {"X-Auth-Token": api_key}
        matches = self._get_matches(headers)
        return self._parse(matches)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=5, max=120),
        reraise=True,
    )
    def _get_matches(self, headers: dict) -> list[dict]:
        """Fetch all WC2026 matches, honouring rate-limit response headers."""
        url = f"{BASE_URL}/competitions/{WC_CODE}/matches"
        params: dict = {"season": WC_SEASON} if WC_SEASON else {}

        with httpx.Client(timeout=30) as client:
            resp = client.get(url, headers=headers, params=params)

        _respect_quota(resp)

        if resp.status_code == 429:
            reset_in = int(resp.headers.get("X-RequestCounter-Reset", 60))
            sleep_for = reset_in + 1
            logger.warning(f"429 from football-data.org — sleeping {sleep_for}s then retrying")
            time.sleep(sleep_for)
            resp.raise_for_status()  # triggers tenacity retry

        resp.raise_for_status()
        return resp.json().get("matches", [])

    def _parse(self, matches: list[dict]) -> pd.DataFrame:
        rows = []
        for m in matches:
            utc_date = m.get("utcDate", "")
            try:
                date = pd.to_datetime(utc_date)
            except Exception:
                continue

            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            if not home or not away:
                continue

            score = m.get("score", {})
            full = score.get("fullTime", {})
            home_score = full.get("home")
            away_score = full.get("away")

            status = m.get("status", "SCHEDULED")
            stage = m.get("stage", "GROUP_STAGE")
            group = m.get("group")

            rows.append(
                {
                    "match_id": make_match_id(str(date.date()), home, away),
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(home_score) if home_score is not None else None,
                    "away_score": float(away_score) if away_score is not None else None,
                    "stage": stage,
                    "group": group,
                    "status": status,
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["match_id", "date", "home_team", "away_team", "home_score", "away_score", "stage", "group", "status"]
        )
