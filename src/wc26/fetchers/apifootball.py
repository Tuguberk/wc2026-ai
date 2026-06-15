"""Fallback fetcher for WC2026 fixtures/results via api-sports.io (API-Football).

Free tier: 100 requests/DAY — treat this as a scarce resource.
We read the daily-budget headers after every response and abort early
rather than draining the quota silently.

API-Sports rate-limit headers:
  x-ratelimit-requests-limit     — daily cap (typically 100)
  x-ratelimit-requests-remaining — calls left today
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from wc26.config import settings
from wc26.fetchers.base import BaseFetcher
from wc26.schemas import make_match_id

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1  # API-Football league ID for FIFA World Cup

# Refuse to make another call when fewer than this many daily requests remain.
# The value is conservative: 5 keeps an emergency buffer for other tools.
_DAILY_BUDGET_FLOOR = 5


def _check_daily_budget(resp: httpx.Response) -> None:
    """Read daily-quota headers and raise if the budget is critically low."""
    try:
        limit = int(resp.headers.get("x-ratelimit-requests-limit", 100))
        remaining = int(resp.headers.get("x-ratelimit-requests-remaining", 999))
    except (ValueError, TypeError):
        return

    used = limit - remaining
    logger.info(
        f"api-sports.io daily budget: {remaining}/{limit} remaining "
        f"({used} used today)"
    )

    if remaining <= _DAILY_BUDGET_FLOOR:
        raise RuntimeError(
            f"api-sports.io daily quota nearly exhausted ({remaining} requests left). "
            f"Aborting to preserve the budget."
        )


class APIFootballFetcher(BaseFetcher):
    """WC2026 fixtures + results from api-sports.io (fallback, 100 req/day)."""

    source_name = "api_football"

    def _fetch(self) -> pd.DataFrame:
        key = settings.api_football_key
        if not key:
            raise ValueError("API_FOOTBALL_KEY not set")

        headers = {"x-apisports-key": key}
        fixtures = self._get_fixtures(headers)
        return self._parse(fixtures)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=5, max=60),
        reraise=True,
    )
    def _get_fixtures(self, headers: dict) -> list[dict]:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{BASE_URL}/fixtures",
                headers=headers,
                params={"league": WC_LEAGUE_ID, "season": 2026},
            )

        _check_daily_budget(resp)
        resp.raise_for_status()
        return resp.json().get("response", [])

    def _parse(self, fixtures: list[dict]) -> pd.DataFrame:
        rows = []
        for f in fixtures:
            fix = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            league = f.get("league", {})

            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            if not home or not away:
                continue

            date = pd.to_datetime(fix.get("date"), errors="coerce")
            if pd.isna(date):
                continue

            status_short = fix.get("status", {}).get("short", "NS")
            status_map = {"FT": "FINISHED", "NS": "SCHEDULED", "1H": "IN_PLAY", "2H": "IN_PLAY"}
            status = status_map.get(status_short, "SCHEDULED")

            rows.append(
                {
                    "match_id": make_match_id(str(date.date()), home, away),
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(goals.get("home")) if goals.get("home") is not None else None,
                    "away_score": float(goals.get("away")) if goals.get("away") is not None else None,
                    "stage": league.get("round", "GROUP_STAGE"),
                    "group": league.get("round") if "Group" in str(league.get("round", "")) else None,
                    "status": status,
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["match_id", "date", "home_team", "away_team", "home_score", "away_score", "stage", "group", "status"]
        )
