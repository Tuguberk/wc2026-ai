"""Live match odds fetcher via The Odds API (https://the-odds-api.com).

Requires `ODDS_API_KEY` in .env. Fetches H2H (3-way) pre-match odds for
WC2026 group stage and knockout matches and returns the best available
decimal odds across bookmakers.

API docs: https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from wc26.models.market_model import RawOdds

logger = logging.getLogger(__name__)

_SPORT_KEY = "soccer_fifa_world_cup"
_BASE_URL = "https://api.the-odds-api.com/v4"

# Bookmakers to prefer in order (all available if not specified)
_PREFERRED_BOOKS = [
    "bet365", "pinnacle", "betfair_ex_eu", "betfair_ex_uk", "unibet_eu",
    "williamhill", "betway", "bwin",
]

# Name normalisations: common bookmaker names → our internal team names
_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "South Korea": "South Korea",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "Ivory Coast": "Ivory Coast",
    "DR Congo": "DR Congo",
}


def _norm(name: str) -> str:
    return _NAME_MAP.get(name, name).strip().lower()


class OddsAPIFetcher:
    """Fetches live WC2026 match odds from The Odds API.

    Call `get_odds(home, away)` to get decimal odds for a specific match.
    The result is cached in memory for the lifetime of the object.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._cache: dict[tuple[str, str], RawOdds | None] = {}
        self._events: list[dict[str, Any]] | None = None

    def _fetch_events(self) -> list[dict[str, Any]]:
        if self._events is not None:
            return self._events

        import httpx

        url = f"{_BASE_URL}/sports/{_SPORT_KEY}/odds/"
        params = {
            "apiKey": self._api_key,
            "regions": "uk,eu,us",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }

        try:
            resp = httpx.get(url, params=params, timeout=15)
            resp.raise_for_status()
            self._events = resp.json()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info(
                f"The Odds API: {len(self._events)} events fetched "
                f"(requests remaining: {remaining})"
            )
        except Exception as exc:
            logger.warning(f"The Odds API fetch failed: {exc}")
            self._events = []

        return self._events or []

    def _best_odds(self, event: dict[str, Any]) -> RawOdds | None:
        """Extract best H/D/A odds across all bookmakers for an event."""
        best_h = best_d = best_a = 0.0

        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                # Outcomes keyed by team name (home / draw / away)
                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")
                draw_key = "Draw"

                h = outcomes.get(home_name, 0.0)
                d = outcomes.get(draw_key, 0.0)
                a = outcomes.get(away_name, 0.0)

                if h > best_h:
                    best_h = h
                if d > best_d:
                    best_d = d
                if a > best_a:
                    best_a = a

        if best_h <= 1.0 or best_d <= 1.0 or best_a <= 1.0:
            return None

        return RawOdds(home=best_h, draw=best_d, away=best_a)

    def get_odds(self, home: str, away: str) -> RawOdds | None:
        """Return decimal odds for the given match, or None if not found."""
        key = (home, away)
        if key in self._cache:
            return self._cache[key]

        h_norm = _norm(home)
        a_norm = _norm(away)

        for event in self._fetch_events():
            ev_home = _norm(event.get("home_team", ""))
            ev_away = _norm(event.get("away_team", ""))

            if (ev_home == h_norm and ev_away == a_norm) or (
                ev_home == a_norm and ev_away == h_norm
            ):
                raw = self._best_odds(event)
                if raw is not None and ev_home == a_norm:
                    # Swap if fetched in reverse order
                    raw = RawOdds(home=raw.away, draw=raw.draw, away=raw.home)
                self._cache[key] = raw
                return raw

        self._cache[key] = None
        return None

    def available_matches(self) -> list[tuple[str, str]]:
        """Return list of (home, away) pairs currently available from the API."""
        return [
            (ev.get("home_team", ""), ev.get("away_team", ""))
            for ev in self._fetch_events()
        ]


def build_odds_fetcher() -> OddsAPIFetcher | None:
    """Build OddsAPIFetcher from settings, or return None if no key configured."""
    from wc26.config import settings

    if not settings.odds_api_key:
        logger.debug("ODDS_API_KEY not set — market odds disabled")
        return None

    return OddsAPIFetcher(api_key=settings.odds_api_key)
