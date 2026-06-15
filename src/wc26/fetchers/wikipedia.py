"""Last-resort fallback: scrape WC2026 match tables from Wikipedia via pandas.read_html.

Most brittle source — only used if both APIs are unavailable.
"""

from __future__ import annotations

import logging
import re

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from wc26.fetchers.base import BaseFetcher
from wc26.schemas import make_match_id

logger = logging.getLogger(__name__)

WC2026_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"


class WikipediaFetcher(BaseFetcher):
    """WC2026 fixtures from Wikipedia (last-resort, most fragile)."""

    source_name = "wikipedia"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=30))
    def _fetch(self) -> pd.DataFrame:
        logger.warning("Using Wikipedia as fixture source — most fragile fallback")
        resp = httpx.get(WC2026_URL, timeout=30, follow_redirects=True)
        resp.raise_for_status()

        tables = pd.read_html(resp.text)
        rows = []
        for tbl in tables:
            parsed = self._try_parse_match_table(tbl)
            rows.extend(parsed)

        if not rows:
            logger.error("Wikipedia: could not parse any match tables")
            return pd.DataFrame(
                columns=["match_id", "date", "home_team", "away_team", "home_score", "away_score", "stage", "group", "status"]
            )

        df = pd.DataFrame(rows)
        return df.drop_duplicates(subset=["match_id"])

    def _try_parse_match_table(self, tbl: pd.DataFrame) -> list[dict]:
        """Best-effort parse of a Wikipedia match table."""
        tbl.columns = [str(c).lower().strip() for c in tbl.columns]
        rows = []

        for _, row in tbl.iterrows():
            row_str = " ".join(str(v) for v in row.values)
            score_match = re.search(r"(\d+)\s*[–\-]\s*(\d+)", row_str)
            if not score_match:
                continue

            date = None
            home_team = ""
            away_team = ""

            for col in ["date", "match date"]:
                if col in tbl.columns and str(row.get(col, "")) not in ("nan", ""):
                    try:
                        date = pd.to_datetime(str(row[col]), errors="coerce")
                    except Exception:
                        pass

            for col in ["home team", "team 1", "home"]:
                if col in tbl.columns and str(row.get(col, "")) not in ("nan", ""):
                    home_team = str(row[col]).strip()
                    break

            for col in ["away team", "team 2", "away"]:
                if col in tbl.columns and str(row.get(col, "")) not in ("nan", ""):
                    away_team = str(row[col]).strip()
                    break

            if not home_team or not away_team or date is None or pd.isna(date):
                continue

            home_score = int(score_match.group(1))
            away_score = int(score_match.group(2))

            rows.append(
                {
                    "match_id": make_match_id(str(date.date()), home_team, away_team),
                    "date": date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                    "stage": "GROUP_STAGE",
                    "group": None,
                    "status": "FINISHED",
                }
            )

        return rows
