"""Fetcher for historical international football results (1872–present).

Primary: Kaggle dataset martj42/international-football-results-from-1872-to-2017
Fallback: GitHub raw CSV mirror (martj42/international_results)
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from wc26.config import settings
from wc26.fetchers.base import BaseFetcher
from wc26.schemas import make_match_id

logger = logging.getLogger(__name__)

GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"


class KaggleHistoricalFetcher(BaseFetcher):
    """Historical international matches: Kaggle primary, GitHub fallback."""

    source_name = "kaggle_historical"

    def _fetch(self) -> pd.DataFrame:
        try:
            return self._fetch_kaggle()
        except Exception as exc:
            logger.warning(f"Kaggle fetch failed ({exc}), trying GitHub mirror")
            return self._fetch_github()

    def _fetch_kaggle(self) -> pd.DataFrame:
        username = settings.kaggle_username
        key = settings.kaggle_key
        if not username or not key:
            raise ValueError("KAGGLE_USERNAME / KAGGLE_KEY not set")

        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = username
        env["KAGGLE_KEY"] = key

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    KAGGLE_DATASET,
                    "-p",
                    tmpdir,
                    "--unzip",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"kaggle CLI failed: {result.stderr}")

            csv_files = list(Path(tmpdir).glob("*.csv"))
            results_csv = next((f for f in csv_files if "results" in f.name.lower()), csv_files[0])
            df = pd.read_csv(results_csv)

        return self._clean(df)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _fetch_github(self) -> pd.DataFrame:
        logger.info("Fetching historical data from GitHub mirror")
        resp = httpx.get(GITHUB_CSV_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize columns and add match_id."""
        df = df.rename(
            columns={
                "date": "date",
                "home_team": "home_team",
                "away_team": "away_team",
                "home_score": "home_score",
                "away_score": "away_score",
                "tournament": "tournament",
                "neutral": "neutral",
            }
        )
        required = ["date", "home_team", "away_team", "home_score", "away_score"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home_team", "away_team"])
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)

        if "tournament" not in df.columns:
            df["tournament"] = "Unknown"
        if "neutral" not in df.columns:
            df["neutral"] = False
        df["neutral"] = df["neutral"].astype(bool)

        df["match_id"] = df.apply(
            lambda r: make_match_id(str(r["date"].date()), r["home_team"], r["away_team"]), axis=1
        )

        return df[
            ["match_id", "date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
        ].copy()
