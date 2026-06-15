"""Build processed tables from raw fetcher outputs.

Responsibilities:
- Clean and merge historical matches
- Assign deterministic match_id, dedup
- Validate against pandera schemas
- Write to data/processed/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from wc26.config import settings
from wc26.schemas import (
    HistoricalMatchSchema,
    WC2026FixtureSchema,
    make_match_id,
)

logger = logging.getLogger(__name__)


def build_historical(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean, dedup, and validate historical match data."""
    df = raw_df.copy()

    # Ensure match_id present
    if "match_id" not in df.columns:
        df["match_id"] = df.apply(
            lambda r: make_match_id(str(r["date"].date()), r["home_team"], r["away_team"]), axis=1
        )

    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)
    if "tournament" not in df.columns:
        df["tournament"] = "Unknown"

    try:
        HistoricalMatchSchema.validate(df, lazy=True)
    except Exception as exc:
        logger.warning(f"Schema validation warning (historical): {exc}")

    return df[
        ["match_id", "date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    ]


def build_fixtures(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean, dedup, and validate WC2026 fixture data."""
    df = raw_df.copy()

    if "match_id" not in df.columns:
        df["match_id"] = df.apply(
            lambda r: make_match_id(str(r["date"].date()), r["home_team"], r["away_team"]), axis=1
        )

    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])

    if "stage" not in df.columns:
        df["stage"] = "GROUP_STAGE"
    if "group" not in df.columns:
        df["group"] = None
    if "status" not in df.columns:
        df["status"] = "SCHEDULED"

    try:
        WC2026FixtureSchema.validate(df, lazy=True)
    except Exception as exc:
        logger.warning(f"Schema validation warning (fixtures): {exc}")

    return df[
        ["match_id", "date", "home_team", "away_team", "home_score", "away_score", "stage", "group", "status"]
    ]


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a DataFrame to data/processed/{name}.parquet."""
    out_dir = settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows → {path}")
    return path


def load_processed(name: str) -> pd.DataFrame | None:
    """Load a processed parquet file, return None if missing."""
    path = settings.processed_dir / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def split_results_scheduled(fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split fixtures into finished results and upcoming matches."""
    finished_mask = fixtures["status"] == "FINISHED"
    # Also handle matches with score but status not updated
    has_score = fixtures["home_score"].notna() & fixtures["away_score"].notna()
    results = fixtures[finished_mask | has_score].copy()
    scheduled = fixtures[~(finished_mask | has_score)].copy()
    return results, scheduled
