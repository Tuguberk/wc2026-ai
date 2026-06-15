"""FIFA World Ranking fetcher.

Primary source: Kaggle dataset `cashncarry/fifaWorldRanking`
  → historical weekly rankings from 1992 to present
  → provides time-series lookup: team + date → rank + points

Fallback: GitHub mirror of the same dataset.

Output schema
-------------
rank_date   : date
team        : str  (country name, normalised)
rank        : int  (1 = strongest)
total_points: float

Usage in feature pipeline:
  lookup = FifaRankingFetcher().fetch()  # call once
  rank, pts = fifa_rank_at(lookup, "Turkey", match_date)
"""

from __future__ import annotations

import logging

import pandas as pd

from wc26.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# GitHub mirror of the Kaggle dataset (raw CSV, updated periodically)
_GITHUB_URL = (
    "https://raw.githubusercontent.com/cnc8/fifa-world-ranking/main/"
    "fifa_ranking-2024-04-04.csv"
)

# Name normalisations: FIFA dataset name → our historical match dataset name
_NAME_MAP: dict[str, str] = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "United States": "United States",
    "Türkiye": "Turkey",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "Antigua and Barbuda": "Antigua and Barbuda",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "São Tomé e Príncipe": "Sao Tome and Principe",
    "Congo DR": "DR Congo",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Northern Ireland": "Northern Ireland",
    "Chinese Taipei": "Chinese Taipei",
    "Curacao": "Curacao",
}


def _normalise_team(name: str) -> str:
    return _NAME_MAP.get(name, name)


class FifaRankingFetcher(BaseFetcher):
    """Fetch historical FIFA rankings and return a time-series DataFrame."""

    source_name = "fifa_ranking"

    def _fetch(self) -> pd.DataFrame:
        # Try Kaggle first (requires kaggle credentials)
        df = self._fetch_kaggle()
        if df is not None and not df.empty:
            return df
        # Fallback: GitHub mirror
        return self._fetch_github()

    def _fetch_kaggle(self) -> pd.DataFrame | None:
        import os, tempfile, subprocess
        from pathlib import Path
        from wc26.config import settings as s

        if not s.kaggle_key:
            return None

        # Build subprocess environment.
        # New Kaggle CLI (v1.6+) uses KAGGLE_API_TOKEN (KGAT_... format).
        # Older CLI uses KAGGLE_USERNAME + KAGGLE_KEY. We set both so either works.
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = s.kaggle_key
        if s.kaggle_username:
            env["KAGGLE_USERNAME"] = s.kaggle_username
            env["KAGGLE_KEY"] = s.kaggle_key

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["kaggle", "datasets", "download",
                     "cashncarry/fifaworldranking", "--unzip", "-p", tmpdir],
                    env=env, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    logger.debug(f"Kaggle FIFA ranking download failed: {result.stderr}")
                    return None
                csvs = list(Path(tmpdir).glob("*.csv"))
                if not csvs:
                    return None
                return _parse_ranking_csv(csvs[0])
        except Exception as exc:
            logger.debug(f"Kaggle FIFA ranking unavailable: {exc}")
            return None

    def _fetch_github(self) -> pd.DataFrame:
        import httpx
        from pathlib import Path as _Path
        import tempfile

        logger.info("Fetching FIFA ranking from GitHub mirror")
        resp = httpx.get(_GITHUB_URL, timeout=30, follow_redirects=True)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(resp.content)
            tmp_path = _Path(f.name)

        try:
            return _parse_ranking_csv(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


def _parse_ranking_csv(path) -> pd.DataFrame:
    from pathlib import Path
    df = pd.read_csv(path)

    # Try to identify columns flexibly
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_")
        if cl in ("rank_date", "date", "ranking_date"):
            col_map["rank_date"] = col
        elif cl in ("country_full", "country", "team", "name", "nation"):
            col_map["team"] = col
        elif cl in ("rank", "ranking", "world_rank"):
            col_map["rank"] = col
        elif "total_points" in cl or cl in ("points", "total_pts", "score"):
            col_map["total_points"] = col

    required = {"rank_date", "team", "rank"}
    missing = required - set(col_map.keys())
    if missing:
        raise ValueError(f"FIFA ranking CSV missing columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame()
    out["rank_date"] = pd.to_datetime(df[col_map["rank_date"]])
    out["team"] = df[col_map["team"]].astype(str).map(_normalise_team).fillna(df[col_map["team"]])
    out["rank"] = pd.to_numeric(df[col_map["rank"]], errors="coerce").fillna(200).astype(int)
    out["total_points"] = (
        pd.to_numeric(df[col_map.get("total_points", col_map["rank"])], errors="coerce").fillna(0)
        if "total_points" in col_map
        else (1600 - out["rank"] * 5).clip(lower=0).astype(float)  # approximate if missing
    )

    out = out.dropna(subset=["rank_date", "team"]).sort_values("rank_date")
    logger.info(
        f"FIFA ranking: {len(out)} rows, "
        f"{out['team'].nunique()} teams, "
        f"{out['rank_date'].min().date()} → {out['rank_date'].max().date()}"
    )
    return out


# ── Lookup helper (used in feature pipeline) ──────────────────────────────────

_FALLBACK_RANK = 150
_FALLBACK_POINTS = 0.0


def build_ranking_lookup(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build per-team sorted ranking time-series for fast asof lookup.

    Returns dict: team_name → DataFrame(rank_date, rank, total_points) sorted by date.
    """
    return {
        team: grp.sort_values("rank_date").reset_index(drop=True)
        for team, grp in df.groupby("team")
    }


def fifa_rank_at(
    lookup: dict[str, pd.DataFrame],
    team: str,
    date: pd.Timestamp,
) -> tuple[int, float]:
    """Return (rank, points) for `team` at or before `date`.

    Falls back to (_FALLBACK_RANK, _FALLBACK_POINTS) if team not found.
    """
    team_df = lookup.get(team)
    if team_df is None or team_df.empty:
        return _FALLBACK_RANK, _FALLBACK_POINTS

    past = team_df[team_df["rank_date"] <= date]
    if past.empty:
        # Use earliest available ranking
        row = team_df.iloc[0]
    else:
        row = past.iloc[-1]

    return int(row["rank"]), float(row["total_points"])
