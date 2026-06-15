"""Phase 2 xG fetcher via soccerdata library (FotMob/Sofascore).

Disabled in MVP (ENABLE_XG=false). When enabled, failures are logged and ignored.
"""

from __future__ import annotations

import logging

import pandas as pd

from wc26.config import settings
from wc26.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class XGFetcher(BaseFetcher):
    """xG data from FotMob via soccerdata (Phase 2, optional)."""

    source_name = "xg_soccerdata"

    def _fetch(self) -> pd.DataFrame:
        if not settings.enable_xg:
            logger.debug("xG fetching disabled (ENABLE_XG=false)")
            return pd.DataFrame(columns=["match_id", "home_xg", "away_xg"])

        try:
            import soccerdata as sd  # type: ignore

            fotmob = sd.FotMob()
            # FotMob competition ID for World Cup 2026 — update when available
            schedule = fotmob.read_schedule(competition="WC", season="2026")
            xg_df = fotmob.read_shot_events(match_id=schedule.index.tolist())

            if xg_df.empty:
                return pd.DataFrame(columns=["match_id", "home_xg", "away_xg"])

            agg = (
                xg_df.groupby(["match_id", "team"])["xg"]
                .sum()
                .unstack(fill_value=0.0)
                .reset_index()
            )
            agg.columns.name = None
            return agg.rename(columns={agg.columns[1]: "home_xg", agg.columns[2]: "away_xg"})

        except Exception as exc:
            logger.warning(f"xG fetch failed (graceful skip): {exc}")
            return pd.DataFrame(columns=["match_id", "home_xg", "away_xg"])
