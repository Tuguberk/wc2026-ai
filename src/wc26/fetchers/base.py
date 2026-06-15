"""Base fetcher interface that all data sources implement."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

from wc26.config import settings

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Common interface for every data source fetcher."""

    source_name: str = "base"

    def fetch(self) -> pd.DataFrame:
        """Fetch data, cache raw output, and return a DataFrame."""
        raw_dir = settings.raw_dir / self.source_name
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = self._fetch()
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            cache_path = raw_dir / f"{ts}.parquet"
            df.to_parquet(cache_path, index=False)
            logger.info(f"[{self.source_name}] fetched {len(df)} rows → {cache_path}")
            return df
        except Exception as exc:
            logger.warning(f"[{self.source_name}] fetch failed: {exc}")
            # Graceful degradation: return last cached file if available
            cached = sorted(raw_dir.glob("*.parquet"))
            if cached:
                logger.info(f"[{self.source_name}] falling back to cache {cached[-1]}")
                return pd.read_parquet(cached[-1])
            raise

    @abstractmethod
    def _fetch(self) -> pd.DataFrame:
        """Subclasses implement this to actually retrieve data."""
        ...

    def _last_cache(self) -> pd.DataFrame | None:
        """Return the most recent cached DataFrame, or None."""
        raw_dir = settings.raw_dir / self.source_name
        cached = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []
        return pd.read_parquet(cached[-1]) if cached else None
