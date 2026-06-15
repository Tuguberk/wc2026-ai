"""Base interface for all prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MatchPrediction:
    """Probabilistic prediction for a single match."""

    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float = 0.0
    exp_away_goals: float = 0.0
    score_matrix: np.ndarray | None = None  # shape (max_goals, max_goals)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.p_home + self.p_draw + self.p_away
        if abs(total - 1.0) > 0.05:
            # Normalize if slightly off
            self.p_home /= total
            self.p_draw /= total
            self.p_away /= total


class BaseModel(ABC):
    """Common interface for prediction models."""

    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train model on historical match data."""
        ...

    @abstractmethod
    def predict_match(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> MatchPrediction:
        """Return probabilistic prediction for a single match."""
        ...

    def predict_batch(
        self,
        matches: pd.DataFrame,
    ) -> list[MatchPrediction]:
        """Predict all rows in a DataFrame with columns home_team, away_team, neutral."""
        results = []
        for _, row in matches.iterrows():
            try:
                pred = self.predict_match(
                    home=row["home_team"],
                    away=row["away_team"],
                    neutral=bool(row.get("neutral", False)),
                )
                results.append(pred)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    f"predict failed for {row.get('home_team')} vs {row.get('away_team')}: {exc}"
                )
        return results

    def is_fitted(self) -> bool:
        return hasattr(self, "_fitted") and self._fitted
