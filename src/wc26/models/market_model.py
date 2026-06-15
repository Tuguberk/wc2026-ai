"""Market odds model: converts bookmaker decimal odds → calibrated probabilities.

De-vig method: Shin (1993) approximation — the most theoretically sound method
for converting overround odds to true implied probabilities.

Usage
-----
model = MarketModel(odds_api_fetcher)   # used at inference time only
pred = model.predict_match(home, away)  # returns None if no odds available
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from wc26.models.base import BaseModel, MatchPrediction

logger = logging.getLogger(__name__)


@dataclass
class RawOdds:
    """Decimal bookmaker odds for a single match (best available across bookmakers)."""

    home: float
    draw: float
    away: float

    def overround(self) -> float:
        return 1 / self.home + 1 / self.draw + 1 / self.away

    def margin(self) -> float:
        return self.overround() - 1.0


def devig_additive(odds: RawOdds) -> tuple[float, float, float]:
    """Simplest de-vig: divide each implied prob by overround (proportional shrinkage)."""
    inv = [1 / odds.home, 1 / odds.draw, 1 / odds.away]
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s


def devig_shin(odds: RawOdds) -> tuple[float, float, float]:
    """Shin (1993) de-vig via iterative Newton's method.

    Solves for z (insider-trading fraction) such that:
        p_i = (sqrt(z^2 + 4*(1-z)*q_i^2/S) - z) / (2*(1-z))
    where q_i = 1/o_i, S = sum(q_i).

    Falls back to additive de-vig if Newton's method doesn't converge.
    """
    q = np.array([1 / odds.home, 1 / odds.draw, 1 / odds.away])
    S = q.sum()
    n = len(q)

    def shin_probs(z: float) -> np.ndarray:
        disc = np.sqrt(z**2 + 4 * (1 - z) * q**2 / S)
        return (disc - z) / (2 * (1 - z))

    def f(z: float) -> float:
        return float(shin_probs(z).sum() - 1.0)

    # Binary search for z in (0, 1)
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break

    z_opt = (lo + hi) / 2
    p = shin_probs(z_opt)

    # Sanity check
    if any(p < 0) or abs(p.sum() - 1.0) > 0.01:
        logger.debug("Shin de-vig failed — falling back to additive")
        return devig_additive(odds)

    return float(p[0]), float(p[1]), float(p[2])


class MarketModel(BaseModel):
    """Wraps an odds fetcher and converts its output to MatchPrediction.

    No training required — uses live market odds directly.
    """

    name = "market"

    def __init__(self, odds_fetcher=None) -> None:
        self.odds_fetcher = odds_fetcher
        self._fitted = odds_fetcher is not None

    def fit(self, df) -> None:
        pass  # no training

    def predict_match(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> MatchPrediction | None:
        """Return de-vigged market probabilities, or None if no odds available."""
        if self.odds_fetcher is None:
            return None

        raw = self.odds_fetcher.get_odds(home, away)
        if raw is None:
            return None

        p_home, p_draw, p_away = devig_shin(raw)
        logger.debug(
            f"Market odds {home} vs {away}: "
            f"H={raw.home:.2f}(→{p_home:.3f}) "
            f"D={raw.draw:.2f}(→{p_draw:.3f}) "
            f"A={raw.away:.2f}(→{p_away:.3f}) "
            f"margin={raw.margin():.1%}"
        )
        return MatchPrediction(
            home_team=home,
            away_team=away,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            metadata={"odds_home": raw.home, "odds_draw": raw.draw, "odds_away": raw.away},
        )
