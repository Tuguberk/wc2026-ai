"""Ensemble: weighted average of Bayesian Poisson and LightGBM models.

Optimal weights are found by minimising Brier score on the holdout
feature DataFrame. The calibrator (if provided) is applied on top.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wc26.models.base import BaseModel, MatchPrediction
from wc26.models.bayesian_poisson import BayesianPoissonModel
from wc26.models.calibration import IsotonicCalibrator
from wc26.models.lgbm import LGBMModel
from wc26.models.market_model import MarketModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Bayesian + LightGBM weighted ensemble with optional isotonic calibration."""

    name = "ensemble"

    def __init__(
        self,
        bayesian: BayesianPoissonModel,
        lgbm: LGBMModel,
        bayesian_weight: float = 0.5,
        calibrator: IsotonicCalibrator | None = None,
        market: MarketModel | None = None,
        market_weight: float = 0.25,
    ) -> None:
        self.bayesian = bayesian
        self.lgbm = lgbm
        self.bayesian_weight = bayesian_weight
        self.calibrator = calibrator
        self.market = market
        self.market_weight = market_weight
        self._fitted = bayesian.is_fitted() and lgbm.is_fitted()

    def fit(self, df: pd.DataFrame) -> None:
        pass  # components are fitted externally

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------

    def fit_weights(self, val_features: pd.DataFrame) -> None:
        """Find Bayesian weight that minimises Brier score on a VALIDATION set.

        Must be called with data held out from both model training AND calibrator
        fitting — i.e. the middle split of a train / val / test partition.
        """
        from scipy.optimize import minimize_scalar

        df = val_features.dropna(subset=["outcome"])
        if len(df) < 20:
            logger.warning("Too few validation samples for weight optimisation — using 0.5")
            return

        lgbm_probs = self.lgbm.predict_proba(df)

        bay_rows = df[["home_team", "away_team"]].copy()
        bay_rows["neutral"] = (
            df["is_neutral"].astype(bool) if "is_neutral" in df.columns else False
        )
        bay_probs = np.array(
            [[p.p_home, p.p_draw, p.p_away]
             for p in self.bayesian.predict_batch(bay_rows)]
        )

        y_true = np.array(
            [[1, 0, 0] if o == "H" else ([0, 1, 0] if o == "D" else [0, 0, 1])
             for o in df["outcome"]],
            dtype=float,
        )

        def brier(w: float) -> float:
            blended = w * bay_probs + (1 - w) * lgbm_probs
            return float(np.mean(np.sum((blended - y_true) ** 2, axis=1)))

        result = minimize_scalar(brier, bounds=(0.1, 0.9), method="bounded")
        self.bayesian_weight = float(result.x)
        best_brier = brier(self.bayesian_weight)
        logger.info(
            f"Ensemble weights: Bayesian={self.bayesian_weight:.3f}, "
            f"LightGBM={1 - self.bayesian_weight:.3f}  "
            f"(Brier on val: {best_brier:.4f})"
        )

    # ------------------------------------------------------------------
    # Calibration fitting
    # ------------------------------------------------------------------

    def fit_calibrator(self, val_features: pd.DataFrame) -> None:
        """Fit the isotonic calibrator on ensemble predictions vs actual outcomes.

        Must use the SAME validation split as fit_weights — never the test set.
        """
        df = val_features.dropna(subset=["outcome"])
        if len(df) < 30:
            logger.warning(
                f"Only {len(df)} val samples — skipping calibration (need ≥30)"
            )
            return

        probs = self._blend_batch(df)
        self.calibrator = IsotonicCalibrator()
        self.calibrator.fit(probs, df["outcome"].tolist())

    def _blend_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return blended (n, 3) probability array."""
        lgbm_probs = self.lgbm.predict_proba(features_df)

        bay_rows = features_df[["home_team", "away_team"]].copy()
        if "is_neutral" in features_df.columns:
            bay_rows["neutral"] = features_df["is_neutral"].astype(bool)
        else:
            bay_rows["neutral"] = False

        bay_probs = np.array(
            [[p.p_home, p.p_draw, p.p_away]
             for p in self.bayesian.predict_batch(bay_rows)]
        )

        w = self.bayesian_weight
        blended = w * bay_probs + (1 - w) * lgbm_probs

        if self.calibrator is not None and self.calibrator.is_fitted:
            blended = self.calibrator.calibrate(blended)

        return blended

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_match(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> MatchPrediction:
        bay = self.bayesian.predict_match(home, away, neutral)
        lgb = self.lgbm.predict_match(home, away, neutral)

        # 2-model base blend (Bayesian + LightGBM)
        w = self.bayesian_weight
        p_home = w * bay.p_home + (1 - w) * lgb.p_home
        p_draw = w * bay.p_draw + (1 - w) * lgb.p_draw
        p_away = w * bay.p_away + (1 - w) * lgb.p_away

        # Optional 3rd component: market odds (only if fetcher has data for this match)
        mkt = self.market.predict_match(home, away, neutral) if self.market is not None else None
        if mkt is not None:
            mw = self.market_weight
            p_home = (1 - mw) * p_home + mw * mkt.p_home
            p_draw = (1 - mw) * p_draw + mw * mkt.p_draw
            p_away = (1 - mw) * p_away + mw * mkt.p_away

        if self.calibrator is not None and self.calibrator.is_fitted:
            p_home, p_draw, p_away = self.calibrator.calibrate_single(p_home, p_draw, p_away)

        return MatchPrediction(
            home_team=home,
            away_team=away,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            exp_home_goals=bay.exp_home_goals,
            exp_away_goals=bay.exp_away_goals,
            score_matrix=bay.score_matrix,
        )

    # ------------------------------------------------------------------
    # Holdout evaluation
    # ------------------------------------------------------------------

    def evaluate_holdout(self, test_features: pd.DataFrame) -> dict[str, float]:
        """Evaluate on a TEST set — must be separate from val set used for fitting."""
        from sklearn.metrics import log_loss

        df = test_features.dropna(subset=["outcome"])
        if df.empty:
            return {"brier": float("nan"), "log_loss": float("nan"), "accuracy": float("nan")}

        probs = self._blend_batch(df)
        actuals = df["outcome"].tolist()
        y_true = np.array(
            [[1, 0, 0] if o == "H" else ([0, 1, 0] if o == "D" else [0, 0, 1])
             for o in actuals],
            dtype=float,
        )
        brier = float(np.mean(np.sum((probs - y_true) ** 2, axis=1)))
        ll = float(log_loss(actuals, probs[:, [2, 1, 0]], labels=["A", "D", "H"]))
        predicted = [["H", "D", "A"][int(np.argmax(p))] for p in probs]
        accuracy = float(np.mean([p == a for p, a in zip(predicted, actuals)]))
        return {"brier": brier, "log_loss": ll, "accuracy": accuracy}
