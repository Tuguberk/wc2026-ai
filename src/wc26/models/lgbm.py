"""LightGBM multiclass model for football match outcome prediction.

Target: 3-class (H = home win, D = draw, A = away win)
Features: Elo ratings, recent form, rest days, match context
Training: strictly time-based — validation set is the chronologically last 20%
          of the training window, ensuring zero future leakage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wc26.features.pipeline import FEATURE_COLS
from wc26.models.base import BaseModel, MatchPrediction

logger = logging.getLogger(__name__)

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
IDX_TO_LABEL = ["H", "D", "A"]


class LGBMModel(BaseModel):
    """LightGBM multiclass football prediction model."""

    name = "lgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.04,
        num_leaves: int = 31,
        min_child_samples: int = 25,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self._fitted = False
        self._model = None
        self._feature_cols: list[str] = []
        # Lookup table built from prediction-context features (upcoming matches)
        self._pred_lookup: dict[tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> None:
        """Train on the output of build_training_features().

        Uses a strict time-based split: last 20% of rows (chronologically)
        form the early-stopping validation set.
        """
        import lightgbm as lgb

        available = [c for c in FEATURE_COLS if c in features_df.columns]
        if not available:
            raise ValueError("No usable feature columns found in features_df")
        self._feature_cols = available

        df = features_df.dropna(subset=["outcome"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            raise ValueError("features_df has no rows with a valid outcome")

        X = df[self._feature_cols].to_numpy(dtype=float)
        y = df["outcome"].map(LABEL_MAP).to_numpy(dtype=int)

        split = max(1, int(len(df) * 0.8))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        train_ds = lgb.Dataset(X_tr, label=y_tr)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
        }

        self._model = lgb.train(
            params,
            train_ds,
            num_boost_round=self.n_estimators,
            valid_sets=[val_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        self._fitted = True
        logger.info(
            f"LightGBM fitted: {self._model.num_trees()} trees, "
            f"{len(self._feature_cols)} features, {split} train / {len(df) - split} val rows"
        )

    # ------------------------------------------------------------------
    # Prediction context (must be called before predict_match in sim loops)
    # ------------------------------------------------------------------

    def set_prediction_features(self, pred_features_df: pd.DataFrame) -> None:
        """Pre-compute and cache probabilities for all upcoming matches.

        Call this once after fit() with the output of
        build_prediction_features(), before running simulations.
        """
        if not self._fitted or self._model is None:
            return
        avail = [c for c in self._feature_cols if c in pred_features_df.columns]
        X = pred_features_df[avail].to_numpy(dtype=float)
        probs = self._model.predict(X)  # (n, 3)
        self._pred_lookup = {}
        for i, row in pred_features_df.iterrows():
            key = (row["home_team"], row["away_team"])
            self._pred_lookup[key] = probs[i]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_match(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> MatchPrediction:
        if not self._fitted:
            raise RuntimeError("LGBMModel not fitted")

        probs = self._pred_lookup.get((home, away))
        if probs is None:
            # Symmetric lookup: if stored as away vs home (shouldn't happen, but guard)
            swapped = self._pred_lookup.get((away, home))
            if swapped is not None:
                probs = swapped[[2, 1, 0]]  # flip H↔A
            else:
                logger.debug(f"LGBMModel: no cached features for {home} vs {away}")
                return MatchPrediction(
                    home_team=home, away_team=away,
                    p_home=1 / 3, p_draw=1 / 3, p_away=1 / 3,
                )

        return MatchPrediction(
            home_team=home,
            away_team=away,
            p_home=float(probs[0]),
            p_draw=float(probs[1]),
            p_away=float(probs[2]),
        )

    # ------------------------------------------------------------------
    # Batch inference (used by ensemble weight optimisation)
    # ------------------------------------------------------------------

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) probability array for a feature DataFrame."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Not fitted")
        avail = [c for c in self._feature_cols if c in features_df.columns]
        X = features_df[avail].to_numpy(dtype=float)
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return LightGBM feature importances (gain) as a DataFrame."""
        if not self._fitted or self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        imp = self._model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Holdout evaluation
    # ------------------------------------------------------------------

    def evaluate_holdout(self, features_df: pd.DataFrame) -> dict[str, float]:
        """Brier + log-loss + accuracy on a feature DataFrame with 'outcome' column."""
        from sklearn.metrics import log_loss

        df = features_df.dropna(subset=["outcome"])
        if df.empty:
            return {"brier": float("nan"), "log_loss": float("nan"), "accuracy": float("nan")}

        probs = self.predict_proba(df)
        actuals = df["outcome"].tolist()
        y_true = np.array(
            [[1, 0, 0] if o == "H" else ([0, 1, 0] if o == "D" else [0, 0, 1]) for o in actuals]
        )
        brier = float(np.mean(np.sum((probs - y_true) ** 2, axis=1)))
        ll = float(log_loss(actuals, probs[:, [2, 1, 0]], labels=["A", "D", "H"]))
        predicted = [IDX_TO_LABEL[int(np.argmax(p))] for p in probs]
        accuracy = float(np.mean([p == a for p, a in zip(predicted, actuals)]))
        return {"brier": brier, "log_loss": ll, "accuracy": accuracy}
