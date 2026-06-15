"""Hierarchical Bayesian Poisson model for football match prediction.

Model:
  attack[t] ~ Normal(mu_att, sigma_att)   — team attack strength (latent)
  defense[t] ~ Normal(mu_def, sigma_def)  — team defense strength (latent)
  home_advantage ~ Normal(0.3, 0.2)       — global home boost

  lambda_home = exp(intercept + home_advantage + attack[home] - defense[away])
  lambda_away = exp(intercept + attack[away] - defense[home])

  home_goals ~ Poisson(lambda_home)
  away_goals ~ Poisson(lambda_away)

Time-weighting: exponential decay with half-life ~2 years applied as
observation weights in the likelihood (via pm.Potential).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from wc26.models.base import BaseModel, MatchPrediction

logger = logging.getLogger(__name__)

MAX_GOALS = 8
HALF_LIFE_DAYS = 365 * 2  # 2-year half-life for time decay


def _time_weight(date: pd.Timestamp, reference: pd.Timestamp, half_life: float = HALF_LIFE_DAYS) -> float:
    """Exponential time-decay weight: 1.0 at reference, 0.5 at half_life ago."""
    days_ago = (reference - date).days
    return float(np.exp(-np.log(2) * days_ago / half_life))


class BayesianPoissonModel(BaseModel):
    """Hierarchical Bayesian Poisson model (PyMC)."""

    name = "bayesian_poisson"

    def __init__(
        self,
        draws: int = 1000,
        tune: int = 500,
        chains: int = 2,
        min_matches: int = 10,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.min_matches = min_matches
        self._fitted = False
        self._trace = None
        self._teams: list[str] = []
        self._team_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Fit model on historical international match data."""
        import pymc as pm  # noqa: PLC0415
        import pytensor.tensor as pt  # noqa: PLC0415

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # Filter to teams with enough data
        team_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        eligible = team_counts[team_counts >= self.min_matches].index.tolist()
        df = df[df["home_team"].isin(eligible) & df["away_team"].isin(eligible)]

        self._teams = sorted(eligible)
        self._team_idx = {t: i for i, t in enumerate(self._teams)}
        n_teams = len(self._teams)

        if n_teams < 2:
            raise ValueError(f"Need ≥2 eligible teams, got {n_teams}")

        home_idx = df["home_team"].map(self._team_idx).values
        away_idx = df["away_team"].map(self._team_idx).values
        home_goals = df["home_score"].values
        away_goals = df["away_score"].values
        neutral = df["neutral"].astype(float).values if "neutral" in df.columns else np.zeros(len(df))

        # Time weights
        ref = df["date"].max()
        weights = df["date"].apply(lambda d: _time_weight(d, ref)).values.astype(float)

        logger.info(
            f"Fitting Bayesian model: {len(df)} matches, {n_teams} teams, "
            f"draws={self.draws}, tune={self.tune}, chains={self.chains}"
        )

        with pm.Model() as model:
            # Hyperpriors
            mu_att = pm.Normal("mu_att", mu=0, sigma=1)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1)
            mu_def = pm.Normal("mu_def", mu=0, sigma=1)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1)

            # Team latent strengths
            attack_offset = pm.Normal("attack_offset", mu=0, sigma=1, shape=n_teams)
            attack = pm.Deterministic("attack", mu_att + attack_offset * sigma_att)

            defense_offset = pm.Normal("defense_offset", mu=0, sigma=1, shape=n_teams)
            defense = pm.Deterministic("defense", mu_def + defense_offset * sigma_def)

            intercept = pm.Normal("intercept", mu=0, sigma=1)
            home_adv = pm.Normal("home_advantage", mu=0.3, sigma=0.2)

            # Goals expected
            log_lam_home = (
                intercept
                + home_adv * (1 - neutral)
                + attack[home_idx]
                - defense[away_idx]
            )
            log_lam_away = intercept + attack[away_idx] - defense[home_idx]

            lam_home = pm.Deterministic("lam_home", pm.math.exp(log_lam_home))
            lam_away = pm.Deterministic("lam_away", pm.math.exp(log_lam_away))

            # Weighted Poisson likelihood via pm.Potential
            home_logp = pm.logp(pm.Poisson.dist(mu=lam_home), home_goals)
            away_logp = pm.logp(pm.Poisson.dist(mu=lam_away), away_goals)
            pm.Potential("weighted_ll", (home_logp + away_logp) * weights)

            self._trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                progressbar=True,
                return_inferencedata=True,
                target_accept=0.9,
            )

        self._model = model
        self._fitted = True
        self._build_params_cache()
        logger.info("Bayesian model fit complete")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_match(
        self,
        home: str,
        away: str,
        neutral: bool = False,
    ) -> MatchPrediction:
        """Predict 1X2 probabilities and score matrix for a match."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        lam_h, lam_a = self._expected_goals(home, away, neutral)
        score_matrix = self._score_matrix(lam_h, lam_a)

        p_home = float(np.sum(np.tril(score_matrix, -1)))
        p_away = float(np.sum(np.triu(score_matrix, 1)))
        p_draw = float(np.sum(np.diag(score_matrix)))

        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        return MatchPrediction(
            home_team=home,
            away_team=away,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            exp_home_goals=float(lam_h),
            exp_away_goals=float(lam_a),
            score_matrix=score_matrix,
        )

    def _build_params_cache(self) -> None:
        """Pre-compute posterior means for all teams once after fitting.

        Avoids re-reading the full trace on every predict_match call,
        which was the bottleneck during Monte Carlo simulation (120k calls).
        """
        trace = self._trace.posterior
        # Flatten chains×draws into a single axis for mean computation
        attack_means = trace["attack"].values.reshape(-1, len(self._teams)).mean(axis=0)
        defense_means = trace["defense"].values.reshape(-1, len(self._teams)).mean(axis=0)
        prior_att = float(trace["mu_att"].values.mean())
        prior_def = float(trace["mu_def"].values.mean())

        self._cache = {
            "intercept": float(trace["intercept"].values.mean()),
            "home_adv": float(trace["home_advantage"].values.mean()),
            "attack": {t: float(attack_means[i]) for t, i in self._team_idx.items()},
            "defense": {t: float(defense_means[i]) for t, i in self._team_idx.items()},
            "prior_att": prior_att,
            "prior_def": prior_def,
        }

    def _expected_goals(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float]:
        """Return posterior mean expected goals for each team (uses cache)."""
        cache = self._cache
        intercept = cache["intercept"]
        home_adv = cache["home_adv"]

        def team_params(name: str) -> tuple[float, float]:
            if name in cache["attack"]:
                return cache["attack"][name], cache["defense"][name]
            logger.debug(f"Unknown team '{name}', using prior mean")
            return cache["prior_att"], cache["prior_def"]

        h_att, h_def = team_params(home)
        a_att, a_def = team_params(away)

        ha = 0.0 if neutral else home_adv
        lam_h = np.exp(intercept + ha + h_att - a_def)
        lam_a = np.exp(intercept + a_att - h_def)
        return float(lam_h), float(lam_a)

    def _score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        """Joint Poisson score probability matrix."""
        from scipy.stats import poisson  # noqa: PLC0415

        goals = np.arange(MAX_GOALS + 1)
        p_h = poisson.pmf(goals, lam_h)
        p_a = poisson.pmf(goals, lam_a)
        return np.outer(p_h, p_a)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_holdout(self, holdout_df: pd.DataFrame) -> dict[str, float]:
        """Compute Brier score, log-loss, and accuracy on a holdout set.

        Only evaluates matches where both teams were seen during training;
        unknown-team fallbacks would distort the metrics.
        """
        from sklearn.metrics import log_loss  # noqa: PLC0415

        known = set(self._team_idx.keys())
        preds = []
        actuals = []

        for _, row in holdout_df.iterrows():
            home, away = row["home_team"], row["away_team"]
            # Skip if either team wasn't in the training set
            if home not in known or away not in known:
                continue
            try:
                pred = self.predict_match(home, away)
                preds.append([pred.p_home, pred.p_draw, pred.p_away])
                h, a = int(row["home_score"]), int(row["away_score"])
                actuals.append("H" if h > a else ("D" if h == a else "A"))
            except Exception:
                continue

        if not preds:
            return {"brier": float("nan"), "log_loss": float("nan"), "accuracy": float("nan")}

        y_pred = np.array(preds)
        y_true_enc = np.array(
            [[1, 0, 0] if o == "H" else ([0, 1, 0] if o == "D" else [0, 0, 1])
             for o in actuals]
        )

        brier = float(np.mean(np.sum((y_pred - y_true_enc) ** 2, axis=1)))
        # sklearn log_loss requires y_prob columns in lexicographic label order.
        # Our predictions are [p_home, p_draw, p_away] = [H, D, A].
        # Lexicographic order is ["A", "D", "H"], so reorder columns: [2, 1, 0].
        ll = float(log_loss(actuals, y_pred[:, [2, 1, 0]], labels=["A", "D", "H"]))

        predicted = [["H", "D", "A"][int(np.argmax(p))] for p in preds]
        accuracy = float(np.mean([p == a for p, a in zip(predicted, actuals)]))

        logger.info(f"Holdout evaluation on {len(preds)} known-team matches")
        return {"brier": brier, "log_loss": ll, "accuracy": accuracy}
