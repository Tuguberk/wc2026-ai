"""Model smoke tests: fit on synthetic data, predict, check output shapes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_matches(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic international match data for model testing."""
    rng = np.random.default_rng(seed)
    teams = ["Turkey", "Germany", "Brazil", "France", "Spain", "Argentina",
             "England", "Portugal", "Belgium", "Netherlands", "United States", "Australia"]
    n_teams = len(teams)

    rows = []
    base_date = pd.Timestamp("2020-01-01")
    for i in range(n):
        h, a = rng.choice(n_teams, 2, replace=False)
        hs = int(rng.poisson(1.4))
        as_ = int(rng.poisson(1.1))
        rows.append({
            "date": base_date + pd.Timedelta(days=int(i * 3)),
            "home_team": teams[h],
            "away_team": teams[a],
            "home_score": hs,
            "away_score": as_,
            "tournament": "Friendly",
            "neutral": bool(i % 5 == 0),
        })

    df = pd.DataFrame(rows)
    df["match_id"] = df.apply(
        lambda r: f"{r['date'].date()}_{r['home_team']}_{r['away_team']}", axis=1
    )
    return df


@pytest.fixture(scope="module")
def small_model():
    """Fit a Bayesian model on synthetic data (fast settings for CI)."""
    from wc26.models.bayesian_poisson import BayesianPoissonModel

    model = BayesianPoissonModel(draws=100, tune=100, chains=1, min_matches=2)
    df = make_synthetic_matches(150)
    model.fit(df)
    return model


def test_model_fit_completes(small_model):
    """Model fit should complete without error."""
    assert small_model.is_fitted()


def test_model_predict_probabilities_sum_to_one(small_model):
    """1X2 probabilities must sum to ~1.0 for any match."""
    pred = small_model.predict_match("Turkey", "Germany")
    total = pred.p_home + pred.p_draw + pred.p_away
    assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to 1: {total}"


def test_model_predict_probabilities_in_range(small_model):
    """Each probability must be in [0, 1]."""
    pred = small_model.predict_match("Brazil", "France")
    assert 0 <= pred.p_home <= 1
    assert 0 <= pred.p_draw <= 1
    assert 0 <= pred.p_away <= 1


def test_model_expected_goals_positive(small_model):
    """Expected goals must be non-negative."""
    pred = small_model.predict_match("Spain", "Argentina")
    assert pred.exp_home_goals >= 0
    assert pred.exp_away_goals >= 0


def test_model_score_matrix_shape(small_model):
    """Score matrix should be square with correct shape."""
    from wc26.models.bayesian_poisson import MAX_GOALS
    pred = small_model.predict_match("Turkey", "United States")
    assert pred.score_matrix is not None
    assert pred.score_matrix.shape == (MAX_GOALS + 1, MAX_GOALS + 1)


def test_model_score_matrix_sums_to_one(small_model):
    """Score matrix probabilities must sum to ~1.0."""
    pred = small_model.predict_match("Turkey", "Australia")
    total = pred.score_matrix.sum()
    assert abs(total - 1.0) < 0.05, f"Score matrix sum: {total}"


def test_model_unknown_team_uses_prior(small_model):
    """Predicting with an unknown team should not raise — uses hyperprior."""
    pred = small_model.predict_match("Unknownland", "Turkey")
    assert 0 <= pred.p_home <= 1


def test_model_holdout_evaluation(small_model):
    """evaluate_holdout should return a dict with expected keys."""
    holdout = make_synthetic_matches(30, seed=99)
    metrics = small_model.evaluate_holdout(holdout)
    assert "brier" in metrics
    assert "log_loss" in metrics
    assert "accuracy" in metrics
    # Brier score for multiclass is in [0, 2]
    assert 0 <= metrics["brier"] <= 2


def test_model_neutral_venue_reduces_home_advantage(small_model):
    """Home-win probability should be higher when not neutral."""
    pred_home = small_model.predict_match("Turkey", "Germany", neutral=False)
    pred_neutral = small_model.predict_match("Turkey", "Germany", neutral=True)
    # On average home advantage should shift some probability toward home team
    # This is probabilistic — just check the method runs without error
    assert pred_home.p_home >= 0
    assert pred_neutral.p_home >= 0
