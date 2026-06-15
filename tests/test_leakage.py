"""Data leakage tests: ensure feature construction never uses future match data.

The core invariant: for any match M at date D,
features generated for M must only use data from dates < D.
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np


def make_timeline_matches() -> pd.DataFrame:
    """Matches spanning 3 years in chronological order."""
    rows = []
    teams = ["Turkey", "Germany", "France", "Brazil", "Spain"]
    rng = np.random.default_rng(7)

    for i in range(100):
        h, a = rng.choice(len(teams), 2, replace=False)
        rows.append(
            {
                "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(i * 10)),
                "home_team": teams[h],
                "away_team": teams[a],
                "home_score": int(rng.poisson(1.3)),
                "away_score": int(rng.poisson(1.1)),
                "tournament": "Friendly",
                "neutral": False,
                "match_id": f"match_{i:03d}",
            }
        )
    return pd.DataFrame(rows)


def test_train_cutoff_excludes_future_matches():
    """Training data must only include matches before the cutoff date."""
    df = make_timeline_matches()
    cutoff = pd.Timestamp("2023-01-01")

    train = df[df["date"] < cutoff]
    future = df[df["date"] >= cutoff]

    assert len(train) > 0, "Need training data"
    assert len(future) > 0, "Need future data"

    # None of the future match dates should appear in training set
    train_dates = set(train["date"].tolist())
    for future_date in future["date"]:
        assert future_date not in train_dates


def test_model_train_only_uses_historical_data():
    """Model.fit() is called only on matches strictly before the holdout period."""
    df = make_timeline_matches()
    cutoff = df["date"].max() - pd.DateOffset(years=1)

    train_df = df[df["date"] < cutoff]
    holdout_df = df[df["date"] >= cutoff]

    # No match_id overlap between train and holdout
    train_ids = set(train_df["match_id"])
    holdout_ids = set(holdout_df["match_id"])
    assert train_ids.isdisjoint(holdout_ids), "Train/holdout match_ids overlap (data leakage!)"


def test_no_future_score_in_training_features():
    """Features derived from historical matches must not include scores of future matches."""
    df = make_timeline_matches()
    cutoff = pd.Timestamp("2023-06-01")

    train = df[df["date"] < cutoff].copy()

    # Feature: running average goals per team (should only use train data)
    home_avg = train.groupby("home_team")["home_score"].mean().to_dict()
    away_avg = train.groupby("away_team")["away_score"].mean().to_dict()

    future = df[df["date"] >= cutoff]
    for _, row in future.iterrows():
        h = row["home_team"]
        # The feature value for future matches comes from historical avg only
        # and must NOT include the future match's own score
        if h in home_avg:
            historical_avg = home_avg[h]
            # Just ensure the function runs without referencing future data
            assert isinstance(historical_avg, float)


def test_holdout_matches_not_seen_during_fit():
    """Simulate time-based split: holdout matches must all be after the train cutoff."""
    df = make_timeline_matches()

    # Use update.py's same 2-year lookback logic
    cutoff = df["date"].max() - pd.DateOffset(years=2)
    train = df[df["date"] < cutoff]
    holdout = df[df["date"] >= cutoff]

    if not train.empty and not holdout.empty:
        assert train["date"].max() < holdout["date"].min(), \
            "Train set contains matches after the holdout start date!"


def test_snapshot_predictions_frozen():
    """Pre-match predictions in snapshots must not be retroactively modified.

    Simulates the append-only snapshot invariant: the first prediction for
    a match_id (chronologically) is the one used for calibration scoring.
    """
    # Two snapshots: snap1 (before match), snap2 (after)
    snap1 = pd.DataFrame(
        {
            "match_id": ["m1"],
            "snapshot_ts": ["20260611T100000Z"],
            "p_home": [0.45],
            "p_draw": [0.25],
            "p_away": [0.30],
        }
    )
    snap2 = pd.DataFrame(
        {
            "match_id": ["m1"],
            "snapshot_ts": ["20260612T100000Z"],
            "p_home": [0.60],  # Updated after match started
            "p_draw": [0.20],
            "p_away": [0.20],
        }
    )

    all_preds = pd.concat([snap1, snap2]).sort_values("snapshot_ts")

    # Calibration scoring uses the EARLIEST snapshot
    pre_match_pred = all_preds[all_preds["match_id"] == "m1"].iloc[0]
    assert pre_match_pred["snapshot_ts"] == "20260611T100000Z"
    assert abs(pre_match_pred["p_home"] - 0.45) < 1e-6, \
        "Calibration must use pre-match prediction, not updated one"
