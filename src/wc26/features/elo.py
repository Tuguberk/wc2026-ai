"""Elo rating system for international football teams.

Standard Elo with:
- K-factor weighted by match importance (World Cup = 60, friendly = 20)
- Home advantage: +100 Elo points to home team's effective rating on neutral=False
- Starting rating: 1500 for all teams (no prior information assumed)
- Ratings computed chronologically; no future data leaks into any match's pre-match rating
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# fmt: off
TOURNAMENT_K: dict[str, float] = {
    "FIFA World Cup":                60.0,
    "Copa América":                  50.0,
    "UEFA Euro":                     50.0,
    "Africa Cup of Nations":         50.0,
    "AFC Asian Cup":                 50.0,
    "CONCACAF Gold Cup":             45.0,
    "Confederations Cup":            45.0,
    "UEFA Nations League":           40.0,
    "FIFA World Cup qualification":  40.0,
    "UEFA Euro qualification":       35.0,
    "CONMEBOL":                      35.0,
    "AFC":                           30.0,
    "CAF":                           30.0,
}
DEFAULT_K: float = 20.0  # friendlies and unknown competitions
HOME_ADV_ELO: float = 100.0  # added to home team's effective rating on non-neutral venues
INITIAL_RATING: float = 1500.0
# fmt: on


def _k(tournament: str) -> float:
    t = tournament.lower()
    for key, k in TOURNAMENT_K.items():
        if key.lower() in t:
            return k
    return DEFAULT_K


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def compute_elo_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre- and post-match Elo ratings for every row in df.

    Processes matches in chronological order. The pre-match rating for
    match M uses only matches strictly before M — no future leakage.

    Returns
    -------
    DataFrame with index aligned to df:
        home_elo_pre, away_elo_pre, home_elo_post, away_elo_post
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}

    home_pre, away_pre, home_post, away_post = [], [], [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", ""))

        r_h = ratings.get(home, INITIAL_RATING)
        r_a = ratings.get(away, INITIAL_RATING)

        home_pre.append(r_h)
        away_pre.append(r_a)

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

        # Add home advantage to effective rating only on non-neutral venues
        r_h_eff = r_h if neutral else r_h + HOME_ADV_ELO
        k = _k(tournament)
        delta = k * (actual - _expected(r_h_eff, r_a))

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

        home_post.append(ratings[home])
        away_post.append(ratings[away])

    return pd.DataFrame(
        {
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "home_elo_post": home_post,
            "away_elo_post": away_post,
        },
        index=df.index,
    )


def current_ratings(df: pd.DataFrame) -> dict[str, float]:
    """Return each team's Elo as of the last match in df (sorted by date)."""
    df = df.sort_values("date").reset_index(drop=True)
    elo = compute_elo_series(df)
    ratings: dict[str, float] = {}
    for i, row in df.iterrows():
        ratings[row["home_team"]] = float(elo.loc[i, "home_elo_post"])
        ratings[row["away_team"]] = float(elo.loc[i, "away_elo_post"])
    return ratings
