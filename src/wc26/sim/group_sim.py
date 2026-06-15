"""Monte Carlo group stage simulation.

Simulates remaining group matches using model probabilities,
computes each team's finishing position distribution.

WC2026 Group format:
- 48 teams, 12 groups of 4
- Top 2 + 8 best third-place teams advance (32 teams total)
- Points: W=3, D=1, L=0 → tiebreakers: GD, GF, random
- Each pair plays ONCE (6 matches per group of 4)
- All WC group stage matches are at neutral venues (neutral=True)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from wc26.models.base import BaseModel

logger = logging.getLogger(__name__)

# WC2026 Group D (Turkey's group)
GROUP_D = ["Turkey", "Australia", "Paraguay", "United States"]

# All WC2026 groups (populated from fixture data as available)
WC2026_GROUPS: dict[str, list[str]] = {
    "A": [], "B": [], "C": [],
    "D": GROUP_D,
    "E": [], "F": [], "G": [], "H": [], "I": [], "J": [], "K": [], "L": [],
}


def _simulate_match(home: str, away: str, model: BaseModel) -> tuple[int, int]:
    """Sample a match result from the model's score matrix or probability vector."""
    pred = model.predict_match(home, away, neutral=True)
    matrix = pred.score_matrix
    if matrix is not None:
        flat = matrix.flatten()
        flat = flat / flat.sum()
        idx = int(np.random.choice(len(flat), p=flat))
        return idx // matrix.shape[1], idx % matrix.shape[1]
    # Fallback: sample outcome, assign canonical scores
    outcome = np.random.choice(["H", "D", "A"], p=[pred.p_home, pred.p_draw, pred.p_away])
    if outcome == "H":
        return 2, 0
    elif outcome == "D":
        return 1, 1
    else:
        return 0, 2


def simulate_group(
    teams: list[str],
    model: BaseModel,
    already_played: pd.DataFrame | None = None,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """Monte Carlo group stage simulation.

    Parameters
    ----------
    teams          : list of team names in the group
    model          : fitted prediction model
    already_played : DataFrame of finished matches with columns
                     [home_team, away_team, home_score, away_score]
    n_simulations  : number of Monte Carlo iterations

    Returns
    -------
    DataFrame with columns [team, p_1st, p_2nd, p_3rd, p_4th, p_advance, avg_points, avg_gd]
    """
    n = len(teams)

    # Each pair plays ONCE — generate unordered pairs only
    all_pairs: list[tuple[str, str]] = [
        (teams[i], teams[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]

    # Fixed results from already-played matches
    played_set: set[tuple[str, str]] = set()
    fixed_results: dict[tuple[str, str], tuple[int, int]] = {}

    if already_played is not None and not already_played.empty:
        for _, row in already_played.iterrows():
            key = (row["home_team"], row["away_team"])
            rev = (row["away_team"], row["home_team"])
            played_set.add(key)
            played_set.add(rev)          # block both directions
            fixed_results[key] = (int(row["home_score"]), int(row["away_score"]))

    # Unplayed pairs — check both directions
    remaining_pairs = [
        p for p in all_pairs
        if p not in played_set and (p[1], p[0]) not in played_set
    ]

    # Position + accumulation counters
    position_counts = {team: [0] * n for team in teams}
    total_points = {team: 0.0 for team in teams}
    total_gd = {team: 0.0 for team in teams}

    for _ in range(n_simulations):
        pts = {t: 0 for t in teams}
        gf  = {t: 0 for t in teams}
        ga  = {t: 0 for t in teams}

        # Apply fixed results
        for (home, away), (hs, as_) in fixed_results.items():
            _apply_result(pts, gf, ga, home, away, hs, as_)

        # Simulate remaining matches
        for home, away in remaining_pairs:
            hs, as_ = _simulate_match(home, away, model)
            _apply_result(pts, gf, ga, home, away, hs, as_)

        # Rank teams
        ranking = _rank_teams(teams, pts, gf, ga)

        for pos, team in enumerate(ranking):
            position_counts[team][pos] += 1
            total_points[team] += pts[team]
            total_gd[team] += gf[team] - ga[team]

    rows = []
    for team in teams:
        cnt = position_counts[team]
        rows.append(
            {
                "team": team,
                "p_1st":     cnt[0] / n_simulations,
                "p_2nd":     cnt[1] / n_simulations,
                "p_3rd":     cnt[2] / n_simulations if n >= 3 else 0.0,
                "p_4th":     cnt[3] / n_simulations if n >= 4 else 0.0,
                "p_advance": (cnt[0] + cnt[1]) / n_simulations,
                "avg_points": total_points[team] / n_simulations,
                "avg_gd":     total_gd[team] / n_simulations,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("avg_points", ascending=False)
        .reset_index(drop=True)
    )


def _apply_result(
    pts: dict, gf: dict, ga: dict,
    home: str, away: str, home_goals: int, away_goals: int,
) -> None:
    gf[home] += home_goals
    ga[home] += away_goals
    gf[away] += away_goals
    ga[away] += home_goals
    if home_goals > away_goals:
        pts[home] += 3
    elif home_goals == away_goals:
        pts[home] += 1
        pts[away] += 1
    else:
        pts[away] += 3


def _rank_teams(
    teams: list[str],
    pts: dict, gf: dict, ga: dict,
) -> list[str]:
    """Rank by: 1) points, 2) GD, 3) GF, 4) random tiebreaker."""
    return sorted(
        teams,
        key=lambda t: (pts[t], gf[t] - ga[t], gf[t], np.random.random()),
        reverse=True,
    )
