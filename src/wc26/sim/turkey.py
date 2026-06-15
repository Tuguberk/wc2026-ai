"""Turkey-specific simulation and probability tracking.

Turkey's WC2026 group: D (Turkey, Australia, Paraguay, United States)
Turkey's WC history: 1954, 2002 (3rd place). Did not qualify in 2022.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wc26.models.base import BaseModel
from wc26.sim.group_sim import GROUP_D, simulate_group

logger = logging.getLogger(__name__)

TURKEY = "Turkey"


def compute_turkey_probs(
    model: BaseModel,
    already_played: pd.DataFrame | None = None,
    n_simulations: int = 10_000,
) -> dict[str, float]:
    """
    Compute Turkey's probabilities in Group D simulation.

    Returns dict with keys:
        p_advance (top 2), p_1st, p_2nd, p_3rd, p_4th,
        avg_points, avg_gd
    """
    group_results = simulate_group(
        teams=GROUP_D,
        model=model,
        already_played=already_played,
        n_simulations=n_simulations,
    )

    turkey_row = group_results[group_results["team"] == TURKEY]
    if turkey_row.empty:
        # Turkey not in results (team name mismatch?) — try partial match
        candidates = group_results[group_results["team"].str.contains("Turk", case=False)]
        if not candidates.empty:
            turkey_row = candidates.iloc[:1]
        else:
            logger.warning("Turkey not found in group simulation results")
            return {
                "p_advance": float("nan"),
                "p_1st": float("nan"),
                "p_2nd": float("nan"),
                "p_3rd": float("nan"),
                "p_4th": float("nan"),
                "avg_points": float("nan"),
                "avg_gd": float("nan"),
            }

    row = turkey_row.iloc[0]
    return {
        "p_advance": float(row.get("p_advance", 0.0)),
        "p_1st": float(row.get("p_1st", 0.0)),
        "p_2nd": float(row.get("p_2nd", 0.0)),
        "p_3rd": float(row.get("p_3rd", 0.0)),
        "p_4th": float(row.get("p_4th", 0.0)),
        "avg_points": float(row.get("avg_points", 0.0)),
        "avg_gd": float(row.get("avg_gd", 0.0)),
    }


def turkey_next_match(fixtures: pd.DataFrame) -> pd.Series | None:
    """Return Turkey's next unplayed match from the fixtures DataFrame."""
    upcoming = fixtures[
        (fixtures["status"] != "FINISHED")
        & ((fixtures["home_team"].str.contains("Turk", case=False))
           | (fixtures["away_team"].str.contains("Turk", case=False)))
    ]
    if upcoming.empty:
        return None
    return upcoming.sort_values("date").iloc[0]


def turkey_match_card(
    model: BaseModel,
    next_match: pd.Series,
) -> dict:
    """Generate the pre-match prediction card for Turkey's next fixture."""
    home = next_match["home_team"]
    away = next_match["away_team"]
    neutral = bool(next_match.get("neutral", False))

    pred = model.predict_match(home, away, neutral)

    # Most likely scores from score matrix
    top_scores = []
    if pred.score_matrix is not None:
        matrix = pred.score_matrix
        n = matrix.shape[0]
        flat_idx = np.argsort(matrix.flatten())[::-1][:5]
        for idx in flat_idx:
            h = idx // n
            a = idx % n
            top_scores.append(
                {"score": f"{h}-{a}", "probability": float(matrix[h, a])}
            )

    return {
        "home_team": home,
        "away_team": away,
        "date": str(next_match.get("date", "")),
        "p_home": pred.p_home,
        "p_draw": pred.p_draw,
        "p_away": pred.p_away,
        "exp_home_goals": pred.exp_home_goals,
        "exp_away_goals": pred.exp_away_goals,
        "top_scores": top_scores,
    }


def turkey_path_probabilities(
    model: BaseModel,
    group_results: pd.DataFrame | None = None,
    n_simulations: int = 10_000,
) -> dict[str, float]:
    """
    Estimate Turkey's probabilities at each knockout stage.

    NOTE: Full bracket simulation requires knowing all 48 teams and the
    WC2026 bracket structure. This function returns group-stage probs and
    placeholder knockout probs until bracket data is available.
    """
    advance_prob = 0.0
    if group_results is not None and not group_results.empty:
        turkey_row = group_results[group_results["team"] == TURKEY]
        if not turkey_row.empty:
            advance_prob = float(turkey_row.iloc[0].get("p_advance", 0.0))

    # Naive power-law decay for later rounds (placeholder for bracket sim)
    # Will be replaced by full bracket simulation in Phase 2
    decay = 0.45  # ~ average win prob per knockout round for a team that advanced
    return {
        "group_advance": advance_prob,
        "round_of_32": advance_prob * (decay**1),
        "quarter_final": advance_prob * (decay**2),
        "semi_final": advance_prob * (decay**3),
        "final": advance_prob * (decay**4),
        "champion": advance_prob * (decay**5),
    }
