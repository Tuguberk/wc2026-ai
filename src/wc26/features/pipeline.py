"""Full feature pipeline: Elo + form + FIFA ranking + match metadata → feature DataFrame.

Two entry points:
- build_training_features(df, ranking_lookup)  — for historical data (leakage-free, includes target)
- build_prediction_features(historical_df, upcoming_df, ranking_lookup)  — inference on upcoming matches

ranking_lookup is optional (pass None to skip FIFA rank features).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from wc26.features.elo import INITIAL_RATING, compute_elo_series, current_ratings
from wc26.features.form import _form_stats, _rest_days

logger = logging.getLogger(__name__)

# fmt: off
TOURNAMENT_IMPORTANCE: dict[str, float] = {
    "FIFA World Cup":                3.0,
    "Copa América":                  2.5,
    "UEFA Euro":                     2.5,
    "Africa Cup of Nations":         2.5,
    "AFC Asian Cup":                 2.5,
    "CONCACAF Gold Cup":             2.0,
    "UEFA Nations League":           2.0,
    "FIFA World Cup qualification":  2.0,
    "UEFA Euro qualification":       1.8,
    "Friendly":                      1.0,
}
# fmt: on

FEATURE_COLS = [
    # Elo
    "home_elo", "away_elo", "elo_diff",
    # Recent form
    "home_win_rate", "home_draw_rate", "home_gf", "home_ga", "home_gd", "home_rest",
    "away_win_rate", "away_draw_rate", "away_gf", "away_ga", "away_gd", "away_rest",
    # FIFA ranking
    "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
    "home_fifa_pts", "away_fifa_pts", "fifa_pts_diff",
    # Match context
    "is_neutral", "tournament_importance",
]


def _importance(tournament: str) -> float:
    t = str(tournament).lower()
    for key, val in TOURNAMENT_IMPORTANCE.items():
        if key.lower() in t:
            return val
    return 1.5


def _get_fifa_features(
    team: str,
    date: pd.Timestamp,
    ranking_lookup: dict | None,
) -> dict[str, float]:
    """Return FIFA rank/points for a team at a given date, or neutral fallback."""
    if ranking_lookup is None:
        return {"rank": 150.0, "pts": 0.0}
    from wc26.fetchers.fifa_ranking import fifa_rank_at
    rank, pts = fifa_rank_at(ranking_lookup, team, date)
    return {"rank": float(rank), "pts": float(pts)}


def build_training_features(
    df: pd.DataFrame,
    ranking_lookup: dict | None = None,
    form_n: int = 5,
) -> pd.DataFrame:
    """Build leakage-free feature matrix for historical matches.

    Returns a DataFrame with FEATURE_COLS + meta columns
    (match_id, date, home_team, away_team, outcome).
    """
    from wc26.features.form import compute_form_features

    df = df.sort_values("date").reset_index(drop=True)

    elo = compute_elo_series(df)
    form = compute_form_features(df, n=form_n)

    out = pd.DataFrame(index=df.index)
    out["home_elo"] = elo["home_elo_pre"]
    out["away_elo"] = elo["away_elo_pre"]
    out["elo_diff"] = elo["home_elo_pre"] - elo["away_elo_pre"]

    for col in form.columns:
        out[col] = form[col]

    # FIFA ranking (leakage-free: use ranking at match date)
    if ranking_lookup is not None:
        home_ranks, away_ranks = [], []
        home_pts_list, away_pts_list = [], []
        for _, row in df.iterrows():
            dt = pd.to_datetime(row["date"])
            h_r = _get_fifa_features(row["home_team"], dt, ranking_lookup)
            a_r = _get_fifa_features(row["away_team"], dt, ranking_lookup)
            home_ranks.append(h_r["rank"])
            away_ranks.append(a_r["rank"])
            home_pts_list.append(h_r["pts"])
            away_pts_list.append(a_r["pts"])
        out["home_fifa_rank"] = home_ranks
        out["away_fifa_rank"] = away_ranks
        out["fifa_rank_diff"] = np.array(home_ranks) - np.array(away_ranks)
        out["home_fifa_pts"] = home_pts_list
        out["away_fifa_pts"] = away_pts_list
        out["fifa_pts_diff"] = np.array(home_pts_list) - np.array(away_pts_list)
    else:
        for col in ["home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
                    "home_fifa_pts", "away_fifa_pts", "fifa_pts_diff"]:
            out[col] = np.nan

    out["is_neutral"] = df["neutral"].astype(float) if "neutral" in df.columns else 0.0
    out["tournament_importance"] = (
        df["tournament"].apply(_importance) if "tournament" in df.columns else 1.5
    )

    # Meta + target
    out["match_id"] = df.get("match_id", "")
    out["date"] = df["date"]
    out["home_team"] = df["home_team"]
    out["away_team"] = df["away_team"]
    out["home_score"] = df["home_score"]
    out["away_score"] = df["away_score"]
    out["outcome"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("D" if r["home_score"] == r["away_score"] else "A"),
        axis=1,
    )

    return out


def build_prediction_features(
    historical_df: pd.DataFrame,
    upcoming_df: pd.DataFrame,
    ranking_lookup: dict | None = None,
    form_n: int = 5,
) -> pd.DataFrame:
    """Build features for upcoming matches using ALL historical data as context."""
    historical_df = historical_df.sort_values("date")

    ratings = current_ratings(historical_df)

    team_history: dict[str, list[dict]] = defaultdict(list)
    for _, row in historical_df.iterrows():
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        dt = row["date"]
        team_history[row["home_team"]].append(
            {"date": dt, "gf": hs, "ga": as_,
             "won": hs > as_, "drew": hs == as_, "lost": hs < as_}
        )
        team_history[row["away_team"]].append(
            {"date": dt, "gf": as_, "ga": hs,
             "won": as_ > hs, "drew": as_ == hs, "lost": as_ < hs}
        )

    rows: list[dict] = []
    for _, row in upcoming_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        dt = pd.to_datetime(row["date"])
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", row.get("stage", "FIFA World Cup")))

        h_elo = ratings.get(home, INITIAL_RATING)
        a_elo = ratings.get(away, INITIAL_RATING)
        h_form = _form_stats(team_history.get(home, []), form_n)
        a_form = _form_stats(team_history.get(away, []), form_n)

        h_fifa = _get_fifa_features(home, dt, ranking_lookup)
        a_fifa = _get_fifa_features(away, dt, ranking_lookup)

        rows.append(
            {
                "match_id": row.get("match_id", ""),
                "home_team": home,
                "away_team": away,
                "home_elo": h_elo,
                "away_elo": a_elo,
                "elo_diff": h_elo - a_elo,
                "home_win_rate": h_form["win_rate"],
                "home_draw_rate": h_form["draw_rate"],
                "home_gf": h_form["gf"],
                "home_ga": h_form["ga"],
                "home_gd": h_form["gd"],
                "home_rest": _rest_days(team_history.get(home, []), dt),
                "away_win_rate": a_form["win_rate"],
                "away_draw_rate": a_form["draw_rate"],
                "away_gf": a_form["gf"],
                "away_ga": a_form["ga"],
                "away_gd": a_form["gd"],
                "away_rest": _rest_days(team_history.get(away, []), dt),
                "home_fifa_rank": h_fifa["rank"],
                "away_fifa_rank": a_fifa["rank"],
                "fifa_rank_diff": h_fifa["rank"] - a_fifa["rank"],
                "home_fifa_pts": h_fifa["pts"],
                "away_fifa_pts": a_fifa["pts"],
                "fifa_pts_diff": h_fifa["pts"] - a_fifa["pts"],
                "is_neutral": float(neutral),
                "tournament_importance": _importance(tournament),
            }
        )

    return pd.DataFrame(rows)
