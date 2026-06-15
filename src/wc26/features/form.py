"""Recent-form feature extraction.

For every match, we compute per-team statistics using ONLY matches that
finished strictly before the match date. The implementation processes
rows in chronological order and appends results to a running history —
this guarantees zero future leakage without any date-index gymnastics.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _form_stats(history: list[dict], n: int) -> dict[str, float]:
    """Aggregate stats from the last n matches. Returns NaN if history is empty."""
    recent = history[-n:]
    if not recent:
        nan = float("nan")
        return {
            "win_rate": nan, "draw_rate": nan, "loss_rate": nan,
            "gf": nan, "ga": nan, "gd": nan,
        }
    k = len(recent)
    return {
        "win_rate": sum(m["won"] for m in recent) / k,
        "draw_rate": sum(m["drew"] for m in recent) / k,
        "loss_rate": sum(m["lost"] for m in recent) / k,
        "gf": sum(m["gf"] for m in recent) / k,
        "ga": sum(m["ga"] for m in recent) / k,
        "gd": (sum(m["gf"] for m in recent) - sum(m["ga"] for m in recent)) / k,
    }


def _rest_days(history: list[dict], match_date: pd.Timestamp) -> float:
    if not history:
        return float("nan")
    return float((match_date - history[-1]["date"]).days)


def compute_form_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return form + rest-day features for each match in df.

    Columns returned:
      home_win_rate, home_draw_rate, home_gf, home_ga, home_gd, home_rest,
      away_win_rate, away_draw_rate, away_gf, away_ga, away_gd, away_rest
    (all computed from matches strictly before each match date)
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_history: dict[str, list[dict]] = defaultdict(list)

    rows: list[dict] = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        dt = row["date"]

        h_form = _form_stats(team_history[home], n)
        a_form = _form_stats(team_history[away], n)

        rows.append(
            {
                "home_win_rate": h_form["win_rate"],
                "home_draw_rate": h_form["draw_rate"],
                "home_gf": h_form["gf"],
                "home_ga": h_form["ga"],
                "home_gd": h_form["gd"],
                "home_rest": _rest_days(team_history[home], dt),
                "away_win_rate": a_form["win_rate"],
                "away_draw_rate": a_form["draw_rate"],
                "away_gf": a_form["gf"],
                "away_ga": a_form["ga"],
                "away_gd": a_form["gd"],
                "away_rest": _rest_days(team_history[away], dt),
            }
        )

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        team_history[home].append(
            {"date": dt, "gf": hs, "ga": as_,
             "won": hs > as_, "drew": hs == as_, "lost": hs < as_}
        )
        team_history[away].append(
            {"date": dt, "gf": as_, "ga": hs,
             "won": as_ > hs, "drew": as_ == hs, "lost": as_ < hs}
        )

    return pd.DataFrame(rows, index=df.index)
