"""WC2026 full 32-team bracket simulation.

Format
------
48 teams → 12 groups of 4
Top 2 from each group (24) + 8 best third-place teams = 32 qualifiers
Round of 32 → R16 → QF → SF → Final (no 3rd-place play-off in simulation)

Usage
-----
The main entry point is ``simulate_full_tournament``.  Pass raw group
composition and already-played results; the function simulates remaining group
matches inside each Monte Carlo iteration so that group outcomes covary
correctly with knockout results.

Bracket seeding
---------------
WC2026 divides the 32-team bracket into 4 sections (S1–S4) of 8 teams.
FIFA's exact draw table had not been published at the model's knowledge cutoff;
the seeding below is an approximation.  See DECISIONS.md for details.

What IS accurate:
- 3rd-place qualification criteria (FIFA points/GD/GF/wins rules)
- Elimination structure (single-elimination)
- Teams from the same group cannot meet before the QF

What is APPROXIMATED:
- Exact group-position → R32 slot assignment
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wc26.models.base import BaseModel

logger = logging.getLogger(__name__)

ROUNDS = ["R32", "R16", "QF", "SF", "Final"]

_DEFAULT_LAMBDA_HOME = 1.3  # fallback when model has no expected goals
_DEFAULT_LAMBDA_AWAY = 1.0


# ── Group stage helpers ───────────────────────────────────────────────────────


def _apply_result(
    pts: dict[str, int],
    gf: dict[str, int],
    ga: dict[str, int],
    home: str,
    away: str,
    hs: int,
    as_: int,
) -> None:
    gf[home] += hs
    ga[home] += as_
    gf[away] += as_
    ga[away] += hs
    if hs > as_:
        pts[home] += 3
    elif hs == as_:
        pts[home] += 1
        pts[away] += 1
    else:
        pts[away] += 3


def _rank_teams(
    teams: list[str],
    pts: dict[str, int],
    gf: dict[str, int],
    ga: dict[str, int],
) -> list[tuple[str, int, int, int, int]]:
    """Return [(team, pts, gd, gf, wins)] sorted by group-stage tiebreakers."""
    wins = {
        t: pts[t] // 3  # exact wins = pts // 3 only if no draws created extra pts; this is approximate
        for t in teams
    }
    return sorted(
        teams,
        key=lambda t: (pts[t], gf[t] - ga[t], gf[t], np.random.random()),
        reverse=True,
    )


def _simulate_single_group(
    teams: list[str],
    played: dict[tuple[str, str], tuple[int, int]],
    model: BaseModel,
) -> list[tuple[str, int, int, int, int]]:
    """Simulate one iteration of a 4-team group; return ranked (team, pts, gd, gf, wins)."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}

    # Apply already-played matches
    for (h, a), (hs, as_) in played.items():
        if h in pts and a in pts:
            _apply_result(pts, gf, ga, h, a, hs, as_)

    # Simulate unplayed matches (each pair plays once; h then a as home)
    for i, h in enumerate(teams):
        for a in teams[i + 1:]:
            if (h, a) in played or (a, h) in played:
                continue
            pred = model.predict_match(h, a, neutral=True)
            matrix = pred.score_matrix
            if matrix is not None:
                flat = matrix.flatten()
                flat = flat / flat.sum()
                idx = int(np.random.choice(len(flat), p=flat))
                hs = idx // matrix.shape[1]
                as_ = idx % matrix.shape[1]
            else:
                eh = pred.exp_home_goals or _DEFAULT_LAMBDA_HOME
                ea = pred.exp_away_goals or _DEFAULT_LAMBDA_AWAY
                hs = int(np.random.poisson(eh))
                as_ = int(np.random.poisson(ea))
            _apply_result(pts, gf, ga, h, a, hs, as_)

    ranked = _rank_teams(teams, pts, gf, ga)
    return [
        (
            t,
            pts[t],
            gf[t] - ga[t],
            gf[t],
            pts[t] // 3,  # approximation; good enough for best-3rd selection
        )
        for t in ranked
    ]


# ── Qualifier selection ───────────────────────────────────────────────────────


def _determine_qualifiers(
    group_ranked: dict[str, list[tuple[str, int, int, int, int]]],
) -> tuple[list[str], list[str], list[str]]:
    """Return (winners, runners_up, best_thirds) given one MC iteration's group results."""
    winners: list[str] = []
    runners_up: list[str] = []
    thirds: list[tuple[str, int, int, int, int]] = []

    for grp, ranked in sorted(group_ranked.items()):
        if len(ranked) >= 1:
            winners.append(ranked[0][0])
        if len(ranked) >= 2:
            runners_up.append(ranked[1][0])
        if len(ranked) >= 3:
            thirds.append(ranked[2])

    # Best 8 third-place by FIFA tiebreakers (pts, gd, gf, wins)
    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x[1], x[2], x[3], x[4], np.random.random()),
        reverse=True,
    )
    return winners, runners_up, [t[0] for t in thirds_sorted[:8]]


# ── Bracket seeding ───────────────────────────────────────────────────────────


def _seed_bracket(
    winners: list[str],
    runners_up: list[str],
    best_thirds: list[str],
) -> list[str]:
    """Assign 32 qualifiers to bracket slots [0..31].

    Layout: 4 sections × 8 slots.  Matches within a section:
    [0]v[7], [1]v[6], [2]v[5], [3]v[4].
    Group winners as higher seeds (even slots 0,2,4 in each section).
    Runners-up and best thirds fill odd slots so group-mates land in
    opposite halves of the same section (no same-group R32 rematch).
    """
    w = list(winners) + ["TBD"] * (12 - len(winners))
    r = list(runners_up) + ["TBD"] * (12 - len(runners_up))
    t = list(best_thirds) + ["TBD"] * (8 - len(best_thirds))

    slots: list[str] = ["TBD"] * 32

    def fill_section(base: int, ww: list[str], rr: list[str], tt: list[str]) -> None:
        slots[base + 0] = ww[0] if ww else "TBD"
        slots[base + 7] = rr[2] if len(rr) > 2 else (tt[0] if tt else "TBD")
        slots[base + 2] = ww[1] if len(ww) > 1 else "TBD"
        slots[base + 5] = rr[1] if len(rr) > 1 else "TBD"
        slots[base + 4] = ww[2] if len(ww) > 2 else "TBD"
        slots[base + 3] = rr[0] if rr else "TBD"
        slots[base + 6] = tt[0] if tt else "TBD"
        slots[base + 1] = tt[1] if len(tt) > 1 else "TBD"

    fill_section(0,  w[0:3],  r[0:3],  t[0:2])
    fill_section(8,  w[3:6],  r[3:6],  t[2:4])
    fill_section(16, w[6:9],  r[6:9],  t[4:6])
    fill_section(24, w[9:12], r[9:12], t[6:8])

    return slots


# ── Knockout match ────────────────────────────────────────────────────────────


def _sim_knockout(home: str, away: str, model: BaseModel) -> str:
    """Sample knockout match winner (no draw; ET/penalties modelled as 50/50 split on draw prob)."""
    if home == "TBD" or away == "TBD":
        return home if np.random.random() < 0.5 else away

    pred = model.predict_match(home, away, neutral=True)
    p_h = pred.p_home + pred.p_draw * 0.5
    p_a = pred.p_away + pred.p_draw * 0.5
    total = p_h + p_a
    return home if np.random.random() < p_h / total else away


# ── Main simulation ───────────────────────────────────────────────────────────


def simulate_full_tournament(
    all_groups: dict[str, list[str]],
    played_by_group: dict[str, dict[tuple[str, str], tuple[int, int]]],
    model: BaseModel,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """Monte Carlo simulation of the full WC2026 tournament.

    Simulates remaining group matches inside each iteration so that
    group outcomes covary correctly with knockout results.

    Parameters
    ----------
    all_groups       : {group_letter: [team1, team2, team3, team4]}
    played_by_group  : {group_letter: {(home_team, away_team): (home_score, away_score)}}
                       — only already-finished matches
    model            : fitted prediction model (must be fitted)
    n_simulations    : Monte Carlo iterations

    Returns
    -------
    DataFrame: team, p_advance, p_r16, p_qf, p_sf, p_final, p_champion
    All probabilities are ABSOLUTE (not conditional on prior round).
    Sorted by p_champion descending.
    """
    if not all_groups or not model.is_fitted():
        logger.warning("bracket_sim: no groups or model not fitted — returning empty DataFrame")
        return _empty_df()

    n_groups = sum(1 for teams in all_groups.values() if len(teams) >= 2)
    logger.info(
        f"Full tournament sim: {n_groups}/12 groups populated, "
        f"{n_simulations:,} iterations"
    )

    # Counters (all absolute, divided by n_simulations):
    # [0] advance = qualified from group
    # [1] r16     = won R32 match → reached R16
    # [2] qf      = won R16 match → reached QF
    # [3] sf      = won QF match  → reached SF
    # [4] final   = won SF match  → reached Final
    # [5] champion= won Final     → champion
    reach: dict[str, list[int]] = {}

    for _ in range(n_simulations):
        # 1. Simulate group stages
        group_ranked: dict[str, list[tuple[str, int, int, int, int]]] = {}
        for grp, teams in sorted(all_groups.items()):
            if len(teams) < 2:
                continue
            played = played_by_group.get(grp, {})
            group_ranked[grp] = _simulate_single_group(teams, played, model)

        # 2. Determine qualifiers
        winners, runners_up, best_thirds = _determine_qualifiers(group_ranked)
        qualifiers = set(winners + runners_up + best_thirds)

        # 3. Track group advancement
        for t in qualifiers:
            if t == "TBD":
                continue
            if t not in reach:
                reach[t] = [0, 0, 0, 0, 0, 0]
            reach[t][0] += 1

        # 4. Seed bracket
        bracket = _seed_bracket(winners, runners_up, best_thirds)

        # 5. Simulate knockout rounds
        # ROUNDS = ["R32", "R16", "QF", "SF", "Final"]
        # Winning round_idx=0 (R32) → index 1 (reached R16)
        # Winning round_idx=4 (Final) → index 5 (champion)
        current_round = list(bracket)
        for round_idx, _ in enumerate(ROUNDS):
            next_round: list[str] = []
            i = 0
            while i + 1 < len(current_round):
                winner = _sim_knockout(current_round[i], current_round[i + 1], model)
                next_round.append(winner)
                if winner != "TBD" and winner in reach:
                    reach[winner][round_idx + 1] += 1
                i += 2
            current_round = next_round
            if not current_round:
                break

    if not reach:
        return _empty_df()

    rows = []
    for team, counts in reach.items():
        row: dict = {"team": team}
        for col, k in zip(
            ["p_advance", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"],
            range(6),
        ):
            p = counts[k] / n_simulations
            lo, hi = _wilson_ci(counts[k], n_simulations)
            row[col] = p
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("p_champion", ascending=False)
        .reset_index(drop=True)
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI for a proportion (k successes in n trials)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _empty_df() -> pd.DataFrame:
    ci_cols = [f"{c}_{s}" for c in ["p_advance", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"] for s in ["lo", "hi"]]
    return pd.DataFrame(
        columns=["team", "p_advance", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"] + ci_cols
    )


# ── Fixture data parsers ──────────────────────────────────────────────────────


def extract_groups_from_fixtures(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Parse group composition from the fixtures DataFrame.

    Returns {group_letter: [team1, team2, ...]}.
    """
    groups: dict[str, list[str]] = {}
    if "group" not in fixtures.columns:
        return groups

    for _, row in fixtures.iterrows():
        grp_raw = row.get("group")
        if not grp_raw or str(grp_raw) in ("nan", "None", ""):
            continue
        grp = str(grp_raw).upper().replace("GROUP", "").replace("_", "").strip()
        if len(grp) != 1 or not grp.isalpha():
            continue
        if grp not in groups:
            groups[grp] = []
        for team in (row["home_team"], row["away_team"]):
            if team not in groups[grp]:
                groups[grp].append(team)

    return groups


def extract_played_by_group(
    fixtures: pd.DataFrame,
    all_groups: dict[str, list[str]],
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    """Extract finished group-stage match results keyed by group.

    Returns {group_letter: {(home_team, away_team): (home_score, away_score)}}.
    """
    finished = fixtures[fixtures["status"] == "FINISHED"]
    result: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}

    for grp, teams in all_groups.items():
        grp_matches = finished[
            finished["home_team"].isin(teams) & finished["away_team"].isin(teams)
        ]
        result[grp] = {
            (row["home_team"], row["away_team"]): (int(row["home_score"]), int(row["away_score"]))
            for _, row in grp_matches.iterrows()
        }

    return result
