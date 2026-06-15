"""End-to-end update pipeline: single command does everything.

Usage:
    python -m wc26.update
    make update

Pipeline (all steps idempotent, each wrapped in try/except):
  1.  fetch_historical          – Kaggle/GitHub CSV
  2.  fetch_wc2026_fixtures     – football-data.org → fallbacks
  3.  fetch_xg (optional)       – soccerdata (ENABLE_XG=true only)
  4.  build_processed_tables    – clean, dedup, match_id, validate
  5.  detect_new_results        – matches finished since last snapshot
  6.  refit_bayesian            – Hierarchical Bayesian Poisson (PyMC)
  7.  build_lgbm_features       – Elo + form feature matrix
  8.  refit_lgbm_ensemble       – LightGBM + ensemble weights + calibration
  9.  evaluate_models           – holdout Brier/LogLoss per model + baselines
  10. generate_predictions      – all future WC2026 matches (ensemble)
  11. score_past_predictions    – calibration: pre-match pred vs actual
  12. run_simulations           – Group D + bracket MC (10 000 runs)
  13. write_snapshot            – data/snapshots/{ts}/
  14. update_outputs            – data/outputs/latest_* + timeseries
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from wc26.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress verbose low-level noise from pytensor/PyMC internals
for _noisy in (
    "pytensor.configparser",
    "pytensor.link.c.cmodule",
    "pytensor.link.c.lazylinker_c",
    "pymc.sampling.mcmc",
    "pymc.stats.convergence",
    "arviz",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("wc26.update")


# ────────────────────────────────────────────────────────────────────────────
# Step helpers
# ────────────────────────────────────────────────────────────────────────────


def _step(name: str):
    """Decorator that wraps a pipeline step in try/except + logging."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            logger.info(f"▶ {name}")
            try:
                result = fn(*args, **kwargs)
                logger.info(f"✓ {name}")
                return result
            except Exception as exc:
                logger.error(f"✗ {name} failed: {exc}", exc_info=True)
                return None

        return wrapper

    return decorator


# ────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ────────────────────────────────────────────────────────────────────────────


@_step("Fetch historical data")
def fetch_historical() -> pd.DataFrame | None:
    from wc26.fetchers.kaggle_historical import KaggleHistoricalFetcher

    fetcher = KaggleHistoricalFetcher()
    return fetcher.fetch()


@_step("Fetch WC2026 fixtures")
def fetch_wc2026_fixtures() -> pd.DataFrame | None:
    source = settings.primary_fixture_source

    if source == "football_data":
        from wc26.fetchers.footballdata import FootballDataFetcher
        try:
            return FootballDataFetcher().fetch()
        except Exception as exc:
            logger.warning(f"football-data.org failed: {exc} — trying API-Football")

    if source in ("football_data", "api_football"):
        from wc26.fetchers.apifootball import APIFootballFetcher
        try:
            return APIFootballFetcher().fetch()
        except Exception as exc:
            logger.warning(f"API-Football failed: {exc} — trying Wikipedia")

    from wc26.fetchers.wikipedia import WikipediaFetcher
    return WikipediaFetcher().fetch()


@_step("Fetch xG data (optional)")
def fetch_xg() -> pd.DataFrame | None:
    if not settings.enable_xg:
        return None
    from wc26.fetchers.xg_soccerdata import XGFetcher
    return XGFetcher().fetch()


@_step("Build processed tables")
def build_processed_tables(
    historical_raw: pd.DataFrame | None,
    fixtures_raw: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    from wc26.data.build import build_fixtures, build_historical, save_processed

    historical = None
    if historical_raw is not None:
        historical = build_historical(historical_raw)
        save_processed(historical, "matches_historical")

    fixtures = None
    if fixtures_raw is not None:
        fixtures = build_fixtures(fixtures_raw)
        save_processed(fixtures, "wc2026_fixtures")

        results = fixtures[fixtures["status"] == "FINISHED"].copy()
        save_processed(results, "wc2026_results")

    return historical, fixtures


@_step("Detect new results")
def detect_new_results(
    fixtures: pd.DataFrame | None,
    last_snapshot_path: Path | None,
) -> list[str]:
    """Return match_ids of results not in the previous snapshot."""
    if fixtures is None:
        return []

    finished = fixtures[fixtures["status"] == "FINISHED"]["match_id"].tolist()

    if last_snapshot_path is None:
        return finished

    prev_pred_path = last_snapshot_path / "predictions.parquet"
    if not prev_pred_path.exists():
        return finished

    prev_ids = set(pd.read_parquet(prev_pred_path)["match_id"].tolist())
    new = [mid for mid in finished if mid not in prev_ids]
    logger.info(f"Detected {len(new)} new results since last snapshot")
    return new


@_step("Refit Bayesian Poisson model")
def refit_bayesian(historical: pd.DataFrame | None) -> object | None:
    if historical is None or historical.empty:
        logger.warning("No historical data — skipping Bayesian refit")
        return None

    from wc26.models.bayesian_poisson import BayesianPoissonModel

    model = BayesianPoissonModel(
        draws=settings.mcmc_draws,
        tune=settings.mcmc_tune,
        chains=settings.mcmc_chains,
    )

    # MCMC training window: last 8 years only (zero weight beyond ~2-yr half-life)
    eight_years_ago = historical["date"].max() - pd.DateOffset(years=8)
    recent = historical[historical["date"] >= eight_years_ago]
    if recent.empty:
        recent = historical

    cutoff = recent["date"].max() - pd.DateOffset(years=2)
    train = recent[recent["date"] < cutoff]
    holdout = recent[recent["date"] >= cutoff]

    if train.empty:
        train = recent

    n_train = len(train)
    n_teams = len(set(train["home_team"]) | set(train["away_team"]))
    logger.info(f"MCMC training: {n_train} matches, ~{n_teams} teams (last 8 years)")

    model.fit(train)

    if not holdout.empty:
        metrics = model.evaluate_holdout(holdout)
        logger.info(
            f"Bayesian holdout — Brier: {metrics['brier']:.4f}  "
            f"LogLoss: {metrics['log_loss']:.4f}  "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

    return model


@_step("Fetch FIFA ranking")
def fetch_fifa_ranking() -> dict | None:
    from wc26.fetchers.fifa_ranking import FifaRankingFetcher, build_ranking_lookup

    df = FifaRankingFetcher().fetch()
    if df is None or df.empty:
        return None
    lookup = build_ranking_lookup(df)
    logger.info(f"FIFA ranking lookup built for {len(lookup)} teams")
    return lookup


@_step("Build LightGBM feature matrix")
def build_lgbm_features(
    historical: pd.DataFrame | None,
    fixtures: pd.DataFrame | None,
    ranking_lookup: dict | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Build training + prediction feature DataFrames for LightGBM.

    Returns (features_df, pred_features_df):
    - features_df      : historical matches with Elo + form + FIFA rank features (includes 'outcome')
    - pred_features_df : all WC team pairs (pre-computed for fast lookup in simulations)
    """
    if historical is None or historical.empty:
        return None, None

    from wc26.features.pipeline import build_prediction_features, build_training_features

    eight_years_ago = historical["date"].max() - pd.DateOffset(years=8)
    recent = historical[historical["date"] >= eight_years_ago]
    if recent.empty:
        recent = historical

    features_df = build_training_features(recent, ranking_lookup=ranking_lookup)
    logger.info(f"Built {len(features_df)} training feature rows")

    pred_features_df = None
    if fixtures is not None and not fixtures.empty:
        wc_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
        pairs = [(h, a) for h, a in permutations(wc_teams, 2)]
        pairs_df = pd.DataFrame(
            {
                "home_team": [h for h, a in pairs],
                "away_team": [a for h, a in pairs],
                "neutral": True,
                "date": historical["date"].max(),
                "tournament": "FIFA World Cup",
                "home_score": np.nan,
                "away_score": np.nan,
            }
        )
        pred_features_df = build_prediction_features(historical, pairs_df, ranking_lookup=ranking_lookup)
        logger.info(
            f"Built prediction features for {len(wc_teams)} WC teams "
            f"({len(pairs)} pairs)"
        )

    return features_df, pred_features_df


@_step("Refit LightGBM + ensemble")
def refit_lgbm_ensemble(
    bayesian_model,
    features_df: pd.DataFrame | None,
    pred_features_df: pd.DataFrame | None,
    odds_fetcher=None,
) -> object | None:
    """Fit LightGBM, optimise ensemble weights, fit isotonic calibrator.

    3-way split to prevent contamination:
      train : years 1–6 of the 8-year window  → fit LightGBM
      val   : year 7                           → optimise weights + calibrator
      test  : year 8 (most recent)             → honest evaluation only (in evaluate_models)
    """
    if bayesian_model is None or features_df is None or features_df.empty:
        logger.warning("Missing Bayesian model or features — skipping LightGBM/ensemble")
        return None

    from wc26.models.ensemble import EnsembleModel
    from wc26.models.lgbm import LGBMModel

    date_max = features_df["date"].max()
    test_cutoff = date_max - pd.DateOffset(years=1)   # most recent 1 yr = test
    val_cutoff  = date_max - pd.DateOffset(years=2)   # preceding 1 yr = val

    train_features = features_df[features_df["date"] < val_cutoff]
    val_features   = features_df[(features_df["date"] >= val_cutoff) & (features_df["date"] < test_cutoff)]
    # test_features used only inside evaluate_models

    if train_features.empty:
        logger.warning("No training features before val cutoff — skipping")
        return None

    n_tr, n_val = len(train_features), len(val_features)
    logger.info(f"3-way split: train={n_tr}, val={n_val}, test={len(features_df) - n_tr - n_val}")

    lgbm = LGBMModel()
    lgbm.fit(train_features)

    if pred_features_df is not None and not pred_features_df.empty:
        lgbm.set_prediction_features(pred_features_df)
        logger.info("LightGBM prediction lookup table populated")

    market_mdl = None
    if odds_fetcher is not None:
        from wc26.models.market_model import MarketModel
        market_mdl = MarketModel(odds_fetcher)
        logger.info("Market odds model attached to ensemble")

    ensemble = EnsembleModel(bayesian=bayesian_model, lgbm=lgbm, market=market_mdl)

    if n_val >= 20:
        ensemble.fit_weights(val_features)
        ensemble.fit_calibrator(val_features)
    else:
        logger.warning(f"Only {n_val} val rows — using default 50/50 weights, no calibration")

    return ensemble


@_step("Evaluate models")
def evaluate_models(
    bayesian_model,
    ensemble_model,
    features_df: pd.DataFrame | None,
    out_dir: Path,
) -> dict:
    """Compute TEST-set metrics for all models and baselines.

    Uses only the most recent 1 year as test (separate from the val year
    used for ensemble weight fitting and calibration).
    """
    if features_df is None or features_df.empty:
        return {}

    # Test set = most recent 1 year (not used in any fitting step)
    cutoff = features_df["date"].max() - pd.DateOffset(years=1)
    holdout = features_df[features_df["date"] >= cutoff].dropna(subset=["outcome"])

    if holdout.empty:
        logger.warning("Empty test set — skipping model evaluation")
        return {}

    from sklearn.metrics import log_loss

    n = len(holdout)
    actuals = holdout["outcome"].tolist()
    y_true = np.array(
        [[1, 0, 0] if o == "H" else ([0, 1, 0] if o == "D" else [0, 0, 1]) for o in actuals],
        dtype=float,
    )

    def _brier(probs: np.ndarray) -> float:
        return float(np.mean(np.sum((probs - y_true) ** 2, axis=1)))

    def _logloss(probs: np.ndarray) -> float:
        return float(log_loss(actuals, probs[:, [2, 1, 0]], labels=["A", "D", "H"]))

    def _acc(probs: np.ndarray) -> float:
        labels = ["H", "D", "A"]
        predicted = [labels[int(np.argmax(p))] for p in probs]
        return float(np.mean([p == a for p, a in zip(predicted, actuals)]))

    metrics: dict = {"n_test": n, "n_holdout": n}  # n_holdout kept for back-compat

    # Bayesian
    if bayesian_model is not None:
        bay_metrics = bayesian_model.evaluate_holdout(holdout)
        metrics["bayesian"] = bay_metrics
        logger.info(
            f"Bayesian  — Brier: {bay_metrics['brier']:.4f}  "
            f"LogLoss: {bay_metrics['log_loss']:.4f}  "
            f"Accuracy: {bay_metrics['accuracy']:.4f}"
        )

    # LightGBM + Ensemble
    if ensemble_model is not None:
        lgbm_metrics = ensemble_model.lgbm.evaluate_holdout(holdout)
        metrics["lgbm"] = lgbm_metrics
        logger.info(
            f"LightGBM  — Brier: {lgbm_metrics['brier']:.4f}  "
            f"LogLoss: {lgbm_metrics['log_loss']:.4f}  "
            f"Accuracy: {lgbm_metrics['accuracy']:.4f}"
        )

        ens_metrics = ensemble_model.evaluate_holdout(holdout)
        metrics["ensemble"] = ens_metrics
        logger.info(
            f"Ensemble  — Brier: {ens_metrics['brier']:.4f}  "
            f"LogLoss: {ens_metrics['log_loss']:.4f}  "
            f"Accuracy: {ens_metrics['accuracy']:.4f}"
        )

        # Save feature importance
        fi = ensemble_model.lgbm.feature_importance()
        if not fi.empty:
            fi.to_parquet(out_dir / "feature_importance.parquet", index=False)

    # Baselines
    equal_probs = np.full((n, 3), 1 / 3)
    home_wins = np.column_stack([
        np.ones(n) * 0.45,
        np.ones(n) * 0.27,
        np.ones(n) * 0.28,
    ])
    h_freq = np.mean([a == "H" for a in actuals])
    d_freq = np.mean([a == "D" for a in actuals])
    a_freq = 1 - h_freq - d_freq
    base_freq = np.column_stack([
        np.full(n, h_freq),
        np.full(n, d_freq),
        np.full(n, a_freq),
    ])
    metrics["baselines"] = {
        "equal_odds": {"brier": _brier(equal_probs), "log_loss": _logloss(equal_probs), "accuracy": _acc(equal_probs)},
        "always_home": {"brier": _brier(home_wins), "log_loss": _logloss(home_wins), "accuracy": _acc(home_wins)},
        "base_rate": {"brier": _brier(base_freq), "log_loss": _logloss(base_freq), "accuracy": _acc(base_freq)},
    }
    logger.info(
        f"Baselines — equal_odds Brier: {metrics['baselines']['equal_odds']['brier']:.4f}  "
        f"always_home Brier: {metrics['baselines']['always_home']['brier']:.4f}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    return metrics


@_step("Generate predictions")
def generate_predictions(
    model,
    fixtures: pd.DataFrame | None,
    snapshot_ts: str,
) -> pd.DataFrame | None:
    if model is None or fixtures is None or fixtures.empty:
        return None

    future = fixtures[fixtures["status"] != "FINISHED"]
    if future.empty:
        logger.info("No upcoming matches to predict")
        return pd.DataFrame(
            columns=["match_id", "snapshot_ts", "home_team", "away_team",
                     "p_home", "p_draw", "p_away", "exp_home_goals", "exp_away_goals"]
        )

    rows = []
    for _, row in future.iterrows():
        try:
            pred = model.predict_match(
                home=row["home_team"],
                away=row["away_team"],
                neutral=bool(row.get("neutral", False)),
            )
            rows.append(
                {
                    "match_id": row["match_id"],
                    "snapshot_ts": snapshot_ts,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "p_home": pred.p_home,
                    "p_draw": pred.p_draw,
                    "p_away": pred.p_away,
                    "exp_home_goals": pred.exp_home_goals,
                    "exp_away_goals": pred.exp_away_goals,
                }
            )
        except Exception as exc:
            logger.warning(f"Prediction failed for {row.get('match_id')}: {exc}")

    return pd.DataFrame(rows) if rows else None


@_step("Score past predictions (calibration)")
def score_past_predictions(
    fixtures: pd.DataFrame | None,
    snapshots_dir: Path,
) -> pd.DataFrame | None:
    """For each finished match, find the snapshot made BEFORE the match and compute Brier."""
    if fixtures is None:
        return None

    results = fixtures[fixtures["status"] == "FINISHED"]
    if results.empty:
        return None

    all_snap_preds: list[pd.DataFrame] = []
    for snap_dir in sorted(snapshots_dir.iterdir()) if snapshots_dir.exists() else []:
        pred_path = snap_dir / "predictions.parquet"
        if pred_path.exists():
            df = pd.read_parquet(pred_path)
            df["snapshot_ts"] = snap_dir.name
            all_snap_preds.append(df)

    if not all_snap_preds:
        return None

    all_preds = pd.concat(all_snap_preds, ignore_index=True)

    calib_rows = []
    for _, match in results.iterrows():
        mid = match["match_id"]
        h, a = int(match["home_score"]), int(match["away_score"])
        outcome = "H" if h > a else ("D" if h == a else "A")

        match_preds = all_preds[all_preds["match_id"] == mid]
        if match_preds.empty:
            continue

        pre_match = match_preds.sort_values("snapshot_ts").iloc[0]
        p_home = float(pre_match["p_home"])
        p_draw = float(pre_match["p_draw"])
        p_away = float(pre_match["p_away"])

        actual_vec = [1.0, 0.0, 0.0] if outcome == "H" else ([0.0, 1.0, 0.0] if outcome == "D" else [0.0, 0.0, 1.0])
        pred_vec = [p_home, p_draw, p_away]
        brier = float(np.sum((np.array(pred_vec) - np.array(actual_vec)) ** 2))

        calib_rows.append(
            {
                "match_id": mid,
                "snapshot_ts": pre_match["snapshot_ts"],
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "actual_outcome": outcome,
                "brier_score": brier,
            }
        )

    return pd.DataFrame(calib_rows) if calib_rows else None


@_step("Run simulations")
def run_simulations(
    model,
    fixtures: pd.DataFrame | None,
    n_simulations: int = 10_000,
) -> dict:
    if model is None:
        return {}

    from wc26.sim.bracket_sim import (
        extract_groups_from_fixtures,
        extract_played_by_group,
        simulate_full_tournament,
    )
    from wc26.sim.group_sim import GROUP_D, simulate_group
    from wc26.sim.turkey import compute_turkey_probs, turkey_next_match

    # ── Group D simulation (detailed) ────────────────────────────────────────
    already_played_d = None
    if fixtures is not None and not fixtures.empty:
        played_d = fixtures[
            (fixtures["status"] == "FINISHED")
            & (fixtures["home_team"].isin(GROUP_D))
            & (fixtures["away_team"].isin(GROUP_D))
        ][["home_team", "away_team", "home_score", "away_score"]]
        if not played_d.empty:
            already_played_d = played_d

    turkey_probs = compute_turkey_probs(
        model=model,
        already_played=already_played_d,
        n_simulations=n_simulations,
    )

    group_d_results = simulate_group(
        teams=GROUP_D,
        model=model,
        already_played=already_played_d,
        n_simulations=n_simulations,
    )

    next_match = turkey_next_match(fixtures) if fixtures is not None else None

    # ── Full bracket simulation ───────────────────────────────────────────────
    bracket_results: pd.DataFrame | None = None
    if fixtures is not None and not fixtures.empty:
        all_groups = extract_groups_from_fixtures(fixtures)
        # Fall back to Group D if fixture data has no group info
        if not all_groups:
            all_groups = {"D": list(GROUP_D)}
        played_by_group = extract_played_by_group(fixtures, all_groups)
        bracket_results = simulate_full_tournament(
            all_groups=all_groups,
            played_by_group=played_by_group,
            model=model,
            n_simulations=settings.bracket_mc_iterations,
        )
        if bracket_results is not None and not bracket_results.empty:
            turkey_row = bracket_results[bracket_results["team"].str.contains("Turk", case=False, na=False)]
            if not turkey_row.empty:
                r = turkey_row.iloc[0]
                turkey_probs.update(
                    {
                        "p_advance_full": float(r.get("p_advance", float("nan"))),
                        "round_of_16": float(r.get("p_r16", float("nan"))),
                        "quarter_final": float(r.get("p_qf", float("nan"))),
                        "semi_final": float(r.get("p_sf", float("nan"))),
                        "final": float(r.get("p_final", float("nan"))),
                        "champion": float(r.get("p_champion", float("nan"))),
                    }
                )
                logger.info(
                    f"Turkey bracket probs — R16: {turkey_probs['round_of_16']:.1%}  "
                    f"QF: {turkey_probs['quarter_final']:.1%}  "
                    f"Champion: {turkey_probs['champion']:.1%}"
                )

    return {
        "turkey_probs": turkey_probs,
        "group_d_results": group_d_results,
        "next_match": next_match,
        "bracket_results": bracket_results,
    }


@_step("Write snapshot")
def write_snapshot(
    snapshot_dir: Path,
    predictions: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    sim_results: dict,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if predictions is not None and not predictions.empty:
        predictions.to_parquet(snapshot_dir / "predictions.parquet", index=False)

    calib_json: dict = {}
    if calibration is not None and not calibration.empty:
        calibration.to_parquet(snapshot_dir / "calibration.parquet", index=False)
        calib_json = {
            "mean_brier": float(calibration["brier_score"].mean()),
            "n_matches_scored": int(len(calibration)),
        }

    group_d_results = sim_results.get("group_d_results")
    if group_d_results is not None:
        group_d_results.to_parquet(snapshot_dir / "turkey_simulation.parquet", index=False)

    bracket_results = sim_results.get("bracket_results")
    if bracket_results is not None and not bracket_results.empty:
        bracket_results.to_parquet(snapshot_dir / "bracket_results.parquet", index=False)

    turkey_probs = sim_results.get("turkey_probs", {})
    meta = {
        "snapshot_ts": snapshot_dir.name,
        "turkey_probs": turkey_probs,
        "calibration": calib_json,
    }
    (snapshot_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info(f"Snapshot written → {snapshot_dir}")


@_step("Update outputs")
def update_outputs(
    snapshot_dir: Path,
    predictions: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    sim_results: dict,
    fixtures: pd.DataFrame | None,
) -> None:
    out_dir = settings.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if predictions is not None and not predictions.empty:
        predictions.to_parquet(out_dir / "latest_predictions.parquet", index=False)

    turkey_probs = sim_results.get("turkey_probs", {})
    calib_data: dict = {"turkey_probs": turkey_probs}
    if calibration is not None and not calibration.empty:
        calib_data["mean_brier"] = float(calibration["brier_score"].mean())
        calib_data["n_matches"] = int(len(calibration))
    (out_dir / "calibration_latest.json").write_text(json.dumps(calib_data, indent=2, default=str))

    group_d_results = sim_results.get("group_d_results")
    if group_d_results is not None:
        group_d_results.to_parquet(out_dir / "turkey_path.parquet", index=False)

    bracket_results = sim_results.get("bracket_results")
    if bracket_results is not None and not bracket_results.empty:
        bracket_results.to_parquet(out_dir / "bracket_results.parquet", index=False)

    if fixtures is not None:
        fixtures.to_parquet(out_dir / "wc2026_fixtures.parquet", index=False)

    # Probability time series — append Turkey's advance prob to the running series
    ts_path = out_dir / "probability_timeseries.parquet"
    ts_row = pd.DataFrame(
        [
            {
                "snapshot_ts": snapshot_dir.name,
                "turkey_advance": turkey_probs.get("p_advance", float("nan")),
                "turkey_champion": turkey_probs.get("champion", float("nan")),
            }
        ]
    )
    if ts_path.exists():
        existing = pd.read_parquet(ts_path)
        existing = existing[existing["snapshot_ts"] != snapshot_dir.name]
        ts = pd.concat([existing, ts_row], ignore_index=True)
    else:
        ts = ts_row
    ts.to_parquet(ts_path, index=False)


# ────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ────────────────────────────────────────────────────────────────────────────


def run_update() -> None:
    snapshot_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = settings.snapshots_dir / snapshot_ts

    snaps_dir = settings.snapshots_dir
    snaps_dir.mkdir(parents=True, exist_ok=True)
    existing_snaps = sorted(snaps_dir.iterdir()) if snaps_dir.exists() else []
    last_snap = existing_snaps[-1] if existing_snaps else None

    out_dir = settings.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1–3: Fetch
    historical_raw = fetch_historical()
    fixtures_raw = fetch_wc2026_fixtures()
    _xg_raw = fetch_xg()  # noqa: F841 — reserved for Phase 3

    # 1.5: Fetch FIFA ranking (independent of other fetches — runs after to not block on failure)
    ranking_lookup = fetch_fifa_ranking()

    # 1.6: Build live odds fetcher (no-op if ODDS_API_KEY not set)
    from wc26.fetchers.odds_live import build_odds_fetcher
    odds_fetcher = build_odds_fetcher()

    # 4: Build processed tables
    historical, fixtures = build_processed_tables(historical_raw, fixtures_raw) or (None, None)

    # 5: Detect new results
    new_result_ids = detect_new_results(fixtures, last_snap)

    # 6: Refit Bayesian
    bayesian_model = refit_bayesian(historical)

    # 7: Build LightGBM feature matrix
    features_df, pred_features_df = build_lgbm_features(historical, fixtures, ranking_lookup) or (None, None)

    # 8: Refit LightGBM + ensemble (attach market model if odds fetcher available)
    ensemble_model = refit_lgbm_ensemble(bayesian_model, features_df, pred_features_df, odds_fetcher)

    # 9: Evaluate all models + save metrics
    evaluate_models(bayesian_model, ensemble_model, features_df, out_dir)

    # Choose best available model for predictions + simulations
    active_model = ensemble_model if ensemble_model is not None else bayesian_model

    # 10: Generate predictions
    predictions = generate_predictions(active_model, fixtures, snapshot_ts)

    # 11: Score past predictions
    calibration = score_past_predictions(fixtures, snaps_dir)

    # 12: Simulations
    sim_results = run_simulations(
        active_model,
        fixtures,
        n_simulations=settings.monte_carlo_iterations,
    ) or {}

    # 13: Write snapshot
    write_snapshot(snapshot_dir, predictions, calibration, sim_results)

    # 14: Update outputs
    update_outputs(snapshot_dir, predictions, calibration, sim_results, fixtures)

    # ── Summary ──────────────────────────────────────────────────────────────
    turkey_probs = sim_results.get("turkey_probs", {})
    advance_pct = turkey_probs.get("p_advance", float("nan"))
    champion_pct = turkey_probs.get("champion", float("nan"))
    mean_brier = (
        float(calibration["brier_score"].mean())
        if calibration is not None and not calibration.empty
        else float("nan")
    )

    def _fmt(v):
        return f"{v:.1%}" if not (v != v) else "—"  # nan check

    print(
        f"\n{'─'*60}\n"
        f"  Snapshot:                   {snapshot_ts}\n"
        f"  New results detected:       {len(new_result_ids)}\n"
        f"  Turkey advance (Group D):   {_fmt(advance_pct)}\n"
        f"  Turkey champion (bracket):  {_fmt(champion_pct)}\n"
        f"  Calibration Brier:          {mean_brier:.4f}\n"
        f"{'─'*60}\n"
    )


if __name__ == "__main__":
    run_update()
