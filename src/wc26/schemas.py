"""Data schemas for validation (pandera DataFrameModel + column contracts)."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import DateTime, Series


class HistoricalMatchSchema(pa.DataFrameModel):
    """Schema for the historical international matches dataset."""

    date: Series[DateTime]
    home_team: Series[str] = pa.Field(nullable=False)
    away_team: Series[str] = pa.Field(nullable=False)
    home_score: Series[int] = pa.Field(ge=0, nullable=False)
    away_score: Series[int] = pa.Field(ge=0, nullable=False)
    tournament: Series[str] = pa.Field(nullable=False)
    neutral: Series[bool] = pa.Field(nullable=False)
    match_id: Series[str] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = False


class WC2026FixtureSchema(pa.DataFrameModel):
    """Schema for WC2026 fixture + result data."""

    match_id: Series[str] = pa.Field(nullable=False)
    date: Series[DateTime]
    home_team: Series[str] = pa.Field(nullable=False)
    away_team: Series[str] = pa.Field(nullable=False)
    home_score: Series[float] = pa.Field(ge=0, nullable=True)  # null = not played yet
    away_score: Series[float] = pa.Field(ge=0, nullable=True)
    stage: Series[str] = pa.Field(nullable=False)  # "Group", "Round of 32", etc.
    group: Series[str] = pa.Field(nullable=True)   # only for group stage
    status: Series[str] = pa.Field(nullable=False)  # "SCHEDULED" | "FINISHED" | "IN_PLAY"

    class Config:
        coerce = True
        strict = False


class PredictionSchema(pa.DataFrameModel):
    """Schema for model output predictions."""

    match_id: Series[str] = pa.Field(nullable=False)
    snapshot_ts: Series[str] = pa.Field(nullable=False)
    home_team: Series[str] = pa.Field(nullable=False)
    away_team: Series[str] = pa.Field(nullable=False)
    p_home: Series[float] = pa.Field(ge=0.0, le=1.0)
    p_draw: Series[float] = pa.Field(ge=0.0, le=1.0)
    p_away: Series[float] = pa.Field(ge=0.0, le=1.0)
    exp_home_goals: Series[float] = pa.Field(ge=0.0, nullable=True)
    exp_away_goals: Series[float] = pa.Field(ge=0.0, nullable=True)

    class Config:
        coerce = True
        strict = False


class CalibrationRecord(pa.DataFrameModel):
    """Per-match calibration tracking: pre-match prediction vs actual result."""

    match_id: Series[str] = pa.Field(nullable=False)
    snapshot_ts: Series[str] = pa.Field(nullable=False)  # snapshot BEFORE the match
    p_home: Series[float] = pa.Field(ge=0.0, le=1.0)
    p_draw: Series[float] = pa.Field(ge=0.0, le=1.0)
    p_away: Series[float] = pa.Field(ge=0.0, le=1.0)
    actual_outcome: Series[str] = pa.Field(isin=["H", "D", "A"])
    brier_score: Series[float] = pa.Field(ge=0.0, le=2.0)

    class Config:
        coerce = True
        strict = False


def make_match_id(date: str, home: str, away: str) -> str:
    """Produce a deterministic, source-agnostic match identifier."""
    import re
    import unicodedata

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-z0-9]", "_", s.lower())
        return re.sub(r"_+", "_", s).strip("_")

    return f"{date}_{normalize(home)}_{normalize(away)}"
