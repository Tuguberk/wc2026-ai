"""Tests for fetcher modules using mocked HTTP responses."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Kaggle Historical fetcher ─────────────────────────────────────────────────


SAMPLE_CSV = """date,home_team,away_team,home_score,away_score,tournament,neutral
2020-01-01,Turkey,Germany,1,2,Friendly,False
2020-03-15,Brazil,Argentina,3,0,Friendly,True
2021-06-01,France,Spain,1,1,Nations League,False
"""


def test_kaggle_fetcher_github_fallback(tmp_path, monkeypatch):
    """KaggleHistoricalFetcher falls back to GitHub CSV when Kaggle is unavailable."""
    from wc26.config import settings
    from wc26.fetchers.kaggle_historical import KaggleHistoricalFetcher

    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = SAMPLE_CSV

    with patch("httpx.get", return_value=mock_resp), \
         patch.object(KaggleHistoricalFetcher, "_fetch_kaggle", side_effect=RuntimeError("no kaggle")):
        fetcher = KaggleHistoricalFetcher()
        df = fetcher._fetch()

    assert len(df) == 3
    assert "match_id" in df.columns
    assert "home_score" in df.columns


def test_kaggle_fetcher_cleans_types(tmp_path, monkeypatch):
    """Fetcher coerces scores to int and date to datetime."""
    from wc26.config import settings
    from wc26.fetchers.kaggle_historical import KaggleHistoricalFetcher

    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = SAMPLE_CSV

    with patch("httpx.get", return_value=mock_resp), \
         patch.object(KaggleHistoricalFetcher, "_fetch_kaggle", side_effect=RuntimeError("no kaggle")):
        fetcher = KaggleHistoricalFetcher()
        df = fetcher._fetch()

    assert pd.api.types.is_integer_dtype(df["home_score"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_kaggle_fetcher_deduplicates(tmp_path, monkeypatch):
    """Match IDs are unique (no duplicates in output)."""
    from wc26.config import settings
    from wc26.fetchers.kaggle_historical import KaggleHistoricalFetcher

    dup_csv = SAMPLE_CSV + "2020-01-01,Turkey,Germany,1,2,Friendly,False\n"
    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = dup_csv

    with patch("httpx.get", return_value=mock_resp), \
         patch.object(KaggleHistoricalFetcher, "_fetch_kaggle", side_effect=RuntimeError("no")):
        fetcher = KaggleHistoricalFetcher()
        df = fetcher._fetch()

    assert df["match_id"].nunique() == len(df)


# ── Football-data.org fetcher ─────────────────────────────────────────────────


SAMPLE_FD_RESPONSE = {
    "matches": [
        {
            "utcDate": "2026-06-11T18:00:00Z",
            "homeTeam": {"name": "Turkey"},
            "awayTeam": {"name": "United States"},
            "score": {"fullTime": {"home": None, "away": None}},
            "status": "SCHEDULED",
            "stage": "GROUP_STAGE",
            "group": "Group D",
        },
        {
            "utcDate": "2026-06-15T15:00:00Z",
            "homeTeam": {"name": "Australia"},
            "awayTeam": {"name": "Paraguay"},
            "score": {"fullTime": {"home": 1, "away": 0}},
            "status": "FINISHED",
            "stage": "GROUP_STAGE",
            "group": "Group D",
        },
    ]
}


def test_football_data_fetcher_parses_matches(tmp_path, monkeypatch):
    """FootballDataFetcher parses API response into correct DataFrame shape."""
    from wc26.config import settings
    from wc26.fetchers.footballdata import FootballDataFetcher

    monkeypatch.setattr(settings, "football_data_api_key", "fake_key")
    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw")

    with patch.object(FootballDataFetcher, "_get_matches", return_value=SAMPLE_FD_RESPONSE["matches"]):
        fetcher = FootballDataFetcher()
        df = fetcher._parse(SAMPLE_FD_RESPONSE["matches"])

    assert len(df) == 2
    assert "match_id" in df.columns
    scheduled = df[df["status"] == "SCHEDULED"]
    assert len(scheduled) == 1
    assert scheduled.iloc[0]["home_score"] is None or pd.isna(scheduled.iloc[0]["home_score"])


def test_football_data_fetcher_empty_response(tmp_path, monkeypatch):
    """FootballDataFetcher returns empty DataFrame on empty API response."""
    from wc26.fetchers.footballdata import FootballDataFetcher

    fetcher = FootballDataFetcher()
    df = fetcher._parse([])
    assert df.empty


# ── Wikipedia fetcher ─────────────────────────────────────────────────────────


def test_wikipedia_fetcher_graceful_on_no_tables(tmp_path, monkeypatch):
    """WikipediaFetcher returns empty DataFrame when no match tables are found."""
    from wc26.config import settings
    from wc26.fetchers.wikipedia import WikipediaFetcher

    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = "<html><body><p>No tables</p></body></html>"

    with patch("httpx.get", return_value=mock_resp), \
         patch("pandas.read_html", return_value=[]):
        fetcher = WikipediaFetcher()
        df = fetcher._fetch()

    assert df.empty
