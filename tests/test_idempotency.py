"""Idempotency tests: running the same data through build/process twice
produces identical results (no duplicate rows, stable match_ids).
"""

from __future__ import annotations

import pandas as pd
import pytest


SAMPLE_HISTORICAL = pd.DataFrame(
    {
        "date": pd.to_datetime(["2020-01-01", "2020-03-15", "2021-06-01"]),
        "home_team": ["Turkey", "Brazil", "France"],
        "away_team": ["Germany", "Argentina", "Spain"],
        "home_score": [1, 3, 1],
        "away_score": [2, 0, 1],
        "tournament": ["Friendly", "Friendly", "Nations League"],
        "neutral": [False, True, False],
    }
)

SAMPLE_FIXTURES = pd.DataFrame(
    {
        "date": pd.to_datetime(["2026-06-11", "2026-06-15"]),
        "home_team": ["Turkey", "Australia"],
        "away_team": ["United States", "Paraguay"],
        "home_score": [None, None],
        "away_score": [None, None],
        "stage": ["GROUP_STAGE", "GROUP_STAGE"],
        "group": ["Group D", "Group D"],
        "status": ["SCHEDULED", "SCHEDULED"],
    }
)


def test_build_historical_idempotent():
    """Running build_historical twice on the same data gives identical match_ids."""
    from wc26.data.build import build_historical

    df1 = build_historical(SAMPLE_HISTORICAL)
    df2 = build_historical(SAMPLE_HISTORICAL)

    assert set(df1["match_id"]) == set(df2["match_id"])
    assert len(df1) == len(df2)


def test_build_historical_no_duplicates():
    """Doubled input produces same number of rows as single input."""
    from wc26.data.build import build_historical

    doubled = pd.concat([SAMPLE_HISTORICAL, SAMPLE_HISTORICAL], ignore_index=True)
    df = build_historical(doubled)
    assert df["match_id"].nunique() == len(SAMPLE_HISTORICAL)
    assert len(df) == len(SAMPLE_HISTORICAL)


def test_build_fixtures_idempotent():
    """build_fixtures is idempotent when called twice on the same data."""
    from wc26.data.build import build_fixtures

    df1 = build_fixtures(SAMPLE_FIXTURES)
    df2 = build_fixtures(SAMPLE_FIXTURES)

    assert set(df1["match_id"]) == set(df2["match_id"])
    assert len(df1) == len(df2)


def test_build_fixtures_no_duplicates():
    """Doubled fixture input produces no duplicate match_ids."""
    from wc26.data.build import build_fixtures

    doubled = pd.concat([SAMPLE_FIXTURES, SAMPLE_FIXTURES], ignore_index=True)
    df = build_fixtures(doubled)
    assert df["match_id"].nunique() == len(SAMPLE_FIXTURES)


def test_match_id_deterministic():
    """make_match_id is deterministic: same inputs always produce same id."""
    from wc26.schemas import make_match_id

    id1 = make_match_id("2026-06-11", "Turkey", "United States")
    id2 = make_match_id("2026-06-11", "Turkey", "United States")
    assert id1 == id2


def test_match_id_normalizes_unicode():
    """make_match_id normalizes unicode (e.g. accented characters)."""
    from wc26.schemas import make_match_id

    id_ascii = make_match_id("2026-06-11", "Cote d'Ivoire", "Brazil")
    id_unicode = make_match_id("2026-06-11", "Côte d'Ivoire", "Brazil")
    # Both should produce the same ID after ASCII normalization
    assert id_ascii == id_unicode


def test_split_results_scheduled():
    """split_results_scheduled correctly separates finished and upcoming matches."""
    from wc26.data.build import split_results_scheduled

    mixed = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "status": ["FINISHED", "SCHEDULED", "FINISHED"],
            "home_score": [1.0, None, 2.0],
            "away_score": [0.0, None, 1.0],
        }
    )
    results, scheduled = split_results_scheduled(mixed)
    assert len(results) == 2
    assert len(scheduled) == 1
    assert scheduled.iloc[0]["match_id"] == "m2"
