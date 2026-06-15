"""Application configuration via pydantic-settings + .env."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kaggle
    kaggle_username: str = ""
    kaggle_key: str = ""

    # football-data.org (primary WC2026 fixtures)
    football_data_api_key: str = ""

    # API-Football fallback
    api_football_key: str = ""

    # The Odds API (https://the-odds-api.com) — for live WC2026 market odds
    odds_api_key: str = ""

    # Fixture source selection
    primary_fixture_source: Literal["football_data", "api_football", "wikipedia"] = "football_data"

    # Feature flags
    enable_xg: bool = False

    # PyMC MCMC
    mcmc_draws: int = 1000
    mcmc_tune: int = 500
    mcmc_chains: int = 2

    # Monte Carlo
    monte_carlo_iterations: int = 10_000        # group stage sim (fast)
    bracket_mc_iterations: int = 25_000         # full tournament bracket sim

    # Logging
    log_level: str = "INFO"

    # Data paths
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"


settings = Settings()
