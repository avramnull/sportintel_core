#!/usr/bin/env python3
"""
Central configuration for sportintel_core.

Single source of truth for paths, season codes, pipeline caps, and feature flags.
Env vars always win over defaults so CI and local runs stay consistent.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "daily_football_data"


def _load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env loader (no dependency). Never overrides existing env."""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        # Local .env is convenience-only; configuration from the process wins.
        pass


_load_dotenv()
SIMS_DIR = SAVE_DIR / "sims"
PARQUET_PATH = ROOT / "master_football_data.parquet"
MODELS_DIR = ROOT / "football_models"
MAP_DIR = MODELS_DIR / "mappings"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def infer_season(today: Optional[date] = None) -> Tuple[str, str]:
    """Infer the football-data.co.uk season from the UTC calendar date."""
    d = today or datetime.now(timezone.utc).date()
    start_year = d.year if d.month >= 7 else d.year - 1
    end_year = start_year + 1
    code = f"{start_year % 100:02d}{end_year % 100:02d}"
    label = f"{start_year}/{end_year}"
    return code, label


def season_code() -> str:
    explicit = _env("FOOTBALL_SEASON")
    if explicit:
        return explicit
    code, _ = infer_season()
    return code


def season_label() -> str:
    explicit = _env("FOOTBALL_SEASON_LABEL")
    if explicit:
        return explicit
    _, label = infer_season()
    return label


# Production defaults. Zero is reserved for explicit "unlimited" behaviour;
# daily CI uses a finite cap to prevent an unexpectedly huge training run.
MAX_TRAIN_TEAMS = _env_int("MAX_TRAIN_TEAMS", 12)
MIN_TEAM_MATCHES = _env_int("MIN_TEAM_MATCHES", 25)
N_SIMULATIONS = _env_int("N_SIMULATIONS", 3000)
ODDS_BLEND = _env_float("ODDS_BLEND", 0.30)
MAX_FIXTURES = _env_int("MAX_FIXTURES", 0)  # 0 = all
TODAY_ONLY = _env_bool("TODAY_ONLY", True)
DAILY_LIGHT = _env_bool("DAILY_LIGHT", True)
SKIP_TRAIN = _env_bool("SKIP_TRAIN", False)
SKIP_SIM = _env_bool("SKIP_SIM", False)
SPORTINTEL_PUSH = _env_bool("SPORTINTEL_PUSH", False)
TRAIN_WORKERS = _env_int("TRAIN_WORKERS", 3)

SPORTINTEL_REPO = _env("SPORTINTEL_REPO", "avramnull/sportintel")
GITHUB_TOKEN = (
    _env("GITHUB_TOKEN")
    or _env("GH_TOKEN")
    or _env("SPORTINTEL_TOKEN")
    or _env("SPORTINTEL_PUSH_TOKEN")
)

# Logging
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
LOG_JSON = _env_bool("LOG_JSON", False)

# API-Football (fixtures + optional standings).
API_FOOTBALL_KEY = _env("API_FOOTBALL_KEY")
API_FOOTBALL_STANDINGS_KEY = _env("API_FOOTBALL_STANDINGS_KEY")
