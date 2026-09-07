#!/usr/bin/env python3
"""Download latest multi-league results CSV from football-data.co.uk.

Industrial version:
  - Retries with exponential backoff + jitter
  - Long read timeout (site is often slow)
  - Falls back to most recent local results_*.csv if download fails
  - Always keeps results_latest.csv stable pointer when possible
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from si_config import SAVE_DIR, season_code  # noqa: E402
from si_http import download_to_files  # noqa: E402
from si_logging import get_logger  # noqa: E402

log = get_logger("scraper")

SEASON_CODE = season_code()
URL = f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/Latest_Results.csv"


def _newest_local() -> Path | None:
    candidates = sorted(SAVE_DIR.glob("results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stat().st_size >= 200:
            return p
    return None


def _write_livescore_api_fallback() -> Path | None:
    """Preferred fallback: API-Football covers every configured league (not
    just the ones openfootball happens to have a current-season file for)
    and updates same-day. Only used when a key is actually configured —
    falls through to the openfootball fallback otherwise."""
    try:
        import pandas as pd
        from api_football_client import ApiFootballClient
        from eur_livescore import fetch_results
        client = ApiFootballClient(purpose="fixtures")
        if not client.available:
            return None
        rows = fetch_results(client)
        if not rows:
            log.warning("livescore API returned no results — falling back further")
            return None
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        out = SAVE_DIR / f"results_{today}.csv"
        stable = SAVE_DIR / "results_latest.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        df.to_csv(stable, index=False)
        log.warning("football-data.co.uk unavailable — using livescore API fallback (%d rows)", len(df))
        return stable
    except Exception as e:
        log.error("livescore API fallback failed: %s", e)
        return None


def _write_openfootball_fallback() -> Path | None:
    """football-data.co.uk is down. Build a results_latest.csv-shaped CSV
    from openfootball instead, so daily_update_parquet.py needs no changes
    at all — it just reads whatever CSV is sitting at this path."""
    try:
        import pandas as pd
        from eur_openfootball import fetch_all_results
        rows = fetch_all_results()
        if not rows:
            log.error("openfootball fallback produced no rows")
            return None
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        out = SAVE_DIR / f"results_{today}.csv"
        stable = SAVE_DIR / "results_latest.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        df.to_csv(stable, index=False)
        log.warning("football-data.co.uk unavailable — using openfootball fallback (%d rows)", len(df))
        return stable
    except Exception as e:
        log.error("openfootball fallback failed: %s", e)
        return None


def download_data() -> Path | None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out = SAVE_DIR / f"results_{today}.csv"
    stable = SAVE_DIR / "results_latest.csv"

    path = download_to_files(
        URL,
        out,
        stable_path=stable,
        timeout=(25, 150),
        max_attempts=7,
        min_bytes=200,
        expect_csv=True,
    )
    if path is not None:
        return path

    fallback = _write_livescore_api_fallback()
    if fallback is not None:
        return fallback

    fallback = _write_openfootball_fallback()
    if fallback is not None:
        return fallback

    # Soft fallback — do not hard-fail the whole pipeline on a flaky source
    local = _newest_local()
    if local is not None:
        log.warning(
            "download failed — reusing local %s (%s bytes) as results_latest",
            local.name,
            f"{local.stat().st_size:,}",
        )
        try:
            if not stable.exists() or stable.stat().st_mtime < local.stat().st_mtime:
                shutil.copy2(local, stable)
            # also copy to today's name so downstream sees a "fresh" path
            if not out.exists():
                shutil.copy2(local, out)
            return stable if stable.exists() else local
        except OSError as e:
            log.error("fallback copy failed: %s", e)
            return local
    log.error("no download and no usable local results_*.csv")
    return None


if __name__ == "__main__":
    path = download_data()
    # Exit 0 on soft-fallback so pipeline can continue with historical data
    raise SystemExit(0 if path else 1)
