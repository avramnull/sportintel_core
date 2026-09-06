#!/usr/bin/env python3
"""Download upcoming fixtures CSV from football-data.co.uk.

Industrial version with retries, long timeouts, and local fallback.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from si_config import SAVE_DIR  # noqa: E402
from si_http import download_to_files  # noqa: E402
from si_logging import get_logger  # noqa: E402

log = get_logger("scraper_fixtures")

URL = "https://www.football-data.co.uk/fixtures.csv"


def _newest_local() -> Path | None:
    candidates = sorted(SAVE_DIR.glob("fixtures_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stat().st_size >= 100:
            return p
    return None


def _write_openfootball_fallback() -> Path | None:
    """football-data.co.uk is down. Build a fixtures_latest.csv-shaped CSV
    from openfootball's upcoming (unplayed) matches instead. No odds
    columns — run_fixture_sims.py already treats a fixture with no market
    data as pure-model (odds_blend=0) rather than dropping it."""
    try:
        import pandas as pd
        from eur_openfootball import fetch_all_upcoming_fixtures
        rows = fetch_all_upcoming_fixtures()
        if not rows:
            log.error("openfootball fixtures fallback produced no rows")
            return None
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        out = SAVE_DIR / f"fixtures_{today}.csv"
        stable = SAVE_DIR / "fixtures_latest.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        df.to_csv(stable, index=False)
        log.warning("football-data.co.uk unavailable — using openfootball fixtures fallback (%d rows)", len(df))
        return stable
    except Exception as e:
        log.error("openfootball fixtures fallback failed: %s", e)
        return None


def download_fixtures() -> Path | None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out = SAVE_DIR / f"fixtures_{today}.csv"
    stable = SAVE_DIR / "fixtures_latest.csv"

    path = download_to_files(
        URL,
        out,
        stable_path=stable,
        timeout=(25, 150),
        max_attempts=7,
        min_bytes=100,
        expect_csv=True,
    )
    if path is not None:
        return path

    fallback = _write_openfootball_fallback()
    if fallback is not None:
        return fallback

    local = _newest_local()
    if local is not None:
        log.warning(
            "fixtures download failed — reusing local %s (%s bytes)",
            local.name,
            f"{local.stat().st_size:,}",
        )
        try:
            if not stable.exists() or stable.stat().st_mtime < local.stat().st_mtime:
                shutil.copy2(local, stable)
            if not out.exists():
                shutil.copy2(local, out)
            return stable if stable.exists() else local
        except OSError as e:
            log.error("fallback copy failed: %s", e)
            return local
    log.error("no download and no usable local fixtures_*.csv")
    return None


if __name__ == "__main__":
    path = download_fixtures()
    raise SystemExit(0 if path else 1)
