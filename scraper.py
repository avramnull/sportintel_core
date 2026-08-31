#!/usr/bin/env python3
"""Download latest multi-league results CSV from football-data.co.uk."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from si_config import SAVE_DIR, season_code  # noqa: E402
from si_logging import get_logger  # noqa: E402

log = get_logger("scraper")

SEASON_CODE = season_code()
URL = f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/Latest_Results.csv"
USER_AGENT = "sportintel-core/1.0 (+https://github.com/avramnull/sportintel_core)"


def download_data():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out = SAVE_DIR / f"results_{today}.csv"
    stable = SAVE_DIR / "results_latest.csv"
    log.info("GET %s", URL)
    try:
        r = requests.get(
            URL,
            timeout=45,
            headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
        )
        r.raise_for_status()
        if len(r.content) < 100 or b"<html" in r.content[:200].lower():
            log.error("response does not look like a CSV (len=%d)", len(r.content))
            return None
        out.write_bytes(r.content)
        stable.write_bytes(r.content)
        log.info("saved %s and %s (%s bytes)", out.name, stable.name, f"{len(r.content):,}")
        return out
    except requests.RequestException as e:
        log.error("download failed: %s", e)
        return None


if __name__ == "__main__":
    path = download_data()
    raise SystemExit(0 if path else 1)
