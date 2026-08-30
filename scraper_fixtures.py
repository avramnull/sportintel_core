#!/usr/bin/env python3
"""Download upcoming fixtures CSV from football-data.co.uk."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import requests

URL = "https://www.football-data.co.uk/fixtures.csv"
SAVE_DIR = Path(__file__).resolve().parent / "daily_football_data"


def download_fixtures():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out = SAVE_DIR / f"fixtures_{today}.csv"
    # also keep a stable name for the pipeline
    stable = SAVE_DIR / "fixtures_latest.csv"
    print(f"GET {URL}")
    try:
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        if len(r.content) < 100 or b"<html" in r.content[:200].lower():
            print("ERROR: response does not look like a CSV")
            return None
        out.write_bytes(r.content)
        stable.write_bytes(r.content)
        print(f"Saved {out} and {stable} ({len(r.content):,} bytes)")
        return stable
    except Exception as e:
        print(f"Fixtures download failed: {e}")
        return None


if __name__ == "__main__":
    path = download_fixtures()
    raise SystemExit(0 if path else 1)
