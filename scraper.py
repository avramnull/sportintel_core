#!/usr/bin/env python3
"""Download latest multi-league results CSV from football-data.co.uk."""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
import requests

SEASON_CODE = os.environ.get("FOOTBALL_SEASON", "2627")
URL = f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/Latest_Results.csv"
SAVE_DIR = Path(__file__).resolve().parent / "daily_football_data"

def download_data():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out = SAVE_DIR / f"results_{today}.csv"
    print(f"GET {URL}")
    try:
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        if len(r.content) < 100 or b"<html" in r.content[:200].lower():
            print("ERROR: response does not look like a CSV")
            return None
        out.write_bytes(r.content)
        print(f"Saved {out} ({len(r.content):,} bytes)")
        return out
    except Exception as e:
        print(f"Download failed: {e}")
        return None

if __name__ == "__main__":
    path = download_data()
    raise SystemExit(0 if path else 1)
