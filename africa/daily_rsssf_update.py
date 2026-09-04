#!/usr/bin/env python3
"""
Clean daily RSSSF → master_africa_football.parquet updater.

- Levels 1 + 2 only
- Strict team identity (no fuzzy)
- Incremental append only
- Precise FT results; HT left null (RSSSF almost never supplies HT)
- No noise, no stories
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from africa.parse_rsssf import fetch_rsssf_levels_1_2  # noqa: E402
from africa.team_mapping import ensure_ids, resolve, clean  # noqa: E402

PARQUET = Path(os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")))


def match_key(row) -> str:
    return "|".join([
        str(row.get("Div", "")),
        str(row.get("Date", ""))[:10],
        clean(str(row.get("HomeTeam", ""))).lower(),
        clean(str(row.get("AwayTeam", ""))).lower(),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily clean RSSSF Africa parquet updater")
    ap.add_argument("--days-back", type=int, default=21)
    ap.add_argument("--out", type=Path, default=PARQUET)
    ap.add_argument("--rsssf-delay", type=float, default=0.4)
    ap.add_argument("--max-pages", type=int, default=150)
    args = ap.parse_args()

    today = date.today()
    year = today.year

    print(f"[daily] fetching RSSSF L1+L2 (years {year-1}–{year+1})")
    rows = fetch_rsssf_levels_1_2(
        year_start=year - 1,
        year_end=year + 1,
        delay=args.rsssf_delay,
        max_pages=args.max_pages,
        probe_years=True,
    )
    if not rows:
        print("[daily] zero rows from RSSSF")
        return

    new = pd.DataFrame(rows)
    new["Date"] = pd.to_datetime(new["Date"], errors="coerce")
    cutoff = pd.Timestamp(today - timedelta(days=args.days_back))
    new = new[new["Date"] >= cutoff].copy()
    if new.empty:
        print("[daily] nothing inside window")
        return

    # Strict identity
    all_names = set(new["HomeTeam"].astype(str)) | set(new["AwayTeam"].astype(str))
    ids = ensure_ids(all_names)
    new["HomeTeam"] = new["HomeTeam"].map(lambda x: resolve(x)[0])
    new["AwayTeam"] = new["AwayTeam"].map(lambda x: resolve(x)[0])
    new["HomeTeamId"] = new["HomeTeam"].map(ids)
    new["AwayTeamId"] = new["AwayTeam"].map(ids)

    # Precise FT only
    new["FTHG"] = pd.to_numeric(new["FTHG"], errors="coerce")
    new["FTAG"] = pd.to_numeric(new["FTAG"], errors="coerce")
    new = new.dropna(subset=["FTHG", "FTAG", "Date", "HomeTeam", "AwayTeam"])
    new = new[(new["FTHG"] >= 0) & (new["FTAG"] >= 0) & (new["FTHG"] <= 15) & (new["FTAG"] <= 15)]
    new["FTHG"] = new["FTHG"].astype(int)
    new["FTAG"] = new["FTAG"].astype(int)
    new["FTR"] = np.where(new["FTHG"] > new["FTAG"], "H",
                  np.where(new["FTHG"] < new["FTAG"], "A", "D"))
    new["HTHG"] = None
    new["HTAG"] = None
    new["TotalGoals"] = new["FTHG"] + new["FTAG"]
    new["Over2_5"] = (new["TotalGoals"] > 2.5).astype(int)
    new["BTTS"] = ((new["FTHG"] > 0) & (new["FTAG"] > 0)).astype(int)
    new["_key"] = new.apply(match_key, axis=1)

    # Load existing master (if any) and append only novel keys
    if args.out.exists():
        try:
            old = pd.read_parquet(args.out)
        except Exception:
            old = pd.read_csv(args.out.with_suffix(".csv")) if args.out.with_suffix(".csv").exists() else pd.DataFrame()
        if len(old):
            old["_key"] = old.apply(match_key, axis=1)
            existing = set(old["_key"])
            novel = new[~new["_key"].isin(existing)].copy()
            print(f"[daily] existing={len(old)}  novel={len(novel)}")
            if novel.empty:
                print("[daily] no new matches to append")
                return
            combined = pd.concat([old.drop(columns=["_key"], errors="ignore"), novel.drop(columns=["_key"])], ignore_index=True)
        else:
            combined = new.drop(columns=["_key"])
    else:
        combined = new.drop(columns=["_key"])
        print(f"[daily] created new master with {len(combined)} rows")

    combined = combined.sort_values("Date").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_parquet(args.out, index=False, compression="zstd")
        print(f"[ok] wrote {args.out}  rows={len(combined)}")
    except Exception as e:
        csv_path = args.out.with_suffix(".csv")
        combined.to_csv(csv_path, index=False)
        print(f"[warn] parquet failed ({e}) — wrote {csv_path}")


if __name__ == "__main__":
    main()
