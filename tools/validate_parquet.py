#!/usr/bin/env python3
"""
Validate master_football_data.parquet for clean model development.

Checks:
  - LFS pointer / size
  - required columns + dtypes
  - null teams / negative scores / FTR consistency
  - duplicate match keys
  - odds sanity ( > 1.01 when present )
  - date range coverage
  - per-division volume
Exit 0 = clean enough to train; 1 = blocking issues.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from si_config import PARQUET_PATH  # noqa: E402


REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
RECOMMENDED = [
    "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A",
    "HTHG", "HTAG", "HTR", "Season", "SeasonCode", "Over2_5", "BTTS",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=PARQUET_PATH)
    ap.add_argument("--max-dup-rate", type=float, default=0.002)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    path: Path = args.path

    errors = []
    warns = []

    if not path.exists():
        print(f"[fail] missing {path}")
        sys.exit(1)
    head = path.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        print("[fail] still an LFS pointer — git lfs pull required")
        sys.exit(1)
    size = path.stat().st_size
    if size < 2_000_000:
        errors.append(f"parquet too small ({size} bytes)")

    print(f"Loading {path} ({size/1e6:.1f} MB)…")
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
    rec_miss = [c for c in RECOMMENDED if c not in df.columns]
    if rec_miss:
        warns.append(f"missing recommended columns: {rec_miss}")

    if missing:
        for e in errors:
            print("[fail]", e)
        sys.exit(1)

    # Null teams
    null_teams = int(df["HomeTeam"].isna().sum() + df["AwayTeam"].isna().sum())
    empty_teams = int(
        (df["HomeTeam"].astype(str).str.strip().isin(["", "nan", "None"])).sum()
        + (df["AwayTeam"].astype(str).str.strip().isin(["", "nan", "None"])).sum()
    )
    if null_teams or empty_teams:
        errors.append(f"null/empty team names: null={null_teams} empty={empty_teams}")

    # Scores
    fthg = pd.to_numeric(df["FTHG"], errors="coerce")
    ftag = pd.to_numeric(df["FTAG"], errors="coerce")
    neg = int(((fthg < 0) | (ftag < 0)).sum())
    if neg:
        errors.append(f"negative scores: {neg}")
    null_scores = int(fthg.isna().sum() + ftag.isna().sum())
    if null_scores:
        warns.append(f"null scores: {null_scores}")

    # FTR consistency
    computed = np.where(fthg > ftag, "H", np.where(fthg < ftag, "A", "D"))
    ftr = df["FTR"].astype(str).str.strip().str.upper()
    valid_mask = fthg.notna() & ftag.notna() & ftr.isin(["H", "D", "A"])
    bad_ftr = int((valid_mask & (ftr.values != computed)).sum())
    if bad_ftr:
        errors.append(f"FTR inconsistent with scores: {bad_ftr}")

    # Dates
    dates = pd.to_datetime(df["Date"], errors="coerce")
    null_dates = int(dates.isna().sum())
    if null_dates:
        errors.append(f"null dates: {null_dates}")
    else:
        print(f"Date range: {dates.min().date()} → {dates.max().date()}")

    # Match key duplicates
    key = (
        df["Div"].astype(str) + "|"
        + dates.dt.strftime("%Y-%m-%d") + "|"
        + df["HomeTeam"].astype(str) + "|"
        + df["AwayTeam"].astype(str)
    )
    dup = int(key.duplicated().sum())
    dup_rate = dup / max(len(df), 1)
    print(f"Duplicate match keys: {dup} ({dup_rate:.4%})")
    if dup_rate > args.max_dup_rate:
        errors.append(f"duplicate key rate {dup_rate:.4%} > {args.max_dup_rate}")

    # Odds sanity
    for col in ("AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = int(((s.notna()) & (s <= 1.0)).sum())
        if bad:
            warns.append(f"{col}: {bad} values <= 1.0")

    # Volume by division
    if "Div" in df.columns:
        vc = df["Div"].value_counts()
        print("Top divisions:")
        for d, c in vc.head(12).items():
            print(f"  {d}: {c:,}")
        thin = vc[vc < 200]
        if len(thin):
            warns.append(f"thin divisions (<200 rows): {dict(thin)}")

    # Same-team home=away
    same = int((df["HomeTeam"].astype(str) == df["AwayTeam"].astype(str)).sum())
    if same:
        errors.append(f"home==away rows: {same}")

    print("---")
    for w in warns:
        print("[warn]", w)
    for e in errors:
        print("[fail]", e)

    if errors:
        print(f"RESULT: FAIL ({len(errors)} blocking, {len(warns)} warnings)")
        sys.exit(1)
    print(f"RESULT: PASS ({len(warns)} warnings)")
    sys.exit(0)


if __name__ == "__main__":
    main()
