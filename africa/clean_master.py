#!/usr/bin/env python3
"""Normalize / validate master_africa_football.{parquet,csv}."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "Country", "League", "Div", "Season", "Source",
]


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED:
        if c not in df.columns:
            if c in ("FTHG", "FTAG"):
                df[c] = np.nan
            elif c == "FTR":
                df[c] = ""
            else:
                df[c] = ""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df = df[(df["HomeTeam"] != "") & (df["AwayTeam"] != "")]
    df = df[df["HomeTeam"].str.lower() != df["AwayTeam"].str.lower()]
    df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0) & (df["FTHG"] <= 15) & (df["FTAG"] <= 15)]
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    ftr = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    bad = ~df["FTR"].astype(str).str.upper().isin(["H", "D", "A"])
    df.loc[bad, "FTR"] = ftr[bad]
    df["FTR"] = df["FTR"].astype(str).str.upper()
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["Over2_5"] = (df["TotalGoals"] > 2.5).astype(int)
    df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    # stable key + dedupe prefer openfootball
    df["_key"] = (
        df["Div"].astype(str) + "|" +
        df["Date"].dt.strftime("%Y-%m-%d") + "|" +
        df["HomeTeam"].str.lower() + "|" +
        df["AwayTeam"].str.lower()
    )
    df["_prio"] = (df["Source"].astype(str) == "openfootball").astype(int)
    df = df.sort_values(["_key", "_prio"], ascending=[True, False]).drop_duplicates("_key", keep="first")
    df = df.drop(columns=["_key", "_prio"])
    df = df.sort_values(["Date", "Country", "HomeTeam"]).reset_index(drop=True)
    # team ids if missing
    if "HomeTeamId" not in df.columns or df["HomeTeamId"].isna().all():
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        t2 = {t: i for i, t in enumerate(teams)}
        df["HomeTeamId"] = df["HomeTeam"].map(t2)
        df["AwayTeamId"] = df["AwayTeam"].map(t2)
    return df


def main():
    candidates = [
        ROOT / "master_africa_football.parquet",
        ROOT / "master_africa_football.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        print("No master_africa file found — skip clean")
        return 0
    print(f"Cleaning {src} ({src.stat().st_size/1e6:.2f} MB)")
    df = clean(load_any(src))
    print(f"  rows={len(df)} countries={df['Country'].nunique()} seasons={df['Season'].nunique()}")
    print(df.groupby("Country").size().sort_values(ascending=False).head(12).to_string())
    out_csv = ROOT / "master_africa_football.csv"
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")
    out_pq = ROOT / "master_africa_football.parquet"
    try:
        df.to_parquet(out_pq, index=False, compression="zstd")
        print(f"  wrote {out_pq}")
    except Exception as e:
        print(f"  parquet skip: {e}")
    meta = {
        "rows": int(len(df)),
        "countries": int(df["Country"].nunique()),
        "seasons": sorted(df["Season"].dropna().astype(str).unique().tolist())[:40],
        "date_min": str(df["Date"].min().date()) if len(df) else None,
        "date_max": str(df["Date"].max().date()) if len(df) else None,
        "sources": df["Source"].value_counts().to_dict() if "Source" in df.columns else {},
    }
    (ROOT / "football_models" / "mappings").mkdir(parents=True, exist_ok=True)
    (ROOT / "football_models" / "mappings" / "africa_master_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("  meta OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
