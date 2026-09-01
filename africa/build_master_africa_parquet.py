#!/usr/bin/env python3
"""
Build master_africa_football.parquet from:
  1) openfootball/world africa/*.txt  (primary, clean)
  2) RSSSF Africa first-level pages   (supplementary)

Usage:
  # clone openfootball next to repo or pass --openfootball-root
  git clone --depth 1 https://github.com/openfootball/world.git data/africa_raw/openfootball_world

  python -m africa.build_master_africa_parquet \
    --openfootball-root data/africa_raw/openfootball_world \
    --fetch-rsssf \
    --out master_africa_football.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from africa.parse_football_txt import parse_openfootball_tree  # noqa: E402
from africa.parse_rsssf import fetch_rsssf_priority  # noqa: E402


def match_key(row) -> str:
    return "|".join([
        str(row.get("Div", "")),
        str(row.get("Date", ""))[:10],
        str(row.get("HomeTeam", "")).strip().lower(),
        str(row.get("AwayTeam", "")).strip().lower(),
    ])


def dedupe_prefer_openfootball(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_key"] = df.apply(match_key, axis=1)
    df["_prio"] = (df["Source"] == "openfootball").astype(int)
    df = df.sort_values(["_key", "_prio"], ascending=[True, False])
    df = df.drop_duplicates("_key", keep="first")
    return df.drop(columns=["_key", "_prio"])


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0) & (df["FTHG"] <= 20) & (df["FTAG"] <= 20)]
    df = df[df["HomeTeam"].astype(str).str.strip() != df["AwayTeam"].astype(str).str.strip()]

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    computed = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    df["FTR"] = np.where(df["FTR"].astype(str).str.upper().isin(["H", "D", "A"]), df["FTR"].astype(str).str.upper(), computed)

    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["GoalDiff"] = df["FTHG"] - df["FTAG"]
    df["Over2_5"] = (df["TotalGoals"] > 2.5).astype(int)
    df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["HomeWin"] = (df["FTR"] == "H").astype(int)
    df["Draw"] = (df["FTR"] == "D").astype(int)
    df["AwayWin"] = (df["FTR"] == "A").astype(int)
    df["HomePoints"] = df["FTR"].map({"H": 3, "D": 1, "A": 0}).astype(int)
    df["AwayPoints"] = df["FTR"].map({"H": 0, "D": 1, "A": 3}).astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Stable team ids within Africa corpus
    teams = sorted(set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str)))
    team2id = {t: i for i, t in enumerate(teams)}
    df["HomeTeamId"] = df["HomeTeam"].map(team2id)
    df["AwayTeamId"] = df["AwayTeam"].map(team2id)

    # Leak-free form + Elo (same philosophy as train.engineer)
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    hist_pts = defaultdict(list)
    hist_gf = defaultdict(list)
    hist_ga = defaultdict(list)
    elo = {}
    K, HOME_ADV = 20.0, 55.0

    for w in (5, 10):
        df[f"HomeFormPts_{w}"] = np.nan
        df[f"AwayFormPts_{w}"] = np.nan
        df[f"HomeFormGD_{w}"] = np.nan
        df[f"AwayFormGD_{w}"] = np.nan
    df["EloHome"] = 1500.0
    df["EloAway"] = 1500.0
    df["EloDiff"] = 0.0

    for i, row in df.iterrows():
        hid, aid = int(row["HomeTeamId"]), int(row["AwayTeamId"])
        rh, ra = elo.get(hid, 1500.0), elo.get(aid, 1500.0)
        df.at[i, "EloHome"] = rh
        df.at[i, "EloAway"] = ra
        df.at[i, "EloDiff"] = rh - ra
        for w in (5, 10):
            if hist_pts[hid]:
                df.at[i, f"HomeFormPts_{w}"] = float(np.mean(hist_pts[hid][-w:]))
                df.at[i, f"HomeFormGD_{w}"] = float(np.mean(np.array(hist_gf[hid][-w:]) - np.array(hist_ga[hid][-w:])))
            if hist_pts[aid]:
                df.at[i, f"AwayFormPts_{w}"] = float(np.mean(hist_pts[aid][-w:]))
                df.at[i, f"AwayFormGD_{w}"] = float(np.mean(np.array(hist_gf[aid][-w:]) - np.array(hist_ga[aid][-w:])))

        # update
        ftr = row["FTR"]
        pts_h = 3 if ftr == "H" else (1 if ftr == "D" else 0)
        pts_a = 3 if ftr == "A" else (1 if ftr == "D" else 0)
        hist_pts[hid].append(pts_h)
        hist_gf[hid].append(int(row["FTHG"]))
        hist_ga[hid].append(int(row["FTAG"]))
        hist_pts[aid].append(pts_a)
        hist_gf[aid].append(int(row["FTAG"]))
        hist_ga[aid].append(int(row["FTHG"]))
        score_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        elo[hid] = rh + K * (score_h - exp_h)
        elo[aid] = ra + K * ((1.0 - score_h) - (1.0 - exp_h))

    df["MatchKey"] = df.apply(match_key, axis=1)
    return df


def main():
    ap = argparse.ArgumentParser(description="Build master_africa_football.parquet")
    ap.add_argument("--openfootball-root", type=Path, default=ROOT / "data" / "africa_raw" / "openfootball_world")
    ap.add_argument("--fetch-rsssf", action="store_true", help="Also scrape RSSSF first-level pages")
    ap.add_argument("--rsssf-max-pages", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "master_africa_football.parquet")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional CSV export")
    args = ap.parse_args()

    frames = []

    of_root = args.openfootball_root
    if of_root.exists():
        print(f"[openfootball] parsing tree under {of_root}")
        rows = parse_openfootball_tree(of_root)
        print(f"[openfootball] {len(rows)} raw matches")
        if rows:
            frames.append(pd.DataFrame(rows))
    else:
        print(f"[openfootball] root missing: {of_root}")
        print("  Clone with:")
        print("  git clone --depth 1 https://github.com/openfootball/world.git data/africa_raw/openfootball_world")

    if args.fetch_rsssf:
        print("[rsssf] fetching priority first-level pages…")
        try:
            rrows = fetch_rsssf_priority(max_pages=args.rsssf_max_pages)
            print(f"[rsssf] {len(rrows)} raw matches")
            if rrows:
                frames.append(pd.DataFrame(rrows))
        except Exception as e:
            print(f"[rsssf] aborted: {e}")

    if not frames:
        raise SystemExit("No data collected — check openfootball root and/or --fetch-rsssf")

    raw = pd.concat(frames, ignore_index=True)
    print(f"[merge] combined raw rows: {len(raw)}")
    clean = dedupe_prefer_openfootball(raw)
    print(f"[merge] after dedupe: {len(clean)}")
    feat = feature_engineer(clean)
    print(f"[features] final rows: {len(feat)} | countries: {feat['Country'].nunique()} | seasons: {feat['Season'].nunique()}")
    print(feat.groupby("Country").size().sort_values(ascending=False).head(15).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.out.suffix.lower()
    if suffix == ".csv":
        feat.to_csv(args.out, index=False)
        print(f"[ok] wrote {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")
    else:
        try:
            feat.to_parquet(args.out, index=False, compression="zstd")
            print(f"[ok] wrote {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")
        except ImportError:
            csv_fallback = args.out.with_suffix(".csv")
            feat.to_csv(csv_fallback, index=False)
            print(f"[warn] parquet engine missing — wrote CSV fallback {csv_fallback}")
    if args.csv_out:
        feat.to_csv(args.csv_out, index=False)
        print(f"[ok] wrote {args.csv_out}")

    # team map for future train/sim
    map_dir = ROOT / "football_models" / "mappings"
    map_dir.mkdir(parents=True, exist_ok=True)
    t2 = {t: int(i) for i, t in enumerate(sorted(set(feat["HomeTeam"]) | set(feat["AwayTeam"])))}
    import json
    (map_dir / "africa_team2id.json").write_text(json.dumps(t2, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] africa teams mapped: {len(t2)}")


if __name__ == "__main__":
    main()
