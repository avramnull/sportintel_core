#!/usr/bin/env python3
"""
Scan fixtures.csv → canonical team names with accurate mapping.

Writes:
  daily_football_data/fixtures_teams.json
  daily_football_data/train_focus_teams.json
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from team_mapping import (
    LEAGUE_MAP,
    clean_name,
    load_aliases,
    load_team2id,
    normalize_team,
    resolve_league,
    slug,
)

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "daily_football_data"
MAP = ROOT / "football_models" / "mappings"
PARQUET = ROOT / "master_football_data.parquet"

# 0 or negative => train ALL unique fixture teams
MAX_TRAIN = int(os.environ.get("MAX_TRAIN_TEAMS", "0"))
MIN_MATCHES = int(os.environ.get("MIN_TEAM_MATCHES", "25"))
# Always include previously trained teams that appear in fixtures (retrain)
RETRAIN_EXISTING = os.environ.get("RETRAIN_EXISTING", "1").strip().lower() in ("1", "true", "yes")

PRIORITY_DIVS = {
    "E0": 100, "SP1": 95, "I1": 95, "D1": 95, "F1": 90,
    "N1": 70, "P1": 70, "B1": 65, "T1": 60, "G1": 55,
    "E1": 50, "SP2": 45, "I2": 45, "D2": 45, "F2": 40,
    "SC0": 50, "E2": 30, "E3": 20, "EC": 15,
    "SC1": 35, "SC2": 25, "SC3": 20,
}


def parquet_team_counts() -> Counter:
    if not PARQUET.exists():
        return Counter()
    try:
        head = PARQUET.read_bytes()[:80]
        if head.startswith(b"version https://git-lfs") or PARQUET.stat().st_size < 1000:
            print("[scan] parquet is LFS pointer — counts from team2id only")
            return Counter({n: 100 for n in load_team2id()})
    except Exception:
        pass
    try:
        df = pd.read_parquet(PARQUET, columns=["HomeTeam", "AwayTeam"])
        c: Counter = Counter()
        for col in ("HomeTeam", "AwayTeam"):
            for v in df[col].dropna().astype(str):
                c[clean_name(v)] += 1
        return c
    except Exception as e:
        print(f"[scan] parquet read failed: {e}")
        return Counter({n: 100 for n in load_team2id()})


def has_model(canon: str) -> bool:
    d = ROOT / "football_models" / "teams" / slug(canon)
    return (d / "models").is_dir() and any((d / "models").iterdir())


def main():
    fx = SAVE / "fixtures_latest.csv"
    if not fx.exists():
        raise SystemExit(f"Missing {fx} — run scraper_fixtures.py first")

    df = pd.read_csv(fx, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise SystemExit("fixtures missing HomeTeam/AwayTeam")

    aliases = load_aliases()
    team2id = load_team2id()
    counts = parquet_team_counts()

    rows = []
    score: Dict[str, float] = {}
    raw_seen: Set[str] = set()
    canon_teams: Set[str] = set()

    for _, r in df.iterrows():
        div = clean_name(str(r.get("Div", "")))
        div_code, league_name = resolve_league(div)
        pri = PRIORITY_DIVS.get(div_code, 5)
        for col in ("HomeTeam", "AwayTeam"):
            raw = clean_name(str(r[col]))
            if not raw or raw.lower() in ("nan", "none"):
                continue
            raw_seen.add(raw)
            canon, status = normalize_team(raw, aliases, team2id)
            canon_teams.add(canon)
            n_hist = int(counts.get(canon, 0)) or int(counts.get(raw, 0))
            in_map = canon in team2id or raw in team2id
            modeled = has_model(canon)
            rows.append({
                "raw": raw,
                "canonical": canon,
                "map_status": status,
                "div": div_code,
                "league": league_name,
                "n_hist": n_hist,
                "in_team2id": in_map,
                "has_model": modeled,
                "priority": pri,
            })
            # Eligible for training if enough history OR already has a model (retrain)
            if n_hist >= MIN_MATCHES or (RETRAIN_EXISTING and modeled):
                boost = 40 if modeled else 20
                score[canon] = max(score.get(canon, 0), pri + boost + min(n_hist, 400) / 40.0)

    # Always put every fixture team in the pool when MAX_TRAIN <= 0
    for c in canon_teams:
        score.setdefault(c, 1.0)

    by_canon = {}
    for row in rows:
        c = row["canonical"]
        if c not in by_canon or row["n_hist"] > by_canon[c]["n_hist"]:
            by_canon[c] = row

    ranked = sorted(score.keys(), key=lambda t: score[t], reverse=True)
    if MAX_TRAIN and MAX_TRAIN > 0:
        focus = ranked[:MAX_TRAIN]
    else:
        focus = ranked  # all unique fixture teams

    unmapped = [r for r in by_canon.values() if r["map_status"] in ("raw",) and not r["in_team2id"]]

    out_teams = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_fixtures": int(len(df)),
        "n_raw_names": len(raw_seen),
        "n_canonical": len(by_canon),
        "teams": sorted(by_canon.keys()),
        "mapping": list(by_canon.values()),
        "unmapped": unmapped,
        "league_map": LEAGUE_MAP,
    }
    SAVE.mkdir(parents=True, exist_ok=True)
    (SAVE / "fixtures_teams.json").write_text(json.dumps(out_teams, indent=2, ensure_ascii=False))

    focus_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_train": MAX_TRAIN if MAX_TRAIN > 0 else len(focus),
        "train_all_fixtures": MAX_TRAIN <= 0,
        "min_matches": MIN_MATCHES,
        "retrain_existing": RETRAIN_EXISTING,
        "focus_teams": focus,
        "scores": {t: score[t] for t in focus},
        "note": "Pass to train.py via FOCUS_TEAMS env (comma-separated)",
    }
    (SAVE / "train_focus_teams.json").write_text(json.dumps(focus_payload, indent=2, ensure_ascii=False))

    print(f"Fixtures: {len(df)}")
    print(f"Mapped {len(raw_seen)} raw → {len(by_canon)} canonical")
    print(f"Unmapped (not in team2id): {len(unmapped)}")
    print(f"Train focus ({len(focus)}): all_fixture_teams={MAX_TRAIN <= 0}")
    for t in focus[:40]:
        print(f"  {t:28} score={score[t]:.1f} hist={counts.get(t,0)} model={has_model(t)}")
    if len(focus) > 40:
        print(f"  … +{len(focus) - 40} more")


if __name__ == "__main__":
    main()
