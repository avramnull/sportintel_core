#!/usr/bin/env python3
"""Scan today's fixtures into canonical teams and a fixture-driven training focus list."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from team_mapping import LEAGUE_MAP, clean_name, load_aliases, load_team2id, normalize_team, resolve_league, slug

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "daily_football_data"
PARQUET = ROOT / "master_football_data.parquet"
MAX_TRAIN = int(os.environ.get("MAX_TRAIN_TEAMS", "0"))
MIN_MATCHES = int(os.environ.get("MIN_TEAM_MATCHES", "25"))
RETRAIN_EXISTING = os.environ.get("RETRAIN_EXISTING", "1").strip().lower() in ("1", "true", "yes")
TODAY_ONLY = os.environ.get("TODAY_ONLY", "1").strip().lower() in ("1", "true", "yes")

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
    except OSError:
        pass
    try:
        df = pd.read_parquet(PARQUET, columns=["HomeTeam", "AwayTeam"])
        aliases = load_aliases()
        team2id = load_team2id()
        counts: Counter = Counter()
        for col in ("HomeTeam", "AwayTeam"):
            for value in df[col].dropna().astype(str):
                raw = clean_name(value)
                canon, _ = normalize_team(raw, aliases, team2id)
                counts[canon] += 1
        return counts
    except Exception as exc:
        print(f"[scan] parquet read failed: {exc}")
        return Counter({n: 100 for n in load_team2id()})


def has_model(canon: str) -> bool:
    d = ROOT / "football_models" / "teams" / slug(canon) / "models"
    return d.is_dir() and any(d.iterdir())


def _filter_today(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise SystemExit("fixtures missing Date")
    df = df.copy()
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    if not TODAY_ONLY:
        print(f"[scan] TODAY_ONLY=0: using all {len(df)} fixtures")
        return df

    before = len(df)
    exact = df[df["_date"].dt.normalize() == today].copy()
    if not exact.empty:
        print(f"[scan] TODAY_ONLY={today.date()}: {before} -> {len(exact)} fixtures")
        return exact

    # Keep scanner and simulator aligned for late-night/WAT feeds when the
    # provider's date is one day either side of the UTC run date.
    window = (df["_date"].dt.normalize() >= today - pd.Timedelta(days=1)) & (
        df["_date"].dt.normalize() <= today + pd.Timedelta(days=1)
    )
    fallback = df[window].copy()
    print(f"[scan] exact UTC day empty; safety window around {today.date()}: {before} -> {len(fallback)} fixtures")
    return fallback


def main():
    fx = SAVE / "fixtures_latest.csv"
    if not fx.exists():
        raise SystemExit(f"Missing {fx} — run scraper_fixtures.py first")

    df = pd.read_csv(fx, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise SystemExit("fixtures missing HomeTeam/AwayTeam")
    df = _filter_today(df)
    df.drop(columns=["_date"], inplace=True, errors="ignore")

    train_divs = [x.strip().upper() for x in os.environ.get("TRAIN_DIVS", "").split(",") if x.strip()]
    if train_divs and "Div" in df.columns:
        before = len(df)
        df = df[df["Div"].astype(str).str.strip().str.upper().isin(train_divs)].copy()
        print(f"[scan] TRAIN_DIVS={train_divs}: {before} -> {len(df)} fixtures")

    aliases = load_aliases()
    team2id = load_team2id()
    counts = parquet_team_counts()
    rows = []
    score: Dict[str, float] = {}
    raw_seen: Set[str] = set()
    canon_teams: Set[str] = set()

    for _, row in df.iterrows():
        div = clean_name(str(row.get("Div", "")))
        div_code, league_name = resolve_league(div)
        priority = PRIORITY_DIVS.get(div_code, 5)
        for col in ("HomeTeam", "AwayTeam"):
            raw = clean_name(str(row[col]))
            if not raw or raw.lower() in ("nan", "none"):
                continue
            raw_seen.add(raw)
            canon, status = normalize_team(raw, aliases, team2id)
            canon_teams.add(canon)
            n_hist = int(counts.get(canon, 0))
            in_map = canon in team2id or raw in team2id
            modeled = has_model(canon)
            rows.append({
                "raw": raw, "canonical": canon, "map_status": status,
                "div": div_code, "league": league_name, "n_hist": n_hist,
                "in_team2id": in_map, "has_model": modeled, "priority": priority,
            })
            if n_hist >= MIN_MATCHES or (RETRAIN_EXISTING and modeled):
                score[canon] = max(score.get(canon, 0), priority + (40 if modeled else 20) + min(n_hist, 400) / 40.0)

    # The daily training population is driven by fixtures, not by a fixed team
    # count. This guarantees every unique fixture team reaches train.py.
    for canon in canon_teams:
        score.setdefault(canon, 1.0)

    by_canon = {}
    for row in rows:
        canon = row["canonical"]
        if canon not in by_canon or row["n_hist"] > by_canon[canon]["n_hist"]:
            by_canon[canon] = row

    ranked = sorted(score, key=score.get, reverse=True)
    focus = ranked[:MAX_TRAIN] if MAX_TRAIN > 0 else ranked
    unmapped = [r for r in by_canon.values() if r["map_status"] == "raw" and not r["in_team2id"]]

    SAVE.mkdir(parents=True, exist_ok=True)
    (SAVE / "fixtures_teams.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_fixtures": int(len(df)), "n_raw_names": len(raw_seen),
        "n_canonical": len(by_canon), "teams": sorted(by_canon),
        "mapping": list(by_canon.values()), "unmapped": unmapped, "league_map": LEAGUE_MAP,
    }, indent=2, ensure_ascii=False))

    (SAVE / "train_focus_teams.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_train": MAX_TRAIN if MAX_TRAIN > 0 else len(focus),
        "train_all_fixtures": MAX_TRAIN <= 0, "min_matches": MIN_MATCHES,
        "retrain_existing": RETRAIN_EXISTING, "today_only": TODAY_ONLY,
        "focus_date": str(pd.Timestamp.now(tz="UTC").date()) if TODAY_ONLY else "all",
        "focus_teams": focus, "scores": {t: score[t] for t in focus},
        "note": "Pass to train.py via FOCUS_TEAMS env (comma-separated)",
    }, indent=2, ensure_ascii=False))

    print(f"Fixtures: {len(df)}")
    print(f"Mapped {len(raw_seen)} raw → {len(by_canon)} canonical")
    print(f"Unmapped (not in team2id): {len(unmapped)}")
    print(f"Train focus ({len(focus)}): all_fixture_teams={MAX_TRAIN <= 0}")
    for team in focus[:40]:
        print(f"  {team:28} score={score[team]:.1f} hist={counts.get(team, 0)} model={has_model(team)}")
    if len(focus) > 40:
        print(f"  … +{len(focus) - 40} more")


if __name__ == "__main__":
    main()
