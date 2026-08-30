#!/usr/bin/env python3
"""
Scan fixtures.csv → canonical team names with accurate mapping.

Uses:
  - football_models/mappings/team_aliases.json (if present)
  - train.TEAM_ALIASES
  - Frequency matching against master parquet team strings
  - team2id.json

Writes:
  daily_football_data/fixtures_teams.json
  daily_football_data/train_focus_teams.json   # ranked list ready for train.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "daily_football_data"
MAP = ROOT / "football_models" / "mappings"
PARQUET = ROOT / "master_football_data.parquet"

# Priority leagues for daily train budget
PRIORITY_DIVS = {
    "E0": 100, "SP1": 95, "I1": 95, "D1": 95, "F1": 90,
    "N1": 70, "P1": 70, "B1": 65, "T1": 60, "G1": 55,
    "E1": 50, "SP2": 45, "I2": 45, "D2": 45, "F2": 40,
    "SC0": 50, "E2": 30, "E3": 20, "EC": 15,
}

MAX_TRAIN = int(os.environ.get("MAX_TRAIN_TEAMS", "12"))
MIN_MATCHES = int(os.environ.get("MIN_TEAM_MATCHES", "40"))


def load_aliases() -> Dict[str, str]:
    aliases = {}
    try:
        import train as tr
        aliases.update({k.lower(): v for k, v in tr.TEAM_ALIASES.items()})
    except Exception:
        pass
    p = MAP / "team_aliases.json"
    if p.exists():
        raw = json.loads(p.read_text())
        for k, v in raw.items():
            aliases[str(k).lower()] = v
    return aliases


def normalize(name: str, aliases: Dict[str, str]) -> str:
    s = " ".join(str(name).strip().split())
    if not s:
        return s
    key = s.lower()
    if key in aliases:
        return aliases[key]
    # title-case fallback for all-upper / all-lower
    if s.isupper() or s.islower():
        return s.title()
    return s


def parquet_team_counts() -> Counter:
    if not PARQUET.exists():
        return Counter()
    # Git LFS pointer files are tiny text — skip until real parquet is present
    try:
        if PARQUET.stat().st_size < 1000 or PARQUET.read_text(errors="ignore").startswith("version https://git-lfs"):
            print("[scan] parquet is LFS pointer or missing — using team2id only")
            c: Counter = Counter()
            t2 = MAP / "team2id.json"
            if t2.exists():
                for name in json.loads(t2.read_text()):
                    c[name] = 100  # enough to pass MIN_MATCHES gate for ranking
            return c
    except Exception:
        pass
    try:
        df = pd.read_parquet(PARQUET, columns=["HomeTeam", "AwayTeam"])
    except Exception as e:
        print(f"[scan] parquet read failed: {e}")
        return Counter()
    c = Counter()
    for col in ("HomeTeam", "AwayTeam"):
        c.update(df[col].dropna().astype(str).str.strip().tolist())
    return c


def map_to_parquet(name: str, aliases: Dict[str, str], counts: Counter) -> Tuple[str, str]:
    """Return (canonical, status) where status is exact|alias|fuzzy|unmapped."""
    n = normalize(name, aliases)
    if n in counts:
        return n, "exact"
    # case-insensitive exact
    lower_map = {k.lower(): k for k in counts}
    if n.lower() in lower_map:
        return lower_map[n.lower()], "exact"
    # fuzzy: startswith / contains for short unique hits
    candidates = [k for k in counts if n.lower() in k.lower() or k.lower() in n.lower()]
    if len(candidates) == 1:
        return candidates[0], "fuzzy"
    if candidates:
        # pick highest frequency
        best = max(candidates, key=lambda k: counts[k])
        return best, "fuzzy"
    return n, "unmapped"


def has_model(canon: str) -> bool:
    slug = "".join(c if c.isalnum() else "_" for c in canon).strip("_").lower()
    d = ROOT / "football_models" / "teams" / slug
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
    counts = parquet_team_counts()
    team2id = {}
    if (MAP / "team2id.json").exists():
        team2id = json.loads((MAP / "team2id.json").read_text())

    rows = []
    score: Dict[str, float] = {}
    raw_seen: Set[str] = set()

    for _, r in df.iterrows():
        div = str(r.get("Div", "")).strip()
        pri = PRIORITY_DIVS.get(div, 5)
        for col in ("HomeTeam", "AwayTeam"):
            raw = str(r[col]).strip()
            if not raw or raw in ("nan", "None"):
                continue
            raw_seen.add(raw)
            canon, status = map_to_parquet(raw, aliases, counts)
            n_hist = int(counts.get(canon, 0))
            in_map = canon in team2id
            modeled = has_model(canon)
            rows.append({
                "raw": raw,
                "canonical": canon,
                "map_status": status,
                "div": div,
                "n_hist": n_hist,
                "in_team2id": in_map,
                "has_model": modeled,
                "priority": pri,
            })
            # score for training: priority league, enough history, prefer missing models
            if n_hist >= MIN_MATCHES and status in ("exact", "fuzzy", "alias"):
                boost = 25 if not modeled else 5
                score[canon] = max(score.get(canon, 0), pri + boost + min(n_hist, 200) / 50.0)

    # unique mapping table
    by_canon = {}
    for row in rows:
        c = row["canonical"]
        if c not in by_canon or row["n_hist"] > by_canon[c]["n_hist"]:
            by_canon[c] = row

    ranked = sorted(score.keys(), key=lambda t: score[t], reverse=True)
    focus = ranked[:MAX_TRAIN]

    out_teams = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_raw": len(raw_seen),
        "n_canonical": len(by_canon),
        "min_matches": MIN_MATCHES,
        "teams": sorted(by_canon.keys()),
        "mapping": list(by_canon.values()),
        "unmapped": [r for r in by_canon.values() if r["map_status"] == "unmapped"],
    }
    SAVE.mkdir(parents=True, exist_ok=True)
    (SAVE / "fixtures_teams.json").write_text(json.dumps(out_teams, indent=2))

    focus_payload = {
        "generated_at": out_teams["generated_at"],
        "max_train": MAX_TRAIN,
        "focus_teams": focus,
        "scores": {t: round(score[t], 2) for t in focus},
        "note": "Pass to train.py via FOCUS_TEAMS env (comma-separated) or --focus",
    }
    (SAVE / "train_focus_teams.json").write_text(json.dumps(focus_payload, indent=2))

    print(f"Mapped {len(raw_seen)} raw → {len(by_canon)} canonical")
    print(f"Unmapped: {len(out_teams['unmapped'])}")
    print(f"Train focus ({len(focus)}): {', '.join(focus)}")
    for t in focus:
        print(f"  {t:28} score={score[t]:.1f} hist={counts.get(t,0)} model={has_model(t)}")


if __name__ == "__main__":
    main()
