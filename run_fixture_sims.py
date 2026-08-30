#!/usr/bin/env python3
"""
Daily batch: fixtures.csv → sim.run_one_match() → full report JSONs (same schema as last_simulation_report).

Writes:
  daily_football_data/sims/<slug>__YYYY-MM-DD.json   one full report each
  daily_football_data/sims/index.json                catalog for admin panel
  daily_football_data/fixtures_teams.json            teams for train focus
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "daily_football_data"
SIMS_DIR = SAVE_DIR / "sims"
sys.path.insert(0, str(ROOT))

# Must set before TensorFlow import (via sim)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sim as sim_mod  # uses upgraded run_one_match

N_SIM = int(os.environ.get("N_SIMULATIONS", "8000"))
ODDS_BLEND = float(os.environ.get("ODDS_BLEND", "0.30"))
SEED = 42
MAX_FIXTURES = int(os.environ.get("MAX_FIXTURES", "0"))  # 0 = all


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    for c in ("Div", "Date", "HomeTeam", "AwayTeam"):
        if c not in df.columns:
            raise SystemExit(f"fixtures missing {c}")
    for side, cols in [
        ("H", ["B365H", "AvgH", "MaxH", "PSH", "BWH"]),
        ("D", ["B365D", "AvgD", "MaxD", "PSD", "BWD"]),
        ("A", ["B365A", "AvgA", "MaxA", "PSA", "BWA"]),
    ]:
        found = next((c for c in cols if c in df.columns), None)
        df[f"odds_{side}"] = pd.to_numeric(df[found], errors="coerce") if found else np.nan
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Time"] = df["Time"].astype(str).str.strip() if "Time" in df.columns else ""
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[(df["odds_H"] > 1.01) & (df["odds_D"] > 1.01) & (df["odds_A"] > 1.01)]
    return df.reset_index(drop=True)


def slug_key(home: str, away: str, date_str: str) -> str:
    raw = f"{home}__{away}__{date_str}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_").lower()[:120]


def main():
    fixtures_path = SAVE_DIR / "fixtures_latest.csv"
    if not fixtures_path.exists():
        from scraper_fixtures import download_fixtures
        fixtures_path = download_fixtures()
        if not fixtures_path:
            raise SystemExit("No fixtures CSV")

    df = load_fixtures(fixtures_path)
    # Align with train: only fixtures on script run day when TODAY_ONLY=1
    today_only = os.environ.get("TODAY_ONLY", "1").strip().lower() in ("1", "true", "yes")
    if today_only and "Date" in df.columns:
        today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        before = len(df)
        df = df[pd.to_datetime(df["Date"], errors="coerce").dt.normalize() == today].copy()
        print(f"TODAY_ONLY={today.date()}: {before} -> {len(df)} fixtures", flush=True)
    if MAX_FIXTURES > 0:
        df = df.head(MAX_FIXTURES)
    print(f"Simulating {len(df)} fixtures via sim.run_one_match …", flush=True)

    SIMS_DIR.mkdir(parents=True, exist_ok=True)
    # clear previous index only; keep old reports optional — overwrite same keys
    index = []
    teams_seen = set()
    ok, fail = 0, 0
    t0_all = time.time()

    for i, row in df.iterrows():
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        teams_seen.add(home)
        teams_seen.add(away)
        d = row["Date"]
        t = row.get("Time") or ""
        date_str = d.strftime("%Y-%m-%d") if pd.notna(d) else "unknown"
        kick = date_str
        if t and t not in ("nan", "None", ""):
            kick = f"{date_str} {t}"
        match_cfg = {
            "league": str(row["Div"]).strip(),
            "home_team": home,
            "away_team": away,
            "match_date": kick,
            "odds_home": float(row["odds_H"]),
            "odds_draw": float(row["odds_D"]),
            "odds_away": float(row["odds_A"]),
            "n_simulations": N_SIM,
            "seed": SEED + int(i),
            "odds_blend": ODDS_BLEND,
        }
        t0 = time.time()
        print(f"  [{ok+fail+1}/{len(df)}] {home} vs {away} …", flush=True)
        try:
            payload = sim_mod.run_one_match(match_cfg, quiet=True, allow_market_only=True)
        except Exception as e:
            print(f"  FAIL {home} vs {away}: {e}", flush=True)
            fail += 1
            continue
        print(f"      ok in {time.time()-t0:.1f}s", flush=True)

        key = slug_key(home, away, date_str)
        out_path = SIMS_DIR / f"{key}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tip = payload.get("locked_tip") or {}
        rep = payload.get("report") or {}
        ft = rep.get("ft_sim") or rep.get("ft_model") or {}
        top_ft = rep.get("top_ft") or []
        cs1 = f"{top_ft[0][0]} ({top_ft[0][1]})" if top_ft else ""
        cs2 = f"{top_ft[1][0]} ({top_ft[1][1]})" if len(top_ft) > 1 else ""
        index.append({
            "id": key,
            "file": f"{key}.json",
            "kickoff": kick,
            "league": payload["resolved"]["league"],
            "div": payload["resolved"]["div"],
            "home": payload["resolved"]["home"],
            "away": payload["resolved"]["away"],
            "fixture": f"{payload['resolved']['home']} vs {payload['resolved']['away']}",
            "odds": f"{match_cfg['odds_home']:.2f} / {match_cfg['odds_draw']:.2f} / {match_cfg['odds_away']:.2f}",
            "odds_h": match_cfg["odds_home"],
            "odds_d": match_cfg["odds_draw"],
            "odds_a": match_cfg["odds_away"],
            "models": payload["resolved"]["models"],
            # rich summary for Sim Lab list
            "ft_h": round(float(ft.get("H", 0)) * 100, 1) if ft.get("H", 0) <= 1 else round(float(ft.get("H", 0)), 1),
            "ft_d": round(float(ft.get("D", 0)) * 100, 1) if ft.get("D", 0) <= 1 else round(float(ft.get("D", 0)), 1),
            "ft_a": round(float(ft.get("A", 0)) * 100, 1) if ft.get("A", 0) <= 1 else round(float(ft.get("A", 0)), 1),
            "over25": round(float(rep.get("over25_sim") or rep.get("over25_model") or 0) * 100, 1)
                      if float(rep.get("over25_sim") or rep.get("over25_model") or 0) <= 1
                      else round(float(rep.get("over25_sim") or rep.get("over25_model") or 0), 1),
            "btts": round(float(rep.get("btts_sim") or rep.get("btts_model") or 0) * 100, 1)
                    if float(rep.get("btts_sim") or rep.get("btts_model") or 0) <= 1
                    else round(float(rep.get("btts_sim") or rep.get("btts_model") or 0), 1),
            "cs_top": cs1,
            "cs_second": cs2,
            "xg_h": (rep.get("xg") or {}).get("home"),
            "xg_a": (rep.get("xg") or {}).get("away"),
            "locked_status": tip.get("status"),
            "locked_section": tip.get("section"),
            "locked_selection": tip.get("selection"),
            "locked_model": tip.get("model"),
            "locked_sim": tip.get("sim"),
            "locked_verdict": tip.get("verdict"),
            "generated_at": payload.get("generated_at"),
        })
        ok += 1
        if ok % 25 == 0:
            print(f"  … {ok}/{len(df)}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    index_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feed_date": today,
        "source": "https://www.football-data.co.uk/fixtures.csv",
        "n_ok": ok,
        "n_fail": fail,
        "n_simulations_each": N_SIM,
        "sims": index,
    }
    (SIMS_DIR / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    (SAVE_DIR / "fixtures_teams.json").write_text(json.dumps({
        "generated_at": index_payload["generated_at"],
        "teams": sorted(teams_seen),
        "n_teams": len(teams_seen),
    }, indent=2), encoding="utf-8")

    print(f"Done: {ok} ok, {fail} fail in {time.time()-t0_all:.0f}s → {SIMS_DIR}/index.json")
    secured = [s for s in index if s.get("locked_status") == "SECURED LOCK"]
    print(f"SECURED LOCKs: {len(secured)}")
    for s in secured[:10]:
        print(f"  {s['fixture']}: {s['locked_section']} → {s['locked_selection']} ({s['locked_model']}%)")


if __name__ == "__main__":
    main()
