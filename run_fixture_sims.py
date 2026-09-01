#!/usr/bin/env python3
"""Run simulations for the complete fixture slate and write an indexed report."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "daily_football_data"
SIMS_DIR = SAVE_DIR / "sims"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sim as sim_mod

N_SIM = int(os.environ.get("N_SIMULATIONS", "8000"))
ODDS_BLEND = float(os.environ.get("ODDS_BLEND", "0.30"))
SEED = 42
MAX_FIXTURES = int(os.environ.get("MAX_FIXTURES", "0"))


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    for c in ("Div", "Date", "HomeTeam", "AwayTeam"):
        if c not in df.columns:
            raise SystemExit(f"fixtures missing {c}")
    for side, cols in (("H", ["B365H", "AvgH", "MaxH", "PSH", "BWH"]),
                       ("D", ["B365D", "AvgD", "MaxD", "PSD", "BWD"]),
                       ("A", ["B365A", "AvgA", "MaxA", "PSA", "BWA"])):
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


def _first_present(*vals, default=0.0):
    for value in vals:
        if value is not None:
            return value
    return default


def _pct(value) -> float:
    value = float(value)
    return round(value * 100, 1) if value <= 1 else round(value, 1)


def _filter_run_day(df: pd.DataFrame) -> pd.DataFrame:
    today_only = os.environ.get("TODAY_ONLY", "1").strip().lower() in ("1", "true", "yes")
    if not today_only:
        return df
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    exact = dates == today
    if int(exact.sum()) == 0:
        exact = (dates >= today - pd.Timedelta(days=1)) & (dates <= today + pd.Timedelta(days=1))
        print(f"TODAY_ONLY exact slate empty; using safety window around {today.date()}", flush=True)
    before = len(df)
    df = df[exact].copy()
    print(f"TODAY_ONLY={today.date()}: {before} -> {len(df)} fixtures", flush=True)
    return df


def main():
    fixtures_path = SAVE_DIR / "fixtures_latest.csv"
    if not fixtures_path.exists():
        from scraper_fixtures import download_fixtures
        fixtures_path = download_fixtures()
        if not fixtures_path:
            raise SystemExit("No fixtures CSV")

    df = _filter_run_day(load_fixtures(fixtures_path))
    if MAX_FIXTURES > 0:
        df = df.head(MAX_FIXTURES)
    print(f"Simulating {len(df)} fixtures via sim.run_one_match …", flush=True)

    SIMS_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    teams_seen = set()
    failures = []
    t0_all = time.time()

    from api_football_client import FDC_DIV_TO_LEAGUE
    from standings_prior import prior_from_league_cache
    standings_bundle = load_standings_bundle()
    tables = standings_bundle.get("tables") or {}
    strength = float(standings_bundle.get("strength") or 0.55)
    print(f"Live standings loaded for {len(tables)} leagues" if tables else "No live standings bundle", flush=True)

    for i, row in df.iterrows():
        home, away = str(row["HomeTeam"]).strip(), str(row["AwayTeam"]).strip()
        teams_seen.update((home, away))
        date_str = row["Date"].strftime("%Y-%m-%d")
        t = row.get("Time") or ""
        kick = f"{date_str} {t}" if t and t not in ("nan", "None", "") else date_str
        div = str(row["Div"]).strip()
        lid = FDC_DIV_TO_LEAGUE.get(div)
        sp = prior_from_league_cache(home, away, lid, tables, strength=strength) if lid else None
        cfg = {
            "league": div, "home_team": home, "away_team": away, "match_date": kick,
            "odds_home": float(row["odds_H"]), "odds_draw": float(row["odds_D"]),
            "odds_away": float(row["odds_A"]), "n_simulations": N_SIM,
            "seed": SEED + int(i), "odds_blend": ODDS_BLEND,
            "standings_prior": sp if (sp and sp.get("matched")) else None,
        }
        print(f"  [{i + 1}/{len(df)}] {home} vs {away} …", flush=True)
        try:
            payload = sim_mod.run_one_match(cfg, quiet=True, allow_market_only=True)
            if not isinstance(payload, dict) or "resolved" not in payload or "report" not in payload:
                raise ValueError("simulation returned an invalid report payload")
        except Exception as exc:
            failures.append({"fixture": f"{home} vs {away}", "error": str(exc)})
            print(f"  FAIL {home} vs {away}: {exc}", flush=True)
            continue

        key = slug_key(home, away, date_str)
        (SIMS_DIR / f"{key}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tip = payload.get("locked_tip") or {}
        rep = payload.get("report") or {}
        ft = rep.get("ft_sim") or rep.get("ft_model") or {}
        top_ft = rep.get("top_ft") or []
        index.append({
            "id": key, "file": f"{key}.json", "kickoff": kick,
            "league": payload["resolved"]["league"], "div": payload["resolved"]["div"],
            "home": payload["resolved"]["home"], "away": payload["resolved"]["away"],
            "fixture": f"{payload['resolved']['home']} vs {payload['resolved']['away']}",
            "odds": f"{cfg['odds_home']:.2f} / {cfg['odds_draw']:.2f} / {cfg['odds_away']:.2f}",
            "odds_h": cfg["odds_home"], "odds_d": cfg["odds_draw"], "odds_a": cfg["odds_away"],
            "models": payload["resolved"]["models"],
            "ft_h": _pct(ft.get("H", 0)), "ft_d": _pct(ft.get("D", 0)), "ft_a": _pct(ft.get("A", 0)),
            "over25": _pct(_first_present(rep.get("over25_sim"), rep.get("over25_model"))),
            "btts": _pct(_first_present(rep.get("btts_sim"), rep.get("btts_model"))),
            "cs_top": f"{top_ft[0][0]} ({top_ft[0][1]})" if top_ft else "",
            "cs_second": f"{top_ft[1][0]} ({top_ft[1][1]})" if len(top_ft) > 1 else "",
            "xg_h": (rep.get("xg") or {}).get("home"), "xg_a": (rep.get("xg") or {}).get("away"),
            "locked_status": tip.get("status"), "locked_section": tip.get("section"),
            "locked_selection": tip.get("selection"), "locked_model": tip.get("model"),
            "locked_sim": tip.get("sim"), "locked_verdict": tip.get("verdict"),
            "xg_total": (rep.get("xg") or {}).get("total"),
            "lambda_h": (rep.get("xg") or {}).get("lambda_home"),
            "lambda_a": (rep.get("xg") or {}).get("lambda_away"),
            "cs_home": (rep.get("clean_sheet") or {}).get("home"),
            "cs_away": (rep.get("clean_sheet") or {}).get("away"),
            "top8_ft": rep.get("top8_ft") or rep.get("top3_ft"),
            "goal_line_25_over": ((rep.get("goal_lines") or {}).get("2.5") or {}).get("over"),
            "ft_vs_model_l1": (rep.get("score_consistency") or {}).get("ft_vs_model_l1"),
            "generated_at": payload.get("generated_at"),
        })

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    index_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "feed_date": today,
        "source": "https://www.football-data.co.uk/fixtures.csv", "n_ok": len(index),
        "n_fail": len(failures), "failures": failures, "n_simulations_each": N_SIM, "sims": index,
    }
    (SIMS_DIR / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    (SAVE_DIR / "fixtures_teams.json").write_text(json.dumps({
        "generated_at": index_payload["generated_at"], "teams": sorted(teams_seen), "n_teams": len(teams_seen),
    }, indent=2), encoding="utf-8")

    elapsed = time.time() - t0_all
    print(f"Done: {len(index)} ok, {len(failures)} fail in {elapsed:.0f}s → {SIMS_DIR}/index.json")
    if failures:
        raise SystemExit(f"Simulation completed with {len(failures)} fixture failures; see sims/index.json")

    secured = [s for s in index if s.get("locked_status") == "SECURED LOCK"]
    print(f"SECURED LOCKs: {len(secured)}")
    for item in secured[:10]:
        print(f"  {item['fixture']}: {item['locked_section']} → {item['locked_selection']} ({item['locked_model']}%)")


if __name__ == "__main__":
    main()
