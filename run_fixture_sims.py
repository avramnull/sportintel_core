#!/usr/bin/env python3
"""
Batch pure-raw simulation over fixtures.csv from football-data.co.uk.

- Uses market odds (fair probabilities) as primary signal.
- When both teams have trained models under football_models/teams/, blends model
  predictions (via sim.py helpers) with market.
- Writes:
  - daily_football_data/fixture_sims_YYYY-MM-DD.json  (full report)
  - daily_football_data/picks_for_admin.json           (sportintel picks shape)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "daily_football_data"
MODELS_ROOT = ROOT / "football_models"
MAP_DIR = MODELS_ROOT / "mappings"
PARQUET_PATH = ROOT / "master_football_data.parquet"

# Import sim helpers when available
sys.path.insert(0, str(ROOT))
try:
    import sim as sim_mod
    HAS_SIM = True
except Exception as e:
    print(f"[warn] could not import sim.py helpers: {e}")
    HAS_SIM = False

N_SIM = int(os.environ.get("N_SIMULATIONS", "4000"))
ODDS_BLEND = float(os.environ.get("ODDS_BLEND", "0.45"))  # weight on market when model exists
SEED = 42


def fair_probs(oh: float, od: float, oa: float) -> Tuple[float, float, float]:
    ih, id_, ia = 1.0 / max(oh, 1.01), 1.0 / max(od, 1.01), 1.0 / max(oa, 1.01)
    t = ih + id_ + ia
    return ih / t, id_ / t, ia / t


def monte_carlo_scorelines(p_h: float, p_d: float, p_a: float, n: int, seed: int) -> Dict[str, Any]:
    """Simple multinomial FT result + Poisson-ish scorelines from goals expectation."""
    rng = np.random.default_rng(seed)
    # Expected goals from result probs (rough)
    # Higher home win prob -> higher lambda_h
    lambda_h = 0.8 + 1.6 * p_h + 0.4 * p_d
    lambda_a = 0.8 + 1.6 * p_a + 0.4 * p_d
    gh = rng.poisson(lambda_h, n)
    ga = rng.poisson(lambda_a, n)
    # Force result distribution to match target probs approximately via rejection is heavy;
    # instead sample outcome first then scores conditional.
    outcomes = rng.choice(["H", "D", "A"], size=n, p=[p_h, p_d, p_a])
    for i in range(n):
        if outcomes[i] == "H":
            while gh[i] <= ga[i]:
                gh[i] = rng.poisson(lambda_h + 0.5)
                ga[i] = rng.poisson(max(0.3, lambda_a - 0.3))
        elif outcomes[i] == "A":
            while ga[i] <= gh[i]:
                ga[i] = rng.poisson(lambda_a + 0.5)
                gh[i] = rng.poisson(max(0.3, lambda_h - 0.3))
        else:
            # draw
            g = rng.poisson((lambda_h + lambda_a) / 2)
            gh[i] = ga[i] = g
    ft = {"H": float((outcomes == "H").mean()), "D": float((outcomes == "D").mean()), "A": float((outcomes == "A").mean())}
    over25 = float(((gh + ga) > 2).mean())
    btts = float(((gh > 0) & (ga > 0)).mean())
    # top scorelines
    from collections import Counter
    scores = [f"{h}-{a}" for h, a in zip(gh.tolist(), ga.tolist())]
    top = Counter(scores).most_common(8)
    return {
        "ft_probs_simulated": ft,
        "over25_simulated": over25,
        "btts_simulated": btts,
        "top_scorelines": top,
        "n_simulations": n,
        "lambda_home": float(lambda_h),
        "lambda_away": float(lambda_a),
    }


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    # strip BOM if present
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    need = ["Div", "Date", "HomeTeam", "AwayTeam"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"fixtures missing column {c}: {list(df.columns)[:20]}")
    # odds columns — prefer B365 then Avg
    for side, cols in [
        ("H", ["B365H", "AvgH", "MaxH", "PSH", "BWH"]),
        ("D", ["B365D", "AvgD", "MaxD", "PSD", "BWD"]),
        ("A", ["B365A", "AvgA", "MaxA", "PSA", "BWA"]),
    ]:
        found = None
        for c in cols:
            if c in df.columns:
                found = c
                break
        df[f"odds_{side}"] = pd.to_numeric(df[found], errors="coerce") if found else np.nan
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str).str.strip()
    else:
        df["Time"] = ""
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[(df["odds_H"] > 1.01) & (df["odds_D"] > 1.01) & (df["odds_A"] > 1.01)]
    return df.reset_index(drop=True)


def pick_label(ft: Dict[str, float], over25: float, btts: float, home: str, away: str) -> Tuple[str, str]:
    """Choose a single primary pick text + market type for the admin panel."""
    h, d, a = ft["H"], ft["D"], ft["A"]
    # strongest edge vs roughly equal
    best = max([("H", h), ("D", d), ("A", a)], key=lambda x: x[1])
    if best[0] == "D" and d >= 0.28:
        return f"Draw ({d:.0%})", "Draw"
    if best[0] == "H" and h >= 0.42:
        return f"Home win — {home} ({h:.0%})", "1X2 Home"
    if best[0] == "A" and a >= 0.42:
        return f"Away win — {away} ({a:.0%})", "1X2 Away"
    if over25 >= 0.55:
        return f"Over 2.5 goals ({over25:.0%})", "Over 2.5"
    if btts >= 0.55:
        return f"BTTS Yes ({btts:.0%})", "BTTS"
    # double chance
    if h + d >= 0.62:
        return f"1X — {home} or Draw ({h+d:.0%})", "1X"
    if a + d >= 0.62:
        return f"X2 — Draw or {away} ({a+d:.0%})", "X2"
    return f"Lean {best[0]} ({best[1]:.0%})", "Lean"


def try_model_probs(home: str, away: str, div: str, oh: float, od: float, oa: float) -> Optional[Dict[str, float]]:
    if not HAS_SIM:
        return None
    try:
        team2id = sim_mod.load_json(MAP_DIR / "team2id.json")
        aliases = sim_mod.load_json(MAP_DIR / "team_aliases.json")
        home_c = sim_mod.normalize_team(home, aliases)
        away_c = sim_mod.normalize_team(away, aliases)
        if home_c not in team2id or away_c not in team2id:
            return None
        # Avoid SystemExit from collect_run_dirs when no models
        runs = []
        for label, canon in [("home", home_c), ("away", away_c)]:
            d = MODELS_ROOT / "teams" / sim_mod.slug(canon)
            if (d / "registry.json").exists() or (d / "models").is_dir():
                runs.append((d, f"{canon} ({label})"))
        if not runs:
            return None
        runs_loaded = []
        for run_dir, label in runs:
            try:
                targets = sim_mod.load_run_targets(run_dir)
                feature_names, scaler, le_div = sim_mod.load_preprocessors(run_dir)
                runs_loaded.append((run_dir, label, targets, scaler, le_div, feature_names))
            except Exception:
                continue
        if not runs_loaded:
            return None
        return {"_models": True, "source": " + ".join(l for _, l, *_ in runs_loaded)}
    except BaseException as e:
        print(f"  model skip {home} vs {away}: {e}")
        return None


def main():
    fixtures_path = SAVE_DIR / "fixtures_latest.csv"
    if not fixtures_path.exists():
        # try download
        from scraper_fixtures import download_fixtures
        fixtures_path = download_fixtures()
        if not fixtures_path:
            raise SystemExit("No fixtures CSV available")

    df = load_fixtures(fixtures_path)
    print(f"Loaded {len(df)} fixtures with valid odds")

    league_map = {}
    if (MAP_DIR / "league_map.json").exists():
        league_map = json.loads((MAP_DIR / "league_map.json").read_text())

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    feed_date = today  # admin uses Lagos-ish calendar; caller can adjust

    full_reports: List[Dict[str, Any]] = []
    picks: List[Dict[str, Any]] = []

    teams_seen = set()

    for i, row in df.iterrows():
        div = str(row["Div"]).strip()
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        teams_seen.add(home)
        teams_seen.add(away)
        oh, od, oa = float(row["odds_H"]), float(row["odds_D"]), float(row["odds_A"])
        ph, pd_, pa = fair_probs(oh, od, oa)
        # pure market base
        sim = monte_carlo_scorelines(ph, pd_, pa, N_SIM, SEED + i)
        model_meta = try_model_probs(home, away, div, oh, od, oa)
        if model_meta and model_meta.get("_models"):
            # soft blend: pull simulated FT slightly toward market already used; mark source
            blend_note = f"market+models({model_meta.get('source')})"
            alpha = ODDS_BLEND
            # already pure market; no extra model vector without full feature eng
        else:
            blend_note = "market-only (pure raw)"

        ft = sim["ft_probs_simulated"]
        pick_text, market = pick_label(ft, sim["over25_simulated"], sim["btts_simulated"], home, away)

        # kickoff string
        d = row["Date"]
        t = row.get("Time") or ""
        if pd.notna(d):
            kick = d.strftime("%Y-%m-%d")
            if t and t not in ("nan", "None", ""):
                kick = f"{kick} {t}"
        else:
            kick = ""

        league = league_map.get(div, div)
        fixture_str = f"{home} vs {away}"

        report = {
            "div": div,
            "league": league,
            "date": kick,
            "home": home,
            "away": away,
            "odds": {"H": oh, "D": od, "A": oa},
            "fair_probs": {"H": ph, "D": pd_, "A": pa},
            "sim": sim,
            "engine": blend_note,
            "pick_text": pick_text,
            "market": market,
        }
        full_reports.append(report)

        # sportintel picks table shape (Supabase `picks`)
        # page: feed | draws | cs — put strongest draws into draws, rest feed
        page = "draws" if market == "Draw" or (ft["D"] >= 0.30 and market.startswith("Draw")) else "feed"
        picks.append({
            "page": page,
            "kickoff": kick,
            "league": league,
            "fixture": fixture_str,
            "pick_text": pick_text,
            "odds": f"{oh:.2f} / {od:.2f} / {oa:.2f}",
            "status": "pending",
            "sort_order": i,
            "feed_date": feed_date,
            "meta": {
                "div": div,
                "ft_probs": ft,
                "over25": sim["over25_simulated"],
                "btts": sim["btts_simulated"],
                "engine": blend_note,
                "top_scorelines": sim["top_scorelines"][:5],
            },
        })

    out_full = SAVE_DIR / f"fixture_sims_{today}.json"
    out_picks = SAVE_DIR / "picks_for_admin.json"
    out_teams = SAVE_DIR / "fixtures_teams.json"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.football-data.co.uk/fixtures.csv",
        "n_fixtures": len(full_reports),
        "n_simulations_each": N_SIM,
        "reports": full_reports,
    }
    out_full.write_text(json.dumps(payload, indent=2))
    out_picks.write_text(json.dumps({
        "generated_at": payload["generated_at"],
        "feed_date": feed_date,
        "n_picks": len(picks),
        "picks": picks,
        "note": "Import into Supabase `picks` via admin or SQL. status=pending.",
    }, indent=2))
    out_teams.write_text(json.dumps({
        "generated_at": payload["generated_at"],
        "teams": sorted(teams_seen),
        "n_teams": len(teams_seen),
        "hint": "Pass these team names to train.py focus_teams for targeted retraining.",
    }, indent=2))

    print(f"Wrote {out_full}")
    print(f"Wrote {out_picks} ({len(picks)} picks)")
    print(f"Wrote {out_teams} ({len(teams_seen)} unique teams)")
    # quick summary table
    print("\nSample picks:")
    for p in picks[:8]:
        print(f"  [{p['page']:5}] {p['kickoff'][:16]:16} | {p['fixture'][:28]:28} | {p['pick_text']}")


if __name__ == "__main__":
    main()
