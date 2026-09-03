#!/usr/bin/env python3
"""Run the complete fixture slate through the calibrated hard simulation engine."""
from __future__ import annotations
import json, os, re, sys, time
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
from hard_simulation import simulate_hard
from industrial_sim_engine import simulate_industrial

# sim.py historically referenced this cache before declaring it.  Initialising
# it here prevents the exception from silently disabling the historical-form
# feature layer for batch runs.
if not hasattr(sim_mod, "_HIST_CACHE"):
    sim_mod._HIST_CACHE = {}

N_SIM = int(os.environ.get("N_SIMULATIONS", "8000"))
ODDS_BLEND = float(os.environ.get("ODDS_BLEND", "0.30"))
SEED = 42
MAX_FIXTURES = int(os.environ.get("MAX_FIXTURES", "0"))
HARD_SIM = os.environ.get("HARD_SIM", "1").strip().lower() in ("1", "true", "yes", "on")
HARD_ATTACK_CV = float(os.environ.get("HARD_ATTACK_CV", "0.13"))
HARD_DEFENSE_CV = float(os.environ.get("HARD_DEFENSE_CV", "0.13"))
HARD_SHARED_CV = float(os.environ.get("HARD_SHARED_CV", "0.07"))


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    for c in ("Div", "Date", "HomeTeam", "AwayTeam"):
        if c not in df.columns:
            raise SystemExit(f"fixtures missing {c}")
    for side, cols in (("H", ["B365H", "AvgH", "MaxH", "PSH", "BWH"]), ("D", ["B365D", "AvgD", "MaxD", "PSD", "BWD"]), ("A", ["B365A", "AvgA", "MaxA", "PSA", "BWA"])):
        found = next((c for c in cols if c in df.columns), None)
        df[f"odds_{side}"] = pd.to_numeric(df[found], errors="coerce") if found else np.nan
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Time"] = df["Time"].astype(str).str.strip() if "Time" in df.columns else ""
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[(df["odds_H"] > 1.01) & (df["odds_D"] > 1.01) & (df["odds_A"] > 1.01)]
    return df.reset_index(drop=True)


def slug_key(home: str, away: str, date_str: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", f"{home}__{away}__{date_str}").strip("_").lower()[:120]


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


def _with_match_clock(match_date: str):
    """Temporarily make sim.py's feature clock match the fixture date.

    Training/live features include calendar and rest-day values. Using the
    runner's current UTC date for a future fixture introduces leakage/drift.
    The wrapper keeps generated_at on the real clock by restoring immediately.
    """
    raw = str(match_date or "").strip()
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return None
    target = parsed.to_pydatetime().replace(tzinfo=None)
    original = getattr(sim_mod, "datetime")

    class MatchClock:
        @classmethod
        def utcnow(cls):
            return target

    sim_mod.datetime = MatchClock
    return original


def _restore_clock(original):
    if original is not None:
        sim_mod.datetime = original


def _apply_hard_engine(payload: dict, seed: int) -> dict:
    rep = payload.get("report") or {}
    def vals(d, default):
        return list((d or default).values())
    args = (
        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),
        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),
        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),
        N_SIM, seed,
    )
    if os.environ.get("INDUSTRIAL_SIM", "1").strip().lower() in ("1", "true", "yes", "on"):
        hard = simulate_industrial(*args, standings_prior=rep.get("standings_prior"))
    else:
        hard = simulate_hard(*args, standings_prior=rep.get("standings_prior"),
                             hardness={"attack_cv":HARD_ATTACK_CV,"defense_cv":HARD_DEFENSE_CV,"shared_cv":HARD_SHARED_CV})
    for key in ("ft_model","ht_model","over25_model","btts_model","ht_over15_model"):
        if key in rep:
            hard[key] = rep[key]
    payload["report"] = hard
    payload.setdefault("metadata", {})["simulation_engine"] = hard["engine"]
    return payload


def _validate_report(payload: dict) -> None:
    """Reject numerically broken reports before they reach Sim Lab."""
    rep = payload.get("report") or {}
    engine = rep.get("engine") or {}
    if not engine.get("version"):
        raise ValueError("simulation engine version missing")
    n = int(rep.get("n", 0))
    if n != N_SIM:
        raise ValueError(f"simulation count mismatch: report={n}, expected={N_SIM}")

    def probs(name, keys):
        d = rep.get(name) or {}
        vals = [float(d[k]) for k in keys]
        if not np.isfinite(vals).all():
            raise ValueError(f"non-finite probabilities in {name}")
        if any(v < -1e-9 or v > 1 + 1e-9 for v in vals):
            raise ValueError(f"probability out of range in {name}: {vals}")
        return vals

    ft = probs("ft_sim", ("H", "D", "A"))
    ht = probs("ht_sim", ("H", "D", "A"))
    if abs(sum(ft) - 1.0) > 1e-6 or abs(sum(ht) - 1.0) > 1e-6:
        raise ValueError("FT/HT probabilities do not sum to 1")

    consistency = rep.get("score_consistency") or {}
    if float(consistency.get("ht_leq_ft", 0.0)) < 1.0 - 1e-12:
        raise ValueError("HT scores are not a subset of FT scores")
    if float(consistency.get("ft_vs_model_l1", 1.0)) > 0.12:
        raise ValueError(f"FT simulation drift too high: {consistency.get('ft_vs_model_l1')}")
    for key in ("over25_abs_error", "btts_abs_error"):
        if float(consistency.get(key, 1.0)) > 0.08:
            raise ValueError(f"{key} too high: {consistency.get(key)}")

    xg = rep.get("xg") or {}
    xgv = [float(xg.get(k, np.nan)) for k in ("home", "away", "total", "lambda_home", "lambda_away")]
    if not np.isfinite(xgv).all() or any(v < 0 for v in xgv):
        raise ValueError("invalid xG/lambda values")

    payload.setdefault("metadata", {})["validation"] = {
        "ok": True,
        "version": "simulation-contract-v1",
        "n": n,
        "ft_sum": round(sum(ft), 10),
        "ht_sum": round(sum(ht), 10),
        "ht_subset_ft": float(consistency.get("ht_leq_ft", 0.0)),
        "ft_l1": float(consistency.get("ft_vs_model_l1", 0.0)),
        "over25_abs_error": float(consistency.get("over25_abs_error", 0.0)),
        "btts_abs_error": float(consistency.get("btts_abs_error", 0.0)),
    }


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
    print(f"Simulating {len(df)} fixtures | hard_engine={HARD_SIM} | n={N_SIM}", flush=True)
    SIMS_DIR.mkdir(parents=True, exist_ok=True)
    index, teams_seen, failures = [], set(), []
    t0_all = time.time()
    for i, row in df.iterrows():
        home, away = str(row["HomeTeam"]).strip(), str(row["AwayTeam"]).strip()
        teams_seen.update((home, away))
        date_str = row["Date"].strftime("%Y-%m-%d")
        t = row.get("Time") or ""
        kick = f"{date_str} {t}" if t and t not in ("nan","None","") else date_str
        cfg = {"league":str(row["Div"]).strip(),"home_team":home,"away_team":away,"match_date":kick,
               "odds_home":float(row["odds_H"]),"odds_draw":float(row["odds_D"]),"odds_away":float(row["odds_A"]),
               "n_simulations":N_SIM,"seed":SEED+int(i),"odds_blend":ODDS_BLEND}
        print(f"  [{i+1}/{len(df)}] {home} vs {away} …", flush=True)
        original_clock = None
        try:
            original_clock = _with_match_clock(kick)
            payload = sim_mod.run_one_match(cfg, quiet=True, allow_market_only=True)
            if not isinstance(payload,dict) or "resolved" not in payload or "report" not in payload:
                raise ValueError("simulation returned an invalid report payload")
            if HARD_SIM:
                payload = _apply_hard_engine(payload, SEED+int(i))
                rows = sim_mod.build_table_rows(payload["report"], payload.get("backends") or {}, payload["resolved"]["home"], payload["resolved"]["away"])
                payload["table"] = rows
                payload["locked_tip"] = sim_mod._locked_tip_silent(rows, payload["report"])
            _validate_report(payload)
        except Exception as exc:
            failures.append({"fixture":f"{home} vs {away}","error":str(exc)})
            print(f"  FAIL {home} vs {away}: {exc}", flush=True)
            continue
        finally:
            _restore_clock(original_clock)
        key = slug_key(home, away, date_str)
        (SIMS_DIR/f"{key}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
        tip, rep = payload.get("locked_tip") or {}, payload.get("report") or {}
        ft = rep.get("ft_sim") or rep.get("ft_model") or {}; top_ft = rep.get("top_ft") or []
        engine = rep.get("engine") or {}
        index.append({"id":key,"file":f"{key}.json","kickoff":kick,"league":payload["resolved"]["league"],"div":payload["resolved"]["div"],
          "home":payload["resolved"]["home"],"away":payload["resolved"]["away"],"fixture":f"{payload['resolved']['home']} vs {payload['resolved']['away']}",
          "odds":f"{cfg['odds_home']:.2f} / {cfg['odds_draw']:.2f} / {cfg['odds_away']:.2f}","odds_h":cfg["odds_home"],"odds_d":cfg["odds_draw"],"odds_a":cfg["odds_away"],"models":payload["resolved"]["models"],
          "ft_h":_pct(ft.get("H",0)),"ft_d":_pct(ft.get("D",0)),"ft_a":_pct(ft.get("A",0)),"over25":_pct(_first_present(rep.get("over25_sim"),rep.get("over25_model"))),"btts":_pct(_first_present(rep.get("btts_sim"),rep.get("btts_model"))),
          "cs_top":f"{top_ft[0][0]} ({top_ft[0][1]})" if top_ft else "","cs_second":f"{top_ft[1][0]} ({top_ft[1][1]})" if len(top_ft)>1 else "","xg_h":(rep.get("xg") or {}).get("home"),"xg_a":(rep.get("xg") or {}).get("away"),
          "locked_status":tip.get("status"),"locked_section":tip.get("section"),"locked_selection":tip.get("selection"),"locked_model":tip.get("model"),"locked_sim":tip.get("sim"),"locked_verdict":tip.get("verdict"),
          "xg_total":(rep.get("xg") or {}).get("total"),"lambda_h":(rep.get("xg") or {}).get("lambda_home"),"lambda_a":(rep.get("xg") or {}).get("lambda_away"),"cs_home":(rep.get("clean_sheet") or {}).get("home"),"cs_away":(rep.get("clean_sheet") or {}).get("away"),
          "top8_ft":rep.get("top8_ft") or rep.get("top3_ft"),"goal_line_25_over":((rep.get("goal_lines") or {}).get("2.5") or {}).get("over"),"ft_vs_model_l1":(rep.get("score_consistency") or {}).get("ft_vs_model_l1"),"generated_at":payload.get("generated_at"),
          "simulation_engine":engine.get("version"),"simulation_method":engine.get("method"),"validation_ok":(payload.get("metadata") or {}).get("validation",{}).get("ok",False)})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    engine_version = "industrial-v4" if os.environ.get("INDUSTRIAL_SIM", "1").strip().lower() in ("1", "true", "yes", "on") else "hard-v2"
    index_payload={"generated_at":datetime.now(timezone.utc).isoformat(),"feed_date":today,"source":"https://www.football-data.co.uk/fixtures.csv","n_ok":len(index),"n_fail":len(failures),"failures":failures,"n_simulations_each":N_SIM,"simulation_engine":engine_version,"sims":index}
    (SIMS_DIR/"index.json").write_text(json.dumps(index_payload,indent=2),encoding="utf-8")
    (SAVE_DIR/"fixtures_teams.json").write_text(json.dumps({"generated_at":index_payload["generated_at"],"teams":sorted(teams_seen),"n_teams":len(teams_seen)},indent=2),encoding="utf-8")
    elapsed=time.time()-t0_all
    print(f"Done: {len(index)} ok, {len(failures)} fail in {elapsed:.0f}s → {SIMS_DIR}/index.json")
    if failures:
        raise SystemExit(f"Simulation completed with {len(failures)} fixture failures; see sims/index.json")
    print(f"Hard simulation engine: {'ON' if HARD_SIM else 'OFF'}")

if __name__ == "__main__":
    main()
