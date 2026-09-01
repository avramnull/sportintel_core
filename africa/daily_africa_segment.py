#!/usr/bin/env python3
"""
Africa daily segment (quota-safe):

  1) Clean/organize master_africa parquet (local)
  2) Fetch TODAY Africa fixtures — 1 API-Football request
  3) Team-specific train for teams on today's board (FOCUS_COUNTRIES + FOCUS teams)
  4) Simulate each fixture → Sim Lab style JSON
  5) Merge into daily_football_data/sims/index.json (region=Africa)

Env:
  API_FOOTBALL_KEY   required for fixtures step (skip fixtures if missing)
  AFRICA_SEGMENT=1   enable from daily_pipeline
  SKIP_AFRICA_TRAIN  1 = only fixtures + sim with existing models
  SKIP_AFRICA_SIM    1 = train only
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SAVE = ROOT / "daily_football_data"
SIMS = SAVE / "sims"


def log(msg: str):
    print(f"[africa] {msg}", flush=True)


def run(cmd, env=None, check=False):
    log("exec: " + " ".join(cmd))
    e = os.environ.copy()
    if env:
        e.update({k: str(v) for k, v in env.items() if v is not None})
    r = subprocess.run(cmd, env=e)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode


def slug(*parts: str) -> str:
    s = "__".join(parts)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80]


def clean_master():
    run([sys.executable, "-m", "africa.clean_master"])


def fetch_fixtures() -> Path | None:
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        log("API_FOOTBALL_KEY not set — skip live Africa fixtures")
        return None
    out = SAVE / "africa_fixtures_today.json"
    rc = run(
        [sys.executable, "-m", "africa.fetch_today_fixtures"],
        env={
            "API_FOOTBALL_KEY": key,
            "AFRICA_FIXTURES_OUT": str(out),
        },
    )
    if rc != 0 or not out.exists():
        log("fixtures fetch failed")
        return None
    return out


def teams_from_fixtures(doc: dict) -> tuple[list[str], list[str]]:
    teams, countries = [], []
    for fx in doc.get("fixtures") or []:
        for k in ("home", "away"):
            t = (fx.get(k) or "").strip()
            if t and t not in teams:
                teams.append(t)
        c = (fx.get("country") or "").strip()
        if c and c not in countries:
            countries.append(c)
    return teams, countries


def train_for_board(teams: list[str], countries: list[str]):
    if os.environ.get("SKIP_AFRICA_TRAIN", "").strip() in ("1", "true", "yes"):
        log("SKIP_AFRICA_TRAIN — using existing models")
        return
    master = ROOT / "master_africa_football.parquet"
    if not master.exists():
        master = ROOT / "master_africa_football.csv"
    if not master.exists():
        log("no Africa master — skip train")
        return
    # Country models for board countries (team-specific in Africa = per-country focus)
    env = {
        "AFRICA_PARQUET": str(master),
        "FOCUS_COUNTRIES": ",".join(countries) if countries else "",
        "MIN_TEAM_MATCHES": os.environ.get("MIN_TEAM_MATCHES", "30"),
        "MAX_BOOST_ROUNDS": os.environ.get("MAX_BOOST_ROUNDS", "600"),
    }
    log(f"train countries={countries} teams_on_board={len(teams)}")
    run([sys.executable, "-m", "africa.train_africa"], env=env)


def sim_reports(doc: dict) -> list[dict]:
    if os.environ.get("SKIP_AFRICA_SIM", "").strip() in ("1", "true", "yes"):
        log("SKIP_AFRICA_SIM")
        return []
    from africa.sim_africa import simulate_match, to_simlab_document

    SIMS.mkdir(parents=True, exist_ok=True)
    entries = []
    for fx in doc.get("fixtures") or []:
        home = (fx.get("home") or "").strip()
        away = (fx.get("away") or "").strip()
        country = (fx.get("country") or "").strip() or None
        if not home or not away:
            continue
        kickoff = (fx.get("date") or "")[:16].replace("T", " ")
        league = fx.get("league") or "Africa"
        try:
            raw = simulate_match(home, away, country, odds_h=fx.get('odds_h'), odds_d=fx.get('odds_d'), odds_a=fx.get('odds_a'))
        except Exception as e:
            log(f"sim fail {home} vs {away}: {e}")
            continue
        payload = to_simlab_document(raw, fx=fx)
        sid = "africa_" + slug(home, away, (kickoff or "na")[:10])
        fname = f"{sid}.json"
        payload["id"] = sid
        (SIMS / fname).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rep = payload["report"]
        ft = rep["ft_sim"]
        top = rep.get("top_ft") or []
        entries.append({
            "id": sid,
            "file": fname,
            "region": "Africa",
            "kickoff": kickoff,
            "league": payload["match"]["league"],
            "div": "AFR",
            "home": home,
            "away": away,
            "fixture": f"{home} vs {away}",
            "odds": (
                f"{fx.get('odds_h')}/{fx.get('odds_d')}/{fx.get('odds_a')}"
                if fx.get("odds_h") and fx.get("odds_d") and fx.get("odds_a")
                else "—"
            ),
            "odds_h": fx.get("odds_h"),
            "odds_d": fx.get("odds_d"),
            "odds_a": fx.get("odds_a"),
            "models": payload["resolved"]["models"],
            "ft_h": round(100 * float(ft["H"]), 1),
            "ft_d": round(100 * float(ft["D"]), 1),
            "ft_a": round(100 * float(ft["A"]), 1),
            "over25": round(100 * float(rep["over25_sim"]), 1),
            "btts": round(100 * float(rep["btts_sim"]), 1),
            "cs_top": top[0][0] if top else "—",
            "cs_second": top[1][0] if len(top) > 1 else "—",
            "xg_h": round(float(rep["xg"]["home"]), 3),
            "xg_a": round(float(rep["xg"]["away"]), 3),
            "xg_total": round(float(rep["xg"]["total"]), 3),
            "cs_home": rep["clean_sheet"]["home"],
            "cs_away": rep["clean_sheet"]["away"],
            "locked_status": payload["locked_tip"]["status"],
            "locked_section": payload["locked_tip"]["section"],
            "locked_selection": payload["locked_tip"]["selection"],
            "locked_model": payload["locked_tip"]["model"],
            "locked_sim": payload["locked_tip"]["sim"],
            "locked_verdict": payload["locked_tip"]["verdict"],
        })
        log(f"sim OK {sid} FT {entries[-1]['ft_h']}/{entries[-1]['ft_d']}/{entries[-1]['ft_a']}")
    return entries



def merge_index(africa_entries: list[dict], feed_date: str):
    SIMS.mkdir(parents=True, exist_ok=True)
    idx_path = SIMS / "index.json"
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            idx = {}
    else:
        idx = {}
    existing = [s for s in (idx.get("sims") or []) if s.get("region") != "Africa"]
    # drop prior africa ids for this feed_date
    merged = existing + africa_entries
    idx["generated_at"] = datetime.now(timezone.utc).isoformat()
    idx["feed_date"] = feed_date or idx.get("feed_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    idx["n_ok"] = len(merged)
    idx["n_africa"] = len(africa_entries)
    idx["sims"] = merged
    # also write dedicated africa index for clarity
    africa_idx = {
        "generated_at": idx["generated_at"],
        "feed_date": idx["feed_date"],
        "source": "api-football (Africa filter)",
        "n_ok": len(africa_entries),
        "sims": africa_entries,
    }
    (SIMS / "index_africa.json").write_text(json.dumps(africa_idx, indent=2), encoding="utf-8")
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    log(f"index merged: total={len(merged)} africa={len(africa_entries)}")


def main():
    log("=== Africa daily segment ===")
    clean_master()
    fx_path = fetch_fixtures()
    if not fx_path:
        log("no fixtures — segment ends after clean")
        return 0
    doc = json.loads(fx_path.read_text(encoding="utf-8"))
    teams, countries = teams_from_fixtures(doc)
    log(f"board: {len(doc.get('fixtures') or [])} fixtures | {len(teams)} teams | countries={countries}")
    if not (doc.get("fixtures") or []):
        log("empty Africa board today")
        merge_index([], doc.get("date") or "")
        return 0
    train_for_board(teams, countries)
    entries = sim_reports(doc)
    merge_index(entries, doc.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    # summary for pipeline
    summary = {
        "date": doc.get("date"),
        "n_fixtures": len(doc.get("fixtures") or []),
        "n_sim_ok": len(entries),
        "countries": countries,
        "teams": teams,
    }
    (SAVE / "africa_segment_last.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("=== Africa segment done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
