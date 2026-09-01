#!/usr/bin/env python3
"""Africa daily segment: fixture board → training → simulations → Sim Lab index."""
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


def run(cmd, env=None, check=True):
    log("exec: " + " ".join(cmd))
    e = os.environ.copy()
    if env:
        e.update({k: str(v) for k, v in env.items() if v is not None})
    r = subprocess.run(cmd, env=e)
    if check and r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def slug(*parts: str) -> str:
    s = "__".join(parts).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80]


def clean_master():
    run([sys.executable, "-m", "africa.clean_master"])


def fetch_fixtures() -> Path:
    """Fetch the authoritative daily board; never use standings credentials."""
    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not backup and not key:
        raise SystemExit("Africa fixtures require API_FOOTBALL_BACKUP_KEY or API_FOOTBALL_KEY")
    out = SAVE / "africa_fixtures_today.json"
    run([sys.executable, "-m", "africa.fetch_today_fixtures"], env={
        "API_FOOTBALL_BACKUP_KEY": backup,
        "API_FOOTBALL_KEY": key,
        "API_FOOTBALL_STANDINGS_KEY": "",
        "AFRICA_FIXTURES_OUT": str(out),
    })
    if not out.exists():
        raise SystemExit("Africa fixture fetch produced no fixture document")
    try:
        doc = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid Africa fixture document: {exc}") from exc
    if doc.get("ok") is not True:
        raise SystemExit("Africa fixture document is not marked ok; refusing downstream work")
    return out


def disable_africa_publication(reason: str):
    """Remove stale Africa rows from the public index when today's board is unavailable."""
    SIMS.mkdir(parents=True, exist_ok=True)
    idx_path = SIMS / "index.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        idx = {}
    sims = [s for s in (idx.get("sims") or []) if s.get("region") != "Africa"]
    idx.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_ok": len(sims),
        "n_africa": 0,
        "africa_status": "unavailable",
        "africa_status_reason": reason[:500],
        "sims": sims,
    })
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    status = {
        "ok": False,
        "status": "unavailable",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    (SAVE / "africa_segment_last.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    log(f"Africa publication disabled for this run: {reason}")


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
    if os.environ.get("SKIP_AFRICA_TRAIN", "").strip().lower() in ("1", "true", "yes"):
        log("SKIP_AFRICA_TRAIN — using existing models")
        return
    master = ROOT / "master_africa_football.parquet"
    if not master.exists():
        master = ROOT / "master_africa_football.csv"
    if not master.exists():
        raise SystemExit("No Africa master dataset available for board training")
    log(f"train countries={countries} teams_on_board={len(teams)}")
    run([sys.executable, "-m", "africa.train_africa"], env={
        "AFRICA_PARQUET": str(master),
        "FOCUS_COUNTRIES": ",".join(countries),
        "MIN_TEAM_MATCHES": os.environ.get("MIN_TEAM_MATCHES", "30"),
        "MAX_BOOST_ROUNDS": os.environ.get("MAX_BOOST_ROUNDS", "600"),
    })


def load_africa_standings() -> tuple[dict, float]:
    cand = SAVE / "standings_latest.json"
    if cand.exists():
        try:
            doc = json.loads(cand.read_text(encoding="utf-8"))
            strength = float(doc.get("strength") or os.environ.get("STANDINGS_STRENGTH", "0.55"))
            tables = {int(k): block.get("rows") or [] for k, block in (doc.get("tables") or {}).items()}
            return tables, strength
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {}, float(os.environ.get("STANDINGS_STRENGTH", "0.55"))


def fetch_standings_for_africa(day: str | None = None):
    if os.environ.get("ENABLE_LIVE_STANDINGS", "").strip().lower() not in ("1", "true", "yes"):
        log("live standings DISABLED")
        return
    if os.environ.get("SKIP_STANDINGS", "").strip().lower() in ("1", "true", "yes"):
        log("SKIP_STANDINGS")
        return
    key = os.environ.get("API_FOOTBALL_KEY", "").strip() or os.environ.get("API_FOOTBALL_STANDINGS_KEY", "").strip()
    env = {
        "API_FOOTBALL_KEY": key,
        "API_FOOTBALL_STANDINGS_KEY": os.environ.get("API_FOOTBALL_STANDINGS_KEY", "").strip() or key,
        "API_FOOTBALL_MAX_STANDINGS": os.environ.get("API_FOOTBALL_MAX_STANDINGS", "0"),
        "STANDINGS_STRENGTH": os.environ.get("STANDINGS_STRENGTH", "0.55"),
    }
    if day:
        env["FIXTURE_DATE"] = day
    run([sys.executable, "fetch_day_standings.py", "--region", "africa", "--merge"], env=env)


def sim_reports(doc: dict) -> list[dict]:
    if os.environ.get("SKIP_AFRICA_SIM", "").strip().lower() in ("1", "true", "yes"):
        log("SKIP_AFRICA_SIM")
        return []
    from africa.sim_africa import simulate_match, to_simlab_document
    from standings_prior import prior_from_league_cache
    tables, strength = load_africa_standings()
    log(f"standings={'available' if tables else 'model-only'} strength={strength}")
    SIMS.mkdir(parents=True, exist_ok=True)
    entries = []
    fixture_count = len(doc.get("fixtures") or [])
    for fx in doc.get("fixtures") or []:
        home, away = (fx.get("home") or "").strip(), (fx.get("away") or "").strip()
        if not home or not away:
            continue
        kickoff = (fx.get("date") or "")[:16].replace("T", " ")
        lid = fx.get("league_id")
        try:
            lid = int(lid) if lid is not None else None
        except (TypeError, ValueError):
            lid = None
        sp = prior_from_league_cache(home, away, lid, tables, strength=strength) if lid else None
        try:
            raw = simulate_match(home, away, (fx.get("country") or "").strip() or None,
                                 odds_h=fx.get("odds_h"), odds_d=fx.get("odds_d"), odds_a=fx.get("odds_a"),
                                 standings_prior=sp if (sp and sp.get("matched")) else None)
        except Exception as exc:
            raise RuntimeError(f"Africa simulation failed for {home} vs {away}: {exc}") from exc
        payload = to_simlab_document(raw, fx=fx)
        sid = "africa_" + slug(home, away, (kickoff or "na")[:10])
        payload["id"] = sid
        (SIMS / f"{sid}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rep, ft = payload["report"], payload["report"]["ft_sim"]
        top = rep.get("top_ft") or []
        entries.append({
            "id": sid, "file": f"{sid}.json", "region": "Africa", "kickoff": kickoff,
            "league": payload["match"]["league"], "div": "AFR", "home": home, "away": away,
            "fixture": f"{home} vs {away}", "odds": "—", "odds_h": fx.get("odds_h"),
            "odds_d": fx.get("odds_d"), "odds_a": fx.get("odds_a"), "models": payload["resolved"]["models"],
            "ft_h": round(100 * float(ft["H"]), 1), "ft_d": round(100 * float(ft["D"]), 1),
            "ft_a": round(100 * float(ft["A"]), 1), "over25": round(100 * float(rep["over25_sim"]), 1),
            "btts": round(100 * float(rep["btts_sim"]), 1), "cs_top": top[0][0] if top else "—",
            "cs_second": top[1][0] if len(top) > 1 else "—", "xg_h": round(float(rep["xg"]["home"]), 3),
            "xg_a": round(float(rep["xg"]["away"]), 3), "xg_total": round(float(rep["xg"]["total"]), 3),
            "cs_home": rep["clean_sheet"]["home"], "cs_away": rep["clean_sheet"]["away"],
            "locked_status": payload["locked_tip"]["status"], "locked_section": payload["locked_tip"]["section"],
            "locked_selection": payload["locked_tip"]["selection"], "locked_model": payload["locked_tip"]["model"],
            "locked_sim": payload["locked_tip"]["sim"], "locked_verdict": payload["locked_tip"]["verdict"],
        })
    if len(entries) != fixture_count:
        raise RuntimeError(f"Africa simulation incomplete: {len(entries)}/{fixture_count} fixtures succeeded")
    return entries


def merge_index(africa_entries: list[dict], feed_date: str):
    SIMS.mkdir(parents=True, exist_ok=True)
    idx_path = SIMS / "index.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        idx = {}
    existing = [s for s in (idx.get("sims") or []) if s.get("region") != "Africa"]
    merged = existing + africa_entries
    generated = datetime.now(timezone.utc).isoformat()
    idx.update({"generated_at": generated, "feed_date": feed_date, "n_ok": len(merged), "n_africa": len(africa_entries), "sims": merged})
    (SIMS / "index_africa.json").write_text(json.dumps({"generated_at": generated, "feed_date": feed_date, "source": "api-football (Africa filter)", "n_ok": len(africa_entries), "sims": africa_entries}, indent=2), encoding="utf-8")
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    log(f"index merged: total={len(merged)} africa={len(africa_entries)}")


def main():
    log("=== Africa daily segment ===")
    clean_master()
    try:
        fx_path = fetch_fixtures()
    except SystemExit as exc:
        if os.environ.get("AFRICA_OPTIONAL", "").strip().lower() in ("1", "true", "yes"):
            reason = str(exc)
            disable_africa_publication(reason)
            log("AFRICA_OPTIONAL=1 — continuing main pipeline without Africa")
            return 0
        raise
    doc = json.loads(fx_path.read_text(encoding="utf-8"))
    teams, countries = teams_from_fixtures(doc)
    log(f"board: {len(doc.get('fixtures') or [])} fixtures | {len(teams)} teams | countries={countries}")
    if not doc.get("fixtures"):
        log("empty Africa board today")
        merge_index([], doc.get("date") or "")
        return 0
    fetch_standings_for_africa(doc.get("date"))
    train_for_board(teams, countries)
    entries = sim_reports(doc)
    merge_index(entries, doc.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    summary = {"ok": True, "status": "complete", "date": doc.get("date"), "n_fixtures": len(doc.get("fixtures") or []), "n_sim_ok": len(entries), "countries": countries, "teams": teams}
    (SAVE / "africa_segment_last.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("=== Africa segment done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
