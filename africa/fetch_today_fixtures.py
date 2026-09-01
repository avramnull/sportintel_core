#!/usr/bin/env python3
"""
Fetch TODAY's African fixtures only — API-Football (api-sports).

Quota-safe design (Free plan ~100 req/day):
  • Exactly ONE request: GET /fixtures?date=YYYY-MM-DD
  • Filter to African countries client-side
  • No per-league loops, no historical seasons

Env:
  API_FOOTBALL_KEY   (required)
  FIXTURE_DATE       optional YYYY-MM-DD (default: today UTC)
  AFRICA_FIXTURES_OUT optional path (default: daily_football_data/africa_fixtures_today.json)

Usage:
  API_FOOTBALL_KEY=xxx python -m africa.fetch_today_fixtures
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://v3.football.api-sports.io"

AFRICA = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
    "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "Congo DR",
    "DR Congo", "Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire", "Djibouti", "Egypt",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
    "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
    "Niger", "Nigeria", "Rwanda", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
    "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
    "Zambia", "Zimbabwe",
}



def fetch_odds_for_fixtures(key: str, fixture_ids: list, delay: float = 0.35) -> dict:
    """
    Best-effort odds pull. Free plan often returns empty for African / current seasons.
    Returns {fixture_id: {H, D, A, bookmaker}} — only when Match Winner exists.
    Uses 1 request per fixture; call only for today's board (small N).
    """
    import time
    out = {}
    if not fixture_ids:
        return out
    # Cap to protect quota (max 15 odds calls per run)
    ids = list(dict.fromkeys(fixture_ids))[:15]
    for fid in ids:
        try:
            r = requests.get(
                f"{API}/odds",
                params={"fixture": fid},
                headers={"x-apisports-key": key, "Accept": "application/json"},
                timeout=30,
            )
            if r.status_code != 200:
                continue
            payload = r.json()
            for block in payload.get("response") or []:
                for bm in block.get("bookmakers") or []:
                    for bet in bm.get("bets") or []:
                        name = (bet.get("name") or "").lower()
                        if name not in ("match winner", "1x2", "full time result"):
                            continue
                        vals = {str(v.get("value")).lower(): float(v.get("odd"))
                                for v in (bet.get("values") or []) if v.get("odd")}
                        # map Home/Draw/Away
                        h = vals.get("home") or vals.get("1")
                        d = vals.get("draw") or vals.get("x")
                        a = vals.get("away") or vals.get("2")
                        if h and d and a:
                            out[int(fid)] = {"H": h, "D": d, "A": a, "bookmaker": bm.get("name")}
                            break
                    if int(fid) in out:
                        break
        except Exception:
            pass
        time.sleep(delay)
    return out


def main() -> int:
    # Load local .env if present (never committed)
    sys.path.insert(0, str(ROOT))
    try:
        import si_config  # noqa: F401 — loads .env
    except Exception:
        pass
    try:
        from api_football_client import key_pool
        keys = key_pool()
    except Exception:
        keys = [k for k in (
            os.environ.get("API_FOOTBALL_KEY", "").strip(),
            os.environ.get("API_FOOTBALL_STANDINGS_KEY", "").strip(),
        ) if k]
    if not keys:
        print("ERROR: set API_FOOTBALL_KEY and/or API_FOOTBALL_STANDINGS_KEY", file=sys.stderr)
        return 2
    key = keys[0]

    day = os.environ.get("FIXTURE_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(os.environ.get(
        "AFRICA_FIXTURES_OUT",
        str(ROOT / "daily_football_data" / "africa_fixtures_today.json"),
    ))
    out.parent.mkdir(parents=True, exist_ok=True)

    # Optional: check remaining quota (costs 1 request — skip unless AFRICA_CHECK_STATUS=1)
    if os.environ.get("AFRICA_CHECK_STATUS", "").strip() in ("1", "true", "yes"):
        st = requests.get(f"{API}/status", headers={"x-apisports-key": key}, timeout=30)
        st.raise_for_status()
        req = (st.json().get("response") or {}).get("requests") or {}
        print(f"[quota] {req.get('current')}/{req.get('limit_day')}")
        if req.get("limit_day") and req.get("current", 0) >= int(req["limit_day"]) - 2:
            print("ERROR: quota almost exhausted — aborting", file=sys.stderr)
            return 3

    print(f"[api-football] ONE request: fixtures?date={day}")
    r = requests.get(
        f"{API}/fixtures",
        params={"date": day},
        headers={"x-apisports-key": key, "Accept": "application/json"},
        timeout=45,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        errs = payload["errors"]
        print(f"API errors: {errs}", file=sys.stderr)
        # Fail over to subordinate key once
        if len(keys) > 1 and key == keys[0]:
            key = keys[1]
            print("[api-football] retrying fixtures with subordinate key", flush=True)
            r = requests.get(
                f"{API}/fixtures",
                params={"date": day},
                headers={"x-apisports-key": key, "Accept": "application/json"},
                timeout=45,
            )
            r.raise_for_status()
            payload = r.json()
        if payload.get("errors"):
            errs = payload["errors"]
            print(f"API errors (after failover): {errs}", file=sys.stderr)
            doc = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "date": day,
                "total_world": 0,
                "total_africa": 0,
                "fixtures": [],
                "api_errors": errs,
            }
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            out.with_suffix(".csv").write_text("", encoding="utf-8")
            return 1

    africa_rows = []
    for x in payload.get("response") or []:
        country = ((x.get("league") or {}).get("country") or "").strip()
        if country not in AFRICA:
            continue
        fx = x.get("fixture") or {}
        lg = x.get("league") or {}
        teams = x.get("teams") or {}
        goals = x.get("goals") or {}
        score = x.get("score") or {}
        africa_rows.append({
            "fixture_id": fx.get("id"),
            "date": (fx.get("date") or "")[:19],
            "timestamp": fx.get("timestamp"),
            "status": (fx.get("status") or {}).get("short"),
            "elapsed": (fx.get("status") or {}).get("elapsed"),
            "country": country,
            "league_id": lg.get("id"),
            "league": lg.get("name"),
            "season": lg.get("season"),
            "round": lg.get("round"),
            "home": (teams.get("home") or {}).get("name"),
            "away": (teams.get("away") or {}).get("name"),
            "home_id": (teams.get("home") or {}).get("id"),
            "away_id": (teams.get("away") or {}).get("id"),
            "goals_home": goals.get("home"),
            "goals_away": goals.get("away"),
            "ht_home": ((score.get("halftime") or {}).get("home")),
            "ht_away": ((score.get("halftime") or {}).get("away")),
            "source": "api-football",
        })

    # Optional odds (AFRICA_FETCH_ODDS=1). Free tier often has none for AF boards.
    if os.environ.get("AFRICA_FETCH_ODDS", "").strip() in ("1", "true", "yes") and africa_rows:
        ids = [r["fixture_id"] for r in africa_rows if r.get("fixture_id")]
        print(f"[api-football] odds pull for {min(len(ids),15)} fixtures (quota-sensitive)")
        odds_map = fetch_odds_for_fixtures(key, ids)
        for r in africa_rows:
            o = odds_map.get(int(r["fixture_id"])) if r.get("fixture_id") else None
            if o:
                r["odds_h"], r["odds_d"], r["odds_a"] = o["H"], o["D"], o["A"]
                r["odds_book"] = o.get("bookmaker")
            else:
                r.setdefault("odds_h", None)
                r.setdefault("odds_d", None)
                r.setdefault("odds_a", None)
        print(f"[api-football] odds found for {sum(1 for r in africa_rows if r.get('odds_h'))}/{len(africa_rows)}")
    else:
        for r in africa_rows:
            r.setdefault("odds_h", None)
            r.setdefault("odds_d", None)
            r.setdefault("odds_a", None)

    doc = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "date": day,
        "total_world": payload.get("results"),
        "total_africa": len(africa_rows),
        "fixtures": africa_rows,
    }
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV twin for scrapers / Sim Lab style
    csv_path = out.with_suffix(".csv")
    if africa_rows:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(africa_rows[0].keys()))
            w.writeheader()
            w.writerows(africa_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    print(f"[ok] Africa fixtures {day}: {len(africa_rows)} (world total {payload.get('results')})")
    for row in africa_rows:
        print(f"  {row['date']} | {row['country']} {row['league']} | {row['home']} vs {row['away']} | {row['status']}")
    print(f"[ok] wrote {out}")
    print(f"[ok] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
