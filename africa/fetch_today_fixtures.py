#!/usr/bin/env python3
"""Fetch TODAY's African fixtures with provider failover.

Primary: API-Football (backup fixture key first, then primary fixture key).
Fallback: Live-score API using LIVE_SCORE_API_KEY/SECRET.
Standings credentials are never used for fixtures.

Authentication/access failures do not become an empty fixture board. If the
primary provider fails, the fallback is attempted; if every provider fails,
this command returns non-zero and writes an explicit ok=false document so the
pipeline cannot publish stale Africa simulations.
"""
from __future__ import annotations

import csv
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


def _keys() -> list[str]:
    """Return API-Football fixture credentials in backup-first order; standings key is never used."""
    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()
    primary = os.environ.get("API_FOOTBALL_KEY", "").strip()
    out = []
    for key in (backup, primary):
        if key and key not in out:
            out.append(key)
    if out:
        return out
    try:
        sys.path.insert(0, str(ROOT))
        from api_football_client import key_pool
        return key_pool()
    except Exception:
        return []


def fetch_odds_for_fixtures(key: str, fixture_ids: list, delay: float = 0.35) -> dict:
    """Best-effort API-Football odds pull for today's board; odds are optional."""
    import time
    out = {}
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
            if payload.get("errors"):
                continue
            for block in payload.get("response") or []:
                for bm in block.get("bookmakers") or []:
                    for bet in bm.get("bets") or []:
                        name = (bet.get("name") or "").lower()
                        if name not in ("match winner", "1x2", "full time result"):
                            continue
                        vals = {
                            str(v.get("value")).lower(): float(v.get("odd"))
                            for v in (bet.get("values") or []) if v.get("odd")
                        }
                        h = vals.get("home") or vals.get("1")
                        d = vals.get("draw") or vals.get("x")
                        a = vals.get("away") or vals.get("2")
                        if h and d and a:
                            out[int(fid)] = {"H": h, "D": d, "A": a, "bookmaker": bm.get("name")}
                            break
                    if int(fid) in out:
                        break
        except (requests.RequestException, ValueError, TypeError):
            pass
        time.sleep(delay)
    return out


def _fetch_fixture_payload(day: str, keys: list[str]) -> tuple[dict, str]:
    if not keys:
        raise RuntimeError("No API-Football fixture key configured")

    last_error = "unknown API error"
    for idx, key in enumerate(keys, start=1):
        try:
            print(f"[api-football] fixtures?date={day} (fixture key {idx}/{len(keys)})", flush=True)
            r = requests.get(
                f"{API}/fixtures",
                params={"date": day},
                headers={"x-apisports-key": key, "Accept": "application/json"},
                timeout=45,
            )
            r.raise_for_status()
            payload = r.json()
            errs = payload.get("errors")
            if errs:
                last_error = str(errs)
                print(f"API errors (fixture key {idx}): {errs}", file=sys.stderr, flush=True)
                continue
            return payload, key
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            print(f"API request failed (fixture key {idx}): {exc}", file=sys.stderr, flush=True)

    raise RuntimeError(
        "API-Football fixtures unavailable after trying all configured fixture keys: "
        + last_error
    )


def _fetch_live_score_fallback(day: str) -> tuple[list[dict], dict]:
    """Fetch today's African fixtures from Live-score API as provider fallback."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_score_api_fixtures import fetch_africa_fixtures
    return fetch_africa_fixtures(day)


def _write_csv(rows: list[dict], csv_path: Path) -> None:
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    try:
        import si_config  # noqa: F401 — loads local .env
    except Exception:
        pass

    day = os.environ.get("FIXTURE_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(os.environ.get(
        "AFRICA_FIXTURES_OUT",
        str(ROOT / "daily_football_data" / "africa_fixtures_today.json"),
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")

    # Provider 1: API-Football.
    keys = _keys()
    if keys:
        try:
            payload, key = _fetch_fixture_payload(day, keys)
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
                    "ht_home": (score.get("halftime") or {}).get("home"),
                    "ht_away": (score.get("halftime") or {}).get("away"),
                    "source": "api-football",
                })

            if os.environ.get("AFRICA_FETCH_ODDS", "").strip().lower() in ("1", "true", "yes") and africa_rows:
                ids = [r["fixture_id"] for r in africa_rows if r.get("fixture_id")]
                print(f"[api-football] odds pull for {min(len(ids), 15)} fixtures (quota-sensitive)")
                odds_map = fetch_odds_for_fixtures(key, ids)
                for r in africa_rows:
                    o = odds_map.get(int(r["fixture_id"])) if r.get("fixture_id") else None
                    r["odds_h"] = o.get("H") if o else None
                    r["odds_d"] = o.get("D") if o else None
                    r["odds_a"] = o.get("A") if o else None
                    r["odds_book"] = o.get("bookmaker") if o else None
            else:
                for r in africa_rows:
                    r.update({"odds_h": None, "odds_d": None, "odds_a": None})

            doc = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "date": day,
                "provider": "api-football",
                "total_world": payload.get("results"),
                "total_africa": len(africa_rows),
                "fixtures": africa_rows,
                "ok": True,
            }
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_csv(africa_rows, csv_path)
            print(f"[ok] Africa fixtures {day}: {len(africa_rows)} (world total {payload.get('results')})")
            for row in africa_rows:
                print(f"  {row['date']} | {row['country']} {row['league']} | {row['home']} vs {row['away']} | {row['status']}")
            print(f"[ok] provider=api-football wrote {out}")
            return 0
        except RuntimeError as exc:
            print(f"[warn] API-Football Africa provider unavailable: {exc}", file=sys.stderr, flush=True)
    else:
        print("[warn] API-Football fixture credentials unavailable; trying Live-score API", file=sys.stderr, flush=True)

    # Provider 2: Live-score API. Credentials are separate from API-Football and
    # are never treated as standings credentials.
    try:
        africa_rows, meta = _fetch_live_score_fallback(day)
        doc = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "provider": "live-score-api",
            "total_world": meta.get("raw_fixtures"),
            "total_africa": len(africa_rows),
            "fixtures": africa_rows,
            "ok": True,
            "fallback": True,
        }
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(africa_rows, csv_path)
        print(f"[ok] Africa fixtures {day}: {len(africa_rows)} via Live-score API fallback")
        for row in africa_rows:
            print(f"  {row['date']} | {row['country']} {row['league']} | {row['home']} vs {row['away']} | {row['status']}")
        print(f"[ok] provider=live-score-api wrote {out}")
        return 0
    except Exception as exc:
        error = f"All African fixture providers failed. API-Football unavailable and Live-score API fallback failed: {exc}"
        print(f"FATAL: {error}", file=sys.stderr)
        doc = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "provider": None,
            "total_world": None,
            "total_africa": None,
            "fixtures": [],
            "api_errors": [error],
            "ok": False,
        }
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
