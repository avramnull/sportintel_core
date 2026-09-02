#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(rel: str, text: str) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_train() -> None:
    p = ROOT / "train.py"
    s = p.read_text(encoding="utf-8")
    s = s.replace('focus_list: List[Optional[str]] = [None]', 'focus_list: List[Optional[str]] = []', 1)
    pat = re.compile(r'\n    # Global always sequential\n    team_jobs = \[\]\n    for focus in focus_list:\n        if focus is None:\n.*?\n            continue\n\n        tid = team2id\[focus\]', re.S)
    ns, n = pat.subn('\n    team_jobs = []\n    for focus in focus_list:\n        tid = team2id[focus]', s, count=1)
    if n:
        s = ns
    # Older variant without the comment is also handled.
    pat2 = re.compile(r'\n    for focus in focus_list:\n        if focus is None:\n.*?\n            continue\n\n        tid = team2id\[focus\]', re.S)
    s, _ = pat2.subn('\n    for focus in focus_list:\n        tid = team2id[focus]', s, count=1)
    p.write_text(s, encoding="utf-8")


def patch_sim() -> None:
    p = ROOT / "sim.py"
    s = p.read_text(encoding="utf-8")
    # Remove old global-model fallback/inclusion completely.
    s = re.sub(
        r'\n    g = MODELS_ROOT / "global"\n    if \(g / "registry\\.json"\)\.exists\(\) or \(g / "models"\)\.is_dir\(\):\n(?:        .*\n)+?    if strict:',
        '\n    if strict:', s, count=1)
    s = s.replace('raise SystemExit(f"No models for {home_canon}/{away_canon} and no global/")',
                  'raise SystemExit(f"No fixture-team models for {home_canon}/{away_canon}")')
    s = s.replace('        reliability = 2.50 if name == "global" else 1.00\n        weights.append(conf * family_w.get(name, 1.0) * reliability)',
                  '        weights.append(conf * family_w.get(name, 1.0))')
    s = s.replace('        reliability = 2.50 if label == "global" else 1.00\n        weights.append(reliability / (0.55 + ent))',
                  '        weights.append(1.0 / (0.55 + ent))')
    p.write_text(s, encoding="utf-8")


LIVE_SCORE = r'''#!/usr/bin/env python3
"""Live-score API source for today's African fixtures.

Only LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET are used. Requests are cached
on disk, pages are deduplicated, and a small request budget prevents quota
waste. No other football provider is used here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

AFRICA_COUNTRIES = {
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
BASE = "https://livescore-api.com/api-client/fixtures/list.json"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "daily_football_data" / "live_score_cache"


def _is_africa(row: dict[str, Any]) -> bool:
    country = row.get("country") or {}
    federation = row.get("federation") or {}
    return str(country.get("name") or "").strip() in AFRICA_COUNTRIES or str(federation.get("name") or "").strip().upper() == "CAF"


def _cache_path(day: str, page: int) -> Path:
    raw = f"fixtures|date={day}|page={page}".encode()
    return CACHE / (hashlib.sha256(raw).hexdigest() + ".json")


def _request(session: requests.Session, key: str, secret: str, day: str, page: int) -> dict[str, Any]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(day, page)
    ttl = int(os.environ.get("LIVE_SCORE_FIXTURE_CACHE_TTL", "21600"))
    if cp.exists() and time.time() - cp.stat().st_mtime < ttl:
        return json.loads(cp.read_text(encoding="utf-8"))
    budget = int(os.environ.get("LIVE_SCORE_MAX_REQUESTS", "20"))
    used = sum(1 for p in CACHE.glob("*.json") if time.time() - p.stat().st_mtime < 86400)
    if used >= budget:
        raise RuntimeError(f"Live-score daily request budget exhausted ({budget})")
    r = session.get(BASE, params={"date": day, "key": key, "secret": secret, "page": page}, timeout=45)
    if r.status_code == 429:
        raise RuntimeError("Live-score API quota/rate limit reached; using cached data only")
    if r.status_code >= 500:
        raise RuntimeError(f"Live-score API server error {r.status_code}")
    r.raise_for_status()
    payload = r.json()
    if payload.get("success") is not True:
        raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))
    cp.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(float(os.environ.get("LIVE_SCORE_MIN_INTERVAL_SEC", "0.5")))
    return payload


def _pages(key: str, secret: str, day: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page, seen = 1, set()
    with requests.Session() as session:
        while page not in seen:
            seen.add(page)
            payload = _request(session, key, secret, day, page)
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                raise RuntimeError("Live-score API data is not an object")
            batch = data.get("fixtures") or []
            if not isinstance(batch, list):
                raise RuntimeError("Live-score API data.fixtures is not a list")
            out.extend(x for x in batch if isinstance(x, dict))
            nxt = data.get("next_page")
            if not nxt:
                break
            try:
                page = int(parse_qs(urlparse(str(nxt)).query).get("page", [page + 1])[0])
            except Exception:
                page += 1
            if page > 100:
                raise RuntimeError("Live-score pagination exceeded safety limit")
    return out


def _odd(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x > 1.0 else None
    except (TypeError, ValueError):
        return None


def fetch_africa_fixtures(day: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()
    secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET are not configured")
    raw = _pages(key, secret, day)
    out, seen = [], set()
    for x in raw:
        if not _is_africa(x):
            continue
        fid = x.get("id") if x.get("id") is not None else x.get("fixture_id")
        try:
            fixture_id = int(fid)
        except (TypeError, ValueError):
            continue
        if fixture_id in seen:
            continue
        seen.add(fixture_id)
        country = x.get("country") or {}
        federation = x.get("federation") or {}
        competition = x.get("competition") or {}
        home, away = x.get("home") or {}, x.get("away") or {}
        odds = (x.get("odds") or {}).get("pre") or {}
        scheduled = str(x.get("scheduled") or x.get("time") or "").strip()[:8]
        fixture_date = f"{day}T{scheduled}" if scheduled else day
        try:
            timestamp = int(datetime.fromisoformat(fixture_date).replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            timestamp = None
        out.append({
            "fixture_id": fixture_id, "date": fixture_date, "timestamp": timestamp,
            "status": "NS", "elapsed": None,
            "country": str(country.get("name") or federation.get("name") or "CAF"),
            "federation": str(federation.get("name") or "CAF"),
            "league_id": competition.get("id"), "league": competition.get("name"),
            "season": None, "round": x.get("round"),
            "home": home.get("name"), "away": away.get("name"),
            "home_id": home.get("id"), "away_id": away.get("id"),
            "goals_home": None, "goals_away": None, "ht_home": None, "ht_away": None,
            "odds_h": _odd(odds.get("1")), "odds_d": _odd(odds.get("X")), "odds_a": _odd(odds.get("2")),
            "odds_book": "live-score-api", "source": "live-score-api",
            "provider": "live-score-api", "schema_version": "africa-fixture-v2",
        })
    return out, {"provider": "live-score-api", "date": day, "raw_fixtures": len(raw), "africa_fixtures": len(out), "fetched_at": datetime.now(timezone.utc).isoformat()}
'''


def patch_africa_segment() -> None:
    p = ROOT / "africa" / "daily_africa_segment.py"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r'def fetch_fixtures\(\) -> Path:.*?\n\ndef disable_africa_publication', '''def fetch_fixtures() -> Path:\n    """Fetch today's African fixture board from Live-score API only."""\n    live_key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()\n    live_secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()\n    if not live_key or not live_secret:\n        raise SystemExit("Africa fixtures require LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET")\n    out = SAVE / "africa_fixtures_today.json"\n    run([sys.executable, "-m", "africa.fetch_today_fixtures"], env={\n        "LIVE_SCORE_API_KEY": live_key, "LIVE_SCORE_API_SECRET": live_secret,\n        "AFRICA_FIXTURES_OUT": str(out),\n    })\n    if not out.exists():\n        raise SystemExit("Africa fixture fetch produced no fixture document")\n    doc = json.loads(out.read_text(encoding="utf-8"))\n    if doc.get("ok") is not True:\n        raise SystemExit("Africa fixture document is not marked ok")\n    return out\n\n\ndef disable_africa_publication''', s, flags=re.S)
    s = re.sub(r'def fetch_standings_for_africa\(day: str \| None = None\):.*?\n\ndef sim_reports', '''def fetch_standings_for_africa(day: str | None = None):\n    log("standings disabled: Live-score fixture board only")\n    return\n\n\ndef sim_reports''', s, flags=re.S)
    s = s.replace('"source": "api-football (Africa filter)"', '"source": "live-score-api"')
    p.write_text(s, encoding="utf-8")


def patch_fetch_today() -> None:
    write("africa/live_score_api_fixtures.py", LIVE_SCORE)
    write("africa/fetch_today_fixtures.py", '''#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport csv\nimport json\nimport os\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n\nfrom .live_score_api_fixtures import fetch_africa_fixtures\n\nROOT = Path(__file__).resolve().parents[1]\n\ndef main() -> int:\n    day = os.environ.get("FIXTURE_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")\n    out = Path(os.environ.get("AFRICA_FIXTURES_OUT", str(ROOT / "daily_football_data" / "africa_fixtures_today.json")))\n    out.parent.mkdir(parents=True, exist_ok=True)\n    csv_path = out.with_suffix(".csv")\n    try:\n        rows, meta = fetch_africa_fixtures(day)\n        doc = {"fetched_at": datetime.now(timezone.utc).isoformat(), "date": day, "provider": "live-score-api", "total_world": meta.get("raw_fixtures"), "total_africa": len(rows), "fixtures": rows, "ok": True}\n        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")\n        if rows:\n            with csv_path.open("w", newline="", encoding="utf-8") as f:\n                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)\n        else:\n            csv_path.write_text("", encoding="utf-8")\n        print(f"[ok] Africa fixtures {day}: {len(rows)} via Live-score API")\n        return 0\n    except Exception as exc:\n        doc = {"fetched_at": datetime.now(timezone.utc).isoformat(), "date": day, "provider": "live-score-api", "total_world": None, "total_africa": None, "fixtures": [], "api_errors": [str(exc)], "ok": False}\n        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")\n        csv_path.write_text("", encoding="utf-8")\n        print(f"FATAL: Live-score API failed: {exc}")\n        return 1\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n''')


def patch_standings_helpers() -> None:
    p = ROOT / "local_standings.py"
    if p.exists():
        s = p.read_text(encoding="utf-8")
        s = re.sub(r'^from api_football_client import FDC_DIV_TO_LEAGUE.*\n', '', s, flags=re.M)
        if 'FDC_DIV_TO_LEAGUE = {' not in s:
            s = 'FDC_DIV_TO_LEAGUE = {"E0":39,"E1":40,"E2":41,"E3":42,"EC":43,"SC0":179,"SC1":180,"D1":78,"D2":79,"SP1":140,"SP2":141,"I1":135,"I2":136,"F1":61,"F2":62,"N1":88,"B1":144,"P1":94,"T1":203,"G1":197}\n\n' + s
        s = s.replace("API-Football", "external provider").replace("api-football", "external-provider")
        p.write_text(s, encoding="utf-8")
    p = ROOT / "standings_prior.py"
    if p.exists():
        s = p.read_text(encoding="utf-8")
        s = re.sub(r'^from api_football_client import _norm, team_lookup\n', '', s, flags=re.M)
        if 'def _norm(name):' not in s:
            marker = 'from __future__ import annotations\n'
            helpers = '''\nimport re\n\ndef _norm(name):\n    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()\n\ndef team_lookup(rows):\n    out = {}\n    for row in rows or []:\n        if isinstance(row, dict):\n            name = row.get("team") or row.get("name") or ""\n            if name:\n                out[_norm(name)] = row\n    return out\n'''
            s = s.replace(marker, marker + helpers, 1)
        s = s.replace("API-Football", "external provider").replace("api-football", "external-provider")
        p.write_text(s, encoding="utf-8")


def scrub_config() -> None:
    p = ROOT / "si_config.py"
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        lines = [x for x in lines if "API_FOOTBALL" not in x and "api-football" not in x.lower()]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    patch_train()
    patch_sim()
    patch_fetch_today()
    patch_africa_segment()
    patch_standings_helpers()
    scrub_config()
    for rel in ("api_football_client.py", "fetch_day_standings.py"):
        p = ROOT / rel
        if p.exists():
            p.unlink()
    global_dir = ROOT / "football_models" / "global"
    if global_dir.exists():
        shutil.rmtree(global_dir)

    forbidden = []
    for rel in ("train.py", "sim.py", "daily_pipeline.py", "local_standings.py", "standings_prior.py", "si_config.py", "africa/fetch_today_fixtures.py", "africa/daily_africa_segment.py", "africa/live_score_api_fixtures.py", "africa/train_africa.py"):
        p = ROOT / rel
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if re.search(r"API_FOOTBALL|api_football_client|api-football", text, re.I):
                forbidden.append(rel)
            compile(text, rel, "exec")
    if forbidden:
        raise SystemExit("provider references remain: " + ", ".join(forbidden))
    print("FINAL MIGRATION OK: fixture-team-only training + Live-score-only Africa + no API-Football")
    try:
        Path(__file__).unlink()
    except OSError:
        pass

if __name__ == "__main__":
    main()
