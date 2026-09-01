#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]

def replace(path: str, old: str, new: str):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    if old not in s:
        raise RuntimeError(f"anchor missing: {path}: {old[:100]!r}")
    p.write_text(s.replace(old, new, 1), encoding="utf-8")

# Daily training is fixture-driven only. No global full-history model is trained.
replace("train.py",
'''    focus_list: List[Optional[str]] = [None]\n    if CONFIG["focus_teams"]:\n''',
'''    focus_list: List[str] = []\n    if CONFIG["focus_teams"]:\n''')
replace("train.py",
'''        if not focus_list:\n            log("ERROR: no valid focus teams")\n            sys.exit(1)\n\n    registry = {\n''',
'''        if not focus_list:\n            log("ERROR: no valid focus teams")\n            sys.exit(1)\n\n    # Production daily training is strictly fixture-team scoped.\n    # A global baseline is intentionally not trained in the daily path.\n    registry = {\n''')
replace("train.py",
'''    # Global always sequential\n    team_jobs = []\n''',
'''    # All daily jobs are fixture-team jobs; never enqueue a global job.\n    team_jobs = []\n''')
replace("train.py",
'''    for focus in focus_list:\n        if focus is None:\n            run_name, run_df = "global", labeled\n''',
'''    for focus in focus_list:\n        if focus is None:\n            continue\n        if focus is None:\n            run_name, run_df = "global", labeled\n''')

# Never load an old global model during simulation either.
replace("sim.py",
'''    if runs:\n        return runs\n    g = MODELS_ROOT / "global"\n    if (g / "registry.json").exists() or (g / "models").is_dir():\n        return [(g, "global")]\n    if strict:\n        raise SystemExit(f"No models for {home_canon}/{away_canon} and no global/")\n''',
'''    if runs:\n        return runs\n    if strict:\n        raise SystemExit(f"No fixture-team models for {home_canon}/{away_canon}")\n''')

# Africa training is also board/country scoped; remove its GLOBAL run.
replace("africa/train_africa.py",
'''    registry = {}\n    # GLOBAL\n    registry["GLOBAL"] = train_scope(df, "GLOBAL", out_root)\n\n    # Per-country if enough rows\n''',
'''    registry = {}\n    # Per-country board scopes only. Never train a GLOBAL Africa model\n    # alongside the country/fixture training jobs.\n''')

# Replace Africa fixture provider with Live-score-only, quota-aware access.
(ROOT / "africa/fetch_today_fixtures.py").write_text(textwrap.dedent('''
#!/usr/bin/env python3
"""Fetch today's African fixtures from Live-score API only.

Quota policy: one cached daily board, follow documented pagination only when
needed, deduplicate fixture IDs, and never retry blindly. API credentials are
read only from LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET.
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
import requests

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "daily_football_data"
BASE = "https://livescore-api.com/api-client/fixtures/list.json"
AFRICA_COUNTRIES = {
    "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon",
    "Cape Verde","Central African Republic","Chad","Comoros","Congo","Congo DR",
    "DR Congo","Ivory Coast","Cote d'Ivoire","Côte d'Ivoire","Djibouti","Egypt",
    "Equatorial Guinea","Eritrea","Eswatini","Ethiopia","Gabon","Gambia","Ghana",
    "Guinea","Guinea-Bissau","Kenya","Lesotho","Liberia","Libya","Madagascar",
    "Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique","Namibia",
    "Niger","Nigeria","Rwanda","Senegal","Seychelles","Sierra Leone","Somalia",
    "South Africa","South Sudan","Sudan","Tanzania","Togo","Tunisia","Uganda",
    "Zambia","Zimbabwe",
}

def _is_africa(x):
    c = x.get("country") or {}
    f = x.get("federation") or {}
    return str(c.get("name") or "").strip() in AFRICA_COUNTRIES or str(f.get("name") or "").strip().upper() == "CAF"

def _odd(v):
    try:
        v = float(v)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None

def _fetch(day, key, secret):
    cache_dir = SAVE / "live_score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"fixtures_{day}.json"
    # Same-day cache is authoritative for the daily training board. This is the
    # main quota saver: repeated pipeline retries do not consume another call.
    if cache.exists():
        try:
            age = datetime.now(timezone.utc).timestamp() - cache.stat().st_mtime
            if age < 20 * 3600:
                return json.loads(cache.read_text(encoding="utf-8")), {"cache": True, "pages": 0}
        except Exception:
            pass
    fixtures, page, seen_pages = [], 1, set()
    session = requests.Session()
    while page not in seen_pages:
        seen_pages.add(page)
        r = session.get(BASE, params={"date": day, "key": key, "secret": secret, "page": page}, timeout=45)
        if r.status_code == 429:
            raise RuntimeError("Live-score rate limit (429); cached data should be used on subsequent runs")
        r.raise_for_status()
        payload = r.json()
        if payload.get("success") is not True:
            raise RuntimeError(str(payload.get("error") or "Live-score API success=false"))
        data = payload.get("data") or {}
        batch = data.get("fixtures") or []
        if not isinstance(batch, list):
            raise RuntimeError("Live-score data.fixtures is not a list")
        fixtures.extend(x for x in batch if isinstance(x, dict))
        nxt = data.get("next_page")
        if not nxt:
            break
        try:
            from urllib.parse import parse_qs, urlparse
            page = int(parse_qs(urlparse(str(nxt)).query).get("page", [page + 1])[0])
        except Exception:
            page += 1
        if page > 100:
            raise RuntimeError("Live-score pagination safety limit exceeded")
        sleep(0.25)
    cache.write_text(json.dumps(fixtures, ensure_ascii=False), encoding="utf-8")
    return fixtures, {"cache": False, "pages": len(seen_pages)}

def main():
    day = os.environ.get("FIXTURE_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key, secret = os.environ.get("LIVE_SCORE_API_KEY", "").strip(), os.environ.get("LIVE_SCORE_API_SECRET", "").strip()
    if not key or not secret:
        raise SystemExit("LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET are required")
    out = Path(os.environ.get("AFRICA_FIXTURES_OUT", str(SAVE / "africa_fixtures_today.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw, meta = _fetch(day, key, secret)
        rows, seen = [], set()
        for x in raw:
            if not _is_africa(x):
                continue
            try: fid = int(x.get("id") if x.get("id") is not None else x.get("fixture_id"))
            except (TypeError, ValueError):
                continue
            if fid in seen: continue
            seen.add(fid)
            c, f, comp = x.get("country") or {}, x.get("federation") or {}, x.get("competition") or {}
            h, a = x.get("home") or {}, x.get("away") or {}
            odds = (x.get("odds") or {}).get("pre") or {}
            scheduled = str(x.get("scheduled") or x.get("time") or "").strip()[:8]
            dt = f"{day}T{scheduled}" if scheduled else day
            try: ts = int(datetime.fromisoformat(dt).replace(tzinfo=timezone.utc).timestamp())
            except Exception: ts = None
            rows.append({
                "fixture_id": fid, "date": dt, "timestamp": ts, "status": "NS", "elapsed": None,
                "country": str(c.get("name") or f.get("name") or "CAF"), "federation": str(f.get("name") or "CAF"),
                "league_id": comp.get("id"), "league": comp.get("name"), "season": None, "round": x.get("round"),
                "home": h.get("name"), "away": a.get("name"), "home_id": h.get("id"), "away_id": a.get("id"),
                "goals_home": None, "goals_away": None, "ht_home": None, "ht_away": None,
                "odds_h": _odd(odds.get("1")), "odds_d": _odd(odds.get("X")), "odds_a": _odd(odds.get("2")),
                "odds_book": "live-score-api", "source": "live-score-api", "provider": "live-score-api",
                "schema_version": "africa-fixture-v2",
            })
        doc = {"fetched_at": datetime.now(timezone.utc).isoformat(), "date": day, "provider": "live-score-api",
               "total_world": len(raw), "total_africa": len(rows), "fixtures": rows, "ok": True, "cache": bool(meta.get("cache"))}
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        csv = out.with_suffix(".csv")
        if rows:
            import csv as csvmod
            with csv.open("w", newline="", encoding="utf-8") as f:
                w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        else: csv.write_text("", encoding="utf-8")
        print(f"[ok] Live-score Africa board {day}: {len(rows)} fixtures | cache={meta.get('cache')} pages={meta.get('pages')}")
        return 0
    except Exception as exc:
        doc = {"fetched_at": datetime.now(timezone.utc).isoformat(), "date": day, "provider": "live-score-api",
               "fixtures": [], "ok": False, "api_errors": [str(exc)]}
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        out.with_suffix(".csv").write_text("", encoding="utf-8")
        print(f"FATAL: Live-score fixture fetch failed: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
'''), encoding="utf-8")

# Africa segment no longer needs API-Football credentials and does not call its standings path.
replace("africa/daily_africa_segment.py",
'''    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()\n    key = os.environ.get("API_FOOTBALL_KEY", "").strip()\n    if not backup and not key:\n        raise SystemExit("Africa fixtures require API_FOOTBALL_BACKUP_KEY or API_FOOTBALL_KEY")\n''',
'''    live_key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()\n    live_secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()\n    if not (live_key and live_secret):\n        raise SystemExit("Africa fixtures require LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET")\n''')
replace("africa/daily_africa_segment.py",
'''        "API_FOOTBALL_BACKUP_KEY": backup,\n        "API_FOOTBALL_KEY": key,\n        "LIVE_SCORE_API_KEY": live_key,\n        "LIVE_SCORE_API_SECRET": live_secret,\n        "API_FOOTBALL_STANDINGS_KEY": "",\n''',
'''        "LIVE_SCORE_API_KEY": live_key,\n        "LIVE_SCORE_API_SECRET": live_secret,\n''')
# Disable the old API standings call completely; Live-score standings will be a separate provider module.
start = "def fetch_standings_for_africa(day: str | None = None):"
s = (ROOT / "africa/daily_africa_segment.py").read_text(encoding="utf-8")
pos = s.find(start)
if pos >= 0:
    end = s.find("\ndef sim_reports", pos)
    if end < 0: raise RuntimeError("standings function end anchor missing")
    s = s[:pos] + '''def fetch_standings_for_africa(day: str | None = None):\n    log("API standings provider removed; Live-score standings are handled separately")\n    return\n\n''' + s[end+1:]
    (ROOT / "africa/daily_africa_segment.py").write_text(s, encoding="utf-8")

# Remove API-Football configuration from the runtime config.
replace("si_config.py",
'''# API-Football: backup key is the preferred fixture credential when supplied.\n# It is intentionally read from the environment/CI secret only; no secret is stored in git.\nAPI_FOOTBALL_BACKUP_KEY = _env("API_FOOTBALL_BACKUP_KEY")\nAPI_FOOTBALL_KEY = _env("API_FOOTBALL_KEY")\n# Standings API integration is disabled by the daily workflow; keep this value\n# available only for compatibility with older callers, never as fixture fallback.\nAPI_FOOTBALL_STANDINGS_KEY = _env("API_FOOTBALL_STANDINGS_KEY")\n''', "")

# Local standings mapping no longer imports the removed API client.
replace("local_standings.py", 'from api_football_client import FDC_DIV_TO_LEAGUE  # noqa: E402\n', '''FDC_DIV_TO_LEAGUE = {\n    "E0":39,"E1":40,"E2":41,"E3":42,"EC":43,"SC0":179,"SC1":180,"D1":78,"D2":79,\n    "SP1":140,"SP2":141,"I1":135,"I2":136,"F1":61,"F2":62,"N1":88,"B1":144,\n    "P1":94,"T1":203,"G1":197,\n}\n''')

# standings_prior keeps its own lightweight normalization instead of importing API-Football.
replace("standings_prior.py", 'from api_football_client import _norm, team_lookup\n', '''def _norm(name):\n    import re\n    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()\n\ndef team_lookup(rows):\n    out = {}\n    for row in rows or []:\n        if not isinstance(row, dict): continue\n        name = row.get("team") or row.get("name") or ""\n        if name: out[_norm(name)] = row\n    return out\n''')

# Remove obsolete API-Football-only modules.
for rel in ("api_football_client.py", "fetch_day_standings.py"):
    p = ROOT / rel
    if p.exists(): p.unlink()

for rel in ("train.py", "sim.py", "africa/train_africa.py", "africa/fetch_today_fixtures.py", "africa/daily_africa_segment.py", "si_config.py", "local_standings.py", "standings_prior.py"):
    compile((ROOT / rel).read_text(encoding="utf-8"), rel, "exec")
print("runtime cleanup applied: fixture-team-only training + Live-score-only Africa + API-Football removed")
Path(__file__).unlink()
