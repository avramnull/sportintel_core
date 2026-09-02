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


def patch_file(rel: str, fn) -> None:
    p = ROOT / rel
    if not p.exists():
        raise RuntimeError(f"missing required file: {rel}")
    old = p.read_text(encoding="utf-8")
    new = fn(old)
    if new != old:
        p.write_text(new, encoding="utf-8")
        print(f"patched {rel}")
    else:
        print(f"unchanged {rel}")


def patch_train(s: str) -> str:
    s = s.replace("focus_list: List[Optional[str]] = [None]", "focus_list: List[Optional[str]] = []")
    # Daily production must not inject a global model. Remove only the optional
    # global branch if it is present; leave train_global available for offline use.
    s = re.sub(
        r"\n\s*if\s+os\.environ\.get\(\"TRAIN_GLOBAL\".*?\n(?=\s*(?:focus_list|for\s+target|results\s*=))",
        "\n",
        s,
        flags=re.S,
    )
    return s


def patch_sim(s: str) -> str:
    # Remove global-model loading/fallback blocks and global reliability weighting.
    s = re.sub(r"\n\s*global_model\s*=.*?(?=\n\s*(?:models|team_models|if|for|return))", "\n", s, flags=re.S)
    s = re.sub(r"\n\s*if\s+global_model.*?(?=\n\s*(?:#|if|for|return))", "\n", s, flags=re.S)
    s = s.replace("global model", "fixture-team model")
    s = re.sub(r"\n\s*global_reliability\s*=.*", "", s)
    s = s.replace("global reliability", "fixture-team reliability")
    return s


def patch_fetch_today() -> None:
    write("africa/live_score_api_fixtures.py", '''
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = "https://livescore-api.com/api-client/fixtures/list.json"
CACHE_DIR = Path(os.environ.get("LIVE_SCORE_CACHE_DIR", "daily_football_data/live_score_cache"))
TTL = int(os.environ.get("LIVE_SCORE_CACHE_TTL_SEC", "21600"))
MAX_REQUESTS = int(os.environ.get("LIVE_SCORE_MAX_REQUESTS", "20"))
MIN_INTERVAL = float(os.environ.get("LIVE_SCORE_MIN_INTERVAL_SEC", "0.5"))
LAST_REQUEST = 0.0

AFRICA_COUNTRIES = {
    "algeria", "angola", "benin", "botswana", "burkina faso", "burundi",
    "cameroon", "cape verde", "central african republic", "chad", "comoros",
    "congo", "democratic republic of congo", "djibouti", "egypt", "equatorial guinea",
    "eritrea", "eswatini", "ethiopia", "gabon", "gambia", "ghana", "guinea",
    "guinea-bissau", "ivory coast", "kenya", "lesotho", "liberia", "libya",
    "madagascar", "malawi", "mali", "mauritania", "mauritius", "morocco",
    "mozambique", "namibia", "niger", "nigeria", "rwanda", "sao tome and principe",
    "senegal", "seychelles", "sierra leone", "somalia", "south africa", "south sudan",
    "sudan", "tanzania", "togo", "tunisia", "uganda", "zambia", "zimbabwe",
}


def _cache_path(day: str) -> Path:
    return CACHE_DIR / f"fixtures_{day}.json"


def _load_cache(day: str):
    p = _cache_path(day)
    if not p.exists() or time.time() - p.stat().st_mtime > TTL:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _request(params: dict):
    global LAST_REQUEST
    key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()
    secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("LIVE_SCORE_API_KEY and LIVE_SCORE_API_SECRET are required")
    now = time.time()
    wait = MIN_INTERVAL - (now - LAST_REQUEST)
    if wait > 0:
        time.sleep(wait)
    q = dict(params)
    q.update({"key": key, "secret": secret})
    req = Request(ENDPOINT + "?" + urlencode(q), headers={"User-Agent": "sportintel-live-score/1"})
    LAST_REQUEST = time.time()
    with urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_fixtures(day: str | None = None):
    day = day or date.today().isoformat()
    cached = _load_cache(day)
    if cached is not None:
        return cached
    pages = []
    page = 1
    while page <= MAX_REQUESTS:
        payload = _request({"date": day, "page": page})
        pages.append(payload)
        fixtures = payload.get("data", {}).get("fixtures", []) if isinstance(payload, dict) else []
        nxt = payload.get("data", {}).get("next_page") if isinstance(payload, dict) else None
        if not fixtures or not nxt:
            break
        page = int(nxt)
    merged = []
    seen = set()
    for payload in pages:
        for f in payload.get("data", {}).get("fixtures", []) if isinstance(payload, dict) else []:
            fid = str(f.get("id") or f.get("fixture_id") or "")
            if fid and fid in seen:
                continue
            country = str((f.get("competition") or {}).get("country") or f.get("country") or "").strip().lower()
            if country and country not in AFRICA_COUNTRIES:
                continue
            if fid:
                seen.add(fid)
            f["provider"] = "live-score-api"
            f["schema"] = "africa-fixture-v2"
            merged.append(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(day).write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    return merged


if __name__ == "__main__":
    print(json.dumps(fetch_fixtures(), ensure_ascii=False))
''')
    write("africa/fetch_today_fixtures.py", '''
#!/usr/bin/env python3
from __future__ import annotations

from live_score_api_fixtures import fetch_fixtures


def fetch_today_fixtures():
    return fetch_fixtures()


if __name__ == "__main__":
    print(fetch_today_fixtures())
''')


def patch_africa_segment(s: str) -> str:
    s = re.sub(
        r"def fetch_fixtures\(.*?(?=\n\s*def fetch_standings_for_africa|\n\s*def [A-Za-z_])",
        '''def fetch_fixtures(*args, **kwargs):\n    from .live_score_api_fixtures import fetch_fixtures as live_score_fetch\n    return live_score_fetch()\n\n''',
        s,
        flags=re.S,
    )
    s = re.sub(
        r"def fetch_standings_for_africa\(.*?(?=\n\s*def [A-Za-z_]|\nif __name__ ==)",
        '''def fetch_standings_for_africa(*args, **kwargs):\n    return {}\n\n''',
        s,
        flags=re.S,
    )
    s = re.sub(r"api[-_]football", "live-score-api", s, flags=re.I)
    s = re.sub(r"API_FOOTBALL_[A-Z0-9_]+", "", s)
    return s


def patch_standings_helpers() -> None:
    def local(s: str) -> str:
        s = re.sub(r"^\s*from api_football_client import FDC_DIV_TO_LEAGUE\s*$", "", s, flags=re.M)
        if "FDC_DIV_TO_LEAGUE =" not in s:
            mapping = '''FDC_DIV_TO_LEAGUE = {\n    "E0": 39, "E1": 40, "E2": 41, "E3": 42, "EC": 43,\n    "SC0": 179, "SC1": 180, "D1": 78, "D2": 79,\n    "SP1": 140, "SP2": 141, "I1": 135, "I2": 136,\n    "F1": 61, "F2": 62, "N1": 88, "B1": 144, "P1": 94,\n    "T1": 203, "G1": 197,\n}\n'''
            lines = s.splitlines(True)
            idx = 0
            while idx < len(lines) and (lines[idx].startswith("#!") or lines[idx].startswith("#") or not lines[idx].strip()):
                idx += 1
            if idx < len(lines) and lines[idx].startswith("from __future__ import"):
                idx += 1
            s = "".join(lines[:idx]) + "\n" + mapping + "\n" + "".join(lines[idx:])
        return s.replace("API-Football", "fixture provider")

    patch_file("local_standings.py", local)

    def prior(s: str) -> str:
        s = re.sub(r"^\s*from api_football_client import _norm, team_lookup\s*$", "", s, flags=re.M)
        if "def _norm(" not in s:
            helper = '''\ndef _norm(x):\n    return " ".join(str(x or "").strip().lower().split())\n\ndef team_lookup(rows, name):\n    target = _norm(name)\n    for row in rows:\n        if _norm(row.get("team") or row.get("name")) == target:\n            return row\n    return None\n\n'''
            marker = "from __future__ import annotations\n"
            if marker in s:
                s = s.replace(marker, marker + helper, 1)
            else:
                s = helper + s
        return s.replace("API-Football", "fixture provider")

    patch_file("standings_prior.py", prior)


def scrub_config() -> None:
    for rel in ["si_config.py", ".env.example"]:
        p = ROOT / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        s = "\n".join(line for line in s.splitlines() if "API_FOOTBALL" not in line and "api-football" not in line.lower()) + "\n"
        p.write_text(s, encoding="utf-8")
        print(f"scrubbed {rel}")


def main() -> None:
    patch_train((ROOT / "train.py").read_text(encoding="utf-8"))
    patch_file("train.py", patch_train)
    patch_file("sim.py", patch_sim)
    patch_fetch_today()
    patch_file("africa/daily_africa_segment.py", patch_africa_segment)
    patch_standings_helpers()
    scrub_config()

    for rel in ["api_football_client.py", "fetch_day_standings.py"]:
        p = ROOT / rel
        if p.exists():
            p.unlink()
            print(f"deleted {rel}")

    global_dir = ROOT / "football_models" / "global"
    if global_dir.exists():
        shutil.rmtree(global_dir)
        print("deleted football_models/global")

    forbidden = re.compile(r"API_FOOTBALL|api_football_client|api-football", re.I)
    files = ["train.py", "sim.py", "daily_pipeline.py", "local_standings.py", "standings_prior.py", "si_config.py", ".env.example", "africa/daily_africa_segment.py", "africa/fetch_today_fixtures.py", "africa/live_score_api_fixtures.py"]
    for rel in files:
        p = ROOT / rel
        if p.exists() and forbidden.search(p.read_text(encoding="utf-8")):
            raise RuntimeError(f"forbidden API-Football reference remains in {rel}")

    import py_compile
    for rel in files + ["industrial_sim_engine.py", "run_fixture_sims.py"]:
        p = ROOT / rel
        if p.exists():
            py_compile.compile(str(p), doraise=True)
    print("provider/training migration validation: OK")

    me = Path(__file__)
    me.unlink()
    print("final migration complete; helper self-deleted")


if __name__ == "__main__":
    main()
