#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def edit(path: str, fn):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    ns = fn(s)
    if ns == s:
        raise RuntimeError(f"no change made: {path}")
    p.write_text(ns, encoding="utf-8")

def repl(path, old, new, required=True):
    def f(s):
        if old not in s:
            if required:
                raise RuntimeError(f"anchor missing: {path}: {old[:120]!r}")
            return s
        return s.replace(old, new, 1)
    edit(path, f)

# 1) Daily football training: fixture-team only. Never enqueue/train GLOBAL.
repl("train.py", 'focus_list: List[Optional[str]] = [None]', 'focus_list: List[Optional[str]] = []')

def remove_train_global(s):
    pat = re.compile(
        r'    for focus in focus_list:\n'
        r'        if focus is None:\n'
        r'            run_name, run_df = "global", labeled\n'
        r'.*?'
        r'            gc\.collect\(\)\n'
        r'            continue\n\n',
        re.S,
    )
    ns, n = pat.subn('', s, count=1)
    if n != 1:
        raise RuntimeError('train.py global training block not found')
    return ns
edit("train.py", remove_train_global)

# 2) Simulator: never fall back to an old global model.
repl("sim.py", '''    if runs:\n        return runs\n    g = MODELS_ROOT / "global"\n    if (g / "registry.json").exists() or (g / "models").is_dir():\n        return [(g, "global")]\n    if strict:\n        raise SystemExit(f"No models for {home_canon}/{away_canon} and no global/")\n''', '''    if runs:\n        return runs\n    if strict:\n        raise SystemExit(f"No fixture-team models for {home_canon}/{away_canon}")\n''', required=False)

# 3) Africa training: no GLOBAL model alongside country scopes.
repl("africa/train_africa.py", '''    registry = {}\n    # GLOBAL\n    registry["GLOBAL"] = train_scope(df, "GLOBAL", out_root)\n\n    # Per-country if enough rows\n''', '''    registry = {}\n    # Per-country board scopes only; never train a GLOBAL Africa model.\n''', required=False)

# 4) Remove the obsolete API-Football client and standalone standings fetcher.
for rel in ("api_football_client.py", "fetch_day_standings.py"):
    p = ROOT / rel
    if p.exists():
        p.unlink()

# 5) Remove API-Football configuration/import dependencies from remaining modules.
p = ROOT / "si_config.py"
s = p.read_text(encoding="utf-8")
ns = re.sub(r'(?ms)^# API-Football:.*?^API_FOOTBALL_STANDINGS_KEY = _env\("API_FOOTBALL_STANDINGS_KEY"\)\n?', '', s)
if ns != s:
    p.write_text(ns, encoding="utf-8")

p = ROOT / "local_standings.py"
s = p.read_text(encoding="utf-8")
if "from api_football_client import FDC_DIV_TO_LEAGUE" in s:
    s = s.replace(
        'from api_football_client import FDC_DIV_TO_LEAGUE  # noqa: E402\n',
        '''FDC_DIV_TO_LEAGUE = {\n    "E0":39,"E1":40,"E2":41,"E3":42,"EC":43,"SC0":179,"SC1":180,"D1":78,"D2":79,\n    "SP1":140,"SP2":141,"I1":135,"I2":136,"F1":61,"F2":62,"N1":88,"B1":144,\n    "P1":94,"T1":203,"G1":197,\n}\n''')
    p.write_text(s, encoding="utf-8")

p = ROOT / "standings_prior.py"
s = p.read_text(encoding="utf-8")
if 'from api_football_client import _norm, team_lookup' in s:
    s = s.replace(
        'from api_football_client import _norm, team_lookup\n',
        '''def _norm(name):\n    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()\n\ndef team_lookup(rows):\n    out = {}\n    for row in rows or []:\n        if not isinstance(row, dict):\n            continue\n        name = row.get("team") or row.get("name") or ""\n        if name:\n            out[_norm(name)] = row\n    return out\n''')
    p.write_text(s, encoding="utf-8")

# 6) Guard against stale API-Football references in runtime source/config.
for rel in ("daily_pipeline.py", "africa/daily_africa_segment.py", "africa/fetch_today_fixtures.py", "africa/train_africa.py", "local_standings.py", "standings_prior.py", "si_config.py"):
    text = (ROOT / rel).read_text(encoding="utf-8")
    if re.search(r'API_FOOTBALL|api_football|api-football', text, re.I):
        raise RuntimeError(f"API-Football reference remains in {rel}")

# 7) Compile the production Python modules before allowing the pipeline to continue.
for rel in ("train.py", "sim.py", "africa/train_africa.py", "africa/fetch_today_fixtures.py", "africa/daily_africa_segment.py", "si_config.py", "local_standings.py", "standings_prior.py"):
    compile((ROOT / rel).read_text(encoding="utf-8"), rel, "exec")

print("runtime cleanup OK: fixture-team-only training + no global fallback + API-Football removed")
