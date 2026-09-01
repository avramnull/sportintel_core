#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def replace(path, old, new):
    p=ROOT/path; s=p.read_text(encoding='utf-8')
    if old not in s:
        raise RuntimeError(f'anchor missing: {path}: {old[:120]!r}')
    p.write_text(s.replace(old,new,1),encoding='utf-8')

# Global full-history anchor always trains beside every fixture-driven team.
replace('train.py','''        if not focus_list:\n            log("ERROR: no valid focus teams")\n            sys.exit(1)\n\n    registry = {''','''        if not focus_list:\n            log("ERROR: no valid focus teams")\n            sys.exit(1)\n\n    if os.environ.get("TRAIN_GLOBAL", "1").strip().lower() in ("1", "true", "yes"):\n        focus_list = [None] + [x for x in focus_list if x is not None]\n        log("Global baseline ENABLED alongside fixture-team models")\n\n    registry = {''')

# Always include global model and reduce overconfidence from peaky runs.
replace('sim.py','''    if runs:\n        return runs\n    g = MODELS_ROOT / "global"\n    if (g / "registry.json").exists() or (g / "models").is_dir():\n        return [(g, "global")]\n''','''    g = MODELS_ROOT / "global"\n    if (g / "registry.json").exists() or (g / "models").is_dir():\n        if not any(str(d) == str(g) for d, _ in runs):\n            runs.append((g, "global"))\n    if runs:\n        return runs\n''')
replace('sim.py','''        conf = 1.0 / (0.35 + ent)  # sharper → higher weight\n        weights.append(conf * family_w.get(name, 1.0))''','''        conf = 1.0 / (0.55 + ent)\n        reliability = 2.50 if name == "global" else 1.00\n        weights.append(conf * family_w.get(name, 1.0) * reliability)''')
replace('sim.py','''        weights.append(1.0 / (0.35 + ent))''','''        reliability = 2.50 if label == "global" else 1.00\n        weights.append(reliability / (0.55 + ent))''')
replace('sim.py','''    pool = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 58]''','''    pool = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 60]''')
replace('sim.py','''    locked = top["Model%"] >= 58 and abs(top["Model%"] - top["Sim%"]) <= 12''','''    engine_count = len([x for x in str(top.get("Agree", "")).split() if x.endswith("eng")])\n    locked = top["Model%"] >= 72 and abs(top["Model%"] - top["Sim%"]) <= 6 and engine_count >= 3''')
replace('sim.py','''        if is_dc(top):\n            locked = top["Model%"] >= 75 and abs(top["Model%"] - top["Sim%"]) <= 10\n        else:\n            locked = top["Model%"] >= 62 and abs(top["Model%"] - top["Sim%"]) <= 10\n''','''        engine_count = len([x for x in str(top.get("Agree", "")).split() if x.endswith("eng")])\n        if is_dc(top):\n            locked = top["Model%"] >= 80 and abs(top["Model%"] - top["Sim%"]) <= 6 and engine_count >= 3\n        else:\n            locked = top["Model%"] >= 72 and abs(top["Model%"] - top["Sim%"]) <= 6 and engine_count >= 3\n''')

# Route fixture simulation through industrial-v3; 120k is the 10x Monte Carlo baseline.
replace('run_fixture_sims.py','from hard_simulation import simulate_hard\n','from industrial_sim_engine import simulate_industrial\n')
replace('run_fixture_sims.py','N_SIM = int(os.environ.get("N_SIMULATIONS", "8000"))','N_SIM = int(os.environ.get("N_SIMULATIONS", "120000"))')
old='''    hard = simulate_hard(\n        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),\n        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),\n        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),\n        N_SIM, seed, standings_prior=rep.get("standings_prior"),\n        hardness={"attack_cv":HARD_ATTACK_CV,"defense_cv":HARD_DEFENSE_CV,"shared_cv":HARD_SHARED_CV})\n'''
new='''    hard = simulate_industrial(\n        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),\n        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),\n        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),\n        N_SIM, seed, standings_prior=rep.get("standings_prior"))\n'''
replace('run_fixture_sims.py',old,new)
replace('run_fixture_sims.py','"simulation_engine":"hard-v2" if HARD_SIM else "sim-v1"','"simulation_engine":"industrial-v3" if HARD_SIM else "sim-v1"')

# Live-score API: documented envelope is success=true + data.fixtures + pagination.
replace('africa/live_score_api_fixtures.py','''            payload = r.json()\n            if payload.get("success") is False:\n                raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))\n            data = payload.get("data") or {}\n            batch = data.get("fixtures") or []\n''','''            payload = r.json()\n            if payload.get("success") is not True:\n                raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))\n            data = payload.get("data") or {}\n            if not isinstance(data, dict):\n                raise RuntimeError("Live-score API data is not an object")\n            batch = data.get("fixtures") or []\n            if not isinstance(batch, list):\n                raise RuntimeError("Live-score API data.fixtures is not a list")\n''')
replace('africa/live_score_api_fixtures.py','''        fid = x.get("id")\n        try:\n            fixture_id = int(fid)\n        except (TypeError, ValueError):\n            continue\n''','''        fid = x.get("id") if x.get("id") is not None else x.get("fixture_id")\n        try:\n            fixture_id = int(fid)\n        except (TypeError, ValueError):\n            continue\n''')
replace('africa/live_score_api_fixtures.py','''        out.append({\n            "fixture_id": fixture_id,\n            "date": str(x.get("date") or ""),\n            "timestamp": None,\n            "status": "NS",\n            "elapsed": None,\n            "country": str(country.get("name") or federation.get("name") or "CAF"),\n            "league_id": competition.get("id"),\n''','''        scheduled = str(x.get("scheduled") or x.get("time") or "").strip()[:8]\n        fixture_date = f"{day}T{scheduled}" if scheduled else day\n        try:\n            ts = int(datetime.fromisoformat(fixture_date).replace(tzinfo=timezone.utc).timestamp())\n        except Exception:\n            ts = None\n        def _odd(v):\n            try:\n                v=float(v); return v if v>1.0 else None\n            except (TypeError,ValueError): return None\n        out.append({\n            "fixture_id": fixture_id,\n            "date": fixture_date,\n            "timestamp": ts,\n            "status": "NS",\n            "elapsed": None,\n            "country": str(country.get("name") or federation.get("name") or "CAF"),\n            "federation": str(federation.get("name") or "CAF"),\n            "league_id": competition.get("id"),\n''')
replace('africa/live_score_api_fixtures.py','''            "odds_h": odds.get("1"),\n            "odds_d": odds.get("X"),\n            "odds_a": odds.get("2"),\n''','''            "odds_h": _odd(odds.get("1")),\n            "odds_d": _odd(odds.get("X")),\n            "odds_a": _odd(odds.get("2")),\n''')
replace('africa/live_score_api_fixtures.py','''            "source": "live-score-api",\n''','''            "source": "live-score-api",\n            "provider": "live-score-api",\n            "schema_version": "africa-fixture-v2",\n''')

# Africa segment accepts Live-score-only fixture access while standings remain disabled.
replace('africa/daily_africa_segment.py','''    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()\n    key = os.environ.get("API_FOOTBALL_KEY", "").strip()\n    if not backup and not key:\n        raise SystemExit("Africa fixtures require API_FOOTBALL_BACKUP_KEY or API_FOOTBALL_KEY")\n''','''    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()\n    key = os.environ.get("API_FOOTBALL_KEY", "").strip()\n    live_key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()\n    live_secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()\n    if not (backup or key or (live_key and live_secret)):\n        raise SystemExit("Africa fixtures require API-Football fixture credentials or LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET")\n''')
replace('africa/daily_africa_segment.py','''        "API_FOOTBALL_KEY": key,\n        "API_FOOTBALL_STANDINGS_KEY": "",\n        "AFRICA_FIXTURES_OUT": str(out),\n''','''        "API_FOOTBALL_KEY": key,\n        "LIVE_SCORE_API_KEY": live_key,\n        "LIVE_SCORE_API_SECRET": live_secret,\n        "API_FOOTBALL_STANDINGS_KEY": "",\n        "AFRICA_FIXTURES_OUT": str(out),\n''')

for f in ('industrial_sim_engine.py','train.py','sim.py','run_fixture_sims.py','africa/live_score_api_fixtures.py','africa/daily_africa_segment.py'):
    compile((ROOT/f).read_text(encoding='utf-8'),f,'exec')
Path(__file__).unlink()
print('industrial-v3 upgrade applied')
