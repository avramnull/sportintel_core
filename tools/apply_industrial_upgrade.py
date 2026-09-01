#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENGINE = r'''#!/usr/bin/env python3
"""Industrial football score simulator v3."""
from __future__ import annotations
from collections import Counter
from math import exp, factorial
import numpy as np
try:
    from scipy.optimize import minimize, brentq
    from scipy.special import gammaln
except Exception:
    minimize = brentq = gammaln = None

ENGINE_VERSION = "industrial-v3"

def _norm(p):
    p=np.asarray(p,dtype=float).ravel()
    if p.size==0: return np.array([1/3,1/3,1/3])
    p=np.nan_to_num(p,nan=1/max(len(p),1),posinf=0,neginf=0)
    p=np.clip(p,1e-9,None); return p/p.sum()

def _pois(k,lam): return exp(-lam)*(lam**int(k))/factorial(int(k))

def _grid(lh,la,common,rho=0.055,max_g=9):
    pmf=np.zeros((max_g+1,max_g+1),float)
    if gammaln is None:
        for h in range(max_g+1):
            for a in range(max_g+1):
                pmf[h,a]=sum(_pois(z,common)*_pois(h-z,lh)*_pois(a-z,la) for z in range(min(h,a)+1))
    else:
        h,a=np.indices(pmf.shape)
        for z in range(max_g+1):
            valid=(h>=z)&(a>=z); hz=h-z; az=a-z
            lp=-common+z*np.log(max(common,1e-12))-gammaln(z+1)
            lp+=-lh+hz*np.log(max(lh,1e-12))-gammaln(hz+1)
            lp+=-la+az*np.log(max(la,1e-12))-gammaln(az+1)
            pmf+=np.where(valid,np.exp(lp),0.0)
    pmf[0,0]*=max(1-lh*la*rho,0.02)
    pmf[0,1]*=max(1+lh*rho,0.02)
    pmf[1,0]*=max(1+la*rho,0.02)
    pmf[1,1]*=max(1-rho,0.02)
    pmf=np.clip(pmf,0,None); return pmf/max(pmf.sum(),1e-15)

def _stats(p):
    h,a=np.indices(p.shape); out=np.where(h>a,0,np.where(h<a,2,1)); total=h+a
    ft=np.array([(p[out==k]).sum() for k in range(3)])
    return ft,float(p[total>=3].sum()),float(p[(h>0)&(a>0)].sum()),float((p*h).sum()),float((p*a).sum())

def _rates(pft,po,pb):
    pft=_norm(pft)[:3]; po=float(np.clip(po,.06,.94)); pb=float(np.clip(pb,.06,.94))
    try: total=brentq(lambda t:1-exp(-t)*(1+t+t*t/2)-po,.25,5.5) if brentq else 2.35+2.8*(po-.5)
    except Exception: total=2.35+2.8*(po-.5)
    share=float(np.clip(.5+.48*(pft[0]-pft[2]),.16,.84))
    x0=np.array([np.log(max(total*share,.15)),np.log(max(total*(1-share),.15)),np.log(.22)])
    def obj(x):
        lh,la=np.exp(x[:2]); common=(1/(1+np.exp(-x[2])))*min(lh,la)*.55
        s=_stats(_grid(lh,la,common)); return 9*np.sum((s[0]-pft)**2)+3*(s[1]-po)**2+2*(s[2]-pb)**2+.001*np.sum(x*x)
    x=x0
    if minimize:
        try:
            r=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':90,'xatol':1e-3,'fatol':1e-6})
            if np.isfinite(r.fun): x=r.x
        except Exception: pass
    lh,la=np.exp(x[:2]); common=(1/(1+np.exp(-x[2])))*min(lh,la)*.55
    return float(np.clip(lh,.12,5)),float(np.clip(la,.12,5)),float(np.clip(common,0,min(lh,la)*.45)),float(total)

def _mixture(lh,la,common):
    q=np.zeros((10,10))
    for w,s in ((.18,.72),(.60,1.0),(.22,1.32)): q+=w*_grid(lh*s,la*s,common*s)
    return q/q.sum()

def _calibrate(p,pft,po,pb,strength=.84):
    h,a=np.indices(p.shape); out=np.where(h>a,0,np.where(h<a,2,1)); over=(h+a)>=3; btts=(h>0)&(a>0)
    f0,o0,b0,_,_=_stats(p); tf=_norm(.84*_norm(pft)[:3]+.16*f0); to=.84*po+.16*o0; tb=.84*pb+.16*b0
    def obj(beta):
        q=p*np.exp(np.clip(beta[0]*(out==0)+beta[1]*(out==1)+beta[2]*over+beta[3]*btts,-10,10)); q/=q.sum()
        ft,ov,bt,_,_=_stats(q); return 8*np.sum((ft-tf)**2)+3*(ov-to)**2+2*(bt-tb)**2+.01*np.sum(beta*beta)
    beta=np.zeros(4)
    if minimize:
        try:
            r=minimize(obj,beta,method='BFGS',options={'maxiter':80}); beta=r.x if np.all(np.isfinite(r.x)) else beta
        except Exception: pass
    beta=np.clip(beta,-2.5,2.5); q=p*np.exp(np.clip(beta[0]*(out==0)+beta[1]*(out==1)+beta[2]*over+beta[3]*btts,-10,10)); q/=q.sum()
    return q,beta.tolist()

def _sample(rng,p,n):
    idx=rng.choice(p.size,size=int(n),p=p.ravel()); return np.column_stack(np.unravel_index(idx,p.shape)).astype(np.int16)

def _ht(rng,scores,pht,phto):
    pht=_norm(pht)[:3]; out=np.zeros_like(scores,dtype=np.int16); base=float(np.clip(.39+.18*(phto-.5),.28,.56)); c=18.; sh=rng.beta(base*c,(1-base)*c,len(scores))
    for i,(hg,ag) in enumerate(scores):
        if hg: out[i,0]=rng.binomial(int(hg),np.clip(sh[i]+.06*(pht[0]-pht[2]),.18,.72))
        if ag: out[i,1]=rng.binomial(int(ag),np.clip(sh[i]+.06*(pht[2]-pht[0]),.18,.72))
    return out

def _line(t,line): return {'over':float(np.mean(t>line)),'under':float(np.mean(t<line)),'push':float(np.mean(t==line))}

def simulate_industrial(p_ft,p_ht,p_over25,p_btts,p_ht_over15,n,seed,standings_prior=None):
    p_ft=_norm(p_ft)[:3]; p_ht=_norm(p_ht if p_ht is not None else [.30,.40,.30])[:3]; po=float(np.clip(p_over25,.05,.95)); pb=float(np.clip(p_btts,.05,.95)); phto=float(np.clip(p_ht_over15 if p_ht_over15 is not None else .30,.05,.95))
    rng=np.random.default_rng(int(seed)); sp=standings_prior or {}
    if sp.get('matched'):
        try:
            from standings_prior import apply_ft_tilt,apply_scalar_tilt
            p_ft=apply_ft_tilt(p_ft,sp.get('ft_tilt') or {}); po=apply_scalar_tilt(po,float(sp.get('over25_tilt') or 0)); pb=apply_scalar_tilt(pb,float(sp.get('btts_tilt') or 0))
        except Exception: pass
    lh,la,common,total_target=_rates(p_ft,po,pb); base=_mixture(lh,la,common); pmf,beta=_calibrate(base,p_ft,po,pb); scores=_sample(rng,pmf,n); ht=_ht(rng,scores,p_ht,phto)
    total=scores.sum(1).astype(float); diff=scores[:,0].astype(float)-scores[:,1].astype(float); htd=ht[:,0].astype(float)-ht[:,1].astype(float)
    ft={'H':float(np.mean(diff>0)),'D':float(np.mean(diff==0)),'A':float(np.mean(diff<0))}; hts={'H':float(np.mean(htd>0)),'D':float(np.mean(htd==0)),'A':float(np.mean(htd<0))}
    top=Counter(map(tuple,scores.tolist())).most_common(12); topht=Counter(map(tuple,ht.tolist())).most_common(8)
    def top_pct(items,k): return [{'score':f'{h}-{a}','count':int(c),'pct':round(100*c/max(len(scores),1),1)} for (h,a),c in items[:k]]
    simft=np.array([ft['H'],ft['D'],ft['A']]); flat=pmf.ravel(); ent=float(-(flat[flat>0]*np.log(flat[flat>0])).sum()); top1=float(top[0][1]/len(scores)) if top else 0; top3=float(sum(c for _,c in top[:3])/len(scores))
    return {'n':int(n),'engine':{'version':ENGINE_VERSION,'method':'bivariate-poisson+dixon-coles+three-regime+soft-maximum-entropy-calibration+beta-binomial-HT','rho':.055,'lambda_home':lh,'lambda_away':la,'common_goal_rate':common,'target_total_lambda':total_target,'calibration_strength':.84,'calibration_beta':beta,'dispersion_regimes':[.72,1.,1.32]},'ft_model':{'H':float(p_ft[0]),'D':float(p_ft[1]),'A':float(p_ft[2])},'ft_sim':ft,'ht_model':{'H':float(p_ht[0]),'D':float(p_ht[1]),'A':float(p_ht[2])},'ht_sim':hts,'over25_model':po,'over25_sim':float(np.mean(total>=3)),'btts_model':pb,'btts_sim':float(np.mean((scores[:,0]>0)&(scores[:,1]>0))),'ht_over15_model':phto,'ht_over15_sim':float(np.mean(ht.sum(1)>=2)),'dc_ft':{'1X':ft['H']+ft['D'],'X2':ft['D']+ft['A'],'12':ft['H']+ft['A']},'dc_ht':{'1X':hts['H']+hts['D'],'X2':hts['D']+hts['A'],'12':hts['H']+hts['A']},'top_ft':[(f'{h}-{a}',int(c)) for (h,a),c in top],'top_ht':[(f'{h}-{a}',int(c)) for (h,a),c in topht],'top3_ft':top_pct(top,3),'top3_ht':top_pct(topht,3),'top8_ft':top_pct(top,8),'xg':{'home':float(scores[:,0].mean()),'away':float(scores[:,1].mean()),'total':float(total.mean()),'lambda_home':lh,'lambda_away':la},'ht_xg':{'home':float(ht[:,0].mean()),'away':float(ht[:,1].mean()),'total':float(ht.sum(1).mean())},'score_consistency':{'ht_leq_ft':float(np.mean((ht[:,0]<=scores[:,0])&(ht[:,1]<=scores[:,1]))),'ft_vs_model_l1':float(np.abs(simft-_norm(p_ft)[:3]).sum()),'over25_abs_error':float(abs(np.mean(total>=3)-po)),'btts_abs_error':float(abs(np.mean((scores[:,0]>0)&(scores[:,1]>0))-pb))},'distribution_diagnostics':{'score_entropy':ent,'top1_mass':top1,'top3_mass':top3,'tail_4plus':float(np.mean(total>=4)),'score_cells':int(pmf.size)},'clean_sheet':{'home':float(np.mean(scores[:,1]==0)),'away':float(np.mean(scores[:,0]==0))},'win_to_nil':{'home':float(np.mean((diff>0)&(scores[:,1]==0))),'away':float(np.mean((diff<0)&(scores[:,0]==0)))},'goal_lines':{str(x):_line(total,x) for x in (.5,1.5,2.5,3.5,4.5)},'asian_handicap':{'home_-0.5':float(np.mean(diff>0)),'home_-1.0':{'cover':float(np.mean(diff>1)),'push':float(np.mean(diff==1))},'home_+0.5':float(np.mean(diff>-1)),'home_+1.0':{'cover':float(np.mean(diff>-1)),'push':float(np.mean(diff==-1))}},'score_matrix_margin':{'home_1_goal':float(np.mean(diff==1)),'home_2_plus':float(np.mean(diff>=2)),'away_1_goal':float(np.mean(diff==-1)),'away_2_plus':float(np.mean(diff<=-2)),'draw':ft['D']}}
'''

def replace(path: str, old: str, new: str):
    p=ROOT/path; s=p.read_text(encoding='utf-8')
    if old not in s:
        raise RuntimeError(f"anchor missing: {path}: {old[:100]!r}")
    p.write_text(s.replace(old,new,1),encoding='utf-8')

# New industrial engine.
(ROOT/'industrial_sim_engine.py').write_text(ENGINE,encoding='utf-8')

# Global model anchor: fixture-driven team count remains uncapped.
replace('train.py','''        if not focus_list:\n            log("ERROR: no valid focus teams")\n            sys.exit(1)\n\n    registry = {''','''        if not focus_list:\n            log("ERROR: no valid focus teams")\n            sys.exit(1)\n\n    if os.environ.get("TRAIN_GLOBAL", "1").strip().lower() in ("1", "true", "yes"):\n        focus_list = [None] + [x for x in focus_list if x is not None]\n        log("Global baseline ENABLED alongside fixture-team models")\n\n    registry = {''')

# Always include global run and temper confidence weighting.
replace('sim.py','''    if runs:\n        return runs\n    g = MODELS_ROOT / "global"\n    if (g / "registry.json").exists() or (g / "models").is_dir():\n        return [(g, "global")]\n''','''    g = MODELS_ROOT / "global"\n    if (g / "registry.json").exists() or (g / "models").is_dir():\n        if not any(str(d) == str(g) for d, _ in runs):\n            runs.append((g, "global"))\n    if runs:\n        return runs\n''')
replace('sim.py','''        conf = 1.0 / (0.35 + ent)  # sharper → higher weight\n        weights.append(conf * family_w.get(name, 1.0))''','''        conf = 1.0 / (0.55 + ent)\n        weights.append(conf * family_w.get(name, 1.0))''')
replace('sim.py','''        weights.append(1.0 / (0.35 + ent))''','''        reliability = 2.50 if label == "global" else 1.00\n        weights.append(reliability / (0.55 + ent))''')
replace('sim.py','''    candidates = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 55]''','''    candidates = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 60]''')
replace('sim.py','''    locked = top["Model%"] >= 58 and abs(top["Model%"] - top["Sim%"] <= 12''','''    engine_count = len([x for x in str(top.get("Agree", "")).split() if x.endswith("eng")])\n    locked = top["Model%"] >= 72 and abs(top["Model%"] - top["Sim%"] <= 6 and engine_count >= 3''')
# The normal console lock block.
replace('sim.py','''        if is_dc(top):\n            locked = top["Model%"] >= 75 and abs(top["Model%"] - top["Sim%"] <= 10\n        else:\n            locked = top["Model%"] >= 62 and abs(top["Model%"] - top["Sim%"] <= 10\n''','''        engine_count = len([x for x in str(top.get("Agree", "")).split() if x.endswith("eng")])\n        if is_dc(top):\n            locked = top["Model%"] >= 80 and abs(top["Model%"] - top["Sim%"] <= 6 and engine_count >= 3\n        else:\n            locked = top["Model%"] >= 72 and abs(top["Model%"] - top["Sim%"] <= 6 and engine_count >= 3\n''')

# Switch batch simulation to industrial-v3 and 10x default volume.
replace('run_fixture_sims.py','from hard_simulation import simulate_hard\n','from industrial_sim_engine import simulate_industrial\n')
replace('run_fixture_sims.py','N_SIM = int(os.environ.get("N_SIMULATIONS", "8000"))','N_SIM = int(os.environ.get("N_SIMULATIONS", "120000"))')
old='''    hard = simulate_hard(\n        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),\n        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),\n        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),\n        N_SIM, seed, standings_prior=rep.get("standings_prior"),\n        hardness={"attack_cv":HARD_ATTACK_CV,"defense_cv":HARD_DEFENSE_CV,"shared_cv":HARD_SHARED_CV})\n'''
new='''    hard = simulate_industrial(\n        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),\n        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),\n        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),\n        N_SIM, seed, standings_prior=rep.get("standings_prior"))\n'''
replace('run_fixture_sims.py',old,new)
replace('run_fixture_sims.py','"simulation_engine":"hard-v2" if HARD_SIM else "sim-v1"','"simulation_engine":"industrial-v3" if HARD_SIM else "sim-v1"')

# Live-score API: validate documented envelope and map calendar fields exactly.
replace('africa/live_score_api_fixtures.py','''            payload = r.json()\n            if payload.get("success") is False:\n                raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))\n            data = payload.get("data") or {}\n            batch = data.get("fixtures") or []\n''','''            payload = r.json()\n            if payload.get("success") is not True:\n                raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))\n            data = payload.get("data") or {}\n            if not isinstance(data, dict):\n                raise RuntimeError("Live-score API data is not an object")\n            batch = data.get("fixtures") or []\n            if not isinstance(batch, list):\n                raise RuntimeError("Live-score API data.fixtures is not a list")\n''')
replace('africa/live_score_api_fixtures.py','''        fid = x.get("id")\n        try:\n            fixture_id = int(fid)\n        except (TypeError, ValueError):\n            continue\n''','''        fid = x.get("id") if x.get("id") is not None else x.get("fixture_id")\n        try:\n            fixture_id = int(fid)\n        except (TypeError, ValueError):\n            continue\n''')
replace('africa/live_score_api_fixtures.py','''        out.append({\n            "fixture_id": fixture_id,\n            "date": str(x.get("date") or ""),\n            "timestamp": None,\n            "status": "NS",\n            "elapsed": None,\n            "country": str(country.get("name") or federation.get("name") or "CAF"),\n            "league_id": competition.get("id"),\n''','''        scheduled = str(x.get("scheduled") or x.get("time") or "").strip()[:8]\n        fixture_date = f"{day}T{scheduled}" if scheduled else day\n        try:\n            ts = int(__import__("datetime").datetime.fromisoformat(fixture_date).replace(tzinfo=timezone.utc).timestamp())\n        except Exception:\n            ts = None\n        def _odd(v):\n            try:\n                v=float(v); return v if v>1.0 else None\n            except (TypeError,ValueError): return None\n        out.append({\n            "fixture_id": fixture_id,\n            "date": fixture_date,\n            "timestamp": ts,\n            "status": "NS",\n            "elapsed": None,\n            "country": str(country.get("name") or federation.get("name") or "CAF"),\n            "federation": str(federation.get("name") or "CAF"),\n            "league_id": competition.get("id"),\n''')
replace('africa/live_score_api_fixtures.py','''            "odds_h": odds.get("1"),\n            "odds_d": odds.get("X"),\n            "odds_a": odds.get("2"),\n''','''            "odds_h": _odd(odds.get("1")),\n            "odds_d": _odd(odds.get("X")),\n            "odds_a": _odd(odds.get("2")),\n''')
replace('africa/live_score_api_fixtures.py','''            "source": "live-score-api",\n''','''            "source": "live-score-api",\n            "provider": "live-score-api",\n            "schema_version": "africa-fixture-v2",\n''')

# Africa segment: Live-score credentials are a valid fixture-only fallback.
replace('africa/daily_africa_segment.py','''    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()\n    key = os.environ.get("API_FOOTBALL_KEY", "").strip()\n    if not backup and not key:\n        raise SystemExit("Africa fixtures require API_FOOTBALL_BACKUP_KEY or API_FOOTBALL_KEY")\n''','''    backup = os.environ.get("API_FOOTBALL_BACKUP_KEY", "").strip()\n    key = os.environ.get("API_FOOTBALL_KEY", "").strip()\n    live_key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()\n    live_secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()\n    if not (backup or key or (live_key and live_secret)):\n        raise SystemExit("Africa fixtures require API-Football fixture credentials or LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET")\n''')
replace('africa/daily_africa_segment.py','''        "API_FOOTBALL_KEY": key,\n        "API_FOOTBALL_STANDINGS_KEY": "",\n        "AFRICA_FIXTURES_OUT": str(out),\n''','''        "API_FOOTBALL_KEY": key,\n        "LIVE_SCORE_API_KEY": live_key,\n        "LIVE_SCORE_API_SECRET": live_secret,\n        "API_FOOTBALL_STANDINGS_KEY": "",\n        "AFRICA_FIXTURES_OUT": str(out),\n''')

# Validate and remove this one-time applier before the normal commit.
for f in ['industrial_sim_engine.py','train.py','sim.py','run_fixture_sims.py','africa/live_score_api_fixtures.py','africa/daily_africa_segment.py']:
    compile((ROOT/f).read_text(encoding='utf-8'),str(f),'exec')
(Path(__file__).resolve()).unlink()
print('industrial upgrade applied and self-removal complete')
