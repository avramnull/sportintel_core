#!/usr/bin/env python3
"""Runtime hardening for Africa: persistent IDs, chronological state and team-model ensemble."""
from __future__ import annotations
import json, re
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from . import sim_africa as engine
from .team_mapping import load_aliases, load_ids, resolve

FEATURE_NUM=["Year","Month","DayOfWeek","IsWeekend","HomeFormPts_5","AwayFormPts_5","HomeFormGD_5","AwayFormGD_5","HomeFormPts_10","AwayFormPts_10","HomeFormGD_10","AwayFormGD_10","HomeFormPts_20","AwayFormPts_20","HomeFormGD_20","AwayFormGD_20","EloHome","EloAway","EloDiff","RestHome","RestAway","RestDiff"]
FEATURE_ID=["HomeTeamId","AwayTeamId"]
_ORIGINAL_SIMULATE_SCORES=engine.simulate_scores
_ORIGINAL_PREDICT_PATHS=engine._predict_paths

def _slug(name): return re.sub(r"[^a-z0-9]+","_",str(name).lower()).strip("_")[:100]

def _safe_resolve(name,hist):
    ids,aliases=load_ids(),load_aliases(); canon,_=resolve(str(name or ""),ids,aliases)
    if canon in ids: return canon
    if hist is not None and len(hist):
        exact={str(t).lower():str(t) for t in set(hist["HomeTeam"].astype(str))|set(hist["AwayTeam"].astype(str))}
        if canon.lower() in exact: return exact[canon.lower()]
    return canon

def _team_form_elo(hist,team,as_of=None):
    out={f"FormPts_{w}":np.nan for w in (5,10,20)}; out.update({f"FormGD_{w}":np.nan for w in (5,10,20)}); out.update({"Elo":1500.0,"TeamId":-1,"Rest":7.0})
    if hist is None or not len(hist): return out
    ids=load_ids(); canon=_safe_resolve(team,hist); tid=int(ids.get(canon,-1)); out["TeamId"]=tid
    if tid<0: return out
    h=hist.copy().sort_values("Date"); dates=pd.to_datetime(h["Date"],errors="coerce")
    if as_of is not None:
        d=pd.to_datetime(as_of,errors="coerce")
        if pd.notna(d): h=h[dates<d].copy()
    elo={}; K,ADV=22.0,55.0
    for _,r in h.iterrows():
        hi,ai=ids.get(str(r["HomeTeam"])),ids.get(str(r["AwayTeam"]))
        if hi is None or ai is None: continue
        rh,ra=elo.get(hi,1500.0),elo.get(ai,1500.0); ftr=str(r.get("FTR","D")).upper(); sh=1.0 if ftr=="H" else (0.5 if ftr=="D" else 0.0)
        eh=1.0/(1.0+10**((ra-(rh+ADV))/400.0)); elo[hi]=rh+K*(sh-eh); elo[ai]=ra+K*((1-sh)-(1-eh))
    out["Elo"]=float(elo.get(tid,1500.0)); sub=h[(h["HomeTeam"]==canon)|(h["AwayTeam"]==canon)].tail(20); pts=[]; gd=[]; last=None
    for _,r in sub.iterrows():
        home=str(r["HomeTeam"])==canon; ftr=str(r.get("FTR","D")).upper(); pts.append(3 if (ftr=="H" if home else ftr=="A") else (1 if ftr=="D" else 0)); gd.append(float(r["FTHG"]-r["FTAG"]) if home else float(r["FTAG"]-r["FTHG"])); last=pd.to_datetime(r["Date"],errors="coerce")
    for w in (5,10,20):
        if pts: out[f"FormPts_{w}"]=float(np.mean(pts[-w:])); out[f"FormGD_{w}"]=float(np.mean(gd[-w:]))
    if last is not None and as_of is not None:
        d=pd.to_datetime(as_of,errors="coerce")
        if pd.notna(d) and pd.notna(last): out["Rest"]=float(np.clip((d-last).days,0,60))
    return out

def _build_feature_row(home,away,hist,now=None):
    now=now or datetime.utcnow(); home=_safe_resolve(home,hist); away=_safe_resolve(away,hist); hf,af=_team_form_elo(hist,home,now),_team_form_elo(hist,away,now)
    row={"Year":now.year,"Month":now.month,"DayOfWeek":now.weekday(),"IsWeekend":int(now.weekday()>=5),"HomeFormPts_5":hf["FormPts_5"],"AwayFormPts_5":af["FormPts_5"],"HomeFormGD_5":hf["FormGD_5"],"AwayFormGD_5":af["FormGD_5"],"HomeFormPts_10":hf["FormPts_10"],"AwayFormPts_10":af["FormPts_10"],"HomeFormGD_10":hf["FormGD_10"],"AwayFormGD_10":af["FormGD_10"],"HomeFormPts_20":hf["FormPts_20"],"AwayFormPts_20":af["FormPts_20"],"HomeFormGD_20":hf["FormGD_20"],"AwayFormGD_20":af["FormGD_20"],"EloHome":hf["Elo"],"EloAway":af["Elo"],"EloDiff":hf["Elo"]-af["Elo"],"RestHome":hf["Rest"],"RestAway":af["Rest"],"RestDiff":hf["Rest"]-af["Rest"],"HomeTeamId":float(hf["TeamId"]),"AwayTeamId":float(af["TeamId"])}
    feats=FEATURE_NUM+FEATURE_ID; return np.array([[float(row.get(c,0.0)) if pd.notna(row.get(c,0.0)) else 0.0 for c in feats]],dtype=np.float64),feats

def _seeded_simulate_scores(seed):
    def wrapped(*args,**kwargs):
        old=np.random.default_rng; np.random.default_rng=lambda *_a,**_k: old(int(seed))
        try: return _ORIGINAL_SIMULATE_SCORES(*args,**kwargs)
        finally: np.random.default_rng=old
    return wrapped

def _load_team_registry(name):
    root=Path(engine.MODELS_ROOT); ids,aliases=load_ids(),load_aliases(); canon,_=resolve(str(name or ""),ids,aliases); d=root/f"team_{_slug(canon)}"; rp=d/"registry.json"; sp=d/"preprocessors"/"feature_stats.json"
    if not rp.exists() or not sp.exists(): raise RuntimeError(f"Missing Africa team model: {canon}")
    return json.loads(rp.read_text(encoding="utf-8")),json.loads(sp.read_text(encoding="utf-8"))

def simulate_match(*args,match_date=None,seed=None,**kwargs):
    engine._resolve_team_name=_safe_resolve; engine._team_form_elo=_team_form_elo; engine.build_feature_row=_build_feature_row
    home=args[0] if args else kwargs.get("home"); away=args[1] if len(args)>1 else kwargs.get("away")
    hist=engine._load_hist(); home_c=_safe_resolve(home,hist); away_c=_safe_resolve(away,hist)
    home_reg,home_stats=_load_team_registry(home_c); away_reg,away_stats=_load_team_registry(away_c)
    def team_predict(_paths,X,target):
        ph=_ORIGINAL_PREDICT_PATHS(home_reg.get("targets",{}).get(target,{}),X,target)
        pa=_ORIGINAL_PREDICT_PATHS(away_reg.get("targets",{}).get(target,{}),X,target)
        if ph is None or pa is None: raise RuntimeError(f"Incomplete Africa team ensemble for {home_c} vs {away_c}: {target}")
        ph=np.asarray(ph,float); pa=np.asarray(pa,float); n=max(len(ph),len(pa)); ph=np.pad(ph,(0,n-len(ph))); pa=np.pad(pa,(0,n-len(pa))); p=(ph+pa)/2.0; return p/p.sum()
    def team_stats(_ignored=None):
        # Both team registries use the same feature schema; fail closed if they diverge.
        if home_stats.get("features")!=away_stats.get("features"): raise RuntimeError(f"Africa feature schema mismatch: {home_c} vs {away_c}")
        return None,home_reg,home_stats
    engine._predict_paths=team_predict
    engine.load_scope=team_stats
    engine.simulate_scores=_seeded_simulate_scores(int(seed if seed is not None else 42))
    return engine.simulate_match(*args,**kwargs)
