#!/usr/bin/env python3
"""Runtime hardening for the legacy Africa simulator without fuzzy team guesses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from . import sim_africa as engine
from .team_mapping import load_aliases, load_ids, resolve

FEATURE_NUM = engine.FEATURE_NUM
FEATURE_ID = engine.FEATURE_ID
_ORIGINAL_SIMULATE_SCORES = engine.simulate_scores


def _safe_resolve(name: str, hist) -> str:
    ids, aliases = load_ids(), load_aliases()
    canon, _ = resolve(name, ids, aliases)
    if canon in ids:
        return canon
    if hist is not None and len(hist):
        exact = {str(t).lower(): str(t) for t in set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str))}
        if canon.lower() in exact:
            return exact[canon.lower()]
    return canon


def _team_form_elo(hist: pd.DataFrame, team: str, as_of=None) -> dict:
    out = {f"FormPts_{w}": np.nan for w in (5, 10, 20)}
    out.update({f"FormGD_{w}": np.nan for w in (5, 10, 20)})
    out.update({"Elo": 1500.0, "TeamId": -1})
    if hist is None or not len(hist): return out
    ids = load_ids(); canon = _safe_resolve(team, hist); tid = int(ids.get(canon, -1)); out["TeamId"] = tid
    if tid < 0: return out
    h = hist.copy().sort_values("Date")
    if as_of is not None:
        d = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(d): h = h[pd.to_datetime(h["Date"], errors="coerce") < d]
    elo, K, ADV = {}, 22.0, 55.0
    for _, r in h.iterrows():
        hi, ai = ids.get(str(r["HomeTeam"])), ids.get(str(r["AwayTeam"]))
        if hi is None or ai is None: continue
        rh, ra = elo.get(hi, 1500.0), elo.get(ai, 1500.0)
        ftr = str(r.get("FTR", "D")).upper(); sh = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        eh = 1.0 / (1.0 + 10 ** ((ra - (rh + ADV)) / 400.0))
        elo[hi] = rh + K * (sh - eh); elo[ai] = ra + K * ((1-sh) - (1-eh))
    out["Elo"] = float(elo.get(tid, 1500.0))
    sub = h[(h["HomeTeam"] == canon) | (h["AwayTeam"] == canon)].tail(20)
    pts, gd = [], []
    for _, r in sub.iterrows():
        home = str(r["HomeTeam"]) == canon; ftr = str(r.get("FTR", "D")).upper()
        pts.append(3 if (ftr == "H" if home else ftr == "A") else (1 if ftr == "D" else 0))
        gd.append(float(r["FTHG"]-r["FTAG"]) if home else float(r["FTAG"]-r["FTHG"]))
    for w in (5,10,20):
        if pts: out[f"FormPts_{w}"] = float(np.mean(pts[-w:])); out[f"FormGD_{w}"] = float(np.mean(gd[-w:]))
    return out


def _build_feature_row(home: str, away: str, hist: Optional[pd.DataFrame], now=None):
    now = now or datetime.utcnow(); home = _safe_resolve(home, hist); away = _safe_resolve(away, hist)
    hf, af = _team_form_elo(hist, home, now), _team_form_elo(hist, away, now)
    row = {"Year":now.year,"Month":now.month,"DayOfWeek":now.weekday(),"IsWeekend":int(now.weekday()>=5),
           "HomeFormPts_5":hf["FormPts_5"],"AwayFormPts_5":af["FormPts_5"],"HomeFormGD_5":hf["FormGD_5"],"AwayFormGD_5":af["FormGD_5"],
           "HomeFormPts_10":hf["FormPts_10"],"AwayFormPts_10":af["FormPts_10"],"HomeFormGD_10":hf["FormGD_10"],"AwayFormGD_10":af["FormGD_10"],
           "HomeFormPts_20":hf["FormPts_20"],"AwayFormPts_20":af["FormPts_20"],"HomeFormGD_20":hf["FormGD_20"],"AwayFormGD_20":af["FormGD_20"],
           "EloHome":hf["Elo"],"EloAway":af["Elo"],"EloDiff":hf["Elo"]-af["Elo"],"RestHome":7.0,"RestAway":7.0,"RestDiff":0.0,
           "HomeTeamId":float(hf["TeamId"]),"AwayTeamId":float(af["TeamId"])}
    feats = FEATURE_NUM + FEATURE_ID
    return np.array([[float(row.get(c,0.0)) if pd.notna(row.get(c,0.0)) else 0.0 for c in feats]],dtype=np.float64), feats


def _seeded_simulate_scores(*args, **kwargs):
    seed = int(kwargs.pop("_seed", 42))
    old = np.random.default_rng
    np.random.default_rng = lambda *_a, **_k: old(seed)
    try: return _ORIGINAL_SIMULATE_SCORES(*args, **kwargs)
    finally: np.random.default_rng = old


def simulate_match(*args, match_date=None, seed=None, **kwargs):
    engine._resolve_team_name = _safe_resolve
    engine._team_form_elo = _team_form_elo
    engine.build_feature_row = _build_feature_row
    engine.simulate_scores = _seeded_simulate_scores
    return engine.simulate_match(*args, **kwargs, _seed=int(seed if seed is not None else 42))
