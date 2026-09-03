#!/usr/bin/env python3
"""Runtime hardening for the legacy Africa simulator without fuzzy team guesses."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from . import sim_africa as engine
from .team_mapping import load_aliases, load_ids, resolve

FEATURE_NUM = engine.FEATURE_NUM
FEATURE_ID = engine.FEATURE_ID


def _safe_resolve(name: str, hist) -> str:
    ids = load_ids()
    aliases = load_aliases()
    canon, _ = resolve(name, ids, aliases)
    if canon in ids:
        return canon
    if hist is not None and len(hist):
        teams = sorted(set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str)))
        exact = {t.lower(): t for t in teams}
        if canon.lower() in exact:
            return exact[canon.lower()]
        # Only accept a unique normalized match through the mapping layer.
        canon2, _ = resolve(canon, {t: i for i, t in enumerate(teams)}, aliases)
        return canon2
    return canon


def _team_form_elo(hist: pd.DataFrame, team: str, as_of=None) -> dict:
    out = {"FormPts_5": np.nan, "FormGD_5": np.nan, "FormPts_10": np.nan, "FormGD_10": np.nan,
           "FormPts_20": np.nan, "FormGD_20": np.nan, "Elo": 1500.0, "TeamId": -1}
    if hist is None or not len(hist):
        return out
    ids = load_ids()
    canon = _safe_resolve(team, hist)
    out["TeamId"] = int(ids.get(canon, -1))
    if out["TeamId"] < 0:
        return out
    h = hist.copy()
    if as_of is not None:
        d = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(d):
            h = h[pd.to_datetime(h["Date"], errors="coerce") < d]
    h = h.sort_values("Date")
    elo = {}
    K, HOME_ADV = 22.0, 55.0
    for _, r in h.iterrows():
        hid, aid = ids.get(str(r["HomeTeam"])), ids.get(str(r["AwayTeam"]))
        if hid is None or aid is None:
            continue
        rh, ra = elo.get(hid, 1500.0), elo.get(aid, 1500.0)
        ftr = str(r.get("FTR", "D")).upper()
        sh = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        elo[hid] = rh + K * (sh - exp_h)
        elo[aid] = ra + K * ((1.0 - sh) - (1.0 - exp_h))
    out["Elo"] = float(elo.get(out["TeamId"], 1500.0))
    mask = (h["HomeTeam"] == canon) | (h["AwayTeam"] == canon)
    sub = h.loc[mask].tail(20)
    pts, gd = [], []
    for _, r in sub.iterrows():
        home = str(r["HomeTeam"]) == canon
        ftr = str(r.get("FTR", "D")).upper()
        pts.append(3 if (ftr == "H" if home else ftr == "A") else (1 if ftr == "D" else 0))
        gd.append(float(r["FTHG"] - r["FTAG"]) if home else float(r["FTAG"] - r["FTHG"]))
    for w in (5, 10, 20):
        if pts:
            out[f"FormPts_{w}"] = float(np.mean(pts[-w:]))
            out[f"FormGD_{w}"] = float(np.mean(gd[-w:]))
    return out


def _build_feature_row(home: str, away: str, hist: Optional[pd.DataFrame], now=None):
    now = now or datetime.utcnow()
    home = _safe_resolve(home, hist)
    away = _safe_resolve(away, hist)
    hf, af = _team_form_elo(hist, home, now), _team_form_elo(hist, away, now)
    row = {
        "Year": now.year, "Month": now.month, "DayOfWeek": now.weekday(), "IsWeekend": int(now.weekday() >= 5),
        "HomeFormPts_5": hf["FormPts_5"], "AwayFormPts_5": af["FormPts_5"],
        "HomeFormGD_5": hf["FormGD_5"], "AwayFormGD_5": af["FormGD_5"],
        "HomeFormPts_10": hf["FormPts_10"], "AwayFormPts_10": af["FormPts_10"],
        "HomeFormGD_10": hf["FormGD_10"], "AwayFormGD_10": af["FormGD_10"],
        "HomeFormPts_20": hf["FormPts_20"], "AwayFormPts_20": af["FormPts_20"],
        "HomeFormGD_20": hf["FormGD_20"], "AwayFormGD_20": af["FormGD_20"],
        "EloHome": hf["Elo"], "EloAway": af["Elo"], "EloDiff": hf["Elo"] - af["Elo"],
        "RestHome": 7.0, "RestAway": 7.0, "RestDiff": 0.0,
        "HomeTeamId": float(hf["TeamId"]), "AwayTeamId": float(af["TeamId"]),
    }
    if hist is not None and len(hist):
        h = hist[pd.to_datetime(hist["Date"], errors="coerce") < pd.Timestamp(now)].sort_values("Date")
        for side, team in (("Home", home), ("Away", away)):
            sub = h[(h["HomeTeam"] == team) | (h["AwayTeam"] == team)].tail(1)
            if len(sub):
                days = max(0, (pd.Timestamp(now) - pd.Timestamp(sub.iloc[0]["Date"])).days)
                row[f"Rest{side}"] = float(min(60, days))
        row["RestDiff"] = row["RestHome"] - row["RestAway"]
    feats = FEATURE_NUM + FEATURE_ID
    X = np.array([[float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in feats]], dtype=np.float64)
    return X, feats


def simulate_match(*args, match_date=None, seed=None, **kwargs):
    engine._resolve_team_name = _safe_resolve
    engine._team_form_elo = _team_form_elo
    engine.build_feature_row = _build_feature_row
    old = os.environ.get("AFRICA_SIM_SEED")
    if seed is not None:
        os.environ["AFRICA_SIM_SEED"] = str(int(seed))
    try:
        return engine.simulate_match(*args, **kwargs)
    finally:
        if old is None:
            os.environ.pop("AFRICA_SIM_SEED", None)
        else:
            os.environ["AFRICA_SIM_SEED"] = old
