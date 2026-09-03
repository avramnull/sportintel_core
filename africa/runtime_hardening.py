#!/usr/bin/env python3
"""Runtime hardening for the Africa simulator: persistent IDs, chronological state and strict feature contract."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import sim_africa as engine
from .team_mapping import load_aliases, load_ids, resolve

FEATURE_NUM = [
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "AwayFormPts_5", "HomeFormGD_5", "AwayFormGD_5",
    "HomeFormPts_10", "AwayFormPts_10", "HomeFormGD_10", "AwayFormGD_10",
    "HomeFormPts_20", "AwayFormPts_20", "HomeFormGD_20", "AwayFormGD_20",
    "EloHome", "EloAway", "EloDiff", "RestHome", "RestAway", "RestDiff",
]
FEATURE_ID = ["HomeTeamId", "AwayTeamId"]
_ORIGINAL_SIMULATE_SCORES = engine.simulate_scores


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:100]


def _safe_resolve(name: str, hist) -> str:
    ids, aliases = load_ids(), load_aliases()
    canon, _ = resolve(str(name or ""), ids, aliases)
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
    out.update({"Elo": 1500.0, "TeamId": -1, "Rest": 7.0})
    if hist is None or not len(hist):
        return out
    ids = load_ids()
    canon = _safe_resolve(team, hist)
    tid = int(ids.get(canon, -1))
    out["TeamId"] = tid
    if tid < 0:
        return out
    h = hist.copy().sort_values("Date")
    dates = pd.to_datetime(h["Date"], errors="coerce")
    if as_of is not None:
        d = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(d):
            h = h[dates < d].copy()
    elo, K, ADV = {}, 22.0, 55.0
    for _, r in h.iterrows():
        hi, ai = ids.get(str(r["HomeTeam"])), ids.get(str(r["AwayTeam"]))
        if hi is None or ai is None:
            continue
        rh, ra = elo.get(hi, 1500.0), elo.get(ai, 1500.0)
        ftr = str(r.get("FTR", "D")).upper()
        sh = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        eh = 1.0 / (1.0 + 10 ** ((ra - (rh + ADV)) / 400.0))
        elo[hi] = rh + K * (sh - eh)
        elo[ai] = ra + K * ((1.0 - sh) - (1.0 - eh))
    out["Elo"] = float(elo.get(tid, 1500.0))
    sub = h[(h["HomeTeam"] == canon) | (h["AwayTeam"] == canon)].tail(20)
    pts, gd, last_date = [], [], None
    for _, r in sub.iterrows():
        home = str(r["HomeTeam"]) == canon
        ftr = str(r.get("FTR", "D")).upper()
        pts.append(3 if (ftr == "H" if home else ftr == "A") else (1 if ftr == "D" else 0))
        gd.append(float(r["FTHG"] - r["FTAG"]) if home else float(r["FTAG"] - r["FTHG"]))
        last_date = pd.to_datetime(r["Date"], errors="coerce")
    for w in (5, 10, 20):
        if pts:
            out[f"FormPts_{w}"] = float(np.mean(pts[-w:]))
            out[f"FormGD_{w}"] = float(np.mean(gd[-w:]))
    if last_date is not None and as_of is not None:
        d = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(d) and pd.notna(last_date):
            out["Rest"] = float(np.clip((d - last_date).days, 0, 60))
    return out


def _build_feature_row(home: str, away: str, hist: Optional[pd.DataFrame], now=None):
    now = now or datetime.utcnow()
    home = _safe_resolve(home, hist)
    away = _safe_resolve(away, hist)
    hf, af = _team_form_elo(hist, home, now), _team_form_elo(hist, away, now)
    row = {
        "Year": now.year, "Month": now.month, "DayOfWeek": now.weekday(), "IsWeekend": int(now.weekday() >= 5),
        "HomeFormPts_5": hf["FormPts_5"], "AwayFormPts_5": af["FormPts_5"], "HomeFormGD_5": hf["FormGD_5"], "AwayFormGD_5": af["FormGD_5"],
        "HomeFormPts_10": hf["FormPts_10"], "AwayFormPts_10": af["FormPts_10"], "HomeFormGD_10": hf["FormGD_10"], "AwayFormGD_10": af["FormGD_10"],
        "HomeFormPts_20": hf["FormPts_20"], "AwayFormPts_20": af["FormPts_20"], "HomeFormGD_20": hf["FormGD_20"], "AwayFormGD_20": af["FormGD_20"],
        "EloHome": hf["Elo"], "EloAway": af["Elo"], "EloDiff": hf["Elo"] - af["Elo"],
        "RestHome": hf["Rest"], "RestAway": af["Rest"], "RestDiff": hf["Rest"] - af["Rest"],
        "HomeTeamId": float(hf["TeamId"]), "AwayTeamId": float(af["TeamId"]),
    }
    feats = FEATURE_NUM + FEATURE_ID
    X = np.array([[float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in feats]], dtype=np.float64)
    return X, feats


def _seeded_simulate_scores(seed):
    def wrapped(*args, **kwargs):
        old = np.random.default_rng
        np.random.default_rng = lambda *_a, **_k: old(int(seed))
        try:
            return _ORIGINAL_SIMULATE_SCORES(*args, **kwargs)
        finally:
            np.random.default_rng = old
    return wrapped


def _strict_load_scope(team_or_country: Optional[str] = None):
    """Load only a canonical team model; never silently fall back to country/global."""
    root = Path(engine.MODELS_ROOT)
    ids, aliases = load_ids(), load_aliases()
    canon, _ = resolve(str(team_or_country or ""), ids, aliases)
    tdir = root / f"team_{_slug(canon)}"
    reg_path = tdir / "registry.json"
    stats_path = tdir / "preprocessors" / "feature_stats.json"
    if not reg_path.exists() or not stats_path.exists():
        return None, None, None
    try:
        return tdir, json.loads(reg_path.read_text(encoding="utf-8")), json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None, None


def simulate_match(*args, match_date=None, seed=None, **kwargs):
    engine._resolve_team_name = _safe_resolve
    engine._team_form_elo = _team_form_elo
    engine.build_feature_row = _build_feature_row
    engine.load_scope = _strict_load_scope
    engine.simulate_scores = _seeded_simulate_scores(int(seed if seed is not None else 42))
    # The legacy signature receives country separately, so pass the canonical home team
    # through the loader by temporarily binding the requested team at call time.
    home = args[0] if args else kwargs.get("home")
    away = args[1] if len(args) > 1 else kwargs.get("away")
    original_loader = engine.load_scope
    def team_loader(_ignored=None):
        return _strict_load_scope(_safe_resolve(str(home or ""), engine._load_hist()))
    engine.load_scope = team_loader
    return engine.simulate_match(*args, **kwargs)
