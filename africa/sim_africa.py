#!/usr/bin/env python3
"""Africa simulation layer — strict mapping + precision-v5 HT/FT scores.

Interfaces required by africa.runtime_hardening and africa.daily_africa_segment:
  - simulate_scores(...)  (overridable by precision engine)
  - _predict_paths(...)
  - load_scope(...)
  - simulate_match(...)
  - to_simlab_document(...)
  - MODELS_ROOT, _load_hist, build_feature_row, _team_form_elo, _resolve_team_name

Output for Sim Lab: precise top_ft / top_ht counts + 1X2 + xG only.
No narrative lock stories as the primary product.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.environ.get("AFRICA_MODELS", str(ROOT / "football_models" / "africa")))
PARQUET = Path(os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")))
N_SIM = int(os.environ.get("N_SIMULATIONS", "150000"))

FEATURE_NUM = [
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "AwayFormPts_5", "HomeFormGD_5", "AwayFormGD_5",
    "HomeFormPts_10", "AwayFormPts_10", "HomeFormGD_10", "AwayFormGD_10",
    "HomeFormPts_20", "AwayFormPts_20", "HomeFormGD_20", "AwayFormGD_20",
    "EloHome", "EloAway", "EloDiff",
    "RestHome", "RestAway", "RestDiff",
]
FEATURE_ID = ["HomeTeamId", "AwayTeamId"]


def _resolve_model_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    s = str(path_str).replace("\\", "/")
    if "football_models/africa/" in s:
        tail = s.split("football_models/africa/")[-1]
        cand = MODELS_ROOT / tail
        if cand.exists():
            return cand
    cand = MODELS_ROOT / s
    if cand.exists():
        return cand
    name = Path(s).name
    hits = list(MODELS_ROOT.rglob(name))
    return hits[0] if hits else p


def _load_hist() -> Optional[pd.DataFrame]:
    path = PARQUET if PARQUET.exists() else PARQUET.with_suffix(".csv")
    if not path.exists():
        return None
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date")


def _resolve_team_name(name: str, hist) -> str:
    """Strict identity only — delegated to team_mapping."""
    from africa.team_mapping import resolve, ensure_ids, load_ids
    if hist is not None and len(hist):
        teams = set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str))
        ensure_ids(teams)
    canon, _ = resolve(str(name or ""))
    return canon


def _team_form_elo(hist: pd.DataFrame, team: str, as_of=None) -> dict:
    out = {
        "FormPts_5": np.nan, "FormGD_5": np.nan,
        "FormPts_10": np.nan, "FormGD_10": np.nan,
        "FormPts_20": np.nan, "FormGD_20": np.nan,
        "Elo": 1500.0, "TeamId": -1, "Rest": 7.0,
    }
    if hist is None or not len(hist):
        return out
    from africa.team_mapping import load_ids
    ids = load_ids()
    team = _resolve_team_name(team, hist)
    tid = int(ids.get(team, -1))
    out["TeamId"] = tid
    if tid < 0:
        return out

    h = hist.copy().sort_values("Date")
    dates = pd.to_datetime(h["Date"], errors="coerce")
    if as_of is not None:
        d = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(d):
            h = h[dates < d].copy()

    elo: Dict[int, float] = {}
    K, ADV = 22.0, 55.0
    for _, r in h.iterrows():
        hi = ids.get(str(r["HomeTeam"]))
        ai = ids.get(str(r["AwayTeam"]))
        if hi is None or ai is None:
            continue
        rh, ra = elo.get(hi, 1500.0), elo.get(ai, 1500.0)
        ftr = str(r.get("FTR", "D")).upper()
        sh = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        eh = 1.0 / (1.0 + 10 ** ((ra - (rh + ADV)) / 400.0))
        elo[hi] = rh + K * (sh - eh)
        elo[ai] = ra + K * ((1 - sh) - (1 - eh))
    out["Elo"] = float(elo.get(tid, 1500.0))

    sub = h[(h["HomeTeam"] == team) | (h["AwayTeam"] == team)].tail(20)
    pts, gd, last = [], [], None
    for _, r in sub.iterrows():
        home = str(r["HomeTeam"]) == team
        ftr = str(r.get("FTR", "D")).upper()
        pts.append(3 if (ftr == "H" if home else ftr == "A") else (1 if ftr == "D" else 0))
        gd.append(float(r["FTHG"] - r["FTAG"]) if home else float(r["FTAG"] - r["FTHG"]))
        last = pd.to_datetime(r["Date"], errors="coerce")
    for w in (5, 10, 20):
        if pts:
            out[f"FormPts_{w}"] = float(np.mean(pts[-w:]))
            out[f"FormGD_{w}"] = float(np.mean(gd[-w:]))
    if last is not None and as_of is not None:
        d = pd.to_datetime(as_of, errors="coerce")
        if pd.notna(d) and pd.notna(last):
            out["Rest"] = float(np.clip((d - last).days, 0, 60))
    return out


def build_feature_row(home: str, away: str, hist: Optional[pd.DataFrame], now=None) -> Tuple[np.ndarray, List[str]]:
    now = now or datetime.utcnow()
    home = _resolve_team_name(home, hist)
    away = _resolve_team_name(away, hist)
    hf = _team_form_elo(hist, home, now) if hist is not None else {"Elo": 1500.0, "TeamId": -1, "Rest": 7.0}
    af = _team_form_elo(hist, away, now) if hist is not None else {"Elo": 1500.0, "TeamId": -1, "Rest": 7.0}
    row = {
        "Year": now.year, "Month": now.month, "DayOfWeek": now.weekday(),
        "IsWeekend": int(now.weekday() >= 5),
        "HomeFormPts_5": hf.get("FormPts_5", np.nan), "AwayFormPts_5": af.get("FormPts_5", np.nan),
        "HomeFormGD_5": hf.get("FormGD_5", np.nan), "AwayFormGD_5": af.get("FormGD_5", np.nan),
        "HomeFormPts_10": hf.get("FormPts_10", np.nan), "AwayFormPts_10": af.get("FormPts_10", np.nan),
        "HomeFormGD_10": hf.get("FormGD_10", np.nan), "AwayFormGD_10": af.get("FormGD_10", np.nan),
        "HomeFormPts_20": hf.get("FormPts_20", np.nan), "AwayFormPts_20": af.get("FormPts_20", np.nan),
        "HomeFormGD_20": hf.get("FormGD_20", np.nan), "AwayFormGD_20": af.get("FormGD_20", np.nan),
        "EloHome": hf.get("Elo", 1500.0), "EloAway": af.get("Elo", 1500.0),
        "EloDiff": float(hf.get("Elo", 1500.0) - af.get("Elo", 1500.0)),
        "RestHome": hf.get("Rest", 7.0), "RestAway": af.get("Rest", 7.0),
        "RestDiff": float(hf.get("Rest", 7.0) - af.get("Rest", 7.0)),
        "HomeTeamId": float(hf.get("TeamId", -1)), "AwayTeamId": float(af.get("TeamId", -1)),
    }
    feats = FEATURE_NUM + FEATURE_ID
    X = np.array([[float(row.get(c) or 0.0) if pd.notna(row.get(c, 0.0)) else 0.0 for c in feats]], dtype=np.float64)
    return X, feats


def _scale(X: np.ndarray, stats: dict) -> np.ndarray:
    feats = stats["features"]
    mu = np.array([stats["mean"].get(c, 0.0) for c in feats], dtype=float)
    sg = np.array([stats["std"].get(c, 1.0) for c in feats], dtype=float)
    sg = np.where(sg == 0, 1.0, sg)
    return (X - mu) / sg


def _predict_paths(paths: dict, X: np.ndarray, target: str) -> Optional[np.ndarray]:
    preds = []
    if not paths:
        return None
    if "xgboost" in paths and _resolve_model_path(paths["xgboost"]).exists():
        try:
            import xgboost as xgb
            m = xgb.Booster()
            m.load_model(str(_resolve_model_path(paths["xgboost"])))
            p = np.asarray(m.predict(xgb.DMatrix(X)))
            preds.append(p[0] if p.ndim > 1 else (np.array([1 - p[0], p[0]]) if p.size == 1 else p))
        except Exception:
            pass
    if "lightgbm" in paths and _resolve_model_path(paths["lightgbm"]).exists():
        try:
            import lightgbm as lgb
            m = lgb.Booster(model_file=str(_resolve_model_path(paths["lightgbm"])))
            p = np.asarray(m.predict(X)).ravel()
            if target in ("ft_result", "ht_result") and p.size >= 3:
                preds.append(p[:3])
            elif p.size == 1:
                preds.append(np.array([1 - float(p[0]), float(p[0])]))
            else:
                preds.append(p)
        except Exception:
            pass
    if "catboost" in paths and _resolve_model_path(paths["catboost"]).exists():
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(_resolve_model_path(paths["catboost"])))
            preds.append(m.predict_proba(X)[0])
        except Exception:
            pass
    if "random_forest" in paths and _resolve_model_path(paths["random_forest"]).exists():
        try:
            with open(_resolve_model_path(paths["random_forest"]), "rb") as f:
                m = pickle.load(f)
            preds.append(m.predict_proba(X)[0])
        except Exception:
            pass
    if not preds:
        return None
    max_len = max(len(p) for p in preds)
    aligned = [np.pad(np.asarray(p, float).ravel(), (0, max(0, max_len - len(p))))[:max_len] for p in preds]
    weights = []
    for v in aligned:
        v = np.clip(v, 1e-9, 1.0)
        v = v / v.sum()
        ent = float(-(v * np.log(v)).sum())
        weights.append(1.0 / (0.35 + ent))
    w = np.asarray(weights)
    w = w / w.sum()
    return (w[:, None] * np.vstack(aligned)).sum(axis=0)


def load_scope(country: Optional[str] = None):
    if country:
        cdir = MODELS_ROOT / f"country_{country.replace(' ', '_')}"
        if (cdir / "registry.json").exists():
            reg = json.loads((cdir / "registry.json").read_text())
            stats = json.loads((cdir / "preprocessors" / "feature_stats.json").read_text())
            return cdir, reg, stats
    gdir = MODELS_ROOT / "GLOBAL"
    if (gdir / "registry.json").exists():
        reg = json.loads((gdir / "registry.json").read_text())
        stats = json.loads((gdir / "preprocessors" / "feature_stats.json").read_text())
        return gdir, reg, stats
    return None, None, None


def simulate_scores(
    p_ft,
    p_ht=None,
    p_over25=0.5,
    p_btts=0.5,
    p_ht_over15=0.30,
    n=None,
    seed=42,
    standings_prior=None,
    odds_ft=None,
    odds_blend=0.0,
    max_goals=12,
    lam_h=None,
    lam_a=None,
    rho=None,
    **_extra,
):
    """Delegate to precision-v5 (deterministic, no Monte-Carlo noise)."""
    from precision_sim_engine import simulate_precision
    return simulate_precision(
        p_ft,
        p_ht=p_ht,
        p_over25=p_over25,
        p_btts=p_btts,
        p_ht_over15=p_ht_over15,
        n=int(n or N_SIM),
        seed=int(seed),
        standings_prior=standings_prior,
        odds_ft=odds_ft,
        odds_blend=odds_blend,
        max_goals=max_goals,
        lam_h=lam_h,
        lam_a=lam_a,
        rho=rho,
    )


def simulate_match(
    home: str,
    away: str,
    country: Optional[str] = None,
    *,
    odds_h: float | None = None,
    odds_d: float | None = None,
    odds_a: float | None = None,
    odds_blend: float | None = None,
    standings_prior: dict | None = None,
    match_date=None,
    seed: int = 42,
) -> dict:
    hist = _load_hist()
    hist_c = hist
    if country and hist is not None and "Country" in hist.columns:
        sub = hist[hist["Country"] == country]
        if len(sub) >= 40:
            hist_c = sub

    prior_ft = np.array([0.42, 0.28, 0.30])
    prior_ht = np.array([0.30, 0.40, 0.30])
    prior_o25, prior_btts, prior_ht_o15 = 0.45, 0.48, 0.30
    if hist_c is not None and len(hist_c) >= 30:
        ftr = hist_c["FTR"].astype(str).str.upper()
        prior_ft = np.array([
            float((ftr == "H").mean()),
            float((ftr == "D").mean()),
            float((ftr == "A").mean()),
        ])
        if prior_ft.sum() > 0:
            prior_ft = prior_ft / prior_ft.sum()
        prior_o25 = float(((hist_c["FTHG"] + hist_c["FTAG"]) > 2.5).mean())
        prior_btts = float(((hist_c["FTHG"] > 0) & (hist_c["FTAG"] > 0)).mean())

    home = _resolve_team_name(home, hist_c)
    away = _resolve_team_name(away, hist_c)
    now = None
    if match_date:
        try:
            now = pd.to_datetime(match_date, errors="coerce")
            if pd.notna(now):
                now = now.to_pydatetime().replace(tzinfo=None)
            else:
                now = None
        except Exception:
            now = None
    X, feats = build_feature_row(home, away, hist_c, now=now)

    try:
        elo_h = float(X[0, feats.index("EloHome")])
        elo_a = float(X[0, feats.index("EloAway")])
        diff = elo_h - elo_a
        lam_h = 1.20 + 0.50 * (1 / (1 + 10 ** (-diff / 400)))
        lam_a = 1.00 + 0.50 * (1 / (1 + 10 ** (diff / 400)))
    except Exception:
        lam_h = lam_a = None

    scope_dir, reg, stats = load_scope(country)
    ft_pred = prior_ft.copy()
    ht_pred = prior_ht.copy()
    o25_pred = prior_o25
    btts_pred = prior_btts

    if reg and stats:
        try:
            Xs = _scale(X, stats)
            targets = reg.get("targets") or {}
            if "ft_result" in targets:
                p = _predict_paths(targets["ft_result"], Xs, "ft_result")
                if p is not None and len(p) >= 3:
                    ft_pred = np.asarray(p[:3], float)
                    ft_pred = ft_pred / ft_pred.sum()
            if "ht_result" in targets:
                p = _predict_paths(targets["ht_result"], Xs, "ht_result")
                if p is not None and len(p) >= 3:
                    ht_pred = np.asarray(p[:3], float)
                    ht_pred = ht_pred / ht_pred.sum()
            if "over25" in targets:
                p = _predict_paths(targets["over25"], Xs, "over25")
                if p is not None:
                    o25_pred = float(p[-1] if len(p) > 1 else p[0])
            if "btts" in targets:
                p = _predict_paths(targets["btts"], Xs, "btts")
                if p is not None:
                    btts_pred = float(p[-1] if len(p) > 1 else p[0])
        except Exception:
            pass

    odds_ft = None
    if odds_h and odds_d and odds_a:
        try:
            odds_ft = (float(odds_h), float(odds_d), float(odds_a))
        except Exception:
            odds_ft = None

    blend = float(odds_blend if odds_blend is not None else os.environ.get("ODDS_BLEND", "0.0") or 0.0)

    sim = simulate_scores(
        ft_pred,
        p_ht=ht_pred,
        p_over25=o25_pred,
        p_btts=btts_pred,
        n=N_SIM,
        seed=seed,
        standings_prior=standings_prior,
        odds_ft=odds_ft,
        odds_blend=blend,
        lam_h=lam_h,
        lam_a=lam_a,
    )

    # Attach identity for document builder
    sim["_home"] = home
    sim["_away"] = away
    sim["_country"] = country or ""
    sim["_engines"] = list((reg or {}).get("targets", {}).keys()) or ["ft_result", "over25", "btts"]
    return sim


def to_simlab_document(raw: dict, fx: dict | None = None) -> dict:
    """Clean Sim Lab document: precise HT + FT score counts, no lock stories."""
    fx = fx or {}
    home = raw.get("_home") or fx.get("home") or "Home"
    away = raw.get("_away") or fx.get("away") or "Away"
    country = raw.get("_country") or fx.get("country") or ""
    league = fx.get("league") or (f"{country}" if country else "Africa")
    kickoff = (fx.get("date") or fx.get("kickoff") or "")[:16].replace("T", " ")

    top_ft = raw.get("top_ft") or []
    top_ht = raw.get("top_ht") or []
    # Normalize to [[score, count], ...]
    def _norm_top(items):
        out = []
        for it in items:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                out.append([str(it[0]), int(it[1])])
            elif isinstance(it, dict):
                out.append([str(it.get("score")), int(it.get("count", 0))])
        return out

    top_ft = _norm_top(top_ft)
    top_ht = _norm_top(top_ht)
    top3_ft = raw.get("top3_ft") or [
        {"score": s, "count": c, "pct": round(100.0 * c / max(raw.get("n", N_SIM), 1), 1)}
        for s, c in top_ft[:3]
    ]

    report = {
        "ft_model": raw.get("ft_model") or raw.get("ft_sim") or {"H": 0.34, "D": 0.33, "A": 0.33},
        "ft_sim": raw.get("ft_sim") or {"H": 0.34, "D": 0.33, "A": 0.33},
        "ht_model": raw.get("ht_model") or raw.get("ht_sim") or {"H": 0.30, "D": 0.40, "A": 0.30},
        "ht_sim": raw.get("ht_sim") or {"H": 0.30, "D": 0.40, "A": 0.30},
        "over25_model": float(raw.get("over25_model") or raw.get("over25_sim") or 0.45),
        "over25_sim": float(raw.get("over25_sim") or 0.45),
        "btts_model": float(raw.get("btts_model") or raw.get("btts_sim") or 0.48),
        "btts_sim": float(raw.get("btts_sim") or 0.48),
        "xg": raw.get("xg") or {"home": 1.2, "away": 1.1, "total": 2.3},
        "ht_xg": raw.get("ht_xg") or {},
        "top_ft": top_ft,
        "top_ht": top_ht,
        "top3_ft": top3_ft,
        "top3_ht": raw.get("top3_ht") or [],
        "score_matrix_top": raw.get("score_matrix_top") or {},
        "score_matrix_top_ht": raw.get("score_matrix_top_ht") or {},
        "clean_sheet": raw.get("clean_sheet") or {"home": float(raw.get("cs_home") or 0), "away": float(raw.get("cs_away") or 0)},
        "ht_clean_sheet": raw.get("ht_clean_sheet") or {},
        "goal_lines": raw.get("goal_lines") or {},
        "goal_line_sim": raw.get("goal_line_sim") or {},
        "score_consistency": raw.get("score_consistency") or {},
        "engine": raw.get("engine") or {"version": "precision-v5"},
        "region": "Africa",
        "engines": raw.get("_engines") or ["ft_result", "over25", "btts"],
    }

    # Minimal locked_tip — Sim Lab must not be dominated by lock stories
    locked_tip = {
        "status": "—",
        "section": "FT CS",
        "selection": top_ft[0][0] if top_ft else "—",
        "model": None,
        "sim": top3_ft[0]["pct"] if top3_ft else None,
        "verdict": "TOP CS",
    }

    return {
        "id": None,
        "region": "Africa",
        "match": {
            "home": home,
            "away": away,
            "kickoff": kickoff,
            "league": league,
            "odds": "—",
            "odds_h": fx.get("odds_h"),
            "odds_d": fx.get("odds_d"),
            "odds_a": fx.get("odds_a"),
            "country": country,
        },
        "resolved": {
            "home": home,
            "away": away,
            "models": ",".join(report["engines"]),
        },
        "report": report,
        "locked_tip": locked_tip,
        "table": [],
        "metadata": {
            "engine": report["engine"],
            "n": int(raw.get("n") or N_SIM),
            "primary": "ht_ft_score_matrix",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--country", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    raw = simulate_match(args.home, args.away, args.country)
    doc = to_simlab_document(raw, {"home": args.home, "away": args.away, "country": args.country or ""})
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
