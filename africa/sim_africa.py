#!/usr/bin/env python3
"""
Africa match simulation — clean precise HT / FT score engine.

- Strict team identity only (no fuzzy)
- Dixon–Coles + IPF score grid
- Output: pure HT and FT score distributions + marginals
- No lock stories, no narrative noise
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from math import exp, factorial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.environ.get("AFRICA_MODELS", str(ROOT / "football_models" / "africa")))
PARQUET = Path(os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")))
N_SIM = int(os.environ.get("N_SIMULATIONS", "8000"))

FEATURE_NUM = [
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "AwayFormPts_5", "HomeFormGD_5", "AwayFormGD_5",
    "HomeFormPts_10", "AwayFormPts_10", "HomeFormGD_10", "AwayFormGD_10",
    "EloHome", "EloAway", "EloDiff",
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
    if hits:
        return hits[0]
    return p


def _load_hist() -> Optional[pd.DataFrame]:
    path = PARQUET if PARQUET.exists() else PARQUET.with_suffix(".csv")
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date")


def _strict_resolve(name: str, hist: Optional[pd.DataFrame]) -> str:
    """Use only the strict team_mapping layer."""
    from africa.team_mapping import resolve, ensure_ids
    if hist is not None and len(hist):
        teams = set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str))
        ensure_ids(teams)
    canon, _ = resolve(name)
    return canon


def _team_form_elo(hist: pd.DataFrame, team: str) -> dict:
    out = {
        "FormPts_5": np.nan, "FormGD_5": np.nan,
        "FormPts_10": np.nan, "FormGD_10": np.nan,
        "Elo": 1500.0, "TeamId": -1,
    }
    if hist is None or not len(hist):
        return out
    teams = sorted(set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str)))
    t2 = {t: i for i, t in enumerate(teams)}
    out["TeamId"] = t2.get(team, -1)

    elo: Dict[int, float] = {}
    K, HOME_ADV = 20.0, 55.0
    for _, r in hist.iterrows():
        hid = t2.get(str(r["HomeTeam"]))
        aid = t2.get(str(r["AwayTeam"]))
        if hid is None or aid is None:
            continue
        rh, ra = elo.get(hid, 1500.0), elo.get(aid, 1500.0)
        ftr = str(r.get("FTR", "D")).upper()
        score_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        elo[hid] = rh + K * (score_h - exp_h)
        elo[aid] = ra + K * ((1.0 - score_h) - (1.0 - exp_h))
    if out["TeamId"] >= 0:
        out["Elo"] = float(elo.get(out["TeamId"], 1500.0))

    mask = (hist["HomeTeam"] == team) | (hist["AwayTeam"] == team)
    sub = hist.loc[mask].tail(15)
    if not len(sub):
        return out
    pts, gd = [], []
    for _, r in sub.iterrows():
        is_h = str(r["HomeTeam"]) == team
        ftr = str(r.get("FTR", "D")).upper()
        if is_h:
            pts.append(3 if ftr == "H" else (1 if ftr == "D" else 0))
            gd.append(float(r["FTHG"]) - float(r["FTAG"]))
        else:
            pts.append(3 if ftr == "A" else (1 if ftr == "D" else 0))
            gd.append(float(r["FTAG"]) - float(r["FTHG"]))
    for w in (5, 10):
        if pts:
            out[f"FormPts_{w}"] = float(np.mean(pts[-w:]))
            out[f"FormGD_{w}"] = float(np.mean(gd[-w:]))
    return out


def build_feature_row(home: str, away: str, hist: Optional[pd.DataFrame], now=None) -> Tuple[np.ndarray, List[str]]:
    now = now or datetime.utcnow()
    hf = _team_form_elo(hist, home) if hist is not None else {"Elo": 1500.0, "TeamId": -1}
    af = _team_form_elo(hist, away) if hist is not None else {"Elo": 1500.0, "TeamId": -1}
    row = {
        "Year": now.year, "Month": now.month, "DayOfWeek": now.weekday(),
        "IsWeekend": int(now.weekday() >= 5),
        "HomeFormPts_5": hf.get("FormPts_5", np.nan),
        "AwayFormPts_5": af.get("FormPts_5", np.nan),
        "HomeFormGD_5": hf.get("FormGD_5", np.nan),
        "AwayFormGD_5": af.get("FormGD_5", np.nan),
        "HomeFormPts_10": hf.get("FormPts_10", np.nan),
        "AwayFormPts_10": af.get("FormPts_10", np.nan),
        "HomeFormGD_10": hf.get("FormGD_10", np.nan),
        "AwayFormGD_10": af.get("FormGD_10", np.nan),
        "EloHome": hf.get("Elo", 1500.0),
        "EloAway": af.get("Elo", 1500.0),
        "EloDiff": float(hf.get("Elo", 1500.0) - af.get("Elo", 1500.0)),
        "HomeTeamId": float(hf.get("TeamId", -1)),
        "AwayTeamId": float(af.get("TeamId", -1)),
    }
    feats = FEATURE_NUM + FEATURE_ID
    X = np.array([[float(row.get(c) or 0.0) for c in feats]], dtype=np.float64)
    return np.nan_to_num(X, nan=0.0), feats


def _scale(X: np.ndarray, stats: dict) -> np.ndarray:
    feats = stats["features"]
    mu = np.array([stats["mean"].get(c, 0.0) for c in feats], dtype=float)
    sg = np.array([stats["std"].get(c, 1.0) for c in feats], dtype=float)
    sg = np.where(sg == 0, 1.0, sg)
    return (X - mu) / sg


def _predict_paths(paths: dict, X: np.ndarray, target: str) -> Optional[np.ndarray]:
    preds = []
    if "xgboost" in paths and _resolve_model_path(paths["xgboost"]).exists():
        try:
            import xgboost as xgb
            m = xgb.Booster()
            m.load_model(str(_resolve_model_path(paths["xgboost"])))
            p = np.asarray(m.predict(xgb.DMatrix(X)))
            if p.ndim == 1:
                preds.append(np.array([1 - p[0], p[0]]))
            else:
                preds.append(p[0])
        except Exception:
            pass
    if "lightgbm" in paths and _resolve_model_path(paths["lightgbm"]).exists():
        try:
            import lightgbm as lgb
            m = lgb.Booster(model_file=str(_resolve_model_path(paths["lightgbm"])))
            p = np.asarray(m.predict(X)).ravel()
            if target == "ft_result" and p.size >= 3:
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
    ft: np.ndarray,
    o25: float,
    btts: float,
    n: int = N_SIM,
    max_goals: int = 8,
    *,
    lam_h: float | None = None,
    lam_a: float | None = None,
    rho: float = -0.08,
) -> dict:
    """Precise Dixon–Coles + IPF score engine. Returns clean HT/FT distributions only."""
    if len(ft) >= 3:
        p_h, p_d, p_a = float(ft[0]), float(ft[1]), float(ft[2])
    else:
        p_h = p_d = p_a = 1 / 3
    s = max(p_h + p_d + p_a, 1e-12)
    p_h, p_d, p_a = p_h / s, p_d / s, p_a / s

    if lam_h is None:
        lam_h = 0.85 + 1.55 * p_h + 0.35 * p_d
    if lam_a is None:
        lam_a = 0.85 + 1.55 * p_a + 0.35 * p_d
    lam_h = float(np.clip(lam_h, 0.35, 3.8))
    lam_a = float(np.clip(lam_a, 0.35, 3.8))

    def pois(k, lam):
        return exp(-lam) * (lam ** k) / factorial(k)

    def dc_tau(i, j, lh, la, r):
        if i == 0 and j == 0:
            return 1 - lh * la * r
        if i == 0 and j == 1:
            return 1 + lh * r
        if i == 1 and j == 0:
            return 1 + la * r
        if i == 1 and j == 1:
            return 1 - r
        return 1.0

    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] = pois(i, lam_h) * pois(j, lam_a) * max(1e-12, dc_tau(i, j, lam_h, lam_a, rho))
    grid /= grid.sum()

    def margins(g):
        ph = sum(g[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > j)
        pd_ = sum(g[i, i] for i in range(max_goals + 1))
        pa = sum(g[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i < j)
        o25p = sum(g[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j >= 3)
        bttsp = sum(g[i, j] for i in range(1, max_goals + 1) for j in range(1, max_goals + 1))
        return ph, pd_, pa, o25p, bttsp

    # IPF alignment (FT + O2.5 + BTTS)
    for _ in range(24):
        ph, pd_, pa, o25p, bttsp = margins(grid)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i > j and ph > 1e-12:
                    grid[i, j] *= p_h / ph
                elif i == j and pd_ > 1e-12:
                    grid[i, j] *= p_d / pd_
                elif i < j and pa > 1e-12:
                    grid[i, j] *= p_a / pa
        grid /= grid.sum()
        ph, pd_, pa, o25p, bttsp = margins(grid)
        over = np.zeros_like(grid)
        under = np.zeros_like(grid)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                (over if i + j >= 3 else under)[i, j] = grid[i, j]
        so, su = over.sum(), under.sum()
        if so > 1e-12 and su > 1e-12:
            grid = over * (o25 / so) + under * ((1 - o25) / su)
            grid /= grid.sum()
        yes = np.zeros_like(grid)
        no = np.zeros_like(grid)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                (yes if i > 0 and j > 0 else no)[i, j] = grid[i, j]
        sy, sn = yes.sum(), no.sum()
        if sy > 1e-12 and sn > 1e-12:
            grid = yes * (btts / sy) + no * ((1 - btts) / sn)
            grid /= grid.sum()

    rng = np.random.default_rng(42)
    flat = grid.ravel()
    idx = rng.choice(flat.size, size=n, p=flat)
    hs = idx // (max_goals + 1)
    aws = idx % (max_goals + 1)

    # FT score counts (exact like the historical table the user showed)
    ft_counts: Dict[str, int] = {}
    for i, j in zip(hs, aws):
        k = f"{i}-{j}"
        ft_counts[k] = ft_counts.get(k, 0) + 1
    ft_top = dict(sorted(ft_counts.items(), key=lambda kv: -kv[1])[:15])

    # HT grid (~45 % intensity, consistent with FT)
    ht_lh = max(0.15, 0.45 * lam_h)
    ht_la = max(0.15, 0.45 * lam_a)
    ht_grid = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            ht_grid[i, j] = pois(i, ht_lh) * pois(j, ht_la) * max(1e-12, dc_tau(i, j, ht_lh, ht_la, rho * 0.7))
    ht_grid /= max(ht_grid.sum(), 1e-12)
    ht_idx = rng.choice(25, size=n, p=ht_grid.ravel())
    hth = np.minimum(ht_idx // 5, hs)
    hta = np.minimum(ht_idx % 5, aws)

    ht_counts: Dict[str, int] = {}
    for i, j in zip(hth, hta):
        k = f"{i}-{j}"
        ht_counts[k] = ht_counts.get(k, 0) + 1
    ht_top = dict(sorted(ht_counts.items(), key=lambda kv: -kv[1])[:12])

    tot = hs + aws
    return {
        "ft_score_counts": ft_top,
        "ht_score_counts": ht_top,
        "ft_1x2": {
            "H": float(np.mean(hs > aws)),
            "D": float(np.mean(hs == aws)),
            "A": float(np.mean(hs < aws)),
        },
        "ht_1x2": {
            "H": float(np.mean(hth > hta)),
            "D": float(np.mean(hth == hta)),
            "A": float(np.mean(hth < hta)),
        },
        "over25": float(np.mean(tot >= 3)),
        "btts": float(np.mean((hs > 0) & (aws > 0))),
        "xg": {
            "home": round(float(hs.mean()), 3),
            "away": round(float(aws.mean()), 3),
            "total": round(float(tot.mean()), 3),
            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),
        },
        "n_sim": n,
    }


def simulate_match(
    home: str,
    away: str,
    country: Optional[str] = None,
) -> dict:
    hist = _load_hist()
    hist_c = hist
    if country and hist is not None and "Country" in hist.columns:
        sub = hist[hist["Country"] == country]
        if len(sub) >= 40:
            hist_c = sub

    # League priors
    prior_ft = np.array([0.42, 0.28, 0.30])
    prior_o25, prior_btts = 0.45, 0.48
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

    home = _strict_resolve(home, hist_c)
    away = _strict_resolve(away, hist_c)
    X, feats = build_feature_row(home, away, hist_c)

    # Elo → lambda
    try:
        elo_h = float(X[0, feats.index("EloHome")])
        elo_a = float(X[0, feats.index("EloAway")])
        diff = elo_h - elo_a
        lam_h = 1.25 + 0.55 * (1 / (1 + 10 ** (-diff / 400)))
        lam_a = 1.05 + 0.55 * (1 / (1 + 10 ** (diff / 400)))
    except Exception:
        lam_h = lam_a = None

    scope_dir, reg, stats = load_scope(country)
    ft_pred = prior_ft.copy()
    o25_pred = prior_o25
    btts_pred = prior_btts

    if reg and stats:
        try:
            Xs = _scale(X, stats)
            if "ft_result" in reg.get("targets", {}):
                p = _predict_paths(reg["targets"]["ft_result"], Xs, "ft_result")
                if p is not None and len(p) >= 3:
                    ft_pred = p[:3] / p[:3].sum()
            if "over25" in reg.get("targets", {}):
                p = _predict_paths(reg["targets"]["over25"], Xs, "over25")
                if p is not None:
                    o25_pred = float(p[-1] if len(p) > 1 else p[0])
            if "btts" in reg.get("targets", {}):
                p = _predict_paths(reg["targets"]["btts"], Xs, "btts")
                if p is not None:
                    btts_pred = float(p[-1] if len(p) > 1 else p[0])
        except Exception:
            pass

    sim = simulate_scores(ft_pred, o25_pred, btts_pred, lam_h=lam_h, lam_a=lam_a)

    return {
        "home": home,
        "away": away,
        "country": country or "",
        "ft_score_counts": sim["ft_score_counts"],
        "ht_score_counts": sim["ht_score_counts"],
        "ft_1x2": {k: round(v, 4) for k, v in sim["ft_1x2"].items()},
        "ht_1x2": {k: round(v, 4) for k, v in sim["ht_1x2"].items()},
        "over25": round(sim["over25"], 4),
        "btts": round(sim["btts"], 4),
        "xg": sim["xg"],
        "n_sim": sim["n_sim"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--country", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = simulate_match(args.home, args.away, args.country)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
