#!/usr/bin/env python3
"""
Africa match simulation — model ensemble + Dixon–Coles-style score grid.

No odds required. Uses football_models/africa registries trained by train_africa.

Usage:
  python -m africa.sim_africa --home "Kano Pillars FC" --away "Enyimba FC" --country Nigeria
  python -m africa.sim_africa --fixtures fixtures.csv --out africa_sim_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.environ.get("AFRICA_MODELS", str(ROOT / "football_models" / "africa")))
PARQUET = Path(os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")))
N_SIM = int(os.environ.get("N_SIMULATIONS", "6000"))

FEATURE_NUM = [
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "AwayFormPts_5", "HomeFormGD_5", "AwayFormGD_5",
    "HomeFormPts_10", "AwayFormPts_10", "HomeFormGD_10", "AwayFormGD_10",
    "EloHome", "EloAway", "EloDiff",
]
FEATURE_ID = ["HomeTeamId", "AwayTeamId"]


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


def _team_form_elo(hist: pd.DataFrame, team: str, as_of=None) -> dict:
    """Compute form + Elo snapshot for a team name from history."""
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

    # Elo walk
    elo = {}
    K, HOME_ADV = 20.0, 55.0
    for _, r in hist.iterrows():
        hid = t2.get(str(r["HomeTeam"])); aid = t2.get(str(r["AwayTeam"]))
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
    from datetime import datetime
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
    X = np.nan_to_num(X, nan=0.0)
    return X, feats


def _scale(X: np.ndarray, stats: dict) -> np.ndarray:
    feats = stats["features"]
    mu = np.array([stats["mean"].get(c, 0.0) for c in feats], dtype=float)
    sg = np.array([stats["std"].get(c, 1.0) for c in feats], dtype=float)
    sg = np.where(sg == 0, 1.0, sg)
    return (X - mu) / sg


def _predict_paths(paths: dict, X: np.ndarray, target: str) -> Optional[np.ndarray]:
    preds = []
    if "xgboost" in paths and Path(paths["xgboost"]).exists():
        try:
            import xgboost as xgb
            m = xgb.Booster(); m.load_model(paths["xgboost"])
            p = np.asarray(m.predict(xgb.DMatrix(X)))
            if p.ndim == 1:
                preds.append(np.array([1 - p[0], p[0]]))
            else:
                preds.append(p[0])
        except Exception as e:
            print(f"  [warn] xgb: {e}")
    if "lightgbm" in paths and Path(paths["lightgbm"]).exists():
        try:
            import lightgbm as lgb
            m = lgb.Booster(model_file=paths["lightgbm"])
            p = np.asarray(m.predict(X)).ravel()
            if target == "ft_result" and p.size >= 3:
                preds.append(p[:3])
            elif p.size == 1:
                preds.append(np.array([1 - float(p[0]), float(p[0])]))
            else:
                preds.append(p)

        except Exception as e:
            print(f"  [warn] lgbm: {e}")
    if "catboost" in paths and Path(paths["catboost"]).exists():
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(); m.load_model(paths["catboost"])
            preds.append(m.predict_proba(X)[0])
        except Exception as e:
            print(f"  [warn] cat: {e}")
    if "random_forest" in paths and Path(paths["random_forest"]).exists():
        try:
            with open(paths["random_forest"], "rb") as f:
                m = pickle.load(f)
            preds.append(m.predict_proba(X)[0])
        except Exception as e:
            print(f"  [warn] rf: {e}")
    if not preds:
        return None
    max_len = max(len(p) for p in preds)
    aligned = [np.pad(np.asarray(p, float).ravel(), (0, max(0, max_len - len(p))))[:max_len] for p in preds]
    # confidence weight
    weights = []
    for v in aligned:
        v = np.clip(v, 1e-9, 1.0); v = v / v.sum()
        ent = float(-(v * np.log(v)).sum())
        weights.append(1.0 / (0.35 + ent))
    w = np.asarray(weights); w = w / w.sum()
    return (w[:, None] * np.vstack(aligned)).sum(axis=0)


def load_scope(country: Optional[str] = None):
    """Prefer country scope, else GLOBAL."""
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
    odds_ft: tuple | None = None,
    odds_blend: float = 0.0,
) -> dict:
    """Industrial score engine: Poisson seeds + Dixon–Coles rho + multi-target IPF."""
    if len(ft) >= 3:
        p_h, p_d, p_a = float(ft[0]), float(ft[1]), float(ft[2])
    else:
        p_h = p_d = p_a = 1 / 3
    s = max(p_h + p_d + p_a, 1e-12)
    p_h, p_d, p_a = p_h / s, p_d / s, p_a / s

    # Optional market prior blend on FT
    if odds_ft and odds_blend > 0:
        oh, od, oa = odds_ft
        if oh and od and oa and min(oh, od, oa) > 1.01:
            ih, id_, ia = 1 / oh, 1 / od, 1 / oa
            z = ih + id_ + ia
            mh, md, ma = ih / z, id_ / z, ia / z
            a = float(np.clip(odds_blend, 0, 0.45))
            p_h = (1 - a) * p_h + a * mh
            p_d = (1 - a) * p_d + a * md
            p_a = (1 - a) * p_a + a * ma
            s = p_h + p_d + p_a
            p_h, p_d, p_a = p_h / s, p_d / s, p_a / s

    # Expected goals: Elo/form-informed lambdas if provided, else from FT identity
    if lam_h is None:
        lam_h = 0.85 + 1.55 * p_h + 0.35 * p_d
    if lam_a is None:
        lam_a = 0.85 + 1.55 * p_a + 0.35 * p_d
    lam_h = float(np.clip(lam_h, 0.35, 3.8))
    lam_a = float(np.clip(lam_a, 0.35, 3.8))

    # Build Dixon–Coles-adjusted probability grid
    from math import exp, factorial

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

    # IPF toward FT / O25 / BTTS (24 passes)
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
    ft_sim = {
        "H": float(np.mean(hs > aws)),
        "D": float(np.mean(hs == aws)),
        "A": float(np.mean(hs < aws)),
    }
    top = {f"{i}-{j}": float(grid[i, j]) for i in range(max_goals + 1) for j in range(max_goals + 1)}
    top_sorted = dict(sorted(top.items(), key=lambda kv: -kv[1])[:12])
    # goal lines
    tot = hs + aws
    return {
        "ft_sim": ft_sim,
        "over25_sim": float(np.mean(tot >= 3)),
        "btts_sim": float(np.mean((hs > 0) & (aws > 0))),
        "xg": {
            "home": float(hs.mean()),
            "away": float(aws.mean()),
            "total": float(tot.mean()),
            "lambda_home": lam_h,
            "lambda_away": lam_a,
        },
        "score_matrix_top": top_sorted,
        "cs_home": float(np.mean(aws == 0)),
        "cs_away": float(np.mean(hs == 0)),
        "goal_line_sim": {
            "1.5": float(np.mean(tot >= 2)),
            "2.5": float(np.mean(tot >= 3)),
            "3.5": float(np.mean(tot >= 4)),
        },
        "rho": rho,
    }



def simulate_match(
    home: str,
    away: str,
    country: Optional[str] = None,
    *,
    odds_h: float | None = None,
    odds_d: float | None = None,
    odds_a: float | None = None,
    odds_blend: float | None = None,
) -> dict:
    hist = _load_hist()
    hist_c = hist
    if country and hist is not None and "Country" in hist.columns:
        sub = hist[hist["Country"] == country]
        if len(sub) >= 40:
            hist_c = sub

    # League priors from history (industrial base rates)
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
        if "Over2_5" in hist_c.columns:
            prior_o25 = float(pd.to_numeric(hist_c["Over2_5"], errors="coerce").mean() or prior_o25)
        else:
            prior_o25 = float(((hist_c["FTHG"] + hist_c["FTAG"]) > 2.5).mean())
        if "BTTS" in hist_c.columns:
            prior_btts = float(pd.to_numeric(hist_c["BTTS"], errors="coerce").mean() or prior_btts)
        else:
            prior_btts = float(((hist_c["FTHG"] > 0) & (hist_c["FTAG"] > 0)).mean())

    X, feats = build_feature_row(home, away, hist_c)
    # Elo-based lambdas from feature row
    try:
        elo_h = float(X[0, feats.index("EloHome")])
        elo_a = float(X[0, feats.index("EloAway")])
    except Exception:
        elo_h, elo_a = 1500.0, 1500.0
    # convert Elo gap to expected goals (home advantage ~0.25)
    gap = (elo_h - elo_a) / 400.0
    base = 1.15
    lam_h = base * (1.08 ** gap) * 1.12  # home bump
    lam_a = base * (1.08 ** (-gap)) * 0.95

    scope_dir, reg, stats = load_scope(country)
    engines = []
    ft = prior_ft.copy()
    o25 = prior_o25
    btts = prior_btts
    if reg and stats:
        Xs = _scale(X, stats)
        targets = reg.get("targets", {})
        p_ft = _predict_paths(targets.get("ft_result", {}), Xs, "ft_result")
        p_o = _predict_paths(targets.get("over25", {}), Xs, "over25")
        p_b = _predict_paths(targets.get("btts", {}), Xs, "btts")
        # Shrink model toward league prior (stabilizes thin Africa samples)
        shrink = float(os.environ.get("AFRICA_PRIOR_SHRINK", "0.25"))
        if p_ft is not None and len(p_ft) >= 3:
            p_ft = np.asarray(p_ft[:3], float)
            p_ft = p_ft / p_ft.sum()
            ft = (1 - shrink) * p_ft + shrink * prior_ft
            ft = ft / ft.sum()
            engines.append("ft_result")
        if p_o is not None:
            o25 = float(p_o[-1] if len(p_o) > 1 else p_o[0])
            o25 = (1 - shrink) * o25 + shrink * prior_o25
            engines.append("over25")
        if p_b is not None:
            btts = float(p_b[-1] if len(p_b) > 1 else p_b[0])
            btts = (1 - shrink) * btts + shrink * prior_btts
            engines.append("btts")
    if not engines:
        engines.append("league-prior+elo")

    blend = odds_blend if odds_blend is not None else float(os.environ.get("ODDS_BLEND", "0.20"))
    odds_tuple = None
    if odds_h and odds_d and odds_a:
        odds_tuple = (float(odds_h), float(odds_d), float(odds_a))
        engines.append("odds-blend")

    sim = simulate_scores(
        ft, o25, btts, n=N_SIM,
        lam_h=lam_h, lam_a=lam_a, rho=-0.10,
        odds_ft=odds_tuple, odds_blend=blend if odds_tuple else 0.0,
    )
    return {
        "home": home,
        "away": away,
        "country": country,
        "engines": engines,
        "model": {
            "ft": {"H": float(ft[0]), "D": float(ft[1]), "A": float(ft[2])},
            "over25": float(o25),
            "btts": float(btts),
            "prior_ft": {"H": float(prior_ft[0]), "D": float(prior_ft[1]), "A": float(prior_ft[2])},
            "lambda_seed": {"home": lam_h, "away": lam_a},
        },
        "odds": {"H": odds_h, "D": odds_d, "A": odds_a} if odds_tuple else None,
        "sim": sim,
        "scope": str(scope_dir) if scope_dir else None,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", type=str)
    ap.add_argument("--away", type=str)
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--fixtures", type=Path, default=None, help="CSV with home,away[,country]")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = []
    if args.fixtures and args.fixtures.exists():
        fx = pd.read_csv(args.fixtures)
        for _, r in fx.iterrows():
            results.append(simulate_match(
                str(r["home"]), str(r["away"]),
                str(r["country"]) if "country" in r and pd.notna(r["country"]) else args.country,
            ))
    elif args.home and args.away:
        results.append(simulate_match(args.home, args.away, args.country))
    else:
        raise SystemExit("Provide --home/--away or --fixtures")

    text = json.dumps(results if len(results) > 1 else results[0], indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()


def to_simlab_document(rep: dict, fx: dict | None = None, n_sim: int = N_SIM) -> dict:
    """Flatten Africa sim output into EU Sim Lab report schema (admin openSimDetail)."""
    fx = fx or {}
    model = rep.get("model") or {}
    sim = rep.get("sim") or {}
    ft_m = model.get("ft") or {}
    ft_s = sim.get("ft_sim") or ft_m
    o25_m = float(model.get("over25", 0.5))
    o25_s = float(sim.get("over25_sim", o25_m))
    btts_m = float(model.get("btts", 0.5))
    btts_s = float(sim.get("btts_sim", btts_m))
    xg = sim.get("xg") or {"home": 0.0, "away": 0.0, "total": 0.0}
    ph, pd_, pa = float(ft_s.get("H", 0)), float(ft_s.get("D", 0)), float(ft_s.get("A", 0))
    top_map = sim.get("score_matrix_top") or {}
    top_ft = []
    for score, pct in top_map.items():
        # pct is probability 0-1 → approximate count
        top_ft.append([score, int(round(float(pct) * n_sim))])
    top_ft = sorted(top_ft, key=lambda x: -x[1])[:12]

    def pct100(x):
        return round(100.0 * float(x), 1)

    def verdict(model_p, sim_p, thresh=55.0):
        mp, sp = pct100(model_p), pct100(sim_p)
        edge = round(sp - mp, 1)
        yes = sp >= thresh and mp >= (thresh - 5)
        return mp, sp, edge, "YES" if yes else ("LEAN" if sp >= 50 else "—")

    table = []
    for section, selection, mp, sp in [
        ("FT", "Home", ft_m.get("H", ph), ph),
        ("FT", "Draw", ft_m.get("D", pd_), pd_),
        ("FT", "Away", ft_m.get("A", pa), pa),
        ("O/U", "Over 2.5", o25_m, o25_s),
        ("O/U", "Under 2.5", 1 - o25_m, 1 - o25_s),
        ("BTTS", "Yes", btts_m, btts_s),
        ("BTTS", "No", 1 - btts_m, 1 - btts_s),
        ("DC", "1X", float(ft_m.get("H", 0)) + float(ft_m.get("D", 0)), ph + pd_),
        ("DC", "X2", float(ft_m.get("A", 0)) + float(ft_m.get("D", 0)), pa + pd_),
        ("DC", "12", float(ft_m.get("H", 0)) + float(ft_m.get("A", 0)), ph + pa),
    ]:
        mp100, sp100, edge, verd = verdict(mp, sp, 55.0 if section != "FT" or selection != "Draw" else 40.0)
        table.append({
            "Section": section,
            "Selection": selection,
            "Model%": mp100,
            "Sim%": sp100,
            "Agree": "Y" if abs(edge) < 5 else "N",
            "Edge": edge,
            "Verdict": verd,
        })

    cs_h = float(sim.get("cs_home") or 0)
    cs_a = float(sim.get("cs_away") or 0)
    report = {
        "ft_model": {"H": float(ft_m.get("H", 0)), "D": float(ft_m.get("D", 0)), "A": float(ft_m.get("A", 0))},
        "ft_sim": {"H": ph, "D": pd_, "A": pa},
        "ht_model": {"H": None, "D": None, "A": None},
        "ht_sim": {"H": None, "D": None, "A": None},
        "over25_model": o25_m,
        "over25_sim": o25_s,
        "btts_model": btts_m,
        "btts_sim": btts_s,
        "xg": {"home": float(xg.get("home", 0)), "away": float(xg.get("away", 0)), "total": float(xg.get("total", 0))},
        "dc_ft": {"1X": ph + pd_, "X2": pa + pd_, "12": ph + pa},
        "dc_ht": {},
        "top_ft": top_ft,
        "top_ht": [],
        "top3_ft": [{"score": s, "count": c, "pct": round(100 * c / max(n_sim, 1), 1)} for s, c in top_ft[:3]],
        "clean_sheet": {"home": cs_h, "away": cs_a},
        "win_to_nil": {"home": ph * cs_h, "away": pa * cs_a},
        "goal_lines": {
            "1.5": {"over": float((sim.get("goal_line_sim") or {}).get("1.5", min(0.95, o25_s + 0.22))),
                    "under": 1 - float((sim.get("goal_line_sim") or {}).get("1.5", min(0.95, o25_s + 0.22)))},
            "2.5": {"over": float((sim.get("goal_line_sim") or {}).get("2.5", o25_s)),
                    "under": 1 - float((sim.get("goal_line_sim") or {}).get("2.5", o25_s))},
            "3.5": {"over": float((sim.get("goal_line_sim") or {}).get("3.5", max(0.05, o25_s - 0.18))),
                    "under": 1 - float((sim.get("goal_line_sim") or {}).get("3.5", max(0.05, o25_s - 0.18)))},
        },
        "score_consistency": {"ft_vs_model_l1": round(
            abs(ph - float(ft_m.get("H", ph))) + abs(pd_ - float(ft_m.get("D", pd_))) + abs(pa - float(ft_m.get("A", pa))),
            4,
        )},
        "region": "Africa",
        "engines": rep.get("engines") or [],
    }

    home = (fx.get("home") or rep.get("home") or "").strip()
    away = (fx.get("away") or rep.get("away") or "").strip()
    kickoff = (fx.get("date") or fx.get("kickoff") or "")[:16].replace("T", " ")
    league = fx.get("league") or ""
    country = fx.get("country") or rep.get("country") or ""
    league_label = f"{country} — {league}" if country and league else (league or country or "Africa")

    # pick locked tip from strongest FT sim
    best = max([("Home", ph), ("Draw", pd_), ("Away", pa)], key=lambda x: x[1])
    locked = {
        "status": "AFRICA BOARD",
        "section": "FT",
        "selection": best[0],
        "model": pct100(ft_m.get({"Home": "H", "Draw": "D", "Away": "A"}[best[0]], best[1])),
        "sim": pct100(best[1]),
        "verdict": "LEAN" if best[1] >= 0.4 else "—",
    }

    return {
        "id": None,  # filled by caller
        "region": "Africa",
        "match": {
            "home": home,
            "away": away,
            "kickoff": kickoff,
            "league": league_label,
            "odds": (
                f"{rep.get('odds',{}).get('H')}/{rep.get('odds',{}).get('D')}/{rep.get('odds',{}).get('A')}"
                if rep.get("odds") else "—"
            ),
            "odds_h": (rep.get("odds") or {}).get("H"),
            "odds_d": (rep.get("odds") or {}).get("D"),
            "odds_a": (rep.get("odds") or {}).get("A"),
            "country": country,
        },
        "resolved": {"models": ",".join(rep.get("engines") or []) or "africa"},
        "locked_tip": locked,
        "report": report,
        "table": table,
    }
