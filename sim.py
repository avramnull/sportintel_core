#!/usr/bin/env python3
"""
LIVE MATCH SIMULATOR — ADVANCED REPORT
Ensembles all backends (XGB/LGBM/CAT/ADA/RF/Torch/TF).
Report: HT + FT tables with model%, sim%, agreement, verdict.
Final LOCKED SECURED TIP.
"""
from __future__ import annotations

import json, pickle, warnings, os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =============================================================================
# MATCH CONFIG — EDIT ONLY THIS
# =============================================================================
MATCH = {
    "league": "La Liga",
    "home_team": "Valencia",
    "away_team": "Betis",
    "odds_home": 2.87,
    "odds_draw": 3.39,
    "odds_away": 2.66,
    "n_simulations": 8000,
    "seed": 42,
    "odds_blend": 0.30,  # 0=pure model, 1=pure market
}

def _find_root() -> Path:
    candidates = []
    if Path("/content").exists():
        candidates.append(Path("/content"))
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd())
    for c in candidates:
        if (c / "football_models" / "mappings" / "team2id.json").exists():
            return c
        if (c / "football_models" / "model_registry.json").exists():
            return c
        if (c / "master_football_data.parquet").exists():
            return c
    return candidates[0]

ROOT = _find_root()
MODELS_ROOT = ROOT / "football_models"
MAP_DIR = MODELS_ROOT / "mappings"
PARQUET_PATH = ROOT / "master_football_data.parquet"


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def normalize_team(name, aliases):
    s = " ".join(str(name).strip().split())
    return aliases.get(s.lower(), s.title() if (s.islower() or s.isupper()) else s)


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def resolve_league(league_input, league_map):
    inv = {v.lower(): k for k, v in league_map.items()}
    s = str(league_input).strip()
    if s in league_map:
        return s, league_map[s]
    if s.lower() in inv:
        code = inv[s.lower()]
        return code, league_map[code]
    for code, name in league_map.items():
        if s.lower() in name.lower() or name.lower() in s.lower():
            return code, name
    raise ValueError(f"Unknown league '{league_input}'")


def implied(odds):
    return 1.0 / max(float(odds), 1.01)


def fair_probs(oh, od, oa):
    ih, id_, ia = implied(oh), implied(od), implied(oa)
    t = ih + id_ + ia
    return ih / t, id_ / t, ia / t


def collect_run_dirs(home_canon, away_canon, strict: bool = True):
    runs = []
    for label, canon in [("home", home_canon), ("away", away_canon)]:
        d = MODELS_ROOT / "teams" / slug(canon)
        if (d / "registry.json").exists() or (d / "models").is_dir():
            runs.append((d, f"{canon} ({label})"))
    if runs:
        return runs
    g = MODELS_ROOT / "global"
    if (g / "registry.json").exists() or (g / "models").is_dir():
        return [(g, "global")]
    if strict:
        raise SystemExit(f"No models for {home_canon}/{away_canon} and no global/")
    return []


def load_run_targets(run_dir):
    reg_path = run_dir / "registry.json"
    if reg_path.exists():
        return load_json(reg_path).get("targets", {})
    targets = {}
    models_dir = run_dir / "models"
    if models_dir.is_dir():
        for p in models_dir.iterdir():
            for backend, prefix in [
                ("xgboost", "xgb_"), ("lightgbm", "lgbm_"), ("catboost", "cat_"),
                ("adaboost", "ada_"), ("random_forest", "rf_"),
                ("pytorch", "torch_"), ("tensorflow", "tf_"),
            ]:
                if p.name.startswith(prefix):
                    tkey = p.name[len(prefix):].rsplit(".", 1)[0]
                    if tkey.endswith(".keras"):
                        tkey = tkey[:-6]
                    targets.setdefault(tkey, {})[backend] = str(p)
    return targets


def load_preprocessors(run_dir, target_key="ft_result"):
    preproc_dir = run_dir / "preprocessors"
    # prefer target-specific, fallback to ft_result
    feat_path = preproc_dir / f"features_{target_key}.json"
    if not feat_path.exists():
        feat_path = preproc_dir / "features_ft_result.json"
    scaler_path = preproc_dir / f"scaler_{target_key}.pkl"
    if not scaler_path.exists():
        scaler_path = preproc_dir / "scaler_ft_result.pkl"
    le_path = preproc_dir / f"le_div_{target_key}.pkl"
    if not le_path.exists():
        le_path = preproc_dir / "le_div_ft_result.pkl"
    feature_names = load_json(feat_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(le_path, "rb") as f:
        le_div = pickle.load(f)
    return feature_names, scaler, le_div


def build_live_features(home_id, away_id, div_code, odds_h, odds_d, odds_a,
                        hist_df, feature_names, le_div):
    now = datetime.utcnow()
    ih, id_, ia = fair_probs(odds_h, odds_d, odds_a)
    imp_sum = ih + id_ + ia
    row = {
        "AvgH": odds_h, "AvgD": odds_d, "AvgA": odds_a,
        "B365H": odds_h, "B365D": odds_d, "B365A": odds_a,
        "LogOddsH": np.log(max(odds_h, 1.01)), "LogOddsD": np.log(max(odds_d, 1.01)),
        "LogOddsA": np.log(max(odds_a, 1.01)),
        "ImpH": implied(odds_h), "ImpD": implied(odds_d), "ImpA": implied(odds_a),
        "OddsMargin": imp_sum - 1.0 if imp_sum else 0.0,
        "HomeOddsEdge": ih, "AwayOddsEdge": ia,
        "Year": now.year, "Month": now.month, "DayOfWeek": now.weekday(),
        "IsWeekend": int(now.weekday() >= 5),
        "H2H_HomeWins_5": 0, "H2H_Draws_5": 0, "H2H_AwayWins_5": 0,
    }
    for prefix, tid in [("Home", home_id), ("Away", away_id)]:
        for w in (5, 10, 20):
            row[f"{prefix}FormPts_{w}"] = np.nan
            row[f"{prefix}FormGF_{w}"] = np.nan
            row[f"{prefix}FormGA_{w}"] = np.nan
            row[f"{prefix}FormGD_{w}"] = np.nan
        if hist_df is not None and len(hist_df):
            if prefix == "Home":
                sub = hist_df[hist_df["HomeTeamId"] == tid].sort_values("Date").tail(20)
                pts = sub["HomePoints"] if "HomePoints" in sub.columns else pd.Series(dtype=float)
                gf = sub["FTHG"] if "FTHG" in sub.columns else pd.Series(dtype=float)
                ga = sub["FTAG"] if "FTAG" in sub.columns else pd.Series(dtype=float)
            else:
                sub = hist_df[hist_df["AwayTeamId"] == tid].sort_values("Date").tail(20)
                pts = sub["AwayPoints"] if "AwayPoints" in sub.columns else pd.Series(dtype=float)
                gf = sub["FTAG"] if "FTAG" in sub.columns else pd.Series(dtype=float)
                ga = sub["FTHG"] if "FTHG" in sub.columns else pd.Series(dtype=float)
            for w in (5, 10, 20):
                row[f"{prefix}FormPts_{w}"] = float(pts.tail(w).mean()) if len(pts) else np.nan
                row[f"{prefix}FormGF_{w}"] = float(gf.tail(w).mean()) if len(gf) else np.nan
                row[f"{prefix}FormGA_{w}"] = float(ga.tail(w).mean()) if len(ga) else np.nan
                if np.isfinite(row[f"{prefix}FormGF_{w}"]) and np.isfinite(row[f"{prefix}FormGA_{w}"]):
                    row[f"{prefix}FormGD_{w}"] = row[f"{prefix}FormGF_{w}"] - row[f"{prefix}FormGA_{w}"]
    vals = []
    for name in feature_names:
        if name == "HomeTeamId":
            vals.append(home_id)
        elif name == "AwayTeamId":
            vals.append(away_id)
        elif name == "DivEnc":
            try:
                vals.append(int(le_div.transform([div_code])[0]))
            except Exception:
                vals.append(0)
        else:
            vals.append(row.get(name, 0.0))
    return np.nan_to_num(np.array(vals, dtype=np.float64).reshape(1, -1), nan=0.0)


def _predict_torch(path, X):
    import torch
    import torch.nn as nn
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    in_dim, hidden, out_dim = ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"]
    task = ckpt["task"]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.0)]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = MLP()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        if task == "multiclass":
            return torch.softmax(logits, dim=1).numpy()[0]
        return np.array([1 - float(torch.sigmoid(logits)[0, 0]), float(torch.sigmoid(logits)[0, 0])])


def _predict_tf(path, X):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    pred = model.predict(X, verbose=0)
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        return pred[0]
    p = float(pred.ravel()[0])
    return np.array([1 - p, p])


def predict_backends(target_key, X_scaled, targets) -> Tuple[Optional[np.ndarray], List[str]]:
    """Returns (mean_prob, list of backend names that contributed)."""
    paths = targets.get(target_key, {})
    if not isinstance(paths, dict) or "error" in paths:
        return None, []
    preds, names = [], []

    if "xgboost" in paths and Path(paths["xgboost"]).exists():
        try:
            import xgboost as xgb
            m = xgb.Booster()
            m.load_model(paths["xgboost"])
            p = np.asarray(m.predict(xgb.DMatrix(X_scaled)))
            preds.append(np.array([1 - p[0], p[0]]) if p.ndim == 1 else p[0])
            names.append("xgb")
        except Exception as e:
            print(f"  [warn] xgb {target_key}: {e}")

    if "lightgbm" in paths and Path(paths["lightgbm"]).exists():
        try:
            import lightgbm as lgb
            m = lgb.Booster(model_file=paths["lightgbm"])
            p = np.asarray(m.predict(X_scaled))
            preds.append(np.array([1 - p[0], p[0]]) if p.ndim == 1 else p[0])
            names.append("lgbm")
        except Exception as e:
            print(f"  [warn] lgbm {target_key}: {e}")

    if "catboost" in paths and Path(paths["catboost"]).exists():
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(paths["catboost"])
            preds.append(m.predict_proba(X_scaled)[0])
            names.append("cat")
        except Exception as e:
            print(f"  [warn] cat {target_key}: {e}")

    if "adaboost" in paths and Path(paths["adaboost"]).exists():
        try:
            with open(paths["adaboost"], "rb") as f:
                m = pickle.load(f)
            p = m.predict_proba(X_scaled)[0]
            if np.isfinite(p).all():
                preds.append(p)
                names.append("ada")
        except Exception as e:
            print(f"  [warn] ada {target_key}: {e}")

    if "random_forest" in paths and Path(paths["random_forest"]).exists():
        try:
            with open(paths["random_forest"], "rb") as f:
                m = pickle.load(f)
            p = m.predict_proba(X_scaled)[0]
            if np.isfinite(p).all():
                preds.append(p)
                names.append("rf")
        except Exception as e:
            print(f"  [warn] rf {target_key}: {e}")

    if "pytorch" in paths and Path(paths["pytorch"]).exists():
        try:
            preds.append(_predict_torch(paths["pytorch"], X_scaled))
            names.append("torch")
        except Exception as e:
            print(f"  [warn] torch {target_key}: {e}")

    if "tensorflow" in paths and Path(paths["tensorflow"]).exists():
        try:
            preds.append(_predict_tf(paths["tensorflow"], X_scaled))
            names.append("tf")
        except Exception as e:
            print(f"  [warn] tf {target_key}: {e}")

    if not preds:
        return None, []
    max_len = max(len(p) for p in preds)
    aligned = [np.pad(np.asarray(p, float).ravel(), (0, max(0, max_len - len(p))))[:max_len] for p in preds]
    return np.mean(aligned, axis=0), names


def predict_across_runs(target_key, runs, home_id, away_id, div_code, oh, od, oa, hist_df):
    preds, all_backends = [], []
    for run_dir, label, targets, scaler, le_div, feature_names in runs:
        X = build_live_features(home_id, away_id, div_code, oh, od, oa, hist_df, feature_names, le_div)
        Xs = X.copy()
        n_scaled = min(Xs.shape[1] - 3, scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else Xs.shape[1] - 3)
        Xs[:, :n_scaled] = scaler.transform(X[:, :n_scaled])
        Xs = np.nan_to_num(Xs, nan=0.0)
        p, backends = predict_backends(target_key, Xs, targets)
        if p is not None:
            preds.append(np.asarray(p, float).ravel())
            all_backends.extend(backends)
            print(f"  {target_key:12s}  [{label}]  backends={backends}  -> {np.round(p, 3)}")
    if not preds:
        return None, []
    max_len = max(len(p) for p in preds)
    aligned = [np.pad(p, (0, max(0, max_len - len(p))))[:max_len] for p in preds]
    return np.mean(aligned, axis=0), sorted(set(all_backends))


def multi_prob(p, n=3):
    if p is None:
        return np.ones(n) / n
    p = np.asarray(p, float).ravel()
    if len(p) < n:
        p = np.pad(p, (0, n - len(p)))
    p = np.clip(p[:n], 1e-6, None)
    return p / p.sum()


def bin_prob(p):
    if p is None:
        return 0.5
    p = np.asarray(p).ravel()
    return float(p[-1]) if len(p) <= 2 else float(p[1])


def blend_with_odds(p_model, p_market, alpha):
    p_model = np.asarray(p_model, float)
    p_market = np.asarray(p_market, float)
    n = max(len(p_model), len(p_market))
    a = np.pad(p_model, (0, max(0, n - len(p_model))))[:n]
    b = np.pad(p_market, (0, max(0, n - len(p_market))))[:n]
    a, b = a / a.sum(), b / b.sum()
    out = (1 - alpha) * a + alpha * b
    return out / out.sum()


def agreement_pct(backends_count: int, total_possible: int = 7) -> str:
    if backends_count <= 0:
        return "0%"
    return f"{100 * backends_count / max(total_possible, 1):.0f}%"


def simulate_match(p_ft, p_ht, p_over25, p_btts, p_ht_over15, n, seed):
    rng = np.random.default_rng(seed)
    p_ft = p_ft / p_ft.sum()
    ft_outcomes = rng.choice(["H", "D", "A"], size=n, p=p_ft)
    scores = []
    for outcome in ft_outcomes:
        over, btts = rng.random() < p_over25, rng.random() < p_btts
        if outcome == "H":
            if over and btts:
                h, a = int(rng.choice([2, 3, 4])), int(rng.choice([1, 2]))
                h = max(h, a + 1)
            elif over:
                h, a = int(rng.choice([3, 4, 5])), 0
            elif btts:
                h, a = int(rng.choice([1, 2])), 1
                h = max(h, a + 1)
            else:
                h, a = int(rng.choice([1, 2])), 0
        elif outcome == "A":
            if over and btts:
                a, h = int(rng.choice([2, 3, 4])), int(rng.choice([1, 2]))
                a = max(a, h + 1)
            elif over:
                a, h = int(rng.choice([3, 4, 5])), 0
            elif btts:
                a, h = int(rng.choice([1, 2])), 1
                a = max(a, h + 1)
            else:
                a, h = int(rng.choice([1, 2])), 0
        else:
            if over and btts:
                h = a = int(rng.choice([2, 3]))
            elif btts:
                h = a = 1
            else:
                h = a = 0
        scores.append((h, a))
    scores = np.array(scores)
    if p_ht is None or len(p_ht) < 3:
        p_ht = np.array([0.35, 0.30, 0.35])
    else:
        p_ht = p_ht / p_ht.sum()
    ht_scores = []
    for _ in range(n):
        o = rng.choice(["H", "D", "A"], p=p_ht)
        if o == "H":
            ht_scores.append((1, 0) if rng.random() > 0.3 else (2, 0) if rng.random() > 0.5 else (2, 1))
        elif o == "A":
            ht_scores.append((0, 1) if rng.random() > 0.3 else (0, 2) if rng.random() > 0.5 else (1, 2))
        else:
            ht_scores.append((0, 0) if rng.random() > 0.4 else (1, 1))
    ht_scores = np.array(ht_scores)
    top = Counter(map(tuple, scores)).most_common(8)
    top_ht = Counter(map(tuple, ht_scores)).most_common(5)
    return {
        "n": n,
        "ft_model": {"H": float(p_ft[0]), "D": float(p_ft[1]), "A": float(p_ft[2])},
        "ft_sim": {
            "H": float((scores[:, 0] > scores[:, 1]).mean()),
            "D": float((scores[:, 0] == scores[:, 1]).mean()),
            "A": float((scores[:, 0] < scores[:, 1]).mean()),
        },
        "ht_model": {"H": float(p_ht[0]), "D": float(p_ht[1]), "A": float(p_ht[2])},
        "ht_sim": {
            "H": float((ht_scores[:, 0] > ht_scores[:, 1]).mean()),
            "D": float((ht_scores[:, 0] == ht_scores[:, 1]).mean()),
            "A": float((ht_scores[:, 0] < ht_scores[:, 1]).mean()),
        },
        "over25_model": float(p_over25),
        "over25_sim": float((scores.sum(1) > 2.5).mean()),
        "btts_model": float(p_btts),
        "btts_sim": float(((scores[:, 0] > 0) & (scores[:, 1] > 0)).mean()),
        "ht_over15_model": float(p_ht_over15),
        "ht_over15_sim": float((ht_scores.sum(1) > 1.5).mean()),
        "dc_ft": {
            "1X": float(p_ft[0] + p_ft[1]),
            "X2": float(p_ft[1] + p_ft[2]),
            "12": float(p_ft[0] + p_ft[2]),
        },
        "dc_ht": {
            "1X": float(p_ht[0] + p_ht[1]),
            "X2": float(p_ht[1] + p_ht[2]),
            "12": float(p_ht[0] + p_ht[2]),
        },
        "top_ft": [(f"{h}-{a}", int(c)) for (h, a), c in top],
        "top_ht": [(f"{h}-{a}", int(c)) for (h, a), c in top_ht],
        "xg": {
            "home": float(scores[:, 0].mean()),
            "away": float(scores[:, 1].mean()),
            "total": float(scores.sum(1).mean()),
        },
    }


def verdict(model_p, threshold=0.52):
    return "YES" if model_p >= threshold else "NO"


def build_table_rows(report, backends_map, home, away):
    """Simple rows: Market | Selection | Model% | Sim% | Agree | Verdict"""
    rows = []

    def add(section, selection, model_p, sim_p, backends, thr=0.52):
        rows.append({
            "Section": section,
            "Selection": selection,
            "Model%": round(model_p * 100, 1),
            "Sim%": round(sim_p * 100, 1),
            "Agree": f"{len(backends)} eng" if backends else "-",
            "Verdict": verdict(model_p, thr),
        })

    ht, hts = report["ht_model"], report["ht_sim"]
    ft, fts = report["ft_model"], report["ft_sim"]
    b_ht = backends_map.get("ht_result", [])
    b_ft = backends_map.get("ft_result", [])
    b_o25 = backends_map.get("over25", [])
    b_btts = backends_map.get("btts", [])
    b_hto = backends_map.get("ht_over15", [])

    # HT winner
    add("HT Winner", f"{home}", ht["H"], hts["H"], b_ht)
    add("HT Winner", "Draw", ht["D"], hts["D"], b_ht)
    add("HT Winner", f"{away}", ht["A"], hts["A"], b_ht)
    # HT Double chance
    add("HT DC", "1X", report["dc_ht"]["1X"], hts["H"] + hts["D"], b_ht, 0.55)
    add("HT DC", "X2", report["dc_ht"]["X2"], hts["D"] + hts["A"], b_ht, 0.55)
    add("HT DC", "12", report["dc_ht"]["12"], hts["H"] + hts["A"], b_ht, 0.55)
    # HT O/U
    add("HT O/U", "Over 1.5", report["ht_over15_model"], report["ht_over15_sim"], b_hto)
    add("HT O/U", "Under 1.5", 1 - report["ht_over15_model"], 1 - report["ht_over15_sim"], b_hto)

    # FT winner
    add("FT Winner", f"{home}", ft["H"], fts["H"], b_ft)
    add("FT Winner", "Draw", ft["D"], fts["D"], b_ft)
    add("FT Winner", f"{away}", ft["A"], fts["A"], b_ft)
    # FT DC
    add("FT DC", "1X", report["dc_ft"]["1X"], fts["H"] + fts["D"], b_ft, 0.55)
    add("FT DC", "X2", report["dc_ft"]["X2"], fts["D"] + fts["A"], b_ft, 0.55)
    add("FT DC", "12", report["dc_ft"]["12"], fts["H"] + fts["A"], b_ft, 0.55)
    # FT O/U
    add("FT O/U", "Over 2.5", report["over25_model"], report["over25_sim"], b_o25)
    add("FT O/U", "Under 2.5", 1 - report["over25_model"], 1 - report["over25_sim"], b_o25)
    # BTTS
    add("BTTS", "Yes", report["btts_model"], report["btts_sim"], b_btts)
    add("BTTS", "No", 1 - report["btts_model"], 1 - report["btts_sim"], b_btts)

    return rows


def print_table(rows):
    print("\n" + "=" * 78)
    print("  MATCH MARKETS TABLE")
    print("=" * 78)
    print(f"{'Section':<10} {'Selection':<18} {'Model%':>8} {'Sim%':>8} {'Agree':>8} {'Verdict':>8}")
    print("-" * 78)
    cur = None
    for r in rows:
        if r["Section"] != cur:
            if cur is not None:
                print("-" * 78)
            cur = r["Section"]
        print(f"{r['Section']:<10} {r['Selection']:<18} {r['Model%']:7.1f}% {r['Sim%']:7.1f}% {r['Agree']:>8} {r['Verdict']:>8}")
    print("=" * 78)


def locked_tip(rows, report, home, away):
    """Hard strong secured tip from highest-confidence YES leans."""
    # Prefer DC / winner / goals with high model% and YES
    candidates = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 55]
    # rank by model% then sim agreement (model close to sim)
    def score(r):
        agree_sim = 100 - abs(r["Model%"] - r["Sim%"])
        return r["Model%"] + 0.15 * agree_sim

    candidates = sorted(candidates, key=score, reverse=True)
    print("\n" + "=" * 78)
    print("  LOCKED SECURED TIP")
    print("=" * 78)
    if not candidates:
        # fallback: highest model% overall
        fallback = max(rows, key=lambda r: r["Model%"])
        print(f"  No high-confidence YES. Soft lean: {fallback['Section']} → {fallback['Selection']}")
        print(f"  Model {fallback['Model%']:.1f}%  Sim {fallback['Sim%']:.1f}%  → NOT SECURED")
        tip = {"status": "NO LOCK", "selection": fallback["Selection"], "section": fallback["Section"],
               "model": fallback["Model%"], "sim": fallback["Sim%"]}
    else:
        top = candidates[0]
        # require model>=58 and |model-sim|<12 for SECURED
        locked = top["Model%"] >= 58 and abs(top["Model%"] - top["Sim%"]) <= 12
        status = "SECURED LOCK" if locked else "STRONG LEAN"
        print(f"  Status     : {status}")
        print(f"  Market     : {top['Section']}")
        print(f"  Selection  : {top['Selection']}")
        print(f"  Model      : {top['Model%']:.1f}%")
        print(f"  Simulation : {top['Sim%']:.1f}%")
        print(f"  Engines    : {top['Agree']}")
        if locked:
            print("  Verdict    : HARD YES — take this side")
        else:
            print("  Verdict    : YES lean — size down if needed")
        # secondary
        if len(candidates) > 1:
            print(f"\n  Secondary  : {candidates[1]['Section']} → {candidates[1]['Selection']} "
                  f"({candidates[1]['Model%']:.1f}%)")
        tip = {
            "status": status,
            "section": top["Section"],
            "selection": top["Selection"],
            "model": top["Model%"],
            "sim": top["Sim%"],
            "agree": top["Agree"],
            "verdict": "HARD YES" if locked else "YES LEAN",
        }
    # scoreline context
    print(f"\n  Top FT scores : {', '.join(f'{s}({c})' for s,c in report['top_ft'][:5])}")
    print(f"  Top HT scores : {', '.join(f'{s}({c})' for s,c in report['top_ht'][:3])}")
    print(f"  Exp goals     : H {report['xg']['home']:.2f}  A {report['xg']['away']:.2f}  T {report['xg']['total']:.2f}")
    print("=" * 78)
    return tip



def run_one_match(match_cfg: dict, quiet: bool = False, allow_market_only: bool = True) -> dict:
    """
    Run a single simulation using the same engine as CLI main().
    Returns the full report dict (match / resolved / report / table / locked_tip / backends / generated_at).
    If models are missing and allow_market_only, uses fair market probs as the model signal.
    """
    def log(*a, **k):
        if not quiet:
            print(*a, **k)

    if not MAP_DIR.exists():
        raise FileNotFoundError(f"Mappings not found: {MAP_DIR}")

    team2id = load_json(MAP_DIR / "team2id.json")
    league_map = load_json(MAP_DIR / "league_map.json")
    aliases = load_json(MAP_DIR / "team_aliases.json")

    # Accept Div code or league name
    league_in = match_cfg.get("league") or match_cfg.get("div") or "E0"
    try:
        div_code, league_name = resolve_league(league_in, league_map)
    except ValueError:
        div_code = str(league_in).strip()
        league_name = league_map.get(div_code, div_code)

    home_canon = normalize_team(match_cfg["home_team"], aliases)
    away_canon = normalize_team(match_cfg["away_team"], aliases)
    home_id = team2id.get(home_canon)
    away_id = team2id.get(away_canon)
    if home_id is None or away_id is None:
        if not allow_market_only:
            raise KeyError(f"Team not in map: home={home_canon} away={away_canon}")
        home_id = home_id if home_id is not None else -1
        away_id = away_id if away_id is not None else -2

    log(f"Resolved: {league_name} ({div_code}) | {home_canon} vs {away_canon}")

    run_dirs = collect_run_dirs(home_canon, away_canon, strict=False)
    runs_loaded = []
    skipped = []
    for run_dir, label in run_dirs:
        try:
            targets = load_run_targets(run_dir)
            feature_names, scaler, le_div = load_preprocessors(run_dir)
            runs_loaded.append((run_dir, label, targets, scaler, le_div, feature_names))
            log(f"  loaded {label}")
        except Exception as e:
            skipped.append(f"{label}: {e}")
            log(f"  SKIP {label}: {e}")

    if runs_loaded:
        model_source = " + ".join(l for _, l, *_ in runs_loaded)
    elif run_dirs:
        model_source = "market-only (models found but failed to load)"
    else:
        model_source = f"market-only (no models for {home_canon} / {away_canon})"
    log(f"Using models: {model_source}")

    if not runs_loaded and not allow_market_only:
        raise RuntimeError("No usable model runs")

    hist_df = None
    if PARQUET_PATH.exists() and runs_loaded:
        try:
            hist_df = pd.read_parquet(PARQUET_PATH, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Div"])
            hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors="coerce")
            hist_df["HomeTeamCanon"] = hist_df["HomeTeam"].map(lambda x: normalize_team(x, aliases))
            hist_df["AwayTeamCanon"] = hist_df["AwayTeam"].map(lambda x: normalize_team(x, aliases))
            hist_df["HomeTeamId"] = hist_df["HomeTeamCanon"].map(team2id)
            hist_df["AwayTeamId"] = hist_df["AwayTeamCanon"].map(team2id)
            hist_df["HomePoints"] = hist_df["FTR"].map({"H": 3, "D": 1, "A": 0})
            hist_df["AwayPoints"] = hist_df["FTR"].map({"H": 0, "D": 1, "A": 3})
        except Exception as e:
            log(f"History warning: {e}")

    oh = float(match_cfg["odds_home"])
    od = float(match_cfg["odds_draw"])
    oa = float(match_cfg["odds_away"])
    alpha = float(match_cfg.get("odds_blend", 0.30))
    n_sim = int(match_cfg.get("n_simulations", 8000))
    seed = int(match_cfg.get("seed", 42))
    p_market = np.array(fair_probs(oh, od, oa))

    backends_map = {}
    if runs_loaded:
        log("\nPer-run predictions:")
        raw_ft, b = predict_across_runs("ft_result", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
        backends_map["ft_result"] = b
        raw_ht, b = predict_across_runs("ht_result", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
        backends_map["ht_result"] = b
        raw_over, b = predict_across_runs("over25", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
        backends_map["over25"] = b
        raw_btts, b = predict_across_runs("btts", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
        backends_map["btts"] = b
        raw_hto, b = predict_across_runs("ht_over15", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
        backends_map["ht_over15"] = b
        p_ft = blend_with_odds(multi_prob(raw_ft, 3), p_market, alpha)
        p_ht = multi_prob(raw_ht, 3)
        p_over = bin_prob(raw_over)
        p_btts = bin_prob(raw_btts)
        p_hto = bin_prob(raw_hto)
    else:
        # Pure market path — same schema
        p_ft = p_market
        p_ht = np.array([0.30, 0.40, 0.30])  # HT lean draw historically
        p_over = float(min(0.75, max(0.25, 0.35 + 0.15 * (1/oh + 1/oa))))
        p_btts = float(min(0.70, max(0.30, 0.45)))
        p_hto = 0.35
        backends_map = {k: ["market"] for k in ("ft_result", "ht_result", "over25", "btts", "ht_over15")}

    report = simulate_match(p_ft, p_ht, p_over, p_btts, p_hto, n_sim, seed)
    # goal_rates optional enrichment matching attached sample
    report.setdefault("goal_rates", {
        "home": float(report["xg"]["home"]),
        "away": float(report["xg"]["away"]),
    })

    log("\n" + "=" * 78)
    log(f"  {home_canon}  vs  {away_canon}   |   {league_name}")
    log(f"  Odds H {oh:.2f}  D {od:.2f}  A {oa:.2f}   |   blend market {alpha:.0%}")
    log(f"  Engines: {model_source}")
    log("=" * 78)

    rows = build_table_rows(report, backends_map, home_canon, away_canon)
    if not quiet:
        print_table(rows)
        tip = locked_tip(rows, report, home_canon, away_canon)
    else:
        # silent locked tip (no console noise)
        tip = _locked_tip_silent(rows, report)

    match_out = dict(match_cfg)
    if match_cfg.get("match_date"):
        match_out["match_date"] = match_cfg["match_date"]

    payload = {
        "match": match_out,
        "resolved": {
            "league": league_name,
            "div": div_code,
            "home": home_canon,
            "away": away_canon,
            "models": model_source,
            "match_date": match_cfg.get("match_date"),
        },
        "report": report,
        "table": rows,
        "locked_tip": tip,
        "backends": backends_map,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return payload


def _locked_tip_silent(rows, report):
    candidates = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 55]
    def score(r):
        agree_sim = 100 - abs(r["Model%"] - r["Sim%"])
        return r["Model%"] + 0.15 * agree_sim
    candidates = sorted(candidates, key=score, reverse=True)
    if not candidates:
        fallback = max(rows, key=lambda r: r["Model%"])
        return {
            "status": "NO LOCK",
            "selection": fallback["Selection"],
            "section": fallback["Section"],
            "model": fallback["Model%"],
            "sim": fallback["Sim%"],
        }
    top = candidates[0]
    locked = top["Model%"] >= 58 and abs(top["Model%"] - top["Sim%"]) <= 12
    return {
        "status": "SECURED LOCK" if locked else "STRONG LEAN",
        "section": top["Section"],
        "selection": top["Selection"],
        "model": top["Model%"],
        "sim": top["Sim%"],
        "agree": top["Agree"],
        "verdict": "HARD YES" if locked else "YES LEAN",
    }



def main():
    print(f"ROOT = {ROOT}")
    payload = run_one_match(MATCH, quiet=False, allow_market_only=False)
    out = ROOT / "last_simulation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
