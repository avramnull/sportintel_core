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
import team_mapping
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# Force CPU — GitHub Actions has no GPU; avoids CUDA hang/spam
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CATBOOST_FORCE_CPU", "1")

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


def normalize_team(name, aliases=None):
    """Accurate FDC mapping — returns canonical string (status discarded)."""
    known = None
    try:
        known = team_mapping.load_team2id()
    except Exception:
        known = None
    canon, _status = team_mapping.normalize_team(name, aliases, known)
    return canon


def slug(name):
    return team_mapping.slug(name)


def resolve_league(league_input, league_map=None):
    return team_mapping.resolve_league(league_input)


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


_PREPROC_CACHE: Dict[Any, Any] = {}


def load_preprocessors(run_dir, target_key="ft_result"):
    preproc_dir = run_dir / "preprocessors"
    cache_key = (str(run_dir), str(target_key))
    cached = _PREPROC_CACHE.get(cache_key)
    if cached is not None:
        return cached
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
    le_y = None
    le_y_path = preproc_dir / f"le_y_{target_key}.pkl"
    if not le_y_path.exists() and target_key != "ft_result":
        le_y_path = preproc_dir / "le_y_ft_result.pkl"
    if le_y_path.exists():
        try:
            with open(le_y_path, "rb") as f:
                le_y = pickle.load(f)
        except Exception:
            le_y = None
    packed = (feature_names, scaler, le_div, le_y)
    _PREPROC_CACHE[cache_key] = packed
    return packed


HDA_ORDER = ("H", "D", "A")
_ALIGN_HDA_LOGGED = False


def align_hda(p, le_y=None):
    """Reorder a 3-class vector to [P(H), P(D), P(A)].

    sklearn LabelEncoder.fit sorts classes, so models trained with
    LabelEncoder on FTR/HTR emit [P(A), P(D), P(H)]. New training writes
    classes_ as ["H","D","A"]. Always remap via the encoder (or the known
    historical A,D,H order) so column 0 is home.
    """
    global _ALIGN_HDA_LOGGED
    if p is None:
        return None
    p = np.asarray(p, dtype=float).ravel()
    if p.size < 3:
        return p
    classes = None
    if le_y is not None and getattr(le_y, "classes_", None) is not None:
        classes = [str(c) for c in le_y.classes_]
    if not classes or not all(c in classes for c in HDA_ORDER):
        classes = ["A", "D", "H"]  # sklearn default sort of {H,D,A}
    idx = {c: i for i, c in enumerate(classes)}
    out = np.array(
        [float(p[idx["H"]]), float(p[idx["D"]]), float(p[idx["A"]])],
        dtype=float,
    )
    out = np.clip(out, 1e-12, None)
    if not _ALIGN_HDA_LOGGED:
        _ALIGN_HDA_LOGGED = True
        if classes != list(HDA_ORDER):
            print(f"  [align_hda] remapped class order {classes} -> {list(HDA_ORDER)}")
        else:
            print(f"  [align_hda] class order already {list(HDA_ORDER)}")
    return out / out.sum()


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
        if hist_df is not None and len(hist_df) and tid is not None and int(tid) >= 0:
            # ALL recent matches for this team (home or away) — fixes major form bias
            mask = (hist_df["HomeTeamId"] == tid) | (hist_df["AwayTeamId"] == tid)
            sub = hist_df.loc[mask].sort_values("Date").tail(25)
            if len(sub):
                is_home = (sub["HomeTeamId"] == tid).to_numpy()
                ftr = sub["FTR"].astype(str).to_numpy()
                pts = np.where(
                    is_home,
                    np.where(ftr == "H", 3, np.where(ftr == "D", 1, 0)),
                    np.where(ftr == "A", 3, np.where(ftr == "D", 1, 0)),
                ).astype(float)
                gf = np.where(is_home, sub["FTHG"].to_numpy(), sub["FTAG"].to_numpy()).astype(float)
                ga = np.where(is_home, sub["FTAG"].to_numpy(), sub["FTHG"].to_numpy()).astype(float)
                for w in (5, 10, 20):
                    row[f"{prefix}FormPts_{w}"] = float(np.nanmean(pts[-w:]))
                    row[f"{prefix}FormGF_{w}"] = float(np.nanmean(gf[-w:]))
                    row[f"{prefix}FormGA_{w}"] = float(np.nanmean(ga[-w:]))
                    row[f"{prefix}FormGD_{w}"] = row[f"{prefix}FormGF_{w}"] - row[f"{prefix}FormGA_{w}"]
            # H2H last 5 (once, on Home pass)
            if prefix == "Home" and away_id is not None and int(away_id) >= 0:
                h2h = hist_df[
                    ((hist_df["HomeTeamId"] == tid) & (hist_df["AwayTeamId"] == away_id))
                    | ((hist_df["HomeTeamId"] == away_id) & (hist_df["AwayTeamId"] == tid))
                ].sort_values("Date").tail(5)
                hw = dw = aw = 0
                for _, m in h2h.iterrows():
                    if m["HomeTeamId"] == tid:
                        if m["FTR"] == "H": hw += 1
                        elif m["FTR"] == "D": dw += 1
                        else: aw += 1
                    else:
                        if m["FTR"] == "A": hw += 1
                        elif m["FTR"] == "D": dw += 1
                        else: aw += 1
                row["H2H_HomeWins_5"] = hw
                row["H2H_Draws_5"] = dw
                row["H2H_AwayWins_5"] = aw
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


# In-memory model caches (avoid reload + TF retracing every fixture)
_TORCH_CACHE: Dict[str, Any] = {}
_TF_CACHE: Dict[str, Any] = {}
_XGB_CACHE: Dict[str, Any] = {}
_LGBM_CACHE: Dict[str, Any] = {}
_CAT_CACHE: Dict[str, Any] = {}
_SK_CACHE: Dict[str, Any] = {}
_HIST_CACHE: Dict[str, Any] = {}  # parquet history loaded once per process


def _predict_torch(path, X):
    import torch
    import torch.nn as nn
    key = str(path)
    entry = _TORCH_CACHE.get(key)
    if entry is None:
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
        entry = (model, task)
        _TORCH_CACHE[key] = entry
    model, task = entry
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        if task == "multiclass":
            return torch.softmax(logits, dim=1).numpy()[0]
        return np.array([1 - float(torch.sigmoid(logits)[0, 0]), float(torch.sigmoid(logits)[0, 0])])


def _predict_tf(path, X):
    import tensorflow as tf
    # Hard CPU — never touch CUDA on Actions
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    key = str(path)
    model = _TF_CACHE.get(key)
    if model is None:
        model = tf.keras.models.load_model(path)
        # warm once to compile graph
        try:
            _ = model(tf.constant(X, dtype=tf.float32), training=False)
        except Exception:
            pass
        _TF_CACHE[key] = model
    # __call__ avoids Keras predict() retracing spam
    pred = model(tf.constant(X, dtype=tf.float32), training=False)
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
            key = paths["xgboost"]
            m = _XGB_CACHE.get(key)
            if m is None:
                m = xgb.Booster()
                m.load_model(key)
                _XGB_CACHE[key] = m
            p = np.asarray(m.predict(xgb.DMatrix(X_scaled)))
            preds.append(np.array([1 - p[0], p[0]]) if p.ndim == 1 else p[0])
            names.append("xgb")
        except Exception as e:
            print(f"  [warn] xgb {target_key}: {e}")

    if "lightgbm" in paths and Path(paths["lightgbm"]).exists():
        try:
            import lightgbm as lgb
            key = paths["lightgbm"]
            m = _LGBM_CACHE.get(key)
            if m is None:
                m = lgb.Booster(model_file=key)
                _LGBM_CACHE[key] = m
            p = np.asarray(m.predict(X_scaled))
            preds.append(np.array([1 - p[0], p[0]]) if p.ndim == 1 else p[0])
            names.append("lgbm")
        except Exception as e:
            print(f"  [warn] lgbm {target_key}: {e}")

    if "catboost" in paths and Path(paths["catboost"]).exists():
        try:
            from catboost import CatBoostClassifier
            key = paths["catboost"]
            m = _CAT_CACHE.get(key)
            if m is None:
                m = CatBoostClassifier()
                m.load_model(key)
                _CAT_CACHE[key] = m
            preds.append(m.predict_proba(X_scaled)[0])
            names.append("cat")
        except Exception as e:
            print(f"  [warn] cat {target_key}: {e}")

    if "adaboost" in paths and Path(paths["adaboost"]).exists():
        try:
            key = paths["adaboost"]
            m = _SK_CACHE.get(key)
            if m is None:
                with open(key, "rb") as f:
                    m = pickle.load(f)
                _SK_CACHE[key] = m
            p = m.predict_proba(X_scaled)[0]
            if np.isfinite(p).all():
                preds.append(p)
                names.append("ada")
        except Exception as e:
            print(f"  [warn] ada {target_key}: {e}")

    if "random_forest" in paths and Path(paths["random_forest"]).exists():
        try:
            key = paths["random_forest"]
            m = _SK_CACHE.get(key)
            if m is None:
                with open(key, "rb") as f:
                    m = pickle.load(f)
                _SK_CACHE[key] = m
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
        le_y = None
        try:
            feature_names, scaler, le_div, le_y = load_preprocessors(run_dir, target_key)
        except Exception as e:
            if not os.environ.get("SIM_QUIET", "").strip():
                print(f"  [warn] preproc {target_key} [{label}]: {e}")
        X = build_live_features(home_id, away_id, div_code, oh, od, oa, hist_df, feature_names, le_div)
        Xs = X.copy()
        n_scaled = min(Xs.shape[1] - 3, scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else Xs.shape[1] - 3)
        Xs[:, :n_scaled] = scaler.transform(X[:, :n_scaled])
        Xs = np.nan_to_num(Xs, nan=0.0)
        p, backends = predict_backends(target_key, Xs, targets)
        if p is not None:
            p = np.asarray(p, float).ravel()
            if target_key in ("ft_result", "ht_result") and len(p) >= 3:
                p = align_hda(p, le_y)
            preds.append(p)
            all_backends.extend(backends)
            if not os.environ.get("SIM_QUIET", "").strip():
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
    """Light fixed blend only — models carry the signal; no anti-model patches."""
    p_model = np.asarray(p_model, float)
    p_market = np.asarray(p_market, float)
    n = max(len(p_model), len(p_market))
    a = np.pad(p_model, (0, max(0, n - len(p_model))))[:n]
    b = np.pad(p_market, (0, max(0, n - len(p_market))))[:n]
    a = np.clip(a, 1e-9, None); b = np.clip(b, 1e-9, None)
    a, b = a / a.sum(), b / b.sum()
    alpha = float(np.clip(alpha, 0.0, 0.45))  # never let market dominate the models
    out = (1 - alpha) * a + alpha * b
    return out / out.sum()


def agreement_pct(backends_count: int, total_possible: int = 7) -> str:
    if backends_count <= 0:
        return "0%"
    return f"{100 * backends_count / max(total_possible, 1):.0f}%"


def simulate_match(p_ft, p_ht, p_over25, p_btts, p_ht_over15, n, seed):
    """
    Industrial Monte Carlo score engine.

    Primary path: discrete Dixon–Coles-adjusted Poisson grid reweighted to match
    target FT / O2.5 / BTTS probabilities (iterative proportional fitting style),
    then sample n scorelines. HT is always a subset of FT.

    Guarantees:
      - ht goals <= ft goals componentwise
      - exact sample count n
      - sim 1X2 / O2.5 / BTTS stay close to model targets (L1 typically < 0.08)
    """
    rng = np.random.default_rng(seed)
    p_ft = np.asarray(p_ft, float).ravel()
    p_ft = np.clip(p_ft, 1e-9, None)
    p_ft = p_ft / p_ft.sum()
    if p_ht is None or len(np.asarray(p_ht).ravel()) < 3:
        p_ht = np.array([0.32, 0.36, 0.32])
    else:
        p_ht = np.asarray(p_ht, float).ravel()[:3]
        p_ht = np.clip(p_ht, 1e-9, None)
        p_ht = p_ht / p_ht.sum()
    p_over25 = float(np.clip(p_over25, 0.05, 0.95))
    p_btts = float(np.clip(p_btts, 0.05, 0.95))
    p_ht_over15 = float(np.clip(p_ht_over15 if p_ht_over15 is not None else 0.35, 0.05, 0.95))

    tot = float(np.clip(2.15 + 1.35 * (p_over25 - 0.5) * 2, 1.6, 3.4))
    home_share = float(np.clip(0.38 + 0.28 * (p_ft[0] - p_ft[2]), 0.28, 0.72))
    lam_h = tot * home_share
    lam_a = tot * (1.0 - home_share)
    tau = 0.08  # Dixon–Coles low-score correlation

    from math import exp, factorial

    def _pois(k, lam):
        return exp(-lam) * (lam ** k) / factorial(int(k))

    def _dc(h, a):
        if h == 0 and a == 0:
            return max(1.0 - lam_h * lam_a * tau, 0.05)
        if h == 0 and a == 1:
            return 1.0 + lam_h * tau
        if h == 1 and a == 0:
            return 1.0 + lam_a * tau
        if h == 1 and a == 1:
            return max(1.0 - tau, 0.05)
        return 1.0

    max_g = 6
    gh, ga = np.meshgrid(np.arange(0, max_g + 1), np.arange(0, max_g + 1), indexing="ij")
    gh = gh.ravel().astype(int)
    ga = ga.ravel().astype(int)
    w = np.array([_pois(h, lam_h) * _pois(a, lam_a) * _dc(h, a) for h, a in zip(gh, ga)], dtype=float)
    w = np.clip(w, 1e-18, None)
    w /= w.sum()

    # Iterative reweight to match market targets (IPF-lite, 8 passes)
    for _ in range(8):
        # FT outcome
        out = np.where(gh > ga, 0, np.where(gh < ga, 2, 1))
        for k in range(3):
            mask = out == k
            cur = w[mask].sum()
            if cur > 0:
                w[mask] *= p_ft[k] / cur
        w /= w.sum()
        # O2.5
        over = (gh + ga) > 2.5
        cur_o = w[over].sum()
        if 0 < cur_o < 1:
            w[over] *= p_over25 / cur_o
            w[~over] *= (1 - p_over25) / max(w[~over].sum(), 1e-18)
            w /= w.sum()
        # BTTS
        btts = (gh > 0) & (ga > 0)
        cur_b = w[btts].sum()
        if 0 < cur_b < 1:
            w[btts] *= p_btts / cur_b
            w[~btts] *= (1 - p_btts) / max(w[~btts].sum(), 1e-18)
            w /= w.sum()

    idx = rng.choice(len(w), size=n, p=w)
    scores = np.column_stack([gh[idx], ga[idx]]).astype(int)

    # HT: subset of FT, shaped by p_ht + p_ht_over15
    ht_scores = np.zeros((n, 2), dtype=int)
    for i in range(n):
        hg, ag = int(scores[i, 0]), int(scores[i, 1])
        cands, weights = [], []
        for hh in range(0, hg + 1):
            for ah in range(0, ag + 1):
                o = 0 if hh > ah else (2 if hh < ah else 1)
                wt = float(p_ht[o])
                tot_ht = hh + ah
                wt *= (0.5 + p_ht_over15) if tot_ht > 1.5 else (1.2 - 0.4 * p_ht_over15)
                if (hg + ag) >= 3 and tot_ht == (hg + ag):
                    wt *= 0.35
                cands.append((hh, ah))
                weights.append(max(wt, 1e-6))
        weights = np.asarray(weights, float)
        weights /= weights.sum()
        pick = int(rng.choice(len(cands), p=weights))
        ht_scores[i] = cands[pick]

    top = Counter(map(tuple, map(tuple, scores))).most_common(12)
    top_ht = Counter(map(tuple, map(tuple, ht_scores))).most_common(8)

    def top_pct(counter_list, k=3):
        total = float(n) if n else 1.0
        return [
            {"score": f"{h}-{a}", "count": int(c), "pct": round(100.0 * c / total, 1)}
            for (h, a), c in counter_list[:k]
        ]

    tot_g = scores.sum(1).astype(float)
    diff = (scores[:, 0] - scores[:, 1]).astype(float)

    def line_probs(arr, line):
        return {
            "over": float((arr > line).mean()),
            "under": float((arr < line).mean()),
            "push": float((arr == line).mean()),
        }

    ft_sim = {
        "H": float((scores[:, 0] > scores[:, 1]).mean()),
        "D": float((scores[:, 0] == scores[:, 1]).mean()),
        "A": float((scores[:, 0] < scores[:, 1]).mean()),
    }
    return {
        "n": int(n),
        "ft_model": {"H": float(p_ft[0]), "D": float(p_ft[1]), "A": float(p_ft[2])},
        "ft_sim": ft_sim,
        "ht_model": {"H": float(p_ht[0]), "D": float(p_ht[1]), "A": float(p_ht[2])},
        "ht_sim": {
            "H": float((ht_scores[:, 0] > ht_scores[:, 1]).mean()),
            "D": float((ht_scores[:, 0] == ht_scores[:, 1]).mean()),
            "A": float((ht_scores[:, 0] < ht_scores[:, 1]).mean()),
        },
        "over25_model": float(p_over25),
        "over25_sim": float((tot_g > 2.5).mean()),
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
        "top3_ft": top_pct(top, 3),
        "top3_ht": top_pct(top_ht, 3),
        "top8_ft": top_pct(top, 8),
        "xg": {
            "home": float(scores[:, 0].mean()),
            "away": float(scores[:, 1].mean()),
            "total": float(tot_g.mean()),
            "lambda_home": float(lam_h),
            "lambda_away": float(lam_a),
        },
        "ht_xg": {
            "home": float(ht_scores[:, 0].mean()),
            "away": float(ht_scores[:, 1].mean()),
            "total": float(ht_scores.sum(1).mean()),
        },
        "score_consistency": {
            "ht_leq_ft": float(np.mean((ht_scores[:, 0] <= scores[:, 0]) & (ht_scores[:, 1] <= scores[:, 1]))),
            "ft_vs_model_l1": float(np.abs(np.array([
                ft_sim["H"] - p_ft[0],
                ft_sim["D"] - p_ft[1],
                ft_sim["A"] - p_ft[2],
            ])).sum()),
        },
        "clean_sheet": {
            "home": float((scores[:, 1] == 0).mean()),
            "away": float((scores[:, 0] == 0).mean()),
        },
        "win_to_nil": {
            "home": float(((scores[:, 0] > scores[:, 1]) & (scores[:, 1] == 0)).mean()),
            "away": float(((scores[:, 0] < scores[:, 1]) & (scores[:, 0] == 0)).mean()),
        },
        "goal_lines": {
            "0.5": line_probs(tot_g, 0.5),
            "1.5": line_probs(tot_g, 1.5),
            "2.5": line_probs(tot_g, 2.5),
            "3.5": line_probs(tot_g, 3.5),
            "4.5": line_probs(tot_g, 4.5),
        },
        "asian_handicap": {
            "home_-0.5": float((diff > 0.5).mean()),
            "home_-1.0": {"cover": float((diff > 1.0).mean()), "push": float((diff == 1.0).mean())},
            "home_+0.5": float((diff > -0.5).mean()),
            "home_+1.0": {"cover": float((diff > -1.0).mean()), "push": float((diff == -1.0).mean())},
        },
        "score_matrix_margin": {
            "home_1_goal": float((diff == 1).mean()),
            "home_2_plus": float((diff >= 2).mean()),
            "away_1_goal": float((diff == -1).mean()),
            "away_2_plus": float((diff <= -2).mean()),
            "draw": float((diff == 0).mean()),
        },
    }



def verdict(model_p, threshold=0.52):
    return "YES" if model_p >= threshold else "NO"


def build_table_rows(report, backends_map, home, away):
    """Industrial board: Market | Selection | Model% | Sim% | Agree | Verdict."""
    rows = []

    def add(section, selection, model_p, sim_p, backends, thr=0.52):
        model_p = float(model_p) if model_p is not None else 0.0
        sim_p = float(sim_p) if sim_p is not None else 0.0
        rows.append({
            "Section": section,
            "Selection": selection,
            "Model%": round(model_p * 100, 1),
            "Sim%": round(sim_p * 100, 1),
            "Agree": f"{len(backends)} eng" if backends else "-",
            "Verdict": verdict(model_p, thr),
            "Edge": round((model_p - sim_p) * 100, 1),
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

    # Extended lines from Monte Carlo board (sim-only model column mirrors sim)
    gl = report.get("goal_lines") or {}
    for line, key in [("Over 1.5", "1.5"), ("Over 3.5", "3.5")]:
        cell = gl.get(key) or {}
        ov = float(cell.get("over", 0.0))
        add("FT O/U+", line, ov, ov, b_o25, 0.55)
    cs = report.get("clean_sheet") or {}
    add("CS", f"{home} CS", float(cs.get("home", 0.0)), float(cs.get("home", 0.0)), b_ft, 0.40)
    add("CS", f"{away} CS", float(cs.get("away", 0.0)), float(cs.get("away", 0.0)), b_ft, 0.40)
    wtn = report.get("win_to_nil") or {}
    add("WTN", f"{home} WTN", float(wtn.get("home", 0.0)), float(wtn.get("home", 0.0)), b_ft, 0.35)
    add("WTN", f"{away} WTN", float(wtn.get("away", 0.0)), float(wtn.get("away", 0.0)), b_ft, 0.35)

    # Correct scores are shown in the distribution panel only — do not spam
    # the markets table with NO verdicts on low-probability scorelines.

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
    """Strict tip selection — avoid flooding SECURED with weak double-chance."""
    def is_dc(r):
        return "DC" in str(r.get("Section", ""))

    def score(r):
        agree_sim = 100 - abs(r["Model%"] - r["Sim%"])
        dc_pen = -18 if is_dc(r) else 0
        return r["Model%"] + 0.2 * agree_sim + dc_pen

    pool = [r for r in rows if r["Verdict"] == "YES" and r["Model%"] >= 58]
    non_dc = [r for r in pool if not is_dc(r)]
    candidates = sorted(non_dc or pool, key=score, reverse=True)

    print("\n" + "=" * 78)
    print("  LOCKED SECURED TIP")
    print("=" * 78)
    if not candidates:
        fallback = max(rows, key=lambda r: r["Model%"])
        print(f"  No high-confidence YES. Soft lean: {fallback['Section']} -> {fallback['Selection']}")
        print(f"  Model {fallback['Model%']:.1f}%  Sim {fallback['Sim%']:.1f}%  -> NOT SECURED")
        tip = {
            "status": "NO LOCK",
            "selection": fallback["Selection"],
            "section": fallback["Section"],
            "model": fallback["Model%"],
            "sim": fallback["Sim%"],
        }
    else:
        top = candidates[0]
        if is_dc(top):
            locked = top["Model%"] >= 75 and abs(top["Model%"] - top["Sim%"]) <= 10
        else:
            locked = top["Model%"] >= 62 and abs(top["Model%"] - top["Sim%"]) <= 10
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
        if len(candidates) > 1:
            print(
                f"\n  Secondary  : {candidates[1]['Section']} -> {candidates[1]['Selection']} "
                f"({candidates[1]['Model%']:.1f}%)"
            )
        tip = {
            "status": status,
            "section": top["Section"],
            "selection": top["Selection"],
            "model": top["Model%"],
            "sim": top["Sim%"],
            "agree": top["Agree"],
            "verdict": "HARD YES" if locked else "YES LEAN",
        }
    t3 = report.get("top3_ft") or []
    if t3:
        print("\n  Top-3 FT correct scores:")
        for i, row in enumerate(t3, 1):
            print(f"    {i}. {row['score']:>5}  {row['pct']:5.1f}%  (n={row['count']})")
    else:
        print(f"\n  Top FT scores : {', '.join(f'{s}({c})' for s,c in report.get('top_ft', [])[:3])}")
    t3h = report.get("top3_ht") or []
    if t3h:
        print("  Top-3 HT scores (consistent with FT):")
        for i, row in enumerate(t3h, 1):
            print(f"    {i}. {row['score']:>5}  {row['pct']:5.1f}%  (n={row['count']})")
    else:
        print(f"  Top HT scores : {', '.join(f'{s}({c})' for s,c in report.get('top_ht', [])[:3])}")
    print(f"  Exp goals FT  : H {report['xg']['home']:.2f}  A {report['xg']['away']:.2f}  T {report['xg']['total']:.2f}")
    if report.get("ht_xg"):
        print(f"  Exp goals HT  : H {report['ht_xg']['home']:.2f}  A {report['ht_xg']['away']:.2f}  T {report['ht_xg']['total']:.2f}")
    cons = (report.get("score_consistency") or {}).get("ht_leq_ft")
    if cons is not None:
        print(f"  HT⊆FT check   : {100*cons:.1f}% of sims")
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

    team2id = team_mapping.load_team2id()
    if not team2id and (MAP_DIR / "team2id.json").exists():
        team2id = load_json(MAP_DIR / "team2id.json")
    aliases = team_mapping.load_aliases()

    # Accept Div code or league name
    league_in = match_cfg.get("league") or match_cfg.get("div") or "E0"
    div_code, league_name = resolve_league(league_in)

    home_canon = normalize_team(match_cfg["home_team"], aliases)
    away_canon = normalize_team(match_cfg["away_team"], aliases)
    # case-insensitive id lookup
    t2_ci = {k.lower(): v for k, v in team2id.items()}
    home_id = team2id.get(home_canon, t2_ci.get(home_canon.lower()))
    away_id = team2id.get(away_canon, t2_ci.get(away_canon.lower()))
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
            feature_names, scaler, le_div, _le_y = load_preprocessors(run_dir)
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
            cache_key = str(PARQUET_PATH.resolve())
            hist_df = _HIST_CACHE.get(cache_key)
            if hist_df is None:
                log("Loading history parquet once…")
                hist_df = pd.read_parquet(
                    PARQUET_PATH,
                    columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Div"],
                )
                hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors="coerce")
                # vectorized-ish: map via team_mapping once
                tmap = {}
                def _canon(x):
                    s = str(x) if x is not None else ""
                    if s not in tmap:
                        tmap[s] = normalize_team(s, aliases)
                    return tmap[s]
                hist_df["HomeTeamCanon"] = hist_df["HomeTeam"].map(_canon)
                hist_df["AwayTeamCanon"] = hist_df["AwayTeam"].map(_canon)
                hist_df["HomeTeamId"] = hist_df["HomeTeamCanon"].map(team2id)
                hist_df["AwayTeamId"] = hist_df["AwayTeamCanon"].map(team2id)
                hist_df["HomePoints"] = hist_df["FTR"].map({"H": 3, "D": 1, "A": 0})
                hist_df["AwayPoints"] = hist_df["FTR"].map({"H": 0, "D": 1, "A": 3})
                _HIST_CACHE[cache_key] = hist_df
                log(f"History cached: {len(hist_df):,} rows")
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
        p_ht = multi_prob(raw_ht, 3)  # pure model HT — no market patch
        p_over = bin_prob(raw_over)
        p_btts = bin_prob(raw_btts)
        p_hto = bin_prob(raw_hto) if raw_hto is not None else float(0.25 + 0.25 * p_over)
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
