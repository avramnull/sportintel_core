#!/usr/bin/env python3
"""
FOOTBALL DEEP TRAINING — MAX POWER
XGBoost + LightGBM + CatBoost + AdaBoost + RandomForest + PyTorch + TensorFlow
Edge-controlled capacity, rich features, per-team or global runs.
"""
from __future__ import annotations

import os, sys, json, pickle, traceback, warnings, gc
from pathlib import Path
import team_mapping
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =============================================================================
# CONFIG
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "master_football_data.parquet").exists() else Path.cwd()

CONFIG = {
    "parquet_path": str(ROOT / "master_football_data.parquet"),
    "out_dir": str(ROOT / "football_models"),

    "train_leagues": [
        "E0", "E1", "E2", "E3", "EC",
        "SP1", "SP2", "I1", "I2", "D1", "D2",
        "F1", "F2", "N1", "B1", "P1", "T1", "G1",
        "SC0", "SC1", "SC2", "SC3",
    ],

    # [] = GLOBAL only; non-empty = per-team models
    "focus_teams": [
        "Valencia",
        "Betis",
    ],

    "min_team_matches": 40,
    "val_fraction": 0.12,
    "seed": 42,
    "n_jobs": max(1, (os.cpu_count() or 2) - 1),

    "max_boost_rounds": 2000,
    "patience_overfit": 50,
    "overfit_gap_warn": 0.15,
    "capacity_growth_steps": 2,

    "use_xgboost": True,
    "use_lightgbm": True,
    "use_catboost": True,
    "use_adaboost": True,
    "use_random_forest": True,
    "use_pytorch": True,
    "use_tensorflow": True,

    # Neural net defaults
    "nn_epochs": 80,
    "nn_batch": 256,
    "nn_patience": 12,
    "nn_hidden": [128, 64, 32],
}


# --- env overrides (daily pipeline) ---
# FOCUS_TEAMS="Arsenal,Liverpool"  MAX_BOOST_ROUNDS=800  DAILY_LIGHT=1
_ft = os.environ.get("FOCUS_TEAMS", "").strip()
if _ft:
    CONFIG["focus_teams"] = [x.strip() for x in _ft.split(",") if x.strip()]
if os.environ.get("MIN_TEAM_MATCHES"):
    CONFIG["min_team_matches"] = int(os.environ["MIN_TEAM_MATCHES"])
if os.environ.get("MAX_BOOST_ROUNDS"):
    CONFIG["max_boost_rounds"] = int(os.environ["MAX_BOOST_ROUNDS"])
if os.environ.get("DAILY_LIGHT", "").strip() in ("1", "true", "yes"):
    # Faster daily train: skip neural nets, fewer growth steps
    CONFIG["use_pytorch"] = False
    CONFIG["use_tensorflow"] = False
    CONFIG["capacity_growth_steps"] = 1
    CONFIG["max_boost_rounds"] = min(CONFIG["max_boost_rounds"], 1200)
    CONFIG["patience_overfit"] = 40
    print("[train] DAILY_LIGHT mode: xgb/lgbm/cat/ada/rf only, reduced rounds")

LEAGUE_MAP = dict(team_mapping.LEAGUE_MAP)

TEAM_ALIASES = dict(team_mapping.TEAM_ALIASES)

TARGETS = {
    "ft_result": {"col": "FTR", "type": "multiclass", "classes": ["H", "D", "A"]},
    "ht_result": {"col": "HTR", "type": "multiclass", "classes": ["H", "D", "A"]},
    "over25":    {"col": "Over2_5", "type": "binary"},
    "btts":      {"col": "BTTS", "type": "binary"},
    "dc_1x":     {"col": "DC_1X", "type": "binary"},
    "dc_x2":     {"col": "DC_X2", "type": "binary"},
    "dc_12":     {"col": "DC_12", "type": "binary"},
    "ht_over15": {"col": "HT_Over1_5", "type": "binary"},
}

FEATURE_NUM = [
    "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A",
    "LogOddsH", "LogOddsD", "LogOddsA", "ImpH", "ImpD", "ImpA",
    "OddsMargin", "HomeOddsEdge", "AwayOddsEdge",
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "HomeFormGF_5", "HomeFormGA_5", "HomeFormGD_5",
    "AwayFormPts_5", "AwayFormGF_5", "AwayFormGA_5", "AwayFormGD_5",
    "HomeFormPts_10", "HomeFormGF_10", "HomeFormGA_10", "HomeFormGD_10",
    "AwayFormPts_10", "AwayFormGF_10", "AwayFormGA_10", "AwayFormGD_10",
    "HomeFormPts_20", "AwayFormPts_20",
    "H2H_HomeWins_5", "H2H_Draws_5", "H2H_AwayWins_5",
]


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    try:
        with open(Path(CONFIG["out_dir"]) / "train.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def normalize_team(name: Any) -> str:
    if name is None or (isinstance(name, float) and str(name) == "nan"):
        return "UNKNOWN"
    try:
        import pandas as _pd
        if _pd.isna(name):
            return "UNKNOWN"
    except Exception:
        pass
    canon, _ = team_mapping.normalize_team(str(name), TEAM_ALIASES, None)
    return canon or "UNKNOWN"

def slug(name: str) -> str:
    return team_mapping.slug(name)


def capacity_params(level: int, base: dict) -> dict:
    p = dict(base)
    p["max_depth"] = min(base.get("max_depth", 6) + level * 2, 14)
    if "num_leaves" in p:
        p["num_leaves"] = min(base["num_leaves"] + level * 32, 255)
    if "iterations" in p:
        p["iterations"] = min(base["iterations"] + level * 400, CONFIG["max_boost_rounds"])
    if "depth" in p:
        p["depth"] = min(base["depth"] + level, 10)
    if "n_estimators" in p:
        p["n_estimators"] = min(base["n_estimators"] + level * 100, 800)
    return p


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for c in ("FTHG", "FTAG", "HTHG", "HTAG", "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for a, b in [("AvgH", "B365H"), ("AvgD", "B365D"), ("AvgA", "B365A")]:
        if a not in df.columns and b in df.columns:
            df[a] = df[b]
        elif a in df.columns and b in df.columns:
            df[a] = df[a].fillna(df[b])

    df["TotalGoals"] = df.get("FTHG", np.nan) + df.get("FTAG", np.nan)
    df["Over2_5"] = (df["TotalGoals"] > 2.5).astype(float)
    df["BTTS"] = ((df.get("FTHG", 0) > 0) & (df.get("FTAG", 0) > 0)).astype(float)
    df["DC_1X"] = df["FTR"].isin(["H", "D"]).astype(float)
    df["DC_X2"] = df["FTR"].isin(["D", "A"]).astype(float)
    df["DC_12"] = df["FTR"].isin(["H", "A"]).astype(float)
    ht_goals = pd.to_numeric(df.get("HTHG"), errors="coerce").fillna(0) + pd.to_numeric(df.get("HTAG"), errors="coerce").fillna(0)
    df["HT_Over1_5"] = (ht_goals > 1.5).astype(float)
    df["HomePoints"] = df["FTR"].map({"H": 3, "D": 1, "A": 0})
    df["AwayPoints"] = df["FTR"].map({"H": 0, "D": 1, "A": 3})

    for side, col in [("H", "AvgH"), ("D", "AvgD"), ("A", "AvgA")]:
        if col in df.columns:
            v = df[col].clip(lower=1.01)
            df[f"LogOdds{side}"] = np.log(v)
            df[f"Imp{side}"] = 1.0 / v

    if all(c in df.columns for c in ("AvgH", "AvgD", "AvgA")):
        imp_sum = (1 / df["AvgH"].clip(1.01)) + (1 / df["AvgD"].clip(1.01)) + (1 / df["AvgA"].clip(1.01))
        df["OddsMargin"] = imp_sum - 1.0
        df["HomeOddsEdge"] = (1 / df["AvgH"].clip(1.01)) / imp_sum
        df["AwayOddsEdge"] = (1 / df["AvgA"].clip(1.01)) / imp_sum
    else:
        df["OddsMargin"] = 0.0
        df["HomeOddsEdge"] = 0.33
        df["AwayOddsEdge"] = 0.33

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    def add_roll(frame, w):
        g = frame.groupby("HomeTeamId", group_keys=False)
        frame[f"HomeFormPts_{w}"] = g["HomePoints"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"HomeFormGF_{w}"] = g["FTHG"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"HomeFormGA_{w}"] = g["FTAG"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"HomeFormGD_{w}"] = frame[f"HomeFormGF_{w}"] - frame[f"HomeFormGA_{w}"]
        g2 = frame.groupby("AwayTeamId", group_keys=False)
        frame[f"AwayFormPts_{w}"] = g2["AwayPoints"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"AwayFormGF_{w}"] = g2["FTAG"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"AwayFormGA_{w}"] = g2["FTHG"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"AwayFormGD_{w}"] = frame[f"AwayFormGF_{w}"] - frame[f"AwayFormGA_{w}"]
        return frame

    for w in (5, 10, 20):
        df = add_roll(df, w)

    df["PairKey"] = df.apply(
        lambda r: tuple(sorted([int(r["HomeTeamId"]), int(r["AwayTeamId"])])), axis=1
    )
    hist = defaultdict(lambda: deque(maxlen=5))
    hw, dr, aw = [], [], []
    for _, r in df.iterrows():
        past = list(hist[r["PairKey"]])
        hw.append(sum(1 for x in past if x == "H"))
        dr.append(sum(1 for x in past if x == "D"))
        aw.append(sum(1 for x in past if x == "A"))
        hist[r["PairKey"]].append(r["FTR"] if pd.notna(r["FTR"]) else "D")
    df["H2H_HomeWins_5"] = hw
    df["H2H_Draws_5"] = dr
    df["H2H_AwayWins_5"] = aw
    return df


def prepare_xy(df, target_key):
    cfg = TARGETS[target_key]
    sub = df.dropna(subset=[cfg["col"]]).copy()
    if len(sub) < 25:
        raise ValueError(f"Too few rows for {target_key}: {len(sub)}")

    num_cols = [c for c in FEATURE_NUM if c in sub.columns]
    X_num = sub[num_cols].apply(pd.to_numeric, errors="coerce")
    valid = [c for c in num_cols if X_num[c].notna().any()]
    X_num = X_num[valid].fillna(X_num[valid].median(numeric_only=True)).fillna(0.0)

    X_teams = sub[["HomeTeamId", "AwayTeamId"]].astype(int).values
    le_div = LabelEncoder()
    div_enc = le_div.fit_transform(sub["Div"].fillna("UNK").astype(str)).reshape(-1, 1)
    X = np.nan_to_num(np.hstack([X_num.values.astype(np.float64), X_teams, div_enc]), nan=0.0)
    feat_names = valid + ["HomeTeamId", "AwayTeamId", "DivEnc"]

    y_raw = sub[cfg["col"]]
    le_y = None
    if cfg["type"] == "multiclass":
        classes = cfg.get("classes")
        if classes:
            mask = y_raw.isin(classes).values
            X, y_raw = X[mask], y_raw[mask]
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw.astype(str).values)
    else:
        y = (pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).values > 0).astype(int)

    ok = np.isfinite(y) if cfg["type"] == "multiclass" else (np.isfinite(y) & np.isin(y, [0, 1]))
    X, y = X[ok], y[ok]
    if len(np.unique(y)) < 2:
        raise ValueError(f"{target_key}: need >=2 classes")
    return X, y, feat_names, le_y, le_div


def _safe_ll(y, p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(log_loss(y, p))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    p = p / p.sum(axis=1, keepdims=True)
    return float(log_loss(y, p))


# ---- Boosting backends (same edge-control pattern) ----
def train_xgboost(Xtr, ytr, Xva, yva, task, n_classes, out_path):
    import xgboost as xgb
    base = dict(
        max_depth=8, learning_rate=0.03, subsample=0.85, colsample_bytree=0.80,
        min_child_weight=4, reg_lambda=1.5, reg_alpha=0.2, gamma=0.1,
        tree_method="hist", random_state=CONFIG["seed"], n_jobs=CONFIG["n_jobs"], verbosity=0,
    )
    if task == "multiclass":
        base.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    else:
        base.update(objective="binary:logistic", eval_metric="logloss")
    best_model, best_val = None, float("inf")
    for level in range(CONFIG["capacity_growth_steps"] + 1):
        params = capacity_params(level, base)
        params["reg_lambda"] = base["reg_lambda"] + level * 0.5
        dtr, dva = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xva, label=yva)
        model = xgb.train(
            params, dtr, num_boost_round=CONFIG["max_boost_rounds"],
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=CONFIG["patience_overfit"], verbose_eval=False,
        )
        tr_ll, va_ll = _safe_ll(ytr, model.predict(dtr)), _safe_ll(yva, model.predict(dva))
        gap = va_ll - tr_ll
        log(f"      XGB L{level}  tr={tr_ll:.4f} va={va_ll:.4f} gap={gap:.4f}")
        if va_ll < best_val:
            best_val, best_model = va_ll, model
        if gap > CONFIG["overfit_gap_warn"]:
            break
        if level < CONFIG["capacity_growth_steps"] and gap < 0.06 and tr_ll > 0.50:
            continue
        break
    best_model.save_model(str(out_path))
    return True


def train_lightgbm(Xtr, ytr, Xva, yva, task, n_classes, out_path):
    import lightgbm as lgb
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float64)
    Xva = np.ascontiguousarray(Xva, dtype=np.float64)
    ytr, yva = np.ascontiguousarray(ytr), np.ascontiguousarray(yva)
    base = dict(
        learning_rate=0.03, num_leaves=96, max_depth=10, min_child_samples=20,
        subsample=0.85, colsample_bytree=0.80, reg_lambda=1.5, reg_alpha=0.2,
        random_state=CONFIG["seed"], n_jobs=CONFIG["n_jobs"], verbose=-1, force_col_wise=True,
    )
    if task == "multiclass":
        base.update(objective="multiclass", num_class=int(n_classes), metric="multi_logloss")
    else:
        base.update(objective="binary", metric="binary_logloss")
    best_model, best_val = None, float("inf")
    for level in range(CONFIG["capacity_growth_steps"] + 1):
        params = capacity_params(level, base)
        dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
        model = lgb.train(
            params, dtr, num_boost_round=CONFIG["max_boost_rounds"],
            valid_sets=[dtr, dva], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(CONFIG["patience_overfit"], verbose=False), lgb.log_evaluation(0)],
        )
        tr_ll, va_ll = _safe_ll(ytr, model.predict(Xtr)), _safe_ll(yva, model.predict(Xva))
        gap = va_ll - tr_ll
        log(f"      LGBM L{level} tr={tr_ll:.4f} va={va_ll:.4f} gap={gap:.4f}")
        if va_ll < best_val:
            best_val, best_model = va_ll, model
        if gap > CONFIG["overfit_gap_warn"]:
            break
        if level < CONFIG["capacity_growth_steps"] and gap < 0.06 and tr_ll > 0.50:
            continue
        break
    best_model.save_model(str(out_path))
    return True


def train_catboost(Xtr, ytr, Xva, yva, task, n_classes, out_path):
    from catboost import CatBoostClassifier, Pool
    base = dict(
        depth=8, learning_rate=0.03, iterations=1800, l2_leaf_reg=4.0,
        random_seed=CONFIG["seed"], od_type="Iter", od_wait=CONFIG["patience_overfit"],
        verbose=False, thread_count=CONFIG["n_jobs"], allow_writing_files=False,
        loss_function="MultiClass" if task == "multiclass" else "Logloss",
    )
    best_model, best_val = None, float("inf")
    for level in range(CONFIG["capacity_growth_steps"] + 1):
        params = capacity_params(level, base)
        keys = ("depth", "learning_rate", "iterations", "l2_leaf_reg", "random_seed",
                "od_type", "od_wait", "verbose", "thread_count", "allow_writing_files", "loss_function")
        model = CatBoostClassifier(**{k: params[k] for k in keys if k in params})
        model.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True, verbose=False)
        tr_ll = _safe_ll(ytr, model.predict_proba(Xtr))
        va_ll = _safe_ll(yva, model.predict_proba(Xva))
        gap = va_ll - tr_ll
        log(f"      CAT L{level}  tr={tr_ll:.4f} va={va_ll:.4f} gap={gap:.4f}")
        if va_ll < best_val:
            best_val, best_model = va_ll, model
        if gap > CONFIG["overfit_gap_warn"]:
            break
        if level < CONFIG["capacity_growth_steps"] and gap < 0.06 and tr_ll > 0.50:
            continue
        break
    best_model.save_model(str(out_path))
    return True


def train_adaboost(Xtr, ytr, Xva, yva, task, out_path):
    Xtr = np.nan_to_num(np.asarray(Xtr, dtype=np.float64), nan=0.0)
    Xva = np.nan_to_num(np.asarray(Xva, dtype=np.float64), nan=0.0)
    ytr, yva = np.asarray(ytr).astype(int), np.asarray(yva).astype(int)
    best_model, best_val = None, float("inf")
    for level in range(CONFIG["capacity_growth_steps"] + 1):
        depth = min(3 + level, 6)
        n_est = min(150 + level * 50, 350)
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=15 + level * 5, random_state=CONFIG["seed"])
        # sklearn API: estimator (1.2+), base_estimator (older); algorithm removed in 1.6+
        model = None
        for kwargs in (
            dict(estimator=tree, n_estimators=n_est, learning_rate=0.4, random_state=CONFIG["seed"]),
            dict(estimator=tree, n_estimators=n_est, learning_rate=0.4, algorithm="SAMME", random_state=CONFIG["seed"]),
            dict(base_estimator=tree, n_estimators=n_est, learning_rate=0.4, algorithm="SAMME", random_state=CONFIG["seed"]),
            dict(base_estimator=tree, n_estimators=n_est, learning_rate=0.4, random_state=CONFIG["seed"]),
        ):
            try:
                model = AdaBoostClassifier(**kwargs)
                break
            except TypeError:
                continue
        if model is None:
            raise TypeError("AdaBoostClassifier: no compatible kwargs for this sklearn version")
        model.fit(Xtr, ytr)
        tr_p, va_p = model.predict_proba(Xtr), model.predict_proba(Xva)
        if not (np.isfinite(tr_p).all() and np.isfinite(va_p).all()):
            break
        tr_ll, va_ll = _safe_ll(ytr, tr_p), _safe_ll(yva, va_p)
        gap = va_ll - tr_ll
        log(f"      ADA L{level}  tr={tr_ll:.4f} va={va_ll:.4f} gap={gap:.4f}")
        if va_ll < best_val:
            best_val, best_model = va_ll, model
        if gap > CONFIG["overfit_gap_warn"] + 0.05:
            break
        if level < CONFIG["capacity_growth_steps"] and gap < 0.08 and tr_ll > 0.55:
            continue
        break
    if best_model is None:
        raise RuntimeError("AdaBoost failed")
    with open(out_path, "wb") as f:
        pickle.dump(best_model, f)
    return True


def train_random_forest(Xtr, ytr, Xva, yva, task, out_path):
    Xtr = np.nan_to_num(np.asarray(Xtr, dtype=np.float64), nan=0.0)
    Xva = np.nan_to_num(np.asarray(Xva, dtype=np.float64), nan=0.0)
    ytr, yva = np.asarray(ytr).astype(int), np.asarray(yva).astype(int)
    best_model, best_val = None, float("inf")
    for level in range(CONFIG["capacity_growth_steps"] + 1):
        n_est = min(200 + level * 150, 700)
        depth = min(12 + level * 2, 20)
        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=4,
            max_features="sqrt", n_jobs=CONFIG["n_jobs"],
            random_state=CONFIG["seed"] + level, class_weight="balanced_subsample",
        )
        model.fit(Xtr, ytr)
        tr_ll = _safe_ll(ytr, model.predict_proba(Xtr))
        va_ll = _safe_ll(yva, model.predict_proba(Xva))
        gap = va_ll - tr_ll
        log(f"      RF  L{level}  tr={tr_ll:.4f} va={va_ll:.4f} gap={gap:.4f} trees={n_est}")
        if va_ll < best_val:
            best_val, best_model = va_ll, model
        if gap > CONFIG["overfit_gap_warn"] + 0.08:
            break
        if level < CONFIG["capacity_growth_steps"] and gap < 0.05 and tr_ll > 0.45:
            continue
        break
    with open(out_path, "wb") as f:
        pickle.dump(best_model, f)
    return True


def train_pytorch(Xtr, ytr, Xva, yva, task, n_classes, out_path):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    if task == "multiclass":
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        yva_t = torch.tensor(yva, dtype=torch.long)
        out_dim = n_classes
    else:
        ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        yva_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)
        out_dim = 1

    in_dim = Xtr.shape[1]
    hidden = CONFIG["nn_hidden"]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.25)]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    crit = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=CONFIG["nn_batch"], shuffle=True)
    best_state, best_val, patience = None, float("inf"), 0

    for epoch in range(CONFIG["nn_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(Xva_t.to(device))
            if task == "multiclass":
                v_prob = torch.softmax(v_logits, dim=1).cpu().numpy()
                va_ll = _safe_ll(yva, v_prob)
            else:
                v_prob = torch.sigmoid(v_logits).cpu().numpy().ravel()
                va_ll = _safe_ll(yva, v_prob)
        sched.step(va_ll)
        if va_ll < best_val - 1e-4:
            best_val = va_ll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG["nn_patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    log(f"      TORCH    va={best_val:.4f} epochs~{epoch+1}")
    # save full checkpoint for inference
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden": hidden,
        "out_dim": out_dim,
        "task": task,
        "n_classes": n_classes if task == "multiclass" else 2,
    }, out_path)
    return True


def train_tensorflow(Xtr, ytr, Xva, yva, task, n_classes, out_path):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(CONFIG["seed"])
    in_dim = Xtr.shape[1]
    hidden = CONFIG["nn_hidden"]

    inputs = keras.Input(shape=(in_dim,))
    x = inputs
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
    if task == "multiclass":
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    else:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        ytr = ytr.astype(np.float32)
        yva = yva.astype(np.float32)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=CONFIG["nn_patience"], restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5),
    ]
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=CONFIG["nn_epochs"], batch_size=CONFIG["nn_batch"],
        verbose=0, callbacks=cb,
    )
    va_loss = min(hist.history["val_loss"])
    log(f"      TF       va_loss={va_loss:.4f}")
    model.save(str(out_path))
    # meta alongside
    meta = {"task": task, "n_classes": int(n_classes) if task == "multiclass" else 2, "in_dim": in_dim}
    with open(str(out_path) + ".meta.json", "w") as f:
        json.dump(meta, f)
    return True


def train_one_target(train_df, val_df, target_key, models_dir, preproc_dir):
    cfg = TARGETS[target_key]
    task = cfg["type"]
    saved = {}

    Xtr, ytr, feat_names, le_y, le_div = prepare_xy(train_df, target_key)
    if task == "multiclass" and le_y is not None:
        val_sub = val_df[val_df[cfg["col"]].astype(str).isin(set(le_y.classes_))].copy()
        Xva, yva_raw, _, _, _ = prepare_xy(val_sub, target_key)
        y_series = val_sub.dropna(subset=[cfg["col"]])[cfg["col"]].astype(str)
        y_series = y_series[y_series.isin(le_y.classes_)]
        yva = le_y.transform(y_series.values)
        m = min(len(Xva), len(yva))
        Xva, yva = Xva[:m], yva[:m]
    else:
        Xva, yva, _, _, _ = prepare_xy(val_df, target_key)

    n_classes = int(len(np.unique(ytr))) if task == "multiclass" else 2
    log(f"    {target_key}: n_train={len(ytr)} n_val={len(yva)} classes={n_classes} feats={Xtr.shape[1]}")

    scaler = StandardScaler()
    Xtr_s, Xva_s = Xtr.copy(), Xva.copy()
    Xtr_s[:, :-3] = scaler.fit_transform(Xtr[:, :-3])
    Xva_s[:, :-3] = scaler.transform(Xva[:, :-3])
    Xtr_s = np.nan_to_num(Xtr_s, nan=0.0)
    Xva_s = np.nan_to_num(Xva_s, nan=0.0)

    preproc_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(preproc_dir / f"scaler_{target_key}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    if le_y is not None:
        with open(preproc_dir / f"le_y_{target_key}.pkl", "wb") as f:
            pickle.dump(le_y, f)
    with open(preproc_dir / f"le_div_{target_key}.pkl", "wb") as f:
        pickle.dump(le_div, f)
    with open(preproc_dir / f"features_{target_key}.json", "w") as f:
        json.dump(feat_names, f)

    backends = [
        ("use_xgboost", "xgboost", lambda p: train_xgboost(Xtr_s, ytr, Xva_s, yva, task, n_classes, p), f"xgb_{target_key}.json"),
        ("use_lightgbm", "lightgbm", lambda p: train_lightgbm(Xtr_s, ytr, Xva_s, yva, task, n_classes, p), f"lgbm_{target_key}.txt"),
        ("use_catboost", "catboost", lambda p: train_catboost(Xtr_s, ytr, Xva_s, yva, task, n_classes, p), f"cat_{target_key}.cbm"),
        ("use_adaboost", "adaboost", lambda p: train_adaboost(Xtr_s, ytr, Xva_s, yva, task, p), f"ada_{target_key}.pkl"),
        ("use_random_forest", "random_forest", lambda p: train_random_forest(Xtr_s, ytr, Xva_s, yva, task, p), f"rf_{target_key}.pkl"),
        ("use_pytorch", "pytorch", lambda p: train_pytorch(Xtr_s, ytr, Xva_s, yva, task, n_classes, p), f"torch_{target_key}.pt"),
        ("use_tensorflow", "tensorflow", lambda p: train_tensorflow(Xtr_s, ytr, Xva_s, yva, task, n_classes, p), f"tf_{target_key}.keras"),
    ]
    for flag, name, fn, fname in backends:
        if not CONFIG.get(flag):
            continue
        try:
            p = models_dir / fname
            fn(p)
            saved[name] = str(p)
        except Exception as e:
            log(f"      {name.upper()} FAILED: {e}")
            traceback.print_exc()

    return saved


def main():
    out_root = Path(CONFIG["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        open(out_root / "train.log", "w").close()
    except Exception:
        pass

    log("=" * 64)
    log("FOOTBALL TRAINING — MAX POWER (RF + Torch + TF + Boosters)")
    log("=" * 64)

    pq_path = Path(CONFIG["parquet_path"])
    if not pq_path.exists():
        log(f"ERROR: parquet not found: {pq_path}")
        sys.exit(1)

    map_dir = out_root / "mappings"
    map_dir.mkdir(parents=True, exist_ok=True)

    log(f"Parquet : {pq_path}")
    log(f"Out dir : {out_root}")
    log(f"Focus   : {CONFIG['focus_teams'] or '(GLOBAL)'}")

    df = pd.read_parquet(pq_path)
    log(f"Loaded  : {df.shape}")
    if CONFIG["train_leagues"]:
        df = df[df["Div"].isin(CONFIG["train_leagues"])].copy()
        log(f"Leagues : {df.shape}")

    df["HomeTeamCanon"] = df["HomeTeam"].map(normalize_team)
    df["AwayTeamCanon"] = df["AwayTeam"].map(normalize_team)
    all_teams = sorted(set(df["HomeTeamCanon"].dropna()) | set(df["AwayTeamCanon"].dropna()))
    team2id = {t: i for i, t in enumerate(all_teams)}
    df["HomeTeamId"] = df["HomeTeamCanon"].map(team2id)
    df["AwayTeamId"] = df["AwayTeamCanon"].map(team2id)

    with open(map_dir / "team2id.json", "w", encoding="utf-8") as f:
        json.dump(team2id, f, indent=2, ensure_ascii=False)
    with open(map_dir / "id2team.json", "w", encoding="utf-8") as f:
        json.dump({str(i): t for t, i in team2id.items()}, f, indent=2, ensure_ascii=False)
    with open(map_dir / "league_map.json", "w", encoding="utf-8") as f:
        json.dump(LEAGUE_MAP, f, indent=2, ensure_ascii=False)
    with open(map_dir / "team_aliases.json", "w", encoding="utf-8") as f:
        json.dump(TEAM_ALIASES, f, indent=2, ensure_ascii=False)
    try:
        team_mapping.write_maps(team2id)
    except Exception as e:
        log(f"team_mapping.write_maps warn: {e}")
    log(f"Teams mapped: {len(team2id)}")

    df = engineer(df)
    labeled = df.dropna(subset=["FTR"]).sort_values("Date")
    log(f"Labeled matches: {len(labeled)}")

    focus_list: List[Optional[str]] = [None]
    if CONFIG["focus_teams"]:
        focus_list = []
        for raw in CONFIG["focus_teams"]:
            canon = normalize_team(raw)
            if canon not in team2id:
                log(f"WARNING: '{raw}' not in data — skip")
                continue
            focus_list.append(canon)
        if not focus_list:
            log("ERROR: no valid focus teams")
            sys.exit(1)

    registry = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "parquet": str(pq_path),
        "n_teams_total": len(team2id),
        "leagues": LEAGUE_MAP,
        "backends": [k.replace("use_", "") for k, v in CONFIG.items() if k.startswith("use_") and v],
        "runs": {},
    }

    for focus in focus_list:
        if focus is None:
            run_name, run_df = "global", labeled
            run_dir = out_root / "global"
        else:
            run_name = slug(focus)
            tid = team2id[focus]
            run_df = labeled[(labeled["HomeTeamId"] == tid) | (labeled["AwayTeamId"] == tid)].copy()
            run_dir = out_root / "teams" / run_name
            log(f"--- Team: {focus}  matches={len(run_df)}  -> {run_dir}")
            if len(run_df) < CONFIG["min_team_matches"]:
                log(f"    SKIP (< {CONFIG['min_team_matches']})")
                continue

        models_dir = run_dir / "models"
        preproc_dir = run_dir / "preprocessors"
        models_dir.mkdir(parents=True, exist_ok=True)
        preproc_dir.mkdir(parents=True, exist_ok=True)

        split = int(len(run_df) * (1 - CONFIG["val_fraction"]))
        split = max(30, min(split, len(run_df) - 20))
        train_df, val_df = run_df.iloc[:split], run_df.iloc[split:]
        log(f"    split train={len(train_df)} val={len(val_df)}")

        run_targets = {}
        for tkey in TARGETS:
            try:
                saved = train_one_target(train_df, val_df, tkey, models_dir, preproc_dir)
                run_targets[tkey] = saved
                log(f"    {tkey} OK -> {list(saved.keys())}")
            except Exception as e:
                log(f"    {tkey} FAILED: {e}")
                traceback.print_exc()
                run_targets[tkey] = {"error": str(e)}

        registry["runs"][run_name] = {
            "team": focus,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "dir": str(run_dir),
            "targets": run_targets,
        }
        with open(run_dir / "registry.json", "w") as f:
            json.dump(registry["runs"][run_name], f, indent=2)
        gc.collect()

    with open(out_root / "model_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    log("=" * 64)
    log(f"DONE -> {out_root / 'model_registry.json'}")
    log("Backends: " + ", ".join(registry["backends"]))
    log("=" * 64)


if __name__ == "__main__":
    main()
