#!/usr/bin/env python3
"""Africa model training — strict, chronological and identity-safe."""
from __future__ import annotations

import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .team_mapping import ensure_ids, load_aliases, resolve

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "parquet_path": os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")),
    "out_dir": str(ROOT / "football_models" / "africa"),
    "min_matches": int(os.environ.get("MIN_TEAM_MATCHES", "25")),
    "val_fraction": 0.20,
    "seed": 42,
    "max_boost_rounds": int(os.environ.get("MAX_BOOST_ROUNDS", "1400")),
}

FEATURE_NUM = [
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "AwayFormPts_5", "HomeFormGD_5", "AwayFormGD_5",
    "HomeFormPts_10", "AwayFormPts_10", "HomeFormGD_10", "AwayFormGD_10",
    "HomeFormPts_20", "AwayFormPts_20", "HomeFormGD_20", "AwayFormGD_20",
    "EloHome", "EloAway", "EloDiff", "RestHome", "RestAway", "RestDiff",
]
FEATURE_ID = ["HomeTeamId", "AwayTeamId"]
TARGETS = {
    "ft_result": ("FTR", "multiclass"),
    "ht_result": ("HTR", "multiclass"),
    "over25": ("Over2_5", "binary"),
    "btts": ("BTTS", "binary"),
    "ht_over15": ("HT_Over1_5", "binary"),
}


def log(msg: str):
    print(msg, flush=True)


def _canonize(df: pd.DataFrame) -> pd.DataFrame:
    ids = ensure_ids(list(df["HomeTeam"].astype(str)) + list(df["AwayTeam"].astype(str)))
    aliases = load_aliases()
    df = df.copy()
    df["HomeTeam"] = df["HomeTeam"].astype(str).map(lambda x: resolve(x, ids, aliases)[0])
    df["AwayTeam"] = df["AwayTeam"].astype(str).map(lambda x: resolve(x, ids, aliases)[0])
    df["HomeTeamId"] = df["HomeTeam"].map(ids)
    df["AwayTeamId"] = df["AwayTeam"].map(ids)
    if df[["HomeTeamId", "AwayTeamId"]].isna().any().any():
        raise ValueError("Africa team identity mapping produced missing IDs")
    return df


def load_africa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Africa master missing columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("FTHG", "FTAG", "HTHG", "HTAG"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()
    if ((df["FTHG"] < 0) | (df["FTAG"] < 0)).any():
        raise ValueError("negative Africa scores found")
    df = df[df["HomeTeam"].astype(str) != df["AwayTeam"].astype(str)].copy()
    if "HTHG" not in df.columns or "HTAG" not in df.columns:
        df["HTHG"], df["HTAG"] = 0.0, 0.0
    df["Over2_5"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
    df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["HTR"] = np.where(df["HTHG"] > df["HTAG"], "H", np.where(df["HTHG"] < df["HTAG"], "A", "D"))
    df["HT_Over1_5"] = ((df["HTHG"] + df["HTAG"]) > 1.5).astype(int)
    df = _canonize(df)
    return df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    n = len(df)
    for c in FEATURE_NUM:
        df[c] = np.nan
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    from collections import defaultdict, deque
    pts = defaultdict(lambda: deque(maxlen=25))
    gd = defaultdict(lambda: deque(maxlen=25))
    elo = {}
    last = {}
    rows = []
    K, HOME_ADV = 22.0, 55.0

    for i, r in df.iterrows():
        h, a = int(r["HomeTeamId"]), int(r["AwayTeamId"])
        rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        out = {"EloHome": rh, "EloAway": ra, "EloDiff": rh - ra, "RestHome": 7.0, "RestAway": 7.0, "RestDiff": 0.0}
        d = r["Date"]
        if h in last:
            out["RestHome"] = float(np.clip((d - last[h]).days, 0, 60))
        if a in last:
            out["RestAway"] = float(np.clip((d - last[a]).days, 0, 60))
        out["RestDiff"] = out["RestHome"] - out["RestAway"]
        for tid, prefix in ((h, "Home"), (a, "Away")):
            vals_p, vals_g = pts[tid], gd[tid]
            for w in (5, 10, 20):
                out[f"{prefix}FormPts_{w}"] = float(np.mean(list(vals_p)[-w:])) if vals_p else np.nan
                out[f"{prefix}FormGD_{w}"] = float(np.mean(list(vals_g)[-w:])) if vals_g else np.nan
        rows.append(out)
        # Update state only after features: strict no-lookahead.
        ftr = str(r["FTR"]).upper()
        sh = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        elo[h] = rh + K * (sh - exp_h)
        elo[a] = ra + K * ((1.0 - sh) - (1.0 - exp_h))
        hp = 3 if ftr == "H" else (1 if ftr == "D" else 0)
        ap = 3 if ftr == "A" else (1 if ftr == "D" else 0)
        pts[h].append(hp); gd[h].append(float(r["FTHG"] - r["FTAG"]))
        pts[a].append(ap); gd[a].append(float(r["FTAG"] - r["FTHG"]))
        last[h] = d; last[a] = d

    state = pd.DataFrame(rows, index=df.index)
    for c in state.columns:
        df[c] = state[c]
    return df


def make_xy(df: pd.DataFrame, target: str):
    col, task = TARGETS[target]
    sub = df.dropna(subset=[col]).copy()
    feats = FEATURE_NUM + FEATURE_ID
    X = sub[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float64)
    if task == "multiclass":
        mapping = {"H": 0, "D": 1, "A": 2}
        y = sub[col].astype(str).str.upper().map(mapping)
        keep = y.notna().to_numpy()
        y = y[keep].astype(int).to_numpy()
        X = X[keep]
        classes = ["H", "D", "A"]
    else:
        y = pd.to_numeric(sub[col], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
        classes = [0, 1]
    return X, y, feats, classes, task


def _scale_fit(X):
    mu = X.mean(axis=0); sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def _train_one(Xtr, ytr, Xva, yva, task, classes, out: Path):
    saved = {}
    try:
        import xgboost as xgb
        params = {"max_depth": 6, "learning_rate": 0.035, "subsample": .88, "colsample_bytree": .85,
                  "min_child_weight": 5, "reg_lambda": 2.0, "reg_alpha": .15, "tree_method": "hist",
                  "random_state": CONFIG["seed"], "n_jobs": max(1, min(4, os.cpu_count() or 2))}
        if task == "multiclass": params.update(objective="multi:softprob", num_class=3, eval_metric="mlogloss")
        else: params.update(objective="binary:logistic", eval_metric="logloss")
        m = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=CONFIG["max_boost_rounds"],
                      evals=[(xgb.DMatrix(Xva, label=yva), "val")], early_stopping_rounds=60, verbose_eval=False)
        p = out / ("xgb_" + task + ".json"); m.save_model(str(p)); saved["xgboost"] = str(p)
    except Exception as e: log(f"  xgb {task} FAILED: {e}")
    try:
        import lightgbm as lgb
        params = {"learning_rate": .035, "num_leaves": 48, "max_depth": 8, "min_child_samples": 15,
                  "reg_lambda": 2.0, "reg_alpha": .15, "random_state": CONFIG["seed"], "verbose": -1}
        if task == "multiclass": params.update(objective="multiclass", num_class=3, metric="multi_logloss")
        else: params.update(objective="binary", metric="binary_logloss")
        m = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=CONFIG["max_boost_rounds"],
                      valid_sets=[lgb.Dataset(Xva, label=yva)], callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        p = out / ("lgbm_" + task + ".txt"); m.save_model(str(p)); saved["lightgbm"] = str(p)
    except Exception as e: log(f"  lgbm {task} FAILED: {e}")
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(loss_function="MultiClass" if task == "multiclass" else "Logloss",
                               depth=7, learning_rate=.035, iterations=min(CONFIG["max_boost_rounds"], 1400),
                               l2_leaf_reg=5.0, random_seed=CONFIG["seed"], verbose=False, allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True, verbose=False)
        p = out / ("cat_" + task + ".cbm"); m.save_model(str(p)); saved["catboost"] = str(p)
    except Exception as e: log(f"  cat {task} FAILED: {e}")
    try:
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=500, max_depth=16, min_samples_leaf=4, max_features="sqrt",
                                   class_weight="balanced_subsample", random_state=CONFIG["seed"], n_jobs=-1)
        m.fit(Xtr, ytr)
        p = out / ("rf_" + task + ".pkl")
        with open(p, "wb") as f: pickle.dump(m, f)
        saved["random_forest"] = str(p)
    except Exception as e: log(f"  rf {task} FAILED: {e}")
    return saved


def train_scope(df: pd.DataFrame, scope: str, out_root: Path):
    run = out_root / scope; models = run / "models"; pre = run / "preprocessors"
    models.mkdir(parents=True, exist_ok=True); pre.mkdir(parents=True, exist_ok=True)
    split = int(len(df) * (1 - CONFIG["val_fraction"]))
    if split < 30 or len(df) - split < 15:
        raise ValueError(f"{scope}: insufficient chronological train/validation rows ({len(df)})")
    tr, va = df.iloc[:split].copy(), df.iloc[split:].copy()
    stats = {}
    for t in TARGETS:
        Xtr, ytr, feats, classes, task = make_xy(tr, t)
        Xva, yva, _, _, _ = make_xy(va, t)
        if len(Xtr) < 30 or len(Xva) < 10 or len(np.unique(ytr)) < 2:
            log(f"  skip {scope}/{t}: insufficient target support")
            continue
        mu, sd = _scale_fit(Xtr)
        Xtr_s, Xva_s = (Xtr - mu) / sd, (Xva - mu) / sd
        stats[t] = {"features": feats, "mean": mu.tolist(), "std": sd.tolist(), "classes": classes}
        saved = _train_one(Xtr_s, ytr, Xva_s, yva, task, classes, models)
        if not saved: log(f"  WARNING {scope}/{t}: no backend saved")
    (pre / "feature_stats.json").write_text(json.dumps({"targets": stats}, indent=2), encoding="utf-8")
    # Compatibility with the existing Africa simulator: expose the FT feature stats at top level.
    if "ft_result" in stats:
        s = stats["ft_result"]
        (pre / "feature_stats.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    reg = {"scope": scope, "n": len(df), "n_train": len(tr), "n_val": len(va),
           "features": FEATURE_NUM + FEATURE_ID, "targets": {},
           "countries": sorted(df["Country"].dropna().astype(str).unique().tolist()) if "Country" in df else []}
    for t in TARGETS:
        reg["targets"][t] = {}
        for backend, prefix, ext in (("xgboost", "xgb_", ".json"), ("lightgbm", "lgbm_", ".txt"), ("catboost", "cat_", ".cbm"), ("random_forest", "rf_", ".pkl")):
            p = models / f"{prefix}{t}{ext}"
            if p.exists(): reg["targets"][t][backend] = str(p)
    (run / "registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


def main():
    path = Path(CONFIG["parquet_path"])
    if not path.exists():
        alt = path.with_suffix(".csv")
        if alt.exists(): path = alt
        else: raise SystemExit(f"Africa master not found: {path}")
    df = engineer(load_africa(path))
    out = Path(CONFIG["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    focus = [x.strip() for x in os.environ.get("FOCUS_COUNTRIES", "").split(",") if x.strip()]
    available = sorted(df["Country"].dropna().astype(str).unique().tolist()) if "Country" in df else []
    scopes = focus or available
    # Do not train a global model when the daily board is country-scoped.
    registry = {"mode": "country_scoped" if focus else "all_countries", "runs": {}}
    for country in scopes:
        sub = df[df["Country"].astype(str) == country].copy() if "Country" in df else df.iloc[0:0]
        if len(sub) < CONFIG["min_matches"]:
            log(f"skip {country}: {len(sub)} < {CONFIG['min_matches']}")
            continue
        try:
            registry["runs"][country] = train_scope(sub, f"country_{country.replace(' ', '_')}", out)
        except Exception as e:
            log(f"country {country} FAILED: {e}")
            traceback.print_exc()
    (out / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    log(f"Africa training complete: scopes={list(registry['runs'])}")


if __name__ == "__main__":
    main()
