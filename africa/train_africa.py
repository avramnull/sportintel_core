#!/usr/bin/env python3
"""
Africa model training — result / O2.5 / BTTS on master_africa_football.parquet.

Designed for odds-sparse African domestic data:
  - Features: Elo, form, calendar, team ids (no market columns required)
  - Backends: XGB / LightGBM / CatBoost / RF (env-toggleable)
  - Scope: GLOBAL Africa + optional per-country folders

Usage:
  AFRICA_PARQUET=master_africa_football.parquet python -m africa.train_africa
  FOCUS_COUNTRIES=Nigeria,Ghana MIN_TEAM_MATCHES=20 python -m africa.train_africa
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "parquet_path": os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")),
    "out_dir": str(ROOT / "football_models" / "africa"),
    "min_matches": int(os.environ.get("MIN_TEAM_MATCHES", "25")),
    "val_fraction": 0.15,
    "seed": 42,
    "max_boost_rounds": int(os.environ.get("MAX_BOOST_ROUNDS", "800")),
    "use_xgboost": os.environ.get("USE_XGBOOST", "1") not in ("0", "false"),
    "use_lightgbm": os.environ.get("USE_LIGHTGBM", "1") not in ("0", "false"),
    "use_catboost": os.environ.get("USE_CATBOOST", "1") not in ("0", "false"),
    "use_random_forest": os.environ.get("USE_RANDOM_FOREST", "1") not in ("0", "false"),
}

FEATURE_NUM = [
    "Year", "Month", "DayOfWeek", "IsWeekend",
    "HomeFormPts_5", "AwayFormPts_5", "HomeFormGD_5", "AwayFormGD_5",
    "HomeFormPts_10", "AwayFormPts_10", "HomeFormGD_10", "AwayFormGD_10",
    "EloHome", "EloAway", "EloDiff",
]
FEATURE_ID = ["HomeTeamId", "AwayTeamId"]

TARGETS = {
    "ft_result": ("FTR", "multiclass"),
    "over25": ("Over2_5", "binary"),
    "btts": ("BTTS", "binary"),
}


def log(msg: str):
    print(msg, flush=True)


def load_africa(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if "Over2_5" not in df.columns:
        df["Over2_5"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
    if "BTTS" not in df.columns:
        df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    for c in FEATURE_NUM:
        if c not in df.columns:
            df[c] = np.nan
    for c in FEATURE_ID:
        if c not in df.columns:
            teams = sorted(set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str)))
            t2 = {t: i for i, t in enumerate(teams)}
            df["HomeTeamId"] = df["HomeTeam"].map(t2)
            df["AwayTeamId"] = df["AwayTeam"].map(t2)
            break
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def make_xy(df: pd.DataFrame, target_key: str):
    col, task = TARGETS[target_key]
    feats = FEATURE_NUM + FEATURE_ID
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if task == "multiclass":
        y = df[col].astype(str).str.upper().map({"H": 0, "D": 1, "A": 2}).to_numpy()
        mask = np.isfinite(y)
    else:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = np.isfinite(y)
    return X[mask], y[mask].astype(int), feats


def train_backends(Xtr, ytr, Xva, yva, task: str, out_dir: Path, target_key: str) -> dict:
    saved = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = CONFIG["max_boost_rounds"]
    seed = CONFIG["seed"]

    if CONFIG["use_xgboost"]:
        try:
            import xgboost as xgb
            if task == "multiclass":
                params = dict(objective="multi:softprob", num_class=3, max_depth=6, eta=0.05,
                              subsample=0.85, colsample_bytree=0.8, seed=seed, eval_metric="mlogloss")
            else:
                params = dict(objective="binary:logistic", max_depth=6, eta=0.05,
                              subsample=0.85, colsample_bytree=0.8, seed=seed, eval_metric="logloss")
            dtr = xgb.DMatrix(Xtr, label=ytr)
            dva = xgb.DMatrix(Xva, label=yva)
            bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "va")],
                            early_stopping_rounds=40, verbose_eval=False)
            path = out_dir / f"xgb_{target_key}.json"
            bst.save_model(str(path))
            saved["xgboost"] = str(path)
            log(f"    xgb {target_key} best_iter={bst.best_iteration}")
        except Exception as e:
            log(f"    xgb FAILED: {e}")

    if CONFIG["use_lightgbm"]:
        try:
            import lightgbm as lgb
            if task == "multiclass":
                params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=48,
                              max_depth=8, seed=seed, verbose=-1)
            else:
                params = dict(objective="binary", learning_rate=0.05, num_leaves=48,
                              max_depth=8, seed=seed, verbose=-1)
            dtr = lgb.Dataset(Xtr, label=ytr)
            dva = lgb.Dataset(Xva, label=yva, reference=dtr)
            bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
            path = out_dir / f"lgbm_{target_key}.txt"
            bst.save_model(str(path))
            saved["lightgbm"] = str(path)
            log(f"    lgbm {target_key} best_iter={bst.best_iteration}")
        except Exception as e:
            log(f"    lgbm FAILED: {e}")

    if CONFIG["use_catboost"]:
        try:
            from catboost import CatBoostClassifier
            if task == "multiclass":
                model = CatBoostClassifier(loss_function="MultiClass", depth=6, learning_rate=0.05,
                                           iterations=min(rounds, 1200), random_seed=seed, verbose=False)
            else:
                model = CatBoostClassifier(loss_function="Logloss", depth=6, learning_rate=0.05,
                                           iterations=min(rounds, 1200), random_seed=seed, verbose=False)
            model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=40, verbose=False)
            path = out_dir / f"cat_{target_key}.cbm"
            model.save_model(str(path))
            saved["catboost"] = str(path)
            log(f"    cat {target_key} OK")
        except Exception as e:
            log(f"    cat FAILED: {e}")

    if CONFIG["use_random_forest"]:
        try:
            import pickle
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=400, max_depth=12, min_samples_leaf=5,
                random_state=seed, n_jobs=-1,
            )
            model.fit(Xtr, ytr)
            path = out_dir / f"rf_{target_key}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            saved["random_forest"] = str(path)
            log(f"    rf {target_key} OK")
        except Exception as e:
            log(f"    rf FAILED: {e}")

    return saved


def train_scope(df: pd.DataFrame, scope_name: str, out_root: Path) -> dict:
    run_dir = out_root / scope_name
    models_dir = run_dir / "models"
    pre_dir = run_dir / "preprocessors"
    models_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)

    split = int(len(df) * (1 - CONFIG["val_fraction"]))
    split = max(40, min(split, len(df) - 30))
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    log(f"--- {scope_name}: n={len(df)} train={len(train_df)} val={len(val_df)}")

    # feature list + simple standardisation stats (mean/std) for sim
    feats = FEATURE_NUM + FEATURE_ID
    Xall = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mean = Xall.mean().to_dict()
    std = Xall.std().replace(0, 1).to_dict()
    with open(pre_dir / "feature_stats.json", "w") as f:
        json.dump({"features": feats, "mean": mean, "std": std}, f, indent=2)

    targets = {}
    for tkey, (_, task) in TARGETS.items():
        try:
            Xtr, ytr, _ = make_xy(train_df, tkey)
            Xva, yva, _ = make_xy(val_df, tkey)
            if len(Xtr) < 30 or len(np.unique(ytr)) < 2:
                log(f"    skip {tkey}: insufficient data")
                continue
            # scale
            mu = np.array([mean[c] for c in feats], dtype=float)
            sg = np.array([std[c] for c in feats], dtype=float)
            Xtr_s = (Xtr - mu) / sg
            Xva_s = (Xva - mu) / sg
            saved = train_backends(Xtr_s, ytr, Xva_s, yva, task, models_dir, tkey)
            targets[tkey] = saved
        except Exception as e:
            log(f"    {tkey} FAILED: {e}")
            traceback.print_exc()
            targets[tkey] = {"error": str(e)}

    entry = {
        "scope": scope_name,
        "n": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "features": feats,
        "targets": targets,
        "countries": sorted(df["Country"].dropna().unique().tolist()) if "Country" in df.columns else [],
    }
    with open(run_dir / "registry.json", "w") as f:
        json.dump(entry, f, indent=2)
    return entry


def main():
    path = Path(CONFIG["parquet_path"])
    if not path.exists():
        # csv fallback
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
        else:
            raise SystemExit(f"Africa master not found: {CONFIG['parquet_path']}")

    df = load_africa(path)
    log(f"Loaded {len(df)} Africa matches from {path}")
    out_root = Path(CONFIG["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    registry = {}
    # GLOBAL
    registry["GLOBAL"] = train_scope(df, "GLOBAL", out_root)

    # Per-country if enough rows
    focus = [x.strip() for x in os.environ.get("FOCUS_COUNTRIES", "").split(",") if x.strip()]
    countries = focus or sorted(df["Country"].dropna().unique().tolist())
    for c in countries:
        sub = df[df["Country"] == c]
        if len(sub) < CONFIG["min_matches"]:
            log(f"skip country {c}: {len(sub)} < {CONFIG['min_matches']}")
            continue
        registry[c] = train_scope(sub, f"country_{c.replace(' ', '_')}", out_root)

    with open(out_root / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    log(f"Done. Models under {out_root}")


if __name__ == "__main__":
    main()
