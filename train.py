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
    # Industrial daily: deeper trees, stronger capacity, still CI-friendly
    CONFIG["capacity_growth_steps"] = int(os.environ.get("CAPACITY_STEPS", "2"))
    CONFIG["max_boost_rounds"] = min(int(os.environ.get("MAX_BOOST_ROUNDS", "1400")), 2000)
    CONFIG["patience_overfit"] = 45
    CONFIG["nn_epochs"] = int(os.environ.get("NN_EPOCHS", "60"))
    CONFIG["nn_patience"] = 12
    CONFIG["nn_batch"] = 256
    CONFIG["nn_hidden"] = [192, 96, 48]  # deeper MLP
    CONFIG["n_jobs"] = max(1, min(3, (os.cpu_count() or 2)))
    CONFIG["overfit_gap_warn"] = 0.12
    if os.environ.get("USE_ADABOOST", "").strip() not in ("1", "true", "yes"):
        CONFIG["use_adaboost"] = False
    if os.environ.get("USE_RANDOM_FOREST", "").strip() in ("0", "false", "no"):
        CONFIG["use_random_forest"] = False
    else:
        CONFIG["use_random_forest"] = True
    print("[train] DAILY ADVANCED: boost<=%s nn_epochs=%s hidden=%s capacity=%s"
          % (CONFIG["max_boost_rounds"], CONFIG["nn_epochs"], CONFIG["nn_hidden"],
             CONFIG["capacity_growth_steps"]))

# Parallel team workers (ProcessPool). 1 = sequential.
CONFIG["train_workers"] = int(os.environ.get("TRAIN_WORKERS", "1"))

# Explicit backend overrides (env wins)
for flag, key in (
    ("USE_XGBOOST", "use_xgboost"),
    ("USE_LIGHTGBM", "use_lightgbm"),
    ("USE_CATBOOST", "use_catboost"),
    ("USE_ADABOOST", "use_adaboost"),
    ("USE_RANDOM_FOREST", "use_random_forest"),
    ("USE_PYTORCH", "use_pytorch"),
    ("USE_TENSORFLOW", "use_tensorflow"),
):
    v = os.environ.get(flag, "").strip().lower()
    if v in ("1", "true", "yes"):
        CONFIG[key] = True
    elif v in ("0", "false", "no"):
        CONFIG[key] = False

LEAGUE_MAP = dict(team_mapping.LEAGUE_MAP)

TEAM_ALIASES = dict(team_mapping.TEAM_ALIASES)

TARGETS = {
    # Class index is positional and MUST stay H=0, D=1, A=2.
    # Never LabelEncoder.fit() these — sklearn sorts to A,D,H and sim.py
    # would read home/away swapped.
    "ft_result": {"col": "FTR", "type": "multiclass", "classes": ["H", "D", "A"]},
    "ht_result": {"col": "HTR", "type": "multiclass", "classes": ["H", "D", "A"]},
    "over25":    {"col": "Over2_5", "type": "binary"},
    "btts":      {"col": "BTTS", "type": "binary"},
    "dc_1x":     {"col": "DC_1X", "type": "binary"},
    "dc_x2":     {"col": "DC_X2", "type": "binary"},
    "dc_12":     {"col": "DC_12", "type": "binary"},
    "ht_over15": {"col": "HT_Over1_5", "type": "binary"},
}
# Daily can train only core markets for speed: DAILY_TARGETS=ft_result,over25,btts,ht_result
_dt = os.environ.get("DAILY_TARGETS", "").strip()
if _dt:
    keep = {x.strip() for x in _dt.split(",") if x.strip()}
    TARGETS = {k: v for k, v in TARGETS.items() if k in keep}
    print(f"[train] DAILY_TARGETS restricted to: {list(TARGETS.keys())}")

FEATURE_NUM = [
    # Market
    "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A",
    "LogOddsH", "LogOddsD", "LogOddsA", "ImpH", "ImpD", "ImpA",
    "OddsMargin", "HomeOddsEdge", "AwayOddsEdge",
    "FairH", "FairD", "FairA", "MarketEntropy",
    # Calendar
    "Year", "Month", "DayOfWeek", "IsWeekend",
    # Global form windows
    "HomeFormPts_5", "HomeFormGF_5", "HomeFormGA_5", "HomeFormGD_5",
    "AwayFormPts_5", "AwayFormGF_5", "AwayFormGA_5", "AwayFormGD_5",
    "HomeFormPts_10", "HomeFormGF_10", "HomeFormGA_10", "HomeFormGD_10",
    "AwayFormPts_10", "AwayFormGF_10", "AwayFormGA_10", "AwayFormGD_10",
    "HomeFormPts_20", "AwayFormPts_20",
    # Venue-specific form
    "HomeHomePts_5", "HomeHomeGD_5", "AwayAwayPts_5", "AwayAwayGD_5",
    # Strength / dynamics
    "EloHome", "EloAway", "EloDiff", "EloExpectHome",
    "RestHome", "RestAway", "RestDiff",
    "HomeStreak", "AwayStreak",
    "HomeFormPts_EW", "AwayFormPts_EW",
    "H2H_HomeWins_5", "H2H_Draws_5", "H2H_AwayWins_5",
    "H2H_HomeGD_5",
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
