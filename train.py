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

    # Per-team chronological form using ALL matches (home+away), leak-free shift(1)
    # Matches sim.build_live_features so train/serve stay aligned.
    df = df.reset_index(drop=True)
    df["_pts_as_home"] = df["FTR"].map({"H": 3, "D": 1, "A": 0}).astype(float)
    df["_pts_as_away"] = df["FTR"].map({"H": 0, "D": 1, "A": 3}).astype(float)
    for w in (5, 10, 20):
        df[f"HomeFormPts_{w}"] = np.nan
        df[f"HomeFormGF_{w}"] = np.nan
        df[f"HomeFormGA_{w}"] = np.nan
        df[f"HomeFormGD_{w}"] = np.nan
        df[f"AwayFormPts_{w}"] = np.nan
        df[f"AwayFormGF_{w}"] = np.nan
        df[f"AwayFormGA_{w}"] = np.nan
        df[f"AwayFormGD_{w}"] = np.nan

    from collections import defaultdict, deque
    hist_pts = defaultdict(lambda: deque(maxlen=25))
    hist_gf = defaultdict(lambda: deque(maxlen=25))
    hist_ga = defaultdict(lambda: deque(maxlen=25))

    home_ids = df["HomeTeamId"].to_numpy()
    away_ids = df["AwayTeamId"].to_numpy()
    fthg = df["FTHG"].to_numpy(dtype=float) if "FTHG" in df.columns else np.zeros(len(df))
    ftag = df["FTAG"].to_numpy(dtype=float) if "FTAG" in df.columns else np.zeros(len(df))
    pts_h = df["_pts_as_home"].to_numpy(dtype=float)
    pts_a = df["_pts_as_away"].to_numpy(dtype=float)

    H_pts = {w: np.full(len(df), np.nan) for w in (5, 10, 20)}
    H_gf = {w: np.full(len(df), np.nan) for w in (5, 10, 20)}
    H_ga = {w: np.full(len(df), np.nan) for w in (5, 10, 20)}
    A_pts = {w: np.full(len(df), np.nan) for w in (5, 10, 20)}
    A_gf = {w: np.full(len(df), np.nan) for w in (5, 10, 20)}
    A_ga = {w: np.full(len(df), np.nan) for w in (5, 10, 20)}

    for i in range(len(df)):
        hid, aid = int(home_ids[i]), int(away_ids[i])
        for w in (5, 10, 20):
            if len(hist_pts[hid]):
                arr = np.asarray(hist_pts[hid])[-w:]
                H_pts[w][i] = float(np.mean(arr))
                H_gf[w][i] = float(np.mean(np.asarray(hist_gf[hid])[-w:]))
                H_ga[w][i] = float(np.mean(np.asarray(hist_ga[hid])[-w:]))
            if len(hist_pts[aid]):
                arr = np.asarray(hist_pts[aid])[-w:]
                A_pts[w][i] = float(np.mean(arr))
                A_gf[w][i] = float(np.mean(np.asarray(hist_gf[aid])[-w:]))
                A_ga[w][i] = float(np.mean(np.asarray(hist_ga[aid])[-w:]))
        # update after features (no leak)
        hist_pts[hid].append(pts_h[i])
        hist_gf[hid].append(fthg[i])
        hist_ga[hid].append(ftag[i])
        hist_pts[aid].append(pts_a[i])
        hist_gf[aid].append(ftag[i])
        hist_ga[aid].append(fthg[i])

    for w in (5, 10, 20):
        df[f"HomeFormPts_{w}"] = H_pts[w]
        df[f"HomeFormGF_{w}"] = H_gf[w]
        df[f"HomeFormGA_{w}"] = H_ga[w]
        df[f"HomeFormGD_{w}"] = H_gf[w] - H_ga[w]
        df[f"AwayFormPts_{w}"] = A_pts[w]
        df[f"AwayFormGF_{w}"] = A_gf[w]
        df[f"AwayFormGA_{w}"] = A_ga[w]
        df[f"AwayFormGD_{w}"] = A_gf[w] - A_ga[w]
    df.drop(columns=["_pts_as_home", "_pts_as_away"], inplace=True, errors="ignore")

    # PairKey groups the two fixtures' worth of history together regardless of
    # who hosted; the counts below are then computed relative to whichever
    # team is HOME in the CURRENT row — matching sim.py's build_live_features,
    # which counts H2H wins for "the team that is home today", not "whoever
    # was home in that particular past meeting".
    df["PairKey"] = df.apply(
        lambda r: tuple(sorted([int(r["HomeTeamId"]), int(r["AwayTeamId"])])), axis=1
    )
    hist = defaultdict(lambda: deque(maxlen=5))  # winner_team_id, or None for draw
    hw, dr, aw = [], [], []
    for _, r in df.iterrows():
        cur_home = int(r["HomeTeamId"])
        past = list(hist[r["PairKey"]])
        hw.append(sum(1 for w in past if w == cur_home))
        dr.append(sum(1 for w in past if w is None))
        aw.append(sum(1 for w in past if w is not None and w != cur_home))
        ftr = r["FTR"] if pd.notna(r["FTR"]) else "D"
        if ftr == "H":
            winner = int(r["HomeTeamId"])
        elif ftr == "A":
            winner = int(r["AwayTeamId"])
        else:
            winner = None
        hist[r["PairKey"]].append(winner)

    df["H2H_HomeWins_5"] = hw
    df["H2H_Draws_5"] = dr
    df["H2H_AwayWins_5"] = aw

    # ---- Industrial strength layer: Elo, rest, venue form, streaks, EW form ----
    n = len(df)
    elo = {}
    elo_h = np.full(n, 1500.0)
    elo_a = np.full(n, 1500.0)
    rest_h = np.full(n, 7.0)
    rest_a = np.full(n, 7.0)
    last_date = {}
    home_home_pts = np.full(n, np.nan)
    home_home_gd = np.full(n, np.nan)
    away_away_pts = np.full(n, np.nan)
    away_away_gd = np.full(n, np.nan)
    hist_hh_pts = defaultdict(list)
    hist_hh_gd = defaultdict(list)
    hist_aa_pts = defaultdict(list)
    hist_aa_gd = defaultdict(list)
    streak = defaultdict(int)
    streak_h = np.zeros(n)
    streak_a = np.zeros(n)
    ew_h = np.full(n, np.nan)
    ew_a = np.full(n, np.nan)
    hist_pts_ew = defaultdict(list)

    K_ELO = 20.0
    HOME_ADV = 60.0
    dates = pd.to_datetime(df["Date"], errors="coerce")
    fthg = pd.to_numeric(df.get("FTHG"), errors="coerce").fillna(0).to_numpy()
    ftag = pd.to_numeric(df.get("FTAG"), errors="coerce").fillna(0).to_numpy()
    ftr = df["FTR"].astype(str).to_numpy()
    hid_a = df["HomeTeamId"].to_numpy()
    aid_a = df["AwayTeamId"].to_numpy()

    for i in range(n):
        hid, aid = int(hid_a[i]), int(aid_a[i])
        rh = elo.get(hid, 1500.0)
        ra = elo.get(aid, 1500.0)
        elo_h[i] = rh
        elo_a[i] = ra
        # rest days
        d = dates.iloc[i]
        if pd.notna(d):
            if hid in last_date and pd.notna(last_date[hid]):
                rest_h[i] = float(min(30, max(0, (d - last_date[hid]).days)))
            if aid in last_date and pd.notna(last_date[aid]):
                rest_a[i] = float(min(30, max(0, (d - last_date[aid]).days)))
            last_date[hid] = d
            last_date[aid] = d
        # venue-specific
        if hist_hh_pts[hid]:
            home_home_pts[i] = float(np.mean(hist_hh_pts[hid][-5:]))
            home_home_gd[i] = float(np.mean(hist_hh_gd[hid][-5:]))
        if hist_aa_pts[aid]:
            away_away_pts[i] = float(np.mean(hist_aa_pts[aid][-5:]))
            away_away_gd[i] = float(np.mean(hist_aa_gd[aid][-5:]))
        streak_h[i] = streak[hid]
        streak_a[i] = streak[aid]
        if hist_pts_ew[hid]:
            w = np.exp(np.linspace(-1.5, 0, len(hist_pts_ew[hid][-10:])))
            ew_h[i] = float(np.average(hist_pts_ew[hid][-10:], weights=w))
        if hist_pts_ew[aid]:
            w = np.exp(np.linspace(-1.5, 0, len(hist_pts_ew[aid][-10:])))
            ew_a[i] = float(np.average(hist_pts_ew[aid][-10:], weights=w))

        # update after features
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        score_h = 1.0 if ftr[i] == "H" else (0.5 if ftr[i] == "D" else 0.0)
        elo[hid] = rh + K_ELO * (score_h - exp_h)
        elo[aid] = ra + K_ELO * ((1.0 - score_h) - (1.0 - exp_h))
        pts_h = 3 if ftr[i] == "H" else (1 if ftr[i] == "D" else 0)
        pts_a = 3 if ftr[i] == "A" else (1 if ftr[i] == "D" else 0)
        hist_hh_pts[hid].append(pts_h)
        hist_hh_gd[hid].append(float(fthg[i] - ftag[i]))
        hist_aa_pts[aid].append(pts_a)
        hist_aa_gd[aid].append(float(ftag[i] - fthg[i]))
        hist_pts_ew[hid].append(pts_h)
        hist_pts_ew[aid].append(pts_a)
        # streak: +win, -loss, 0 draw resets toward 0
        if ftr[i] == "H":
            streak[hid] = streak[hid] + 1 if streak[hid] >= 0 else 1
            streak[aid] = streak[aid] - 1 if streak[aid] <= 0 else -1
        elif ftr[i] == "A":
            streak[aid] = streak[aid] + 1 if streak[aid] >= 0 else 1
            streak[hid] = streak[hid] - 1 if streak[hid] <= 0 else -1
        else:
            streak[hid] = 0
            streak[aid] = 0

    df["EloHome"] = elo_h
    df["EloAway"] = elo_a
    df["EloDiff"] = elo_h - elo_a
    df["EloExpectHome"] = 1.0 / (1.0 + np.power(10.0, (elo_a - (elo_h + HOME_ADV)) / 400.0))
    df["RestHome"] = rest_h
    df["RestAway"] = rest_a
    df["RestDiff"] = rest_h - rest_a
    df["HomeHomePts_5"] = home_home_pts
    df["HomeHomeGD_5"] = home_home_gd
    df["AwayAwayPts_5"] = away_away_pts
    df["AwayAwayGD_5"] = away_away_gd
    df["HomeStreak"] = streak_h
    df["AwayStreak"] = streak_a
    df["HomeFormPts_EW"] = ew_h
    df["AwayFormPts_EW"] = ew_a
    # market fair probs + entropy
    if all(c in df.columns for c in ("AvgH", "AvgD", "AvgA")):
        ih = 1.0 / df["AvgH"].clip(1.01)
        id_ = 1.0 / df["AvgD"].clip(1.01)
        ia = 1.0 / df["AvgA"].clip(1.01)
        s = ih + id_ + ia
        df["FairH"] = ih / s
        df["FairD"] = id_ / s
        df["FairA"] = ia / s
        p = np.clip(np.column_stack([df["FairH"], df["FairD"], df["FairA"]]), 1e-9, 1)
        df["MarketEntropy"] = -(p * np.log(p)).sum(axis=1)
    else:
        df["FairH"] = df["FairD"] = df["FairA"] = np.nan
        df["MarketEntropy"] = np.nan
    # H2H goal diff from same past meetings (approx via form already); set 0 placeholder filled above structure
    df["H2H_HomeGD_5"] = df.get("H2H_HomeWins_5", 0) - df.get("H2H_AwayWins_5", 0)

    return df


def prepare_xy(df, target_key, le_div_in=None, le_y_in=None):
    """
    le_div_in / le_y_in: pass the TRAIN-fitted encoders when preparing a
    validation split, so Div and class encodings match training exactly.
    Leave None when preparing the training split (fits fresh encoders).
    """
    cfg = TARGETS[target_key]
    sub = df.dropna(subset=[cfg["col"]]).copy()
    if len(sub) < 25:
        raise ValueError(f"Too few rows for {target_key}: {len(sub)}")

    num_cols = [c for c in FEATURE_NUM if c in sub.columns]
    X_num = sub[num_cols].apply(pd.to_numeric, errors="coerce")
    valid = [c for c in num_cols if X_num[c].notna().any()]
    X_num = X_num[valid].fillna(X_num[valid].median(numeric_only=True)).fillna(0.0)

    X_teams = sub[["HomeTeamId", "AwayTeamId"]].astype(int).values

    if le_div_in is not None:
        # Reuse the training encoding — never refit on the val split, or the
        # same division could get a different integer code than in Xtr.
        le_div = le_div_in
        div_map = {c: i for i, c in enumerate(le_div.classes_)}
        oov = len(le_div.classes_)  # unseen division -> dedicated OOV bucket
        div_enc = (
            sub["Div"].fillna("UNK").astype(str).map(div_map).fillna(oov).astype(int).values.reshape(-1, 1)
        )
    else:
        le_div = LabelEncoder()
        div_enc = le_div.fit_transform(sub["Div"].fillna("UNK").astype(str)).reshape(-1, 1)

    X = np.nan_to_num(np.hstack([X_num.values.astype(np.float64), X_teams, div_enc]), nan=0.0)
    feat_names = valid + ["HomeTeamId", "AwayTeamId", "DivEnc"]

    y_raw = sub[cfg["col"]]
    le_y = le_y_in
    if cfg["type"] == "multiclass":
        classes = list(le_y_in.classes_) if le_y_in is not None else list(cfg.get("classes") or ["H", "D", "A"])
        if classes:
            mask = y_raw.isin(classes).values
            X, y_raw = X[mask], y_raw[mask]
        mapping = {str(c): i for i, c in enumerate(classes)}
        y = np.array([mapping[str(v)] for v in y_raw.astype(str).values], dtype=int)
        if le_y is None:
            # Pin class order to cfg (H,D,A). Do NOT fit LabelEncoder —
            # .fit() sorts alphabetically to A,D,H and inverts home/away.
            le_y = LabelEncoder()
            le_y.classes_ = np.asarray(classes)
    else:
        y = (pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).values > 0).astype(int)

    ok = np.isfinite(y) if cfg["type"] == "multiclass" else (np.isfinite(y) & np.isin(y, [0, 1]))
    X, y = X[ok], y[ok]
    if le_y_in is None and len(np.unique(y)) < 2:
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
        Xva, yva, _, _, _ = prepare_xy(val_df, target_key, le_div_in=le_div, le_y_in=le_y)
    else:
        Xva, yva, _, _, _ = prepare_xy(val_df, target_key, le_div_in=le_div)

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



def _train_team_job(args):
    """Worker for parallel daily team training (picklable)."""
    focus, run_df_dict, team2id_focus_tid, out_root_str, min_matches = args
    import shutil
    out_root = Path(out_root_str)
    run_name = slug(focus)
    run_dir = out_root / "teams" / run_name
    run_df = pd.DataFrame(run_df_dict)
    if "Date" in run_df.columns:
        run_df["Date"] = pd.to_datetime(run_df["Date"], errors="coerce")
    log(f"--- Team: {focus}  matches={len(run_df)}  -> {run_dir}")
    if len(run_df) < min_matches:
        log(f"    SKIP (< {min_matches})")
        return run_name, {"team": focus, "skipped": True, "n": len(run_df)}

    models_dir = run_dir / "models"
    preproc_dir = run_dir / "preprocessors"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    if preproc_dir.exists():
        shutil.rmtree(preproc_dir)
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

    entry = {
        "team": focus,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "dir": str(run_dir),
        "targets": run_targets,
    }
    with open(run_dir / "registry.json", "w") as f:
        json.dump(entry, f, indent=2)
    gc.collect()
    return run_name, entry


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
    # --- data quality gate (critical for clean models) ---
    req = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        log(f"ERROR: parquet missing columns {miss}")
        sys.exit(1)
    null_teams = int(df["HomeTeam"].isna().sum() + df["AwayTeam"].isna().sum())
    if null_teams:
        log(f"ERROR: {null_teams} null team names — clean parquet first")
        sys.exit(1)
    fthg = pd.to_numeric(df["FTHG"], errors="coerce")
    ftag = pd.to_numeric(df["FTAG"], errors="coerce")
    if int(((fthg < 0) | (ftag < 0)).sum()):
        log("ERROR: negative scores in parquet")
        sys.exit(1)
    same = int((df["HomeTeam"].astype(str) == df["AwayTeam"].astype(str)).sum())
    if same:
        log(f"WARNING: dropping {same} rows where HomeTeam==AwayTeam")
        df = df[df["HomeTeam"].astype(str) != df["AwayTeam"].astype(str)].copy()
    # drop rows with null essential fields before feature engineering
    before = len(df)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"]).copy()
    if len(df) < before:
        log(f"Dropped {before - len(df)} rows with null Date/Team/FTR")
    if CONFIG["train_leagues"]:
        df = df[df["Div"].isin(CONFIG["train_leagues"])].copy()
        log(f"Leagues : {df.shape}")

    df["HomeTeamCanon"] = df["HomeTeam"].map(normalize_team)
    df["AwayTeamCanon"] = df["AwayTeam"].map(normalize_team)
    all_teams = sorted(set(df["HomeTeamCanon"].dropna()) | set(df["AwayTeamCanon"].dropna()))
    # Stable IDs: preserve prior mapping so models/sim stay aligned across days
    team2id = {}
    prev_path = map_dir / "team2id.json"
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            team2id = {str(k): int(v) for k, v in prev.items()}
        except Exception:
            team2id = {}
    next_id = (max(team2id.values()) + 1) if team2id else 0
    for t in all_teams:
        if t not in team2id:
            team2id[t] = next_id
            next_id += 1
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

    workers = max(1, int(CONFIG.get("train_workers", 1)))
    # Global always sequential
    team_jobs = []
    for focus in focus_list:
        if focus is None:
            run_name, run_df = "global", labeled
            run_dir = out_root / "global"
            models_dir = run_dir / "models"
            preproc_dir = run_dir / "preprocessors"
            import shutil
            if models_dir.exists():
                shutil.rmtree(models_dir)
            if preproc_dir.exists():
                shutil.rmtree(preproc_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            preproc_dir.mkdir(parents=True, exist_ok=True)
            split = int(len(run_df) * (1 - CONFIG["val_fraction"]))
            split = max(30, min(split, len(run_df) - 20))
            train_df, val_df = run_df.iloc[:split], run_df.iloc[split:]
            log(f"--- GLOBAL  matches={len(run_df)} train={len(train_df)} val={len(val_df)}")
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
            registry["runs"]["global"] = {
                "team": None, "n_train": len(train_df), "n_val": len(val_df),
                "dir": str(run_dir), "targets": run_targets,
            }
            with open(run_dir / "registry.json", "w") as f:
                json.dump(registry["runs"]["global"], f, indent=2)
            gc.collect()
            continue

        tid = team2id[focus]
        run_df = labeled[(labeled["HomeTeamId"] == tid) | (labeled["AwayTeamId"] == tid)].copy()
        # serialize for workers (avoid huge pickle of full frame)
        keep = set(FEATURE_NUM) | {
            "Div", "Date", "HomeTeam", "AwayTeam", "HomeTeamId", "AwayTeamId",
            "FTR", "HTR", "FTHG", "FTAG", "Over2_5", "BTTS", "DC_1X", "DC_X2", "DC_12",
            "HT_Over1_5", "HomeTeamCanon", "AwayTeamCanon",
            "Year", "Month", "DayOfWeek", "IsWeekend",
            "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A",
            "HomeOddsEdge", "AwayOddsEdge", "LogOddsH", "LogOddsD", "LogOddsA",
            "ImpH", "ImpD", "ImpA", "OddsMargin",
        }
        cols = [
            c for c in run_df.columns
            if c in keep
            or str(c).startswith(("HomeForm", "AwayForm", "H2H", "LogOdds", "Imp", "Odds"))
        ]
        # fallback: all columns if filter empty
        if len(cols) < 10:
            cols = list(run_df.columns)
        slim = run_df[cols].copy()
        team_jobs.append((focus, slim.to_dict(orient="list"), tid, str(out_root), CONFIG["min_team_matches"]))

    if not team_jobs:
        pass
    elif workers <= 1 or len(team_jobs) == 1:
        log(f"Training {len(team_jobs)} teams sequential")
        for job in team_jobs:
            name, entry = _train_team_job(job)
            if not entry.get("skipped"):
                registry["runs"][name] = entry
            gc.collect()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Cap workers to avoid RAM blow-up on Actions
        w = min(workers, len(team_jobs), 4)
        log(f"Training {len(team_jobs)} teams in parallel workers={w}")
        # Limit BLAS/OMP threads inside each worker
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        with ProcessPoolExecutor(max_workers=w) as pool:
            futs = {pool.submit(_train_team_job, job): job[0] for job in team_jobs}
            for fut in as_completed(futs):
                team = futs[fut]
                try:
                    name, entry = fut.result()
                    if not entry.get("skipped"):
                        registry["runs"][name] = entry
                    log(f"  [done] {team}")
                except Exception as e:
                    log(f"  [FAIL] {team}: {e}")
                    traceback.print_exc()

    with open(out_root / "model_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    log("=" * 64)
    log(f"DONE -> {out_root / 'model_registry.json'}")
    log("Backends: " + ", ".join(registry["backends"]))
    log("=" * 64)


if __name__ == "__main__":
    main()
