#!/usr/bin/env python3
"""
================================================================================
LIVE MATCH SIMULATOR — ADVANCED REPORT BUILD
================================================================================
Edit MATCH only.

  - Ensembles HOME + AWAY team models (when both exist)
  - Odds-blend option for calibration
  - Full report + PRECISE JUDGEMENT TABLE (edge vs market, stars, pick)

Works on Colab and local.
================================================================================
"""

from __future__ import annotations

import json, pickle, warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# MATCH CONFIG
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
    # blend model with implied odds (0=pure model, 1=pure market)
    "odds_blend": 0.35,
}

def _find_root() -> Path:
    candidates = []
    if Path("/content").exists(): candidates.append(Path("/content"))
    try: candidates.append(Path(__file__).resolve().parent)
    except NameError: pass
    candidates.append(Path.cwd())
    for c in candidates:
        if (c/"football_models"/"mappings"/"team2id.json").exists(): return c
        if (c/"football_models"/"model_registry.json").exists(): return c
    return candidates[0]

ROOT = _find_root()
MODELS_ROOT = ROOT / "football_models"
MAP_DIR = MODELS_ROOT / "mappings"
PARQUET_PATH = ROOT / "master_football_data.parquet"

def load_json(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def normalize_team(name, aliases):
    s = " ".join(str(name).strip().split())
    return aliases.get(s.lower(), s.title() if (s.islower() or s.isupper()) else s)

def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()

def resolve_league(league_input, league_map):
    inv = {v.lower(): k for k,v in league_map.items()}
    s = str(league_input).strip()
    if s in league_map: return s, league_map[s]
    if s.lower() in inv:
        code = inv[s.lower()]; return code, league_map[code]
    for code, name in league_map.items():
        if s.lower() in name.lower() or name.lower() in s.lower():
            return code, name
    raise ValueError(f"Unknown league '{league_input}'")

def implied(odds):
    return 1.0 / max(float(odds), 1.01)

def fair_probs(oh, od, oa):
    ih, id_, ia = implied(oh), implied(od), implied(oa)
    t = ih + id_ + ia
    return ih/t, id_/t, ia/t

def collect_run_dirs(home_canon, away_canon):
    runs = []
    for label, canon in [("home", home_canon), ("away", away_canon)]:
        d = MODELS_ROOT / "teams" / slug(canon)
        if (d/"registry.json").exists() or (d/"models").is_dir():
            runs.append((d, f"{canon} ({label})"))
    if runs: return runs
    g = MODELS_ROOT / "global"
    if (g/"registry.json").exists() or (g/"models").is_dir():
        return [(g, "global")]
    raise SystemExit(f"No models for {home_canon}/{away_canon} and no global/")

def load_run_targets(run_dir):
    reg_path = run_dir / "registry.json"
    if reg_path.exists():
        return load_json(reg_path).get("targets", {})
    targets = {}
    models_dir = run_dir / "models"
    if models_dir.is_dir():
        for p in models_dir.iterdir():
            for backend, prefix in [("xgboost","xgb_"),("lightgbm","lgbm_"),
                                    ("catboost","cat_"),("adaboost","ada_")]:
                if p.name.startswith(prefix):
                    tkey = p.name[len(prefix):].rsplit(".",1)[0]
                    targets.setdefault(tkey, {})[backend] = str(p)
    return targets

def load_preprocessors(run_dir):
    preproc_dir = run_dir / "preprocessors"
    if not (preproc_dir/"features_ft_result.json").exists():
        reg = run_dir / "registry.json"
        if reg.exists():
            preproc_dir = Path(load_json(reg).get("dir", run_dir)) / "preprocessors"
    feature_names = load_json(preproc_dir / "features_ft_result.json")
    with open(preproc_dir/"scaler_ft_result.pkl","rb") as f: scaler = pickle.load(f)
    with open(preproc_dir/"le_div_ft_result.pkl","rb") as f: le_div = pickle.load(f)
    return feature_names, scaler, le_div

def build_live_features(home_id, away_id, div_code, odds_h, odds_d, odds_a,
                        hist_df, feature_names, le_div):
    now = datetime.utcnow()
    ih, id_, ia = fair_probs(odds_h, odds_d, odds_a)
    imp_sum = ih+id_+ia
    row = {
        "AvgH":odds_h,"AvgD":odds_d,"AvgA":odds_a,
        "B365H":odds_h,"B365D":odds_d,"B365A":odds_a,
        "LogOddsH":np.log(max(odds_h,1.01)),"LogOddsD":np.log(max(odds_d,1.01)),
        "LogOddsA":np.log(max(odds_a,1.01)),
        "ImpH":implied(odds_h),"ImpD":implied(odds_d),"ImpA":implied(odds_a),
        "OddsMargin":imp_sum-1.0 if imp_sum else 0.0,
        "HomeOddsEdge":ih,"AwayOddsEdge":ia,
        "Year":now.year,"Month":now.month,"DayOfWeek":now.weekday(),
        "IsWeekend":int(now.weekday()>=5),
        "H2H_HomeWins_5":0,"H2H_Draws_5":0,"H2H_AwayWins_5":0,
    }
    for prefix, tid in [("Home",home_id),("Away",away_id)]:
        for w in (5,10,20):
            row[f"{prefix}FormPts_{w}"]=np.nan
            row[f"{prefix}FormGF_{w}"]=np.nan
            row[f"{prefix}FormGA_{w}"]=np.nan
            row[f"{prefix}FormGD_{w}"]=np.nan
        if hist_df is not None and len(hist_df):
            if prefix=="Home":
                sub = hist_df[hist_df["HomeTeamId"]==tid].sort_values("Date").tail(20)
                pts = sub["HomePoints"] if "HomePoints" in sub.columns else pd.Series(dtype=float)
                gf = sub["FTHG"] if "FTHG" in sub.columns else pd.Series(dtype=float)
                ga = sub["FTAG"] if "FTAG" in sub.columns else pd.Series(dtype=float)
            else:
                sub = hist_df[hist_df["AwayTeamId"]==tid].sort_values("Date").tail(20)
                pts = sub["AwayPoints"] if "AwayPoints" in sub.columns else pd.Series(dtype=float)
                gf = sub["FTAG"] if "FTAG" in sub.columns else pd.Series(dtype=float)
                ga = sub["FTHG"] if "FTHG" in sub.columns else pd.Series(dtype=float)
            for w in (5,10,20):
                row[f"{prefix}FormPts_{w}"] = float(pts.tail(w).mean()) if len(pts) else np.nan
                row[f"{prefix}FormGF_{w}"] = float(gf.tail(w).mean()) if len(gf) else np.nan
                row[f"{prefix}FormGA_{w}"] = float(ga.tail(w).mean()) if len(ga) else np.nan
                if np.isfinite(row[f"{prefix}FormGF_{w}"]) and np.isfinite(row[f"{prefix}FormGA_{w}"]):
                    row[f"{prefix}FormGD_{w}"] = row[f"{prefix}FormGF_{w}"]-row[f"{prefix}FormGA_{w}"]
    vals = []
    for name in feature_names:
        if name=="HomeTeamId": vals.append(home_id)
        elif name=="AwayTeamId": vals.append(away_id)
        elif name=="DivEnc":
            try: vals.append(int(le_div.transform([div_code])[0]))
            except Exception: vals.append(0)
        else: vals.append(row.get(name, 0.0))
    return np.nan_to_num(np.array(vals,dtype=np.float64).reshape(1,-1), nan=0.0)

def predict_backends(target_key, X_scaled, targets):
    paths = targets.get(target_key, {})
    if not isinstance(paths, dict) or "error" in paths: return None
    preds = []
    if "xgboost" in paths and Path(paths["xgboost"]).exists():
        try:
            import xgboost as xgb
            m = xgb.Booster(); m.load_model(paths["xgboost"])
            p = np.asarray(m.predict(xgb.DMatrix(X_scaled)))
            preds.append(np.array([1-p[0],p[0]]) if p.ndim==1 else p[0])
        except Exception as e: print(f"  [warn] xgb {target_key}: {e}")
    if "lightgbm" in paths and Path(paths["lightgbm"]).exists():
        try:
            import lightgbm as lgb
            m = lgb.Booster(model_file=paths["lightgbm"])
            p = np.asarray(m.predict(X_scaled))
            preds.append(np.array([1-p[0],p[0]]) if p.ndim==1 else p[0])
        except Exception as e: print(f"  [warn] lgbm {target_key}: {e}")
    if "catboost" in paths and Path(paths["catboost"]).exists():
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(); m.load_model(paths["catboost"])
            preds.append(m.predict_proba(X_scaled)[0])
        except Exception as e: print(f"  [warn] cat {target_key}: {e}")
    if "adaboost" in paths and Path(paths["adaboost"]).exists():
        try:
            with open(paths["adaboost"],"rb") as f: m = pickle.load(f)
            p = m.predict_proba(X_scaled)[0]
            if np.isfinite(p).all(): preds.append(p)
        except Exception as e: print(f"  [warn] ada {target_key}: {e}")
    if not preds: return None
    max_len = max(len(p) for p in preds)
    aligned = [np.pad(np.asarray(p,float).ravel(),(0,max(0,max_len-len(p))))[:max_len] for p in preds]
    return np.mean(aligned, axis=0)

def predict_across_runs(target_key, runs, home_id, away_id, div_code, oh, od, oa, hist_df):
    preds = []
    for run_dir, label, targets, scaler, le_div, feature_names in runs:
        X = build_live_features(home_id,away_id,div_code,oh,od,oa,hist_df,feature_names,le_div)
        Xs = X.copy(); Xs[:,:-3]=scaler.transform(X[:,:-3]); Xs=np.nan_to_num(Xs,nan=0.0)
        p = predict_backends(target_key, Xs, targets)
        if p is not None:
            preds.append(np.asarray(p,float).ravel())
            print(f"  {target_key:12s}  [{label}]  -> {np.round(p,3)}")
    if not preds: return None
    max_len = max(len(p) for p in preds)
    aligned = [np.pad(p,(0,max(0,max_len-len(p))))[:max_len] for p in preds]
    return np.mean(aligned, axis=0)

def multi_prob(p, n=3):
    if p is None: return np.ones(n)/n
    p = np.asarray(p,float).ravel()
    if len(p)<n: p = np.pad(p,(0,n-len(p)))
    p = np.clip(p[:n],1e-6,None); return p/p.sum()

def bin_prob(p):
    if p is None: return 0.5
    p = np.asarray(p).ravel()
    return float(p[-1]) if len(p)<=2 else float(p[1])

def blend_with_odds(p_model, p_market, alpha):
    """alpha = weight on market (0..1)."""
    p_model = np.asarray(p_model,float); p_market = np.asarray(p_market,float)
    n = max(len(p_model), len(p_market))
    a = np.pad(p_model,(0,max(0,n-len(p_model))))[:n]
    b = np.pad(p_market,(0,max(0,n-len(p_market))))[:n]
    a = a/a.sum(); b = b/b.sum()
    out = (1-alpha)*a + alpha*b
    return out/out.sum()

# =============================================================================
# Simulation
# =============================================================================
def simulate_match(p_ft, p_ht, p_over25, p_btts, p_dc1x, p_dcx2, p_dc12, n, seed):
    rng = np.random.default_rng(seed)
    p_ft = p_ft/p_ft.sum()
    ft_outcomes = rng.choice(["H","D","A"], size=n, p=p_ft)
    scores = []
    for outcome in ft_outcomes:
        over, btts = rng.random()<p_over25, rng.random()<p_btts
        if outcome=="H":
            if over and btts:
                h,a = int(rng.choice([2,3,4])), int(rng.choice([1,2])); h=max(h,a+1)
            elif over: h,a = int(rng.choice([3,4,5])), 0
            elif btts: h,a = int(rng.choice([1,2])),1; h=max(h,a+1)
            else: h,a = int(rng.choice([1,2])),0
        elif outcome=="A":
            if over and btts:
                a,h = int(rng.choice([2,3,4])), int(rng.choice([1,2])); a=max(a,h+1)
            elif over: a,h = int(rng.choice([3,4,5])), 0
            elif btts: a,h = int(rng.choice([1,2])),1; a=max(a,h+1)
            else: a,h = int(rng.choice([1,2])),0
        else:
            if over and btts: h=a=int(rng.choice([2,3]))
            elif btts: h=a=1
            else: h=a=0
        scores.append((h,a))
    scores = np.array(scores)
    if p_ht is None or len(p_ht)<3: p_ht = np.array([0.35,0.30,0.35])
    else: p_ht = p_ht/p_ht.sum()
    top = Counter(map(tuple, scores)).most_common(10)
    # HT score proxy from HT probs + goals intensity
    ht_scores = []
    for _ in range(n):
        o = rng.choice(["H","D","A"], p=p_ht)
        if o=="H": ht_scores.append((1,0) if rng.random()>0.3 else (2,0) if rng.random()>0.5 else (2,1))
        elif o=="A": ht_scores.append((0,1) if rng.random()>0.3 else (0,2) if rng.random()>0.5 else (1,2))
        else: ht_scores.append((0,0) if rng.random()>0.4 else (1,1))
    ht_scores = np.array(ht_scores)
    return {
        "n_simulations": n,
        "ft_probs_model": {"H":float(p_ft[0]),"D":float(p_ft[1]),"A":float(p_ft[2])},
        "ft_probs_simulated": {
            "H":float((scores[:,0]>scores[:,1]).mean()),
            "D":float((scores[:,0]==scores[:,1]).mean()),
            "A":float((scores[:,0]<scores[:,1]).mean()),
        },
        "ht_probs_model": {"H":float(p_ht[0]),"D":float(p_ht[1]),"A":float(p_ht[2])},
        "ht_probs_simulated": {
            "H":float((ht_scores[:,0]>ht_scores[:,1]).mean()),
            "D":float((ht_scores[:,0]==ht_scores[:,1]).mean()),
            "A":float((ht_scores[:,0]<ht_scores[:,1]).mean()),
        },
        "over25_model": float(p_over25),
        "over25_simulated": float((scores.sum(1)>2.5).mean()),
        "btts_model": float(p_btts),
        "btts_simulated": float(((scores[:,0]>0)&(scores[:,1]>0)).mean()),
        "double_chance": {"1X":float(p_dc1x),"X2":float(p_dcx2),"12":float(p_dc12)},
        "top_scorelines": [(f"{h}-{a}",int(c)) for (h,a),c in top],
        "top_ht_scorelines": [(f"{h}-{a}",int(c)) for (h,a),c in Counter(map(tuple,ht_scores)).most_common(5)],
        "expected_goals": {
            "home":float(scores[:,0].mean()),"away":float(scores[:,1].mean()),
            "total":float(scores.sum(1).mean()),
        },
        "score_stats": {
            "home_clean_sheet": float((scores[:,1]==0).mean()),
            "away_clean_sheet": float((scores[:,0]==0).mean()),
            "home_win_to_nil": float(((scores[:,0]>scores[:,1])&(scores[:,1]==0)).mean()),
            "away_win_to_nil": float(((scores[:,1]>scores[:,0])&(scores[:,0]==0)).mean()),
            "over_1_5": float((scores.sum(1)>1.5).mean()),
            "over_3_5": float((scores.sum(1)>3.5).mean()),
            "under_2_5": float((scores.sum(1)<2.5).mean()),
        },
    }

# =============================================================================
# Judgement table
# =============================================================================
def stars(edge_pp):
    """edge in percentage points (model - market)."""
    a = abs(edge_pp)
    if a >= 12: return "*****"
    if a >= 8: return "****"
    if a >= 5: return "***"
    if a >= 3: return "**"
    if a >= 1.5: return "*"
    return "-"

def judgement_table(report, oh, od, oa, home, away):
    mh, md, ma = fair_probs(oh, od, oa)
    ft = report["ft_probs_model"]
    rows = []

    def add(market, selection, model_p, market_p, odds=None):
        edge = (model_p - market_p) * 100
        # value if model > market by meaningful amount
        if edge >= 5: verdict = "STRONG VALUE"
        elif edge >= 2.5: verdict = "VALUE"
        elif edge >= 1: verdict = "SLIGHT VALUE"
        elif edge <= -5: verdict = "FADE"
        elif edge <= -2.5: verdict = "LEAN AGAINST"
        else: verdict = "NEUTRAL"
        rows.append({
            "Market": market,
            "Selection": selection,
            "Model%": round(model_p*100, 1),
            "Market%": round(market_p*100, 1),
            "Edge_pp": round(edge, 1),
            "Odds": odds,
            "Stars": stars(edge),
            "Verdict": verdict,
        })

    add("FT 1X2", f"{home} Win", ft["H"], mh, oh)
    add("FT 1X2", "Draw", ft["D"], md, od)
    add("FT 1X2", f"{away} Win", ft["A"], ma, oa)

    # DC from model (or derived)
    dc = report["double_chance"]
    add("Double Chance", "1X (Home/Draw)", dc["1X"], mh+md)
    add("Double Chance", "X2 (Draw/Away)", dc["X2"], md+ma)
    add("Double Chance", "12 (Home/Away)", dc["12"], mh+ma)

    # Goals — market O/U not always given; use model only vs 50% baseline if no line
    # assume fair O2.5 ~ from typical; without OU odds use 50% as neutral
    add("Goals", "Over 2.5", report["over25_model"], 0.50)
    add("Goals", "Under 2.5", 1-report["over25_model"], 0.50)
    add("Goals", "BTTS Yes", report["btts_model"], 0.50)
    add("Goals", "BTTS No", 1-report["btts_model"], 0.50)

    ht = report["ht_probs_model"]
    # no HT market odds — compare to FT shape / flat
    add("HT Result", f"{home} HT", ht["H"], 0.33)
    add("HT Result", "HT Draw", ht["D"], 0.34)
    add("HT Result", f"{away} HT", ht["A"], 0.33)

    return rows

def print_judgement_table(rows):
    print("\n" + "=" * 88)
    print("  PRECISE JUDGEMENT TABLE  (Model vs Market)")
    print("=" * 88)
    hdr = f"{'Market':<16} {'Selection':<22} {'Model%':>7} {'Mkt%':>7} {'Edge':>7} {'Odds':>7} {'Stars':>6}  Verdict"
    print(hdr)
    print("-" * 88)
    for r in rows:
        odds_s = f"{r['Odds']:.2f}" if r["Odds"] is not None else "  -  "
        edge_s = f"{r['Edge_pp']:+.1f}"
        print(f"{r['Market']:<16} {r['Selection']:<22} {r['Model%']:6.1f}% {r['Market%']:6.1f}% "
              f"{edge_s:>7} {odds_s:>7} {r['Stars']:>6}  {r['Verdict']}")
    print("=" * 88)
    # Top picks
    ranked = sorted(rows, key=lambda x: -x["Edge_pp"])
    print("\n  TOP MODEL LEANS (by edge)")
    for r in ranked[:5]:
        if r["Edge_pp"] > 0:
            print(f"    • {r['Market']} | {r['Selection']}  edge {r['Edge_pp']:+.1f}pp  {r['Stars']}  → {r['Verdict']}")
    fades = [r for r in ranked if r["Edge_pp"] < -2.5]
    if fades:
        print("\n  FADE / LEAN AGAINST")
        for r in fades[:3]:
            print(f"    • {r['Market']} | {r['Selection']}  edge {r['Edge_pp']:+.1f}pp  → {r['Verdict']}")
    print()

def print_report(league_name, home, away, report, odds, model_source, blend_alpha):
    oh,od,oa = odds
    print("\n" + "=" * 72)
    print("  LIVE MATCH SIMULATION REPORT (ADVANCED)")
    print("=" * 72)
    print(f"  League      : {league_name}")
    print(f"  Match       : {home}  vs  {away}")
    print(f"  1X2 odds    : H {oh:.2f}  |  D {od:.2f}  |  A {oa:.2f}")
    print(f"  Models used : {model_source}")
    print(f"  Odds blend  : {blend_alpha:.0%} market / {1-blend_alpha:.0%} model")
    print(f"  Simulations : {report['n_simulations']:,}")
    print("-" * 72)
    m,s = report["ft_probs_model"], report["ft_probs_simulated"]
    print(f"\n  FULL TIME")
    print(f"    Model  →  H {m['H']:.1%}  D {m['D']:.1%}  A {m['A']:.1%}")
    print(f"    Sim    →  H {s['H']:.1%}  D {s['D']:.1%}  A {s['A']:.1%}")
    mh,md,ma = fair_probs(oh,od,oa)
    print(f"    Market →  H {mh:.1%}  D {md:.1%}  A {ma:.1%}")
    ht,hts = report["ht_probs_model"], report["ht_probs_simulated"]
    print(f"\n  HALF TIME")
    print(f"    Model  →  H {ht['H']:.1%}  D {ht['D']:.1%}  A {ht['A']:.1%}")
    print(f"    Sim    →  H {hts['H']:.1%}  D {hts['D']:.1%}  A {hts['A']:.1%}")
    print(f"\n  GOALS")
    print(f"    Over 2.5  model {report['over25_model']:.1%}  sim {report['over25_simulated']:.1%}")
    print(f"    BTTS      model {report['btts_model']:.1%}  sim {report['btts_simulated']:.1%}")
    st = report["score_stats"]
    print(f"    Over 1.5 {st['over_1_5']:.1%}   Over 3.5 {st['over_3_5']:.1%}   Under 2.5 {st['under_2_5']:.1%}")
    print(f"    Clean sheet home {st['home_clean_sheet']:.1%}  away {st['away_clean_sheet']:.1%}")
    print(f"    Win to nil home {st['home_win_to_nil']:.1%}  away {st['away_win_to_nil']:.1%}")
    eg = report["expected_goals"]
    print(f"    Exp goals home {eg['home']:.2f}  away {eg['away']:.2f}  total {eg['total']:.2f}")
    dc = report["double_chance"]
    print(f"\n  DOUBLE CHANCE  1X {dc['1X']:.1%}  X2 {dc['X2']:.1%}  12 {dc['12']:.1%}")
    print("\n  TOP FT SCORELINES")
    for sc,cnt in report["top_scorelines"][:8]:
        print(f"    {sc:>5}  {cnt:5d}  ({100*cnt/report['n_simulations']:5.1f}%)")
    print("\n  TOP HT SCORELINES")
    for sc,cnt in report["top_ht_scorelines"]:
        print(f"    {sc:>5}  {cnt:5d}  ({100*cnt/report['n_simulations']:5.1f}%)")
    # narrative
    print("\n  NARRATIVE")
    fav = max(m, key=m.get)
    if fav=="H": print(f"    Pre-match lean: {home} ({m['H']:.0%}).")
    elif fav=="A": print(f"    Pre-match lean: {away} ({m['A']:.0%}).")
    else: print("    Pre-match lean: draw territory.")
    if report["over25_model"]>0.55: print("    Goals: open game expected (Over lean).")
    elif report["over25_model"]<0.42: print("    Goals: tight / low-scoring lean.")
    else: print("    Goals: balanced total.")
    if report["btts_model"]>0.55: print("    Both teams likely on the scoresheet.")
    print("=" * 72)

def main():
    print(f"ROOT = {ROOT}")
    if not MAP_DIR.exists():
        raise SystemExit(f"Mappings not found: {MAP_DIR}")
    team2id = load_json(MAP_DIR/"team2id.json")
    league_map = load_json(MAP_DIR/"league_map.json")
    aliases = load_json(MAP_DIR/"team_aliases.json")
    div_code, league_name = resolve_league(MATCH["league"], league_map)
    home_canon = normalize_team(MATCH["home_team"], aliases)
    away_canon = normalize_team(MATCH["away_team"], aliases)
    if home_canon not in team2id: raise SystemExit(f"Home not in map: {home_canon}")
    if away_canon not in team2id: raise SystemExit(f"Away not in map: {away_canon}")
    home_id, away_id = team2id[home_canon], team2id[away_canon]
    print(f"Resolved: {league_name} ({div_code}) | {home_canon} vs {away_canon}")

    run_dirs = collect_run_dirs(home_canon, away_canon)
    model_source = " + ".join(l for _,l in run_dirs)
    print(f"Using models: {model_source}")
    runs_loaded = []
    for run_dir, label in run_dirs:
        try:
            targets = load_run_targets(run_dir)
            feature_names, scaler, le_div = load_preprocessors(run_dir)
            runs_loaded.append((run_dir, label, targets, scaler, le_div, feature_names))
            print(f"  loaded {label}")
        except Exception as e:
            print(f"  SKIP {label}: {e}")
    if not runs_loaded: raise SystemExit("No usable runs")

    hist_df = None
    if PARQUET_PATH.exists():
        try:
            hist_df = pd.read_parquet(PARQUET_PATH, columns=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","Div"])
            hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors="coerce")
            hist_df["HomeTeamCanon"] = hist_df["HomeTeam"].map(lambda x: normalize_team(x, aliases))
            hist_df["AwayTeamCanon"] = hist_df["AwayTeam"].map(lambda x: normalize_team(x, aliases))
            hist_df["HomeTeamId"] = hist_df["HomeTeamCanon"].map(team2id)
            hist_df["AwayTeamId"] = hist_df["AwayTeamCanon"].map(team2id)
            hist_df["HomePoints"] = hist_df["FTR"].map({"H":3,"D":1,"A":0})
            hist_df["AwayPoints"] = hist_df["FTR"].map({"H":0,"D":1,"A":3})
        except Exception as e:
            print(f"History warning: {e}")

    oh,od,oa = MATCH["odds_home"], MATCH["odds_draw"], MATCH["odds_away"]
    alpha = float(MATCH.get("odds_blend", 0.35))
    print("\nPer-run predictions:")
    raw_ft = predict_across_runs("ft_result", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
    raw_ht = predict_across_runs("ht_result", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
    raw_over = predict_across_runs("over25", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
    raw_btts = predict_across_runs("btts", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
    raw_1x = predict_across_runs("dc_1x", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
    raw_x2 = predict_across_runs("dc_x2", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)
    raw_12 = predict_across_runs("dc_12", runs_loaded, home_id, away_id, div_code, oh, od, oa, hist_df)

    p_ft = multi_prob(raw_ft, 3)
    p_market = np.array(fair_probs(oh, od, oa))
    p_ft = blend_with_odds(p_ft, p_market, alpha)
    p_ht = multi_prob(raw_ht, 3)
    p_over = bin_prob(raw_over)
    p_btts = bin_prob(raw_btts)
    # DC: blend with market-derived
    p_1x = (1-alpha)*bin_prob(raw_1x) + alpha*(p_market[0]+p_market[1])
    p_x2 = (1-alpha)*bin_prob(raw_x2) + alpha*(p_market[1]+p_market[2])
    p_12 = (1-alpha)*bin_prob(raw_12) + alpha*(p_market[0]+p_market[2])

    report = simulate_match(p_ft, p_ht, p_over, p_btts, p_1x, p_x2, p_12,
                            MATCH["n_simulations"], MATCH["seed"])
    print_report(league_name, home_canon, away_canon, report, (oh,od,oa), model_source, alpha)
    rows = judgement_table(report, oh, od, oa, home_canon, away_canon)
    print_judgement_table(rows)

    out = ROOT / "last_simulation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "match": MATCH,
            "resolved": {"league":league_name,"div":div_code,"home":home_canon,"away":away_canon,"models":model_source},
            "report": report,
            "judgement_table": rows,
            "generated_at": datetime.utcnow().isoformat()+"Z",
        }, f, indent=2)
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()
