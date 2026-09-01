#!/usr/bin/env python3
"""High-rigor football score simulation engine.

Calibrated bivariate-Poisson scoring + Dixon-Coles low-score correction,
latent match-state uncertainty and bounded goal-rate over-dispersion.
The model probabilities are calibration targets, not hard quotas.
"""
from __future__ import annotations
from collections import Counter
from math import exp, factorial
import numpy as np
try:
    from scipy.optimize import minimize, brentq
except Exception:
    minimize = None
    brentq = None


def _norm(p):
    p = np.asarray(p, dtype=float).ravel()
    p = np.nan_to_num(p, nan=1/max(len(p), 1), posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


def _pois(k, lam):
    return exp(-lam) * (lam ** int(k)) / factorial(int(k))


def _joint_grid(lh, la, common=0.0, tau=0.055, max_g=10):
    pmf = np.zeros((max_g + 1, max_g + 1), dtype=float)
    zmax = min(max_g, int(max(8, common * 8 + 6)))
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            s = sum(_pois(z, common) * _pois(h-z, lh) * _pois(a-z, la)
                    for z in range(min(h, a, zmax) + 1))
            if h == 0 and a == 0:
                s *= max(1.0 - lh * la * tau, 0.05)
            elif h == 0 and a == 1:
                s *= 1.0 + lh * tau
            elif h == 1 and a == 0:
                s *= 1.0 + la * tau
            elif h == 1 and a == 1:
                s *= max(1.0 - tau, 0.05)
            pmf[h, a] = s
    return pmf / max(pmf.sum(), 1e-15)


def _stats(pmf):
    h, a = np.indices(pmf.shape)
    out = np.where(h > a, 0, np.where(h < a, 2, 1))
    ft = np.array([(pmf[out == k]).sum() for k in range(3)])
    total = h + a
    return ft, float(pmf[total >= 3].sum()), float(pmf[(h > 0) & (a > 0)].sum())


def _solve_rates(p_ft, p_over, p_btts):
    p_ft = _norm(p_ft)[:3]
    p_over = float(np.clip(p_over, 0.08, 0.92))
    p_btts = float(np.clip(p_btts, 0.08, 0.92))
    def over_for_total(total):
        return 1.0 - exp(-total) * (1.0 + total + total*total/2.0)
    if brentq is not None:
        total = brentq(lambda x: over_for_total(x) - p_over, 0.55, 5.5)
    else:
        total = float(np.clip(2.4 + 3.0*(p_over - 0.5), 0.55, 5.5))
    def objective(x):
        share = 1.0 / (1.0 + np.exp(-x[0]))
        common = min(total * (0.18/(1.0 + np.exp(-x[1]))), 0.45)
        base = max(total-common, 0.40)
        lh = np.clip(base*share, 0.15, 4.8)
        la = np.clip(base-lh, 0.15, 4.8)
        ft, ov, bt = _stats(_joint_grid(lh, la, common))
        return float(8*np.sum((ft-p_ft)**2) + 2.5*(ov-p_over)**2 + 1.5*(bt-p_btts)**2)
    x = np.array([0.0, -1.5])
    if minimize is not None:
        try:
            x = minimize(objective, x, method="Nelder-Mead",
                         options={"maxiter": 180, "xatol": 1e-5, "fatol": 1e-7}).x
        except Exception:
            pass
    share = 1.0/(1.0+np.exp(-x[0]))
    common = min(total*(0.18/(1.0+np.exp(-x[1]))), 0.45)
    base = max(total-common, 0.40)
    lh = float(np.clip(base*share, 0.15, 4.8))
    la = float(np.clip(base-lh, 0.15, 4.8))
    return lh, la, float(common), float(total)


def _sample_joint(rng, lh, la, common, n, hardness):
    cv_h = float(np.clip(hardness.get("attack_cv", 0.13), 0.04, 0.30))
    cv_a = float(np.clip(hardness.get("defense_cv", 0.13), 0.04, 0.30))
    sh = 1.0/(cv_h*cv_h); sa = 1.0/(cv_a*cv_a)
    state_h = rng.gamma(sh, 1.0/sh, size=n)
    state_a = rng.gamma(sa, 1.0/sa, size=n)
    shared_cv = float(np.clip(hardness.get("shared_cv", 0.07), 0.0, 0.20))
    if shared_cv:
        ss = 1.0/(shared_cv*shared_cv)
        shared = rng.gamma(ss, 1.0/ss, size=n)
    else:
        shared = np.ones(n)
    z = rng.poisson(np.clip(common*shared, 0, 2.0))
    h = z + rng.poisson(np.clip(lh*state_h*shared, 0, 8.0))
    a = z + rng.poisson(np.clip(la*state_a*shared, 0, 8.0))
    return np.column_stack([h, a]).astype(int)


def simulate_hard(p_ft, p_ht, p_over25, p_btts, p_ht_over15, n, seed,
                  standings_prior=None, hardness=None):
    hardness = {"attack_cv":0.13, "defense_cv":0.13, "shared_cv":0.07,
                "dc_tau":0.055, **(hardness or {})}
    rng = np.random.default_rng(int(seed))
    p_ft = _norm(p_ft)[:3]
    p_ht = _norm(p_ht if p_ht is not None else [0.30,0.40,0.30])[:3]
    p_over25 = float(np.clip(p_over25, 0.08, 0.92))
    p_btts = float(np.clip(p_btts, 0.08, 0.92))
    p_ht_over15 = float(np.clip(p_ht_over15 if p_ht_over15 is not None else 0.30, 0.08, 0.92))
    sp = standings_prior or {}
    if sp.get("matched"):
        try:
            from standings_prior import apply_ft_tilt, apply_scalar_tilt
            p_ft = apply_ft_tilt(p_ft, sp.get("ft_tilt") or {})
            p_over25 = apply_scalar_tilt(p_over25, float(sp.get("over25_tilt") or 0.0))
            p_btts = apply_scalar_tilt(p_btts, float(sp.get("btts_tilt") or 0.0))
        except Exception:
            pass
    lh, la, common, total_target = _solve_rates(p_ft, p_over25, p_btts)
    scores = _sample_joint(rng, lh, la, common, int(n), hardness)
    ht_scores = np.zeros_like(scores)
    ht_share = float(np.clip(0.43 + 0.10*(p_ht_over15-0.5), 0.30, 0.56))
    for i, (hg, ag) in enumerate(scores):
        if hg:
            ht_scores[i,0] = rng.binomial(int(hg), np.clip(ht_share + 0.03*(p_ht[0]-p_ht[2]), 0.20, 0.65))
        if ag:
            ht_scores[i,1] = rng.binomial(int(ag), np.clip(ht_share + 0.03*(p_ht[2]-p_ht[0]), 0.20, 0.65))
    tot = scores.sum(axis=1); diff = scores[:,0]-scores[:,1]
    ft_h=float(np.mean(diff>0)); ft_d=float(np.mean(diff==0)); ft_a=float(np.mean(diff<0))
    ht_h=float(np.mean(ht_scores[:,0]>ht_scores[:,1])); ht_d=float(np.mean(ht_scores[:,0]==ht_scores[:,1])); ht_a=float(np.mean(ht_scores[:,0]<ht_scores[:,1]))
    top=Counter(map(tuple,scores.tolist())).most_common(12); top_ht=Counter(map(tuple,ht_scores.tolist())).most_common(8)
    def top_pct(items,k):
        return [{"score":f"{h}-{a}","count":int(c),"pct":round(100*c/max(len(scores),1),1)} for (h,a),c in items[:k]]
    def line(x):
        return {"over":float(np.mean(tot>x)),"under":float(np.mean(tot<x)),"push":float(np.mean(tot==x))}
    return {
      "n":int(n), "engine":{"version":"hard-v2","method":"calibrated-bivariate-poisson+dixon-coles+latent-state","attack_cv":hardness["attack_cv"],"defense_cv":hardness["defense_cv"],"shared_state_cv":hardness["shared_cv"],"common_goal_rate":common,"lambda_home":lh,"lambda_away":la,"target_total_lambda":total_target},
      "ft_model":{"H":float(p_ft[0]),"D":float(p_ft[1]),"A":float(p_ft[2])}, "ft_sim":{"H":ft_h,"D":ft_d,"A":ft_a},
      "ht_model":{"H":float(p_ht[0]),"D":float(p_ht[1]),"A":float(p_ht[2])}, "ht_sim":{"H":ht_h,"D":ht_d,"A":ht_a},
      "over25_model":p_over25,"over25_sim":float(np.mean(tot>=3)),"btts_model":p_btts,"btts_sim":float(np.mean((scores[:,0]>0)&(scores[:,1]>0))),
      "ht_over15_model":p_ht_over15,"ht_over15_sim":float(np.mean(ht_scores.sum(axis=1)>=2)),
      "dc_ft":{"1X":ft_h+ft_d,"X2":ft_d+ft_a,"12":ft_h+ft_a}, "dc_ht":{"1X":ht_h+ht_d,"X2":ht_d+ht_a,"12":ht_h+ht_a},
      "top_ft":[(f"{h}-{a}",int(c)) for (h,a),c in top], "top_ht":[(f"{h}-{a}",int(c)) for (h,a),c in top_ht],
      "top3_ft":top_pct(top,3),"top3_ht":top_pct(top_ht,3),"top8_ft":top_pct(top,8),
      "xg":{"home":float(scores[:,0].mean()),"away":float(scores[:,1].mean()),"total":float(tot.mean()),"lambda_home":lh,"lambda_away":la},
      "ht_xg":{"home":float(ht_scores[:,0].mean()),"away":float(ht_scores[:,1].mean()),"total":float(ht_scores.sum(axis=1).mean())},
      "score_consistency":{"ht_leq_ft":float(np.mean((ht_scores[:,0]<=scores[:,0])&(ht_scores[:,1]<=scores[:,1]))),"ft_vs_model_l1":float(np.abs(np.array([ft_h-p_ft[0],ft_d-p_ft[1],ft_a-p_ft[2]])).sum())},
      "clean_sheet":{"home":float(np.mean(scores[:,1]==0)),"away":float(np.mean(scores[:,0]==0))},
      "win_to_nil":{"home":float(np.mean((scores[:,0]>scores[:,1])&(scores[:,1]==0))),"away":float(np.mean((scores[:,1]>scores[:,0])&(scores[:,0]==0)))},
      "goal_lines":{str(x):line(x) for x in (0.5,1.5,2.5,3.5,4.5)},
      "asian_handicap":{"home_-0.5":float(np.mean(diff>0)),"home_-1.0":{"cover":float(np.mean(diff>1)),"push":float(np.mean(diff==1))},"home_+0.5":float(np.mean(diff>-1)),"home_+1.0":{"cover":float(np.mean(diff>-1)),"push":float(np.mean(diff==-1))}},
      "score_matrix_margin":{"home_1_goal":float(np.mean(diff==1)),"home_2_plus":float(np.mean(diff>=2)),"away_1_goal":float(np.mean(diff==-1)),"away_2_plus":float(np.mean(diff<=-2)),"draw":ft_d},
    }
