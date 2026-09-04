#!/usr/bin/env python3
"""Precision football score engine.

Deterministic, low-noise score-distribution engine shared by normal and Africa
simulation paths.  It does not manufacture Monte-Carlo noise for the published
probabilities: primary probabilities are computed from an explicitly fitted
joint score distribution.  Random sampling is optional and only used to expose
legacy-compatible nominal counts.
"""
from __future__ import annotations

from collections import Counter
from math import exp, lgamma
from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

ENGINE_VERSION = "precision-v5"
HDA = ("H", "D", "A")
EPS = 1e-12


def _norm(p, k=3):
    a = np.asarray(p if p is not None else [], dtype=float).ravel()
    if a.size < k:
        a = np.pad(a, (0, k - a.size), constant_values=1.0 / k)
    a = np.nan_to_num(a[:k], nan=1.0 / k, posinf=0.0, neginf=0.0)
    a = np.clip(a, EPS, None)
    return a / a.sum()


def _clip01(x, lo=0.02, hi=0.98):
    try:
        return float(np.clip(float(x), lo, hi))
    except Exception:
        return (lo + hi) / 2.0


def _pois_vec(lam, max_goals):
    k = np.arange(max_goals + 1, dtype=float)
    logp = -float(lam) + k * np.log(max(float(lam), EPS)) - np.array([lgamma(x + 1.0) for x in k])
    return np.exp(logp)


def _dc_grid(lh, la, common=0.0, rho=-0.055, max_goals=12):
    """Normalized bivariate-Poisson + conservative Dixon-Coles grid."""
    ph = _pois_vec(lh, max_goals)
    pa = _pois_vec(la, max_goals)
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    zmax = min(max_goals, max(0, int(common * 10.0 + 8)))
    for z in range(zmax + 1):
        hz = np.arange(max_goals + 1) - z
        az = np.arange(max_goals + 1) - z
        vh = hz >= 0
        va = az >= 0
        if not vh.any() or not va.any():
            continue
        hzv = hz[vh].astype(int)
        azv = az[va].astype(int)
        common_p = exp(-common) * (max(common, EPS) ** z) / exp(lgamma(z + 1.0)) if common > 0 else (1.0 if z == 0 else 0.0)
        if common_p <= 0:
            continue
        grid[np.ix_(np.where(vh)[0], np.where(va)[0])] += common_p * ph[hzv, None] * pa[None, azv]
    # Standard low-score DC structure, bounded so malformed rates cannot create negatives.
    grid[0, 0] *= max(0.05, 1.0 - lh * la * rho)
    grid[0, 1] *= max(0.05, 1.0 + lh * rho)
    grid[1, 0] *= max(0.05, 1.0 + la * rho)
    grid[1, 1] *= max(0.05, 1.0 - rho)
    grid = np.clip(grid, 0.0, None)
    s = float(grid.sum())
    return grid / max(s, EPS)


def _stats(g):
    h, a = np.indices(g.shape)
    total = h + a
    ft = np.array([g[h > a].sum(), g[h == a].sum(), g[h < a].sum()], dtype=float)
    return {
        "ft": ft,
        "over25": float(g[total >= 3].sum()),
        "btts": float(g[(h > 0) & (a > 0)].sum()),
        "xg_h": float((g * h).sum()),
        "xg_a": float((g * a).sum()),
        "total": float((g * total).sum()),
    }


def _total_from_over25(p):
    """Invert P(Poisson(T)>=3)=p with a monotone bisection."""
    target = _clip01(p, 0.03, 0.97)
    lo, hi = 0.05, 7.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        over = 1.0 - exp(-mid) * (1.0 + mid + mid * mid / 2.0)
        if over < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _fit_ft(p_ft, p_over25, p_btts, max_goals=12):
    """Fit a coherent joint score distribution to all available aggregate targets.

    The fit is deliberately regularized: common-goal correlation and DC correction
    are weakly bounded rather than being allowed to chase one noisy target.
    """
    p_ft = _norm(p_ft)
    p_over25 = _clip01(p_over25)
    p_btts = _clip01(p_btts)
    target_total = _total_from_over25(p_over25)
    share0 = float(np.clip(p_ft[0] / max(p_ft[0] + p_ft[2], EPS), 0.12, 0.88))
    lh0 = max(0.12, target_total * share0 * 0.92)
    la0 = max(0.12, target_total * (1.0 - share0) * 0.92)

    def unpack(x):
        lh = float(np.exp(np.clip(x[0], -3.0, 1.8)))
        la = float(np.exp(np.clip(x[1], -3.0, 1.8)))
        common = float(0.38 * (1.0 / (1.0 + np.exp(-x[2]))))
        rho = float(0.16 * np.tanh(x[3]))
        return lh, la, common, rho

    def objective(x):
        lh, la, common, rho = unpack(x)
        g = _dc_grid(lh, la, common, rho, max_goals)
        s = _stats(g)
        # Match outcome, totals and BTTS; retain a conservative prior against
        # excessive shared-goal correlation or DC distortion.
        e = (
            24.0 * np.sum((s["ft"] - p_ft) ** 2)
            + 7.0 * (s["over25"] - p_over25) ** 2
            + 6.0 * (s["btts"] - p_btts) ** 2
            + 3.0 * ((s["total"] - target_total) / 2.5) ** 2
            + 0.20 * (common / 0.18) ** 2
            + 0.10 * (rho / 0.055) ** 2
        )
        return float(e)

    x0 = np.array([np.log(lh0), np.log(la0), -1.7, -0.35], dtype=float)
    best = x0
    if minimize is not None:
        starts = [
            x0,
            np.array([np.log(max(lh0, .2)), np.log(max(la0, .2)), -2.2, -0.7]),
            np.array([np.log(max(lh0, .2)), np.log(max(la0, .2)), -1.1, 0.0]),
        ]
        for start in starts:
            try:
                r = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 260, "xatol": 1e-6, "fatol": 1e-9})
                if np.isfinite(r.fun) and objective(best) > float(r.fun):
                    best = r.x
            except Exception:
                pass
    lh, la, common, rho = unpack(best)
    g = _dc_grid(lh, la, common, rho, max_goals)
    s = _stats(g)
    return g, {"lambda_home": lh, "lambda_away": la, "common_goal_rate": common, "rho": rho, "target_total_lambda": target_total, "fit": s}


def _binom_row(n, q):
    if n <= 0:
        return np.array([1.0])
    k = np.arange(n + 1)
    # recurrence avoids factorial overflow for large score grids
    out = np.empty(n + 1, dtype=float)
    out[0] = (1.0 - q) ** n
    for i in range(1, n + 1):
        out[i] = out[i - 1] * (n - i + 1) / i * q / max(1.0 - q, EPS)
    return out / max(out.sum(), EPS)


def _ht_distribution(ft_grid, p_ht=None, p_ht_over15=None):
    """Exact HT distribution conditional on FT scores; guarantees HT <= FT."""
    target = _norm(p_ht if p_ht is not None else [0.30, 0.40, 0.30])
    target_o = _clip01(p_ht_over15 if p_ht_over15 is not None else 0.30)

    def build(qh, qa):
        ht = np.zeros_like(ft_grid)
        for h in range(ft_grid.shape[0]):
            bh = _binom_row(h, qh)
            for a in range(ft_grid.shape[1]):
                w = ft_grid[h, a]
                if w <= 0:
                    continue
                ba = _binom_row(a, qa)
                for x, px in enumerate(bh):
                    for y, py in enumerate(ba):
                        ht[x, y] += w * px * py
        return ht / max(ht.sum(), EPS)

    def loss(x):
        qh = float(np.clip(1.0 / (1.0 + np.exp(-x[0])), 0.18, 0.62))
        qa = float(np.clip(1.0 / (1.0 + np.exp(-x[1])), 0.18, 0.62))
        htg = build(qh, qa)
        st = _stats(htg)
        return float(18.0 * np.sum((st["ft"] - target) ** 2) + 6.0 * (st["over25"] - target_o) ** 2 + 0.06 * ((qh - .43) ** 2 + (qa - .43) ** 2))

    x = np.array([np.log(.43 / .57), np.log(.43 / .57)], dtype=float)
    if minimize is not None:
        try:
            r = minimize(loss, x, method="Nelder-Mead", options={"maxiter": 160, "xatol": 1e-6, "fatol": 1e-9})
            if np.all(np.isfinite(r.x)):
                x = r.x
        except Exception:
            pass
    qh = float(np.clip(1.0 / (1.0 + np.exp(-x[0])), 0.18, 0.62))
    qa = float(np.clip(1.0 / (1.0 + np.exp(-x[1])), 0.18, 0.62))
    ht = build(qh, qa)
    return ht, {"home_share": qh, "away_share": qa, "target": target.tolist(), "target_over15": target_o}


def _line_probs(g, line):
    h, a = np.indices(g.shape)
    t = h + a
    return {"over": float(g[t > line].sum()), "under": float(g[t < line].sum()), "push": float(g[t == line].sum())}


def _top(g, n=12, nominal_n=150000):
    flat = [(float(g[h, a]), h, a) for h in range(g.shape[0]) for a in range(g.shape[1])]
    flat.sort(reverse=True)
    return [
        (f"{h}-{a}", int(round(p * nominal_n))) for p, h, a in flat[:n]
    ]


def _top_pct(g, n=8):
    flat = [(float(g[h, a]), h, a) for h in range(g.shape[0]) for a in range(g.shape[1])]
    flat.sort(reverse=True)
    return [{"score": f"{h}-{a}", "count": int(round(p * 100000)), "pct": round(100.0 * p, 2)} for p, h, a in flat[:n]]


def simulate_precision(
    p_ft,
    p_ht=None,
    p_over25=0.5,
    p_btts=0.5,
    p_ht_over15=0.30,
    n=150000,
    seed=42,
    standings_prior=None,
    odds_ft=None,
    odds_blend=0.0,
    max_goals=12,
):
    """Return a deterministic, coherent score report with legacy-compatible keys."""
    p_ft = _norm(p_ft)
    p_ht = _norm(p_ht if p_ht is not None else [0.30, 0.40, 0.30])
    po = _clip01(p_over25)
    pb = _clip01(p_btts)
    phto = _clip01(p_ht_over15)

    sp = standings_prior or {}
    if sp.get("matched"):
        try:
            from standings_prior import apply_ft_tilt, apply_scalar_tilt
            p_ft = _norm(apply_ft_tilt(p_ft, sp.get("ft_tilt") or {}))
            po = _clip01(apply_scalar_tilt(po, float(sp.get("over25_tilt") or 0.0)))
            pb = _clip01(apply_scalar_tilt(pb, float(sp.get("btts_tilt") or 0.0)))
        except Exception:
            pass

    if odds_ft and odds_blend > 0:
        try:
            oh, od, oa = [float(x) for x in odds_ft]
            if min(oh, od, oa) > 1.01:
                m = _norm([1.0 / oh, 1.0 / od, 1.0 / oa])
                a = float(np.clip(odds_blend, 0.0, 0.35))
                p_ft = _norm((1.0 - a) * p_ft + a * m)
        except Exception:
            pass

    g, fit = _fit_ft(p_ft, po, pb, max_goals=max_goals)
    ht, htfit = _ht_distribution(g, p_ht, phto)
    fs = _stats(g)
    hs = _stats(ht)
    h, a = np.indices(g.shape)
    diff = h - a
    hd, ad = np.indices(ht.shape)
    hdiff = hd - ad
    trunc = float(1.0 - g.sum())
    fit_l1 = float(np.abs(fs["ft"] - p_ft).sum())
    over_err = float(abs(fs["over25"] - po))
    btts_err = float(abs(fs["btts"] - pb))
    ht_l1 = float(np.abs(hs["ft"] - p_ht).sum())

    top = _top(g, 12, int(n))
    topht = _top(ht, 8, int(n))
    top3 = _top_pct(g, 3)
    top8 = _top_pct(g, 8)
    # Convert percentage diagnostic counts to the requested nominal simulation size.
    for item in top3 + top8:
        item["count"] = int(round(item["pct"] / 100.0 * int(n)))
    top3ht = _top_pct(ht, 3)
    for item in top3ht:
        item["count"] = int(round(item["pct"] / 100.0 * int(n)))

    ent = float(-(g[g > 0] * np.log(g[g > 0])).sum())
    return {
        "n": int(n),
        "engine": {
            "version": ENGINE_VERSION,
            "method": "deterministic-hierarchical-bivariate-poisson+dixon-coles+joint-target-fit+exact-HT-conditional",
            "probability_mode": "exact_distribution_no_monte_carlo_noise",
            "lambda_home": float(fit["lambda_home"]),
            "lambda_away": float(fit["lambda_away"]),
            "common_goal_rate": float(fit["common_goal_rate"]),
            "rho": float(fit["rho"]),
            "target_total_lambda": float(fit["target_total_lambda"]),
            "ht_home_share": float(htfit["home_share"]),
            "ht_away_share": float(htfit["away_share"]),
            "max_goals": int(max_goals),
        },
        "ft_model": {"H": float(p_ft[0]), "D": float(p_ft[1]), "A": float(p_ft[2])},
        "ft_sim": {"H": float(fs["ft"][0]), "D": float(fs["ft"][1]), "A": float(fs["ft"][2])},
        "ht_model": {"H": float(p_ht[0]), "D": float(p_ht[1]), "A": float(p_ht[2])},
        "ht_sim": {"H": float(hs["ft"][0]), "D": float(hs["ft"][1]), "A": float(hs["ft"][2])},
        "over25_model": po,
        "over25_sim": float(fs["over25"]),
        "btts_model": pb,
        "btts_sim": float(fs["btts"]),
        "ht_over15_model": phto,
        "ht_over15_sim": float(hs["over25"]),
        "dc_ft": {"1X": float(fs["ft"][0] + fs["ft"][1]), "X2": float(fs["ft"][1] + fs["ft"][2]), "12": float(fs["ft"][0] + fs["ft"][2])},
        "dc_ht": {"1X": float(hs["ft"][0] + hs["ft"][1]), "X2": float(hs["ft"][1] + hs["ft"][2]), "12": float(hs["ft"][0] + hs["ft"][2])},
        "top_ft": top,
        "top_ht": topht,
        "top3_ft": top3,
        "top3_ht": top3ht,
        "top8_ft": top8,
        "xg": {"home": float(fs["xg_h"]), "away": float(fs["xg_a"]), "total": float(fs["total"]), "lambda_home": float(fit["lambda_home"]), "lambda_away": float(fit["lambda_away"])},
        "ht_xg": {"home": float(hs["xg_h"]), "away": float(hs["xg_a"]), "total": float(hs["total"])},
        "score_consistency": {
            "ht_leq_ft": 1.0,
            "ft_vs_model_l1": fit_l1,
            "over25_abs_error": over_err,
            "btts_abs_error": btts_err,
            "ht_vs_model_l1": ht_l1,
            "truncation_mass": trunc,
            "entropy": ent,
        },
        "clean_sheet": {"home": float(g[:, 0].sum()), "away": float(g[0, :].sum())},
        "win_to_nil": {"home": float(g[(h > a) & (a == 0)].sum()), "away": float(g[(a > h) & (h == 0)].sum())},
        "goal_lines": {str(x): _line_probs(g, x) for x in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "asian_handicap": {
            "home_-0.5": float(g[diff > 0].sum()),
            "home_-1.0": {"cover": float(g[diff > 1].sum()), "push": float(g[diff == 1].sum())},
            "home_+0.5": float(g[diff > -1].sum()),
            "home_+1.0": {"cover": float(g[diff > -1].sum()), "push": float(g[diff == -1].sum())},
        },
        "score_matrix_margin": {
            "home_1_goal": float(g[diff == 1].sum()), "home_2_plus": float(g[diff >= 2].sum()),
            "away_1_goal": float(g[diff == -1].sum()), "away_2_plus": float(g[diff <= -2].sum()), "draw": float(fs["ft"][1]),
        },
    }


def simulate_industrial(*args, **kwargs):
    """Compatibility name used by the normal fixture pipeline."""
    return simulate_precision(*args, **kwargs)


def simulate_africa(*args, **kwargs):
    """Compatibility name used by the hardened Africa runtime."""
    return simulate_precision(*args, **kwargs)
