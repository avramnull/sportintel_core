#!/usr/bin/env python3
"""Industrial football score simulation engine — v4.

Stronger bivariate-Poisson + Dixon-Coles + multi-regime dispersion +
soft maximum-entropy calibration + correlated HT model.
Produces sharper scoreline mass and more decisive lock confidence.
"""
from __future__ import annotations
from collections import Counter
from math import exp, factorial
import numpy as np
try:
    from scipy.optimize import minimize, brentq
    from scipy.special import gammaln
except Exception:
    minimize = brentq = gammaln = None

ENGINE_VERSION = "industrial-v4"

def _norm(p):
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        return np.array([1/3, 1/3, 1/3])
    p = np.nan_to_num(p, nan=1/max(len(p), 1), posinf=0, neginf=0)
    p = np.clip(p, 1e-9, None)
    return p / p.sum()

def _pois(k, lam):
    return exp(-lam) * (lam ** int(k)) / factorial(int(k))

def _grid(lh, la, common, rho=0.052, max_g=10):
    """Bivariate Poisson grid with Dixon-Coles low-score correction."""
    pmf = np.zeros((max_g + 1, max_g + 1), float)
    if gammaln is None:
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                pmf[h, a] = sum(
                    _pois(z, common) * _pois(h - z, lh) * _pois(a - z, la)
                    for z in range(min(h, a) + 1)
                )
    else:
        h, a = np.indices(pmf.shape)
        for z in range(max_g + 1):
            valid = (h >= z) & (a >= z)
            hz = h - z
            az = a - z
            lp = -common + z * np.log(max(common, 1e-12)) - gammaln(z + 1)
            lp += -lh + hz * np.log(max(lh, 1e-12)) - gammaln(hz + 1)
            lp += -la + az * np.log(max(la, 1e-12)) - gammaln(az + 1)
            pmf += np.where(valid, np.exp(lp), 0.0)
    # Dixon-Coles adjustments (slightly softer than classic to keep mass mobile)
    pmf[0, 0] *= max(1 - lh * la * rho, 0.03)
    pmf[0, 1] *= max(1 + lh * rho * 0.95, 0.03)
    pmf[1, 0] *= max(1 + la * rho * 0.95, 0.03)
    pmf[1, 1] *= max(1 - rho * 0.85, 0.03)
    pmf = np.clip(pmf, 0, None)
    return pmf / max(pmf.sum(), 1e-15)

def _stats(p):
    h, a = np.indices(p.shape)
    out = np.where(h > a, 0, np.where(h < a, 2, 1))
    total = h + a
    ft = np.array([(p[out == k]).sum() for k in range(3)])
    return (
        ft,
        float(p[total >= 3].sum()),
        float(p[(h > 0) & (a > 0)].sum()),
        float((p * h).sum()),
        float((p * a).sum()),
    )

def _rates(pft, po, pb):
    """Solve goal rates so the joint matches FT / O2.5 / BTTS targets."""
    pft = _norm(pft)[:3]
    po = float(np.clip(po, 0.06, 0.94))
    pb = float(np.clip(pb, 0.06, 0.94))
    try:
        total = (
            brentq(lambda t: 1 - exp(-t) * (1 + t + t * t / 2) - po, 0.25, 5.8)
            if brentq
            else 2.35 + 2.8 * (po - 0.5)
        )
    except Exception:
        total = 2.35 + 2.8 * (po - 0.5)
    share = float(np.clip(0.5 + 0.50 * (pft[0] - pft[2]), 0.14, 0.86))
    x0 = np.array([
        np.log(max(total * share, 0.15)),
        np.log(max(total * (1 - share), 0.15)),
        np.log(0.20),
    ])

    def obj(x):
        lh, la = np.exp(x[:2])
        common = (1 / (1 + np.exp(-x[2]))) * min(lh, la) * 0.52
        s = _stats(_grid(lh, la, common))
        return (
            10 * np.sum((s[0] - pft) ** 2)
            + 3.5 * (s[1] - po) ** 2
            + 2.5 * (s[2] - pb) ** 2
            + 0.0008 * np.sum(x * x)
        )

    x = x0
    if minimize:
        try:
            r = minimize(
                obj, x0, method="Nelder-Mead",
                options={"maxiter": 120, "xatol": 1e-4, "fatol": 1e-7},
            )
            if np.isfinite(r.fun):
                x = r.x
        except Exception:
            pass
    lh, la = np.exp(x[:2])
    common = (1 / (1 + np.exp(-x[2]))) * min(lh, la) * 0.52
    return (
        float(np.clip(lh, 0.12, 5.2)),
        float(np.clip(la, 0.12, 5.2)),
        float(np.clip(common, 0, min(lh, la) * 0.48)),
        float(total),
    )

def _mixture(lh, la, common):
    """Five-regime dispersion mixture → sharper central mass + realistic tails."""
    q = np.zeros((11, 11))
    regimes = (
        (0.10, 0.68),
        (0.22, 0.85),
        (0.36, 1.00),
        (0.22, 1.18),
        (0.10, 1.40),
    )
    for w, s in regimes:
        q += w * _grid(lh * s, la * s, common * s)
    return q / q.sum()

def _calibrate(p, pft, po, pb):
    """Soft maximum-entropy tilt toward model targets without killing scoreline diversity."""
    h, a = np.indices(p.shape)
    out = np.where(h > a, 0, np.where(h < a, 2, 1))
    over = (h + a) >= 3
    btts = (h > 0) & (a > 0)
    f0, o0, b0, _, _ = _stats(p)
    # Blend targets so we never over-fit pure model noise
    tf = _norm(0.80 * _norm(pft)[:3] + 0.20 * f0)
    to = 0.80 * po + 0.20 * o0
    tb = 0.80 * pb + 0.20 * b0

    def obj(beta):
        q = p * np.exp(
            np.clip(
                beta[0] * (out == 0)
                + beta[1] * (out == 1)
                + beta[2] * over
                + beta[3] * btts,
                -9, 9,
            )
        )
        q /= q.sum()
        ft, ov, bt, _, _ = _stats(q)
        # entropy bonus keeps distributions from collapsing to 1-2 scorelines
        flat = q.ravel()
        ent = -float((flat[flat > 0] * np.log(flat[flat > 0])).sum())
        return (
            9 * np.sum((ft - tf) ** 2)
            + 3.2 * (ov - to) ** 2
            + 2.2 * (bt - tb) ** 2
            + 0.012 * np.sum(beta * beta)
            - 0.004 * ent
        )

    beta = np.zeros(4)
    if minimize:
        try:
            r = minimize(obj, beta, method="BFGS", options={"maxiter": 100})
            if np.all(np.isfinite(r.x)):
                beta = r.x
        except Exception:
            pass
    beta = np.clip(beta, -2.2, 2.2)
    q = p * np.exp(
        np.clip(
            beta[0] * (out == 0)
            + beta[1] * (out == 1)
            + beta[2] * over
            + beta[3] * btts,
            -9, 9,
        )
    )
    q /= q.sum()
    return q, beta.tolist()

def _sample(rng, p, n):
    idx = rng.choice(p.size, size=int(n), p=p.ravel())
    return np.column_stack(np.unravel_index(idx, p.shape)).astype(np.int16)

def _ht(rng, scores, pht, phto):
    """Correlated half-time model — HT goals are binomial shares of FT with mild edge."""
    pht = _norm(pht)[:3]
    out = np.zeros_like(scores, dtype=np.int16)
    base = float(np.clip(0.40 + 0.16 * (phto - 0.5), 0.28, 0.55))
    c = 22.0
    sh = rng.beta(base * c, (1 - base) * c, len(scores))
    edge = 0.055 * (pht[0] - pht[2])
    for i, (hg, ag) in enumerate(scores):
        if hg:
            out[i, 0] = rng.binomial(int(hg), np.clip(sh[i] + edge, 0.18, 0.70))
        if ag:
            out[i, 1] = rng.binomial(int(ag), np.clip(sh[i] - edge, 0.18, 0.70))
    return out

def _line(t, line):
    return {
        "over": float(np.mean(t > line)),
        "under": float(np.mean(t < line)),
        "push": float(np.mean(t == line)),
    }

def simulate_industrial(
    p_ft, p_ht, p_over25, p_btts, p_ht_over15, n, seed, standings_prior=None
):
    p_ft = _norm(p_ft)[:3]
    p_ht = _norm(p_ht if p_ht is not None else [0.30, 0.40, 0.30])[:3]
    po = float(np.clip(p_over25, 0.05, 0.95))
    pb = float(np.clip(p_btts, 0.05, 0.95))
    phto = float(np.clip(p_ht_over15 if p_ht_over15 is not None else 0.30, 0.05, 0.95))
    rng = np.random.default_rng(int(seed))
    sp = standings_prior or {}
    if sp.get("matched"):
        try:
            from standings_prior import apply_ft_tilt, apply_scalar_tilt
            p_ft = apply_ft_tilt(p_ft, sp.get("ft_tilt") or {})
            po = apply_scalar_tilt(po, float(sp.get("over25_tilt") or 0))
            pb = apply_scalar_tilt(pb, float(sp.get("btts_tilt") or 0))
        except Exception:
            pass

    lh, la, common, total_target = _rates(p_ft, po, pb)
    base = _mixture(lh, la, common)
    pmf, beta = _calibrate(base, p_ft, po, pb)
    scores = _sample(rng, pmf, n)
    ht = _ht(rng, scores, p_ht, phto)

    total = scores.sum(1).astype(float)
    diff = scores[:, 0].astype(float) - scores[:, 1].astype(float)
    htd = ht[:, 0].astype(float) - ht[:, 1].astype(float)

    ft = {
        "H": float(np.mean(diff > 0)),
        "D": float(np.mean(diff == 0)),
        "A": float(np.mean(diff < 0)),
    }
    hts = {
        "H": float(np.mean(htd > 0)),
        "D": float(np.mean(htd == 0)),
        "A": float(np.mean(htd < 0)),
    }

    top = Counter(map(tuple, scores.tolist())).most_common(12)
    topht = Counter(map(tuple, ht.tolist())).most_common(8)

    def top_pct(items, k):
        return [
            {
                "score": f"{h}-{a}",
                "count": int(c),
                "pct": round(100 * c / max(len(scores), 1), 1),
            }
            for (h, a), c in items[:k]
        ]

    simft = np.array([ft["H"], ft["D"], ft["A"]])
    flat = pmf.ravel()
    ent = float(-(flat[flat > 0] * np.log(flat[flat > 0])).sum())
    top1 = float(top[0][1] / len(scores)) if top else 0.0
    top3 = float(sum(c for _, c in top[:3]) / len(scores))

    return {
        "n": int(n),
        "engine": {
            "version": ENGINE_VERSION,
            "method": (
                "bivariate-poisson+dixon-coles+five-regime+soft-maxent-calibration"
                "+correlated-beta-binomial-HT"
            ),
            "rho": 0.052,
            "lambda_home": lh,
            "lambda_away": la,
            "common_goal_rate": common,
            "target_total_lambda": total_target,
            "calibration_strength": 0.80,
            "calibration_beta": beta,
            "dispersion_regimes": [0.68, 0.85, 1.00, 1.18, 1.40],
        },
        "ft_model": {"H": float(p_ft[0]), "D": float(p_ft[1]), "A": float(p_ft[2])},
        "ft_sim": ft,
        "ht_model": {"H": float(p_ht[0]), "D": float(p_ht[1]), "A": float(p_ht[2])},
        "ht_sim": hts,
        "over25_model": po,
        "over25_sim": float(np.mean(total >= 3)),
        "btts_model": pb,
        "btts_sim": float(np.mean((scores[:, 0] > 0) & (scores[:, 1] > 0))),
        "ht_over15_model": phto,
        "ht_over15_sim": float(np.mean(ht.sum(1) >= 2)),
        "dc_ft": {
            "1X": ft["H"] + ft["D"],
            "X2": ft["D"] + ft["A"],
            "12": ft["H"] + ft["A"],
        },
        "dc_ht": {
            "1X": hts["H"] + hts["D"],
            "X2": hts["D"] + hts["A"],
            "12": hts["H"] + hts["A"],
        },
        "top_ft": [(f"{h}-{a}", int(c)) for (h, a), c in top],
        "top_ht": [(f"{h}-{a}", int(c)) for (h, a), c in topht],
        "top3_ft": top_pct(top, 3),
        "top3_ht": top_pct(topht, 3),
        "top8_ft": top_pct(top, 8),
        "xg": {
            "home": float(scores[:, 0].mean()),
            "away": float(scores[:, 1].mean()),
            "total": float(total.mean()),
            "lambda_home": lh,
            "lambda_away": la,
        },
        "ht_xg": {
            "home": float(ht[:, 0].mean()),
            "away": float(ht[:, 1].mean()),
            "total": float(ht.sum(1).mean()),
        },
        "score_consistency": {
            "ht_leq_ft": float(
                np.mean((ht[:, 0] <= scores[:, 0]) & (ht[:, 1] <= scores[:, 1]))
            ),
            "ft_vs_model_l1": float(np.abs(simft - _norm(p_ft)[:3]).sum()),
            "over25_abs_error": float(abs(np.mean(total >= 3) - po)),
            "btts_abs_error": float(
                abs(np.mean((scores[:, 0] > 0) & (scores[:, 1] > 0)) - pb)
            ),
        },
        "distribution_diagnostics": {
            "score_entropy": ent,
            "top1_mass": top1,
            "top3_mass": top3,
            "tail_4plus": float(np.mean(total >= 4)),
            "score_cells": int(pmf.size),
        },
        "clean_sheet": {
            "home": float(np.mean(scores[:, 1] == 0)),
            "away": float(np.mean(scores[:, 0] == 0)),
        },
        "win_to_nil": {
            "home": float(np.mean((diff > 0) & (scores[:, 1] == 0))),
            "away": float(np.mean((diff < 0) & (scores[:, 0] == 0))),
        },
        "goal_lines": {str(x): _line(total, x) for x in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "asian_handicap": {
            "home_-0.5": float(np.mean(diff > 0)),
            "home_-1.0": {
                "cover": float(np.mean(diff > 1)),
                "push": float(np.mean(diff == 1)),
            },
            "home_+0.5": float(np.mean(diff > -1)),
            "home_+1.0": {
                "cover": float(np.mean(diff > -1)),
                "push": float(np.mean(diff == -1)),
            },
        },
        "score_matrix_margin": {
            "home_1_goal": float(np.mean(diff == 1)),
            "home_2_plus": float(np.mean(diff >= 2)),
            "away_1_goal": float(np.mean(diff == -1)),
            "away_2_plus": float(np.mean(diff <= -2)),
            "draw": ft["D"],
        },
    }
