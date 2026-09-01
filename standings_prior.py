#!/usr/bin/env python3
"""
Live league standings → industrial simulation priors.

Takes API-Football (or compatible) table rows and produces:
  - rank / PPG / GD-per-game gap between home and away
  - lambda multipliers and FT probability tilt
  - optional form-string score (W/D/L sequence)

Used by EUR `sim.run_one_match` and Africa `sim_africa.simulate_match`.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

from api_football_client import _norm, team_lookup


def form_score(form: str, n: int = 5) -> float:
    """Map last form chars (W/D/L) to points-per-game proxy in [0, 3]."""
    if not form:
        return float("nan")
    chars = re.findall(r"[WDL]", str(form).upper())[-n:]
    if not chars:
        return float("nan")
    pts = sum(3 if c == "W" else (1 if c == "D" else 0) for c in chars)
    return pts / len(chars)


def _ppg(row: Optional[dict]) -> float:
    if not row:
        return float("nan")
    played = row.get("played") or 0
    pts = row.get("points")
    if not played or pts is None:
        return float("nan")
    return float(pts) / max(int(played), 1)


def _gd_pg(row: Optional[dict]) -> float:
    if not row:
        return float("nan")
    played = row.get("played") or 0
    gd = row.get("gd")
    if gd is None:
        gf, ga = row.get("gf"), row.get("ga")
        if gf is None or ga is None:
            return float("nan")
        gd = float(gf) - float(ga)
    if not played:
        return float("nan")
    return float(gd) / max(int(played), 1)


def _gf_pg(row: Optional[dict]) -> float:
    if not row:
        return float("nan")
    played = row.get("played") or 0
    gf = row.get("gf")
    if gf is None or not played:
        return float("nan")
    return float(gf) / max(int(played), 1)


def _ga_pg(row: Optional[dict]) -> float:
    if not row:
        return float("nan")
    played = row.get("played") or 0
    ga = row.get("ga")
    if ga is None or not played:
        return float("nan")
    return float(ga) / max(int(played), 1)


def match_team(lookup: Dict[str, dict], name: str) -> Optional[dict]:
    if not name or not lookup:
        return None
    key = _norm(name)
    if key in lookup:
        return lookup[key]
    low = name.lower().strip()
    if low in lookup:
        return lookup[low]
    # fuzzy contains
    for k, row in lookup.items():
        if key and (key in k or k in key) and abs(len(k) - len(key)) <= 8:
            return row
    # token overlap
    tokens = set(key.split())
    best, best_n = None, 0
    for k, row in lookup.items():
        kt = set(k.split())
        n = len(tokens & kt)
        if n >= 2 and n > best_n:
            best, best_n = row, n
    return best


def build_prior(
    home: str,
    away: str,
    table_rows: list,
    *,
    strength: float = 0.55,
) -> Dict[str, Any]:
    """
    Build standings-informed prior for one fixture.

    strength ∈ [0, 1]: how hard to push lambdas / FT toward table signal.
    """
    lookup = team_lookup(table_rows or [])
    h = match_team(lookup, home)
    a = match_team(lookup, away)
    out: Dict[str, Any] = {
        "home_row": _public_row(h),
        "away_row": _public_row(a),
        "matched": bool(h and a),
        "strength": float(strength),
        "lambda_mult_home": 1.0,
        "lambda_mult_away": 1.0,
        "ft_tilt": {"H": 0.0, "D": 0.0, "A": 0.0},
        "over25_tilt": 0.0,
        "btts_tilt": 0.0,
        "signal": {},
    }
    if not h or not a:
        return out

    s = max(0.0, min(1.0, float(strength)))
    rank_h = float(h.get("rank") or 0)
    rank_a = float(a.get("rank") or 0)
    # lower rank is better; positive rank_gap means home is stronger
    rank_gap = (rank_a - rank_h) if (rank_h and rank_a) else 0.0
    n_teams = max(len(table_rows), 10)
    rank_gap_n = rank_gap / max(n_teams - 1, 1)

    ppg_h, ppg_a = _ppg(h), _ppg(a)
    gd_h, gd_a = _gd_pg(h), _gd_pg(a)
    gf_h, gf_a = _gf_pg(h), _gf_pg(a)
    ga_h, ga_a = _ga_pg(h), _ga_pg(a)
    form_h, form_a = form_score(h.get("form") or ""), form_score(a.get("form") or "")

    ppg_gap = 0.0
    if not math.isnan(ppg_h) and not math.isnan(ppg_a):
        ppg_gap = (ppg_h - ppg_a) / 3.0  # scale to ~[-1,1]

    gd_gap = 0.0
    if not math.isnan(gd_h) and not math.isnan(gd_a):
        gd_gap = max(-1.5, min(1.5, gd_h - gd_a)) / 1.5

    form_gap = 0.0
    if not math.isnan(form_h) and not math.isnan(form_a):
        form_gap = (form_h - form_a) / 3.0

    # Composite strength in ~[-1, 1]
    composite = (
        0.35 * rank_gap_n
        + 0.30 * ppg_gap
        + 0.20 * gd_gap
        + 0.15 * form_gap
    )
    composite = max(-1.0, min(1.0, composite))

    # Lambda multipliers: stronger side scores more / concedes less
    # home already has venue edge elsewhere; table only scales relative force
    lam_h = math.exp(0.28 * s * composite)
    lam_a = math.exp(-0.28 * s * composite)
    # Goal volume from combined attack rates
    volume = 0.0
    if not math.isnan(gf_h) and not math.isnan(gf_a):
        volume = ((gf_h + gf_a) / 2.0) - 1.25  # ~0 at average scoring leagues
        volume = max(-0.8, min(0.8, volume))
    vol_mult = math.exp(0.18 * s * volume)
    out["lambda_mult_home"] = float(lam_h * vol_mult)
    out["lambda_mult_away"] = float(lam_a * vol_mult)

    # FT tilt (added after model, before renormalize) — mild
    tilt = 0.10 * s * composite
    out["ft_tilt"] = {"H": float(tilt), "D": float(-0.35 * abs(tilt)), "A": float(-tilt)}

    # Over/BTTS from combined gf+ga
    o_tilt = 0.0
    b_tilt = 0.0
    if not math.isnan(gf_h) and not math.isnan(ga_h) and not math.isnan(gf_a) and not math.isnan(ga_a):
        expected = 0.5 * (gf_h + ga_a + gf_a + ga_h)
        o_tilt = 0.08 * s * max(-1.0, min(1.0, (expected - 2.5) / 1.2))
        # both teams scoring: product of attack vs opp defence proxies
        b_tilt = 0.06 * s * max(-1.0, min(1.0, ((gf_h * ga_a) + (gf_a * ga_h)) / 4.0 - 1.0))
    out["over25_tilt"] = float(o_tilt)
    out["btts_tilt"] = float(b_tilt)

    out["signal"] = {
        "rank_home": h.get("rank"),
        "rank_away": a.get("rank"),
        "rank_gap": rank_gap,
        "ppg_home": None if math.isnan(ppg_h) else round(ppg_h, 3),
        "ppg_away": None if math.isnan(ppg_a) else round(ppg_a, 3),
        "gd_pg_home": None if math.isnan(gd_h) else round(gd_h, 3),
        "gd_pg_away": None if math.isnan(gd_a) else round(gd_a, 3),
        "form_home": h.get("form"),
        "form_away": a.get("form"),
        "composite": round(composite, 4),
        "league_id": h.get("league_id") or a.get("league_id"),
        "league": h.get("league") or a.get("league"),
    }
    return out


def apply_ft_tilt(ft, tilt: dict):
    """Apply additive tilt to H/D/A probs and renormalize. Accepts array or dict."""
    import numpy as np
    if isinstance(ft, dict):
        h = float(ft.get("H", 1 / 3)) + float(tilt.get("H", 0))
        d = float(ft.get("D", 1 / 3)) + float(tilt.get("D", 0))
        a = float(ft.get("A", 1 / 3)) + float(tilt.get("A", 0))
        h, d, a = max(h, 1e-6), max(d, 1e-6), max(a, 1e-6)
        s = h + d + a
        return {"H": h / s, "D": d / s, "A": a / s}
    arr = np.asarray(ft, float).ravel()[:3].copy()
    if len(arr) < 3:
        arr = np.array([1 / 3, 1 / 3, 1 / 3], float)
    arr[0] += float(tilt.get("H", 0))
    arr[1] += float(tilt.get("D", 0))
    arr[2] += float(tilt.get("A", 0))
    arr = np.clip(arr, 1e-6, None)
    return arr / arr.sum()


def apply_scalar_tilt(p: float, tilt: float) -> float:
    return float(max(0.05, min(0.95, float(p) + float(tilt))))


def _public_row(row: Optional[dict]) -> Optional[dict]:
    if not row:
        return None
    keys = ("team", "team_id", "rank", "points", "played", "won", "draw", "lost", "gf", "ga", "gd", "form")
    return {k: row.get(k) for k in keys}


def prior_from_league_cache(
    home: str,
    away: str,
    league_id: Optional[int],
    cache: Dict[int, list],
    *,
    strength: float = 0.55,
) -> Dict[str, Any]:
    if not league_id or not cache:
        return build_prior(home, away, [], strength=strength)
    rows = cache.get(int(league_id)) or []
    return build_prior(home, away, rows, strength=strength)
