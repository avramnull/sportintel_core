#!/usr/bin/env python3
"""Core security contract for published football simulation tips.

A secured tip is not tied to a particular market. Any market may qualify, but
only after the same hard gate: strong model probability, strong simulation
probability, small model/simulation disagreement, and a healthy simulation
consistency envelope. CS and HT-CS are represented by one best side each.
"""
from __future__ import annotations

import math
import re
from typing import Any

CORE_MIN_MODEL = 0.78
CORE_MIN_SIM = 0.78
CORE_MAX_GAP = 0.06
CORE_MIN_AGREE = 0.72
CORE_MIN_SCORE = 0.78


def _p(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x / 100.0 if x > 1.0 else x


def _agreement(r: dict) -> float:
    s = str(r.get("Agree", ""))
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        return min(1.0, n / 3.0)
    return 1.0 if s.upper() in {"Y", "YES", "TRUE"} else 0.0


def _healthy(report: dict) -> bool:
    c = report.get("score_consistency") or {}
    if c and float(c.get("ht_leq_ft", 1.0)) < 0.999999:
        return False
    for k in ("ft_vs_model_l1", "over25_abs_error", "btts_abs_error"):
        if k in c and float(c[k]) > 0.10:
            return False
    return True


def _candidate_score(row: dict) -> float:
    m = _p(row.get("Model%"))
    s = _p(row.get("Sim%"))
    gap = abs(m - s)
    agree = _agreement(row)
    return 0.45 * min(m, s) + 0.30 * ((m + s) / 2) + 0.15 * agree + 0.10 * max(0.0, 1.0 - gap / CORE_MAX_GAP)


def select_fixed_clean_sheet(report: dict, home: str, away: str) -> dict | None:
    cs = report.get("clean_sheet") or {}
    h, a = _p(cs.get("home")), _p(cs.get("away"))
    if h <= 0 and a <= 0:
        return None
    if h >= a:
        return {"Section": "CS", "Selection": f"{home} CS", "Model%": round(h * 100, 1), "Sim%": round(h * 100, 1), "Agree": "simulation", "Edge": 0.0, "Verdict": "YES" if h >= CORE_MIN_SCORE else "LEAN"}
    return {"Section": "CS", "Selection": f"{away} CS", "Model%": round(a * 100, 1), "Sim%": round(a * 100, 1), "Agree": "simulation", "Edge": 0.0, "Verdict": "YES" if a >= CORE_MIN_SCORE else "LEAN"}


def select_fixed_ht_cs(report: dict, home: str, away: str) -> dict | None:
    cs = report.get("ht_clean_sheet") or {}
    h, a = _p(cs.get("home")), _p(cs.get("away"))
    if h <= 0 and a <= 0:
        return None
    if h >= a:
        return {"Section": "HT CS", "Selection": f"{home} HT CS", "Model%": round(h * 100, 1), "Sim%": round(h * 100, 1), "Agree": "simulation", "Edge": 0.0, "Verdict": "YES" if h >= CORE_MIN_SCORE else "LEAN"}
    return {"Section": "HT CS", "Selection": f"{away} HT CS", "Model%": round(a * 100, 1), "Sim%": round(a * 100, 1), "Agree": "simulation", "Edge": 0.0, "Verdict": "YES" if a >= CORE_MIN_SCORE else "LEAN"}


def secure_any_market(rows: list[dict], report: dict, *, require_backend: bool = True) -> dict:
    """Return exactly one SECURED LOCK only when the core gate is satisfied."""
    if not rows or not _healthy(report):
        return {"status": "NO LOCK", "section": "—", "selection": "—", "model": None, "sim": None, "verdict": "—", "security": "CORE_BLOCK"}

    candidates = []
    for r in rows:
        m, s = _p(r.get("Model%")), _p(r.get("Sim%"))
        gap = abs(m - s)
        agree = _agreement(r)
        if m < CORE_MIN_MODEL or s < CORE_MIN_SIM or gap > CORE_MAX_GAP:
            continue
        if require_backend and r.get("Agree") not in ("simulation", "market") and agree < CORE_MIN_AGREE:
            continue
        score = _candidate_score(r)
        if score >= CORE_MIN_SCORE:
            candidates.append((score, r))

    if not candidates:
        return {"status": "NO LOCK", "section": "—", "selection": "—", "model": None, "sim": None, "verdict": "—", "security": "CORE_BLOCK"}

    _, top = max(candidates, key=lambda x: x[0])
    return {
        "status": "SECURED LOCK",
        "section": top.get("Section", "—"),
        "selection": top.get("Selection", "—"),
        "model": top.get("Model%"),
        "sim": top.get("Sim%"),
        "agree": top.get("Agree"),
        "verdict": "HARD YES",
        "security": "CORE_LOCKED",
        "security_score": round(_candidate_score(top), 4),
    }


def enforce_payload(payload: dict) -> dict:
    """Normalize CS rows and replace any legacy weak locked-tip logic."""
    resolved = payload.get("resolved") or {}
    home = resolved.get("home") or (payload.get("match") or {}).get("home") or "Home"
    away = resolved.get("away") or (payload.get("match") or {}).get("away") or "Away"
    report = payload.get("report") or {}
    rows = list(payload.get("table") or [])

    rows = [r for r in rows if str(r.get("Section")) not in {"CS", "HT CS"}]
    cs = select_fixed_clean_sheet(report, home, away)
    htcs = select_fixed_ht_cs(report, home, away)
    if cs:
        rows.append(cs)
    if htcs:
        rows.append(htcs)
    payload["table"] = rows

    require_backend = not str(resolved.get("models", "")).lower().startswith("market-only")
    payload["locked_tip"] = secure_any_market(rows, report, require_backend=require_backend)
    payload.setdefault("metadata", {})["security"] = {
        "version": "core-lock-v1",
        "min_model": CORE_MIN_MODEL,
        "min_sim": CORE_MIN_SIM,
        "max_model_sim_gap": CORE_MAX_GAP,
        "healthy_simulation_required": True,
        "any_market_eligible": True,
        "single_cs": True,
        "single_ht_cs": True,
        "status": payload["locked_tip"]["status"],
    }
    return payload
