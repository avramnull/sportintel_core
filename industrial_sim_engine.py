#!/usr/bin/env python3
"""Industrial compatibility layer for the precision-v5 simulation engine."""
from __future__ import annotations

from precision_sim_engine import ENGINE_VERSION, simulate_precision


def simulate_industrial(*args, **kwargs):
    report = simulate_precision(*args, **kwargs)
    top = report.get("top3_ft") or []
    top1 = float(top[0]["pct"] / 100.0) if top else 0.0
    top3 = float(sum(x.get("pct", 0.0) for x in top) / 100.0)
    report["distribution_diagnostics"] = {
        "top1_mass": top1,
        "top3_mass": top3,
        "tail_mass_at_12": float((report.get("score_consistency") or {}).get("truncation_mass", 0.0)),
        "monte_carlo_noise": 0.0,
    }
    return report


__all__ = ["ENGINE_VERSION", "simulate_precision", "simulate_industrial"]
