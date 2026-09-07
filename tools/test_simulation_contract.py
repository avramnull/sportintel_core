#!/usr/bin/env python3
"""Fast deterministic contract tests for the production simulation engine."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from industrial_sim_engine import ENGINE_VERSION, simulate_industrial  # noqa: E402


def check_case(ft, ht, over25, btts, ht_over15, seed):
    n = 5000
    r = simulate_industrial(ft, ht, over25, btts, ht_over15, n, seed)
    assert r["engine"]["version"] == ENGINE_VERSION
    assert r["n"] == n

    for key in ("ft_sim", "ht_sim"):
        vals = np.array([r[key]["H"], r[key]["D"], r[key]["A"]], dtype=float)
        assert np.isfinite(vals).all()
        assert np.all((vals >= 0) & (vals <= 1))
        assert abs(vals.sum() - 1.0) < 1e-9

    consistency = r["score_consistency"]
    assert consistency["ht_leq_ft"] == 1.0
    assert consistency["ft_vs_model_l1"] <= 0.12
    assert consistency["over25_abs_error"] <= 0.08
    assert consistency["btts_abs_error"] <= 0.08

    xg = r["xg"]
    assert all(math.isfinite(float(xg[k])) and float(xg[k]) >= 0 for k in ("home", "away", "total", "lambda_home", "lambda_away"))

    return r


def main() -> None:
    cases = [
        ([.60, .25, .15], [.48, .34, .18], .72, .64, .42, 11),
        ([.20, .35, .45], [.22, .36, .42], .34, .41, .29, 22),
        ([.34, .33, .33], [.30, .40, .30], .50, .50, .30, 33),
    ]
    for case in cases:
        check_case(*case)
    print("simulation contract: OK")


if __name__ == "__main__":
    main()
