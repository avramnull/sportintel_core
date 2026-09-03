#!/usr/bin/env python3
from __future__ import annotations

from security_core import enforce_payload, secure_any_market


def main():
    report = {
        "score_consistency": {"ht_leq_ft": 1.0, "ft_vs_model_l1": 0.02, "over25_abs_error": 0.02, "btts_abs_error": 0.02},
        "clean_sheet": {"home": 0.81, "away": 0.44},
        "ht_clean_sheet": {"home": 0.86, "away": 0.61},
    }
    rows = [
        {"Section":"FT O/U", "Selection":"Under 2.5", "Model%":82.0, "Sim%":80.0, "Agree":"7 eng"},
        {"Section":"BTTS", "Selection":"No", "Model%":70.0, "Sim%":69.0, "Agree":"7 eng"},
    ]
    p = enforce_payload({"resolved":{"home":"A","away":"B","models":"A (home) + B (away)"},"report":report,"table":rows})
    assert sum(r["Section"] == "CS" for r in p["table"]) == 1
    assert sum(r["Section"] == "HT CS" for r in p["table"]) == 1
    assert p["locked_tip"]["status"] == "SECURED LOCK"
    assert p["locked_tip"]["selection"] == "Under 2.5"

    # A simulation-only CS row must never self-certify as a core lock.
    blocked = secure_any_market([{"Section":"CS","Selection":"A CS","Model%":99,"Sim%":99,"Agree":"simulation","IndependentModel":False}], report, require_backend=True)
    assert blocked["status"] == "NO LOCK"
    print("security-contract: OK")


if __name__ == "__main__":
    main()
