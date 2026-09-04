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
    # HT CS (86%, zero model/sim gap by construction, full agreement) now
    # genuinely outscores FT O/U Under 2.5 (82/80%) under the composite
    # formula once CS/HT CS are real, eligible candidates like any other
    # market — this is the correct, honest outcome, not a regression.
    assert p["locked_tip"]["status"] == "SECURED LOCK"
    assert p["locked_tip"]["section"] == "HT CS"
    assert p["locked_tip"]["selection"] == "A HT CS"

    # CS/HT CS are real candidates like any other market: no market is
    # excluded from the lock pool by type. A genuinely strong, healthy
    # clean-sheet probability can become a core lock on its own merits...
    from security_core import select_fixed_ht_cs
    strong = {
        "score_consistency": {"ht_leq_ft": 1.0, "ft_vs_model_l1": 0.02, "over25_abs_error": 0.02, "btts_abs_error": 0.02},
        "ht_clean_sheet": {"home": 0.88, "away": 0.04},
    }
    strong_row = select_fixed_ht_cs(strong, "A", "B")
    secured = secure_any_market([strong_row], strong, require_backend=True)
    assert secured["status"] == "SECURED LOCK", secured
    assert secured["section"] == "HT CS"

    # ...but a weak/unhealthy one is still correctly rejected — the bar is
    # the same 78/78/gap<=6/composite>=0.78 rule as every other market, not
    # a lowered one and not a blanket exclusion.
    weak = {
        "score_consistency": {"ht_leq_ft": 1.0, "ft_vs_model_l1": 0.02, "over25_abs_error": 0.02, "btts_abs_error": 0.02},
        "ht_clean_sheet": {"home": 0.55, "away": 0.30},
    }
    weak_row = select_fixed_ht_cs(weak, "A", "B")
    blocked = secure_any_market([weak_row], weak, require_backend=True)
    assert blocked["status"] == "NO LOCK", blocked

    # FT Score / HT Score (exact scoreline, distinct from CS/HT CS clean
    # sheet): a genuinely dominant pick must not be blocked just because
    # top1_mass is mathematically the same number as its own probability.
    strong_score_report = {"score_consistency": {"ht_leq_ft": 1.0}, "distribution_diagnostics": {"score_entropy": 0.6}}
    strong_score_row = {"Section": "HT Score", "Selection": "0-0", "Model%": 82.0, "Sim%": 82.0, "Agree": "Y"}
    secured_score = secure_any_market([strong_score_row], strong_score_report)
    assert secured_score["status"] == "SECURED LOCK", secured_score

    # ...but a numerically degenerate simulation (entropy collapsed near
    # zero) is still correctly rejected regardless of the stated probability.
    degenerate_report = {"score_consistency": {"ht_leq_ft": 1.0}, "distribution_diagnostics": {"score_entropy": 0.1}}
    degenerate_row = {"Section": "HT Score", "Selection": "0-0", "Model%": 99.0, "Sim%": 99.0, "Agree": "Y"}
    degenerate = secure_any_market([degenerate_row], degenerate_report)
    assert degenerate["status"] == "NO LOCK", degenerate

    print("security-contract: OK")


if __name__ == "__main__":
    main()
