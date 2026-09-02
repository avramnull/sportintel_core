#!/usr/bin/env python3
"""Calibrate generated locks without double-counting model-conditioned simulation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIMS_DIR = ROOT / "daily_football_data" / "sims"


def _num(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def calibrate(payload: dict) -> bool:
    """Turn an existing candidate into a lock only when its own probability is high.

    The simulator is conditioned on the model, so its agreement is a diagnostic,
    not an independent confidence multiplier. We therefore preserve the model's
    calibrated probability and use simulation diagnostics only as vetoes.
    """
    tip = payload.get("locked_tip")
    report = payload.get("report") or {}
    rows = payload.get("table") or []
    if not isinstance(tip, dict):
        return False

    section = str(tip.get("section") or "").upper()
    selection = str(tip.get("selection") or "")
    model = _num(tip.get("model"))
    sim = _num(tip.get("sim"))
    for row in rows:
        if (str(row.get("Section", "")).upper() == section and
                str(row.get("Selection", "")) == selection):
            model = _num(row.get("Model%"), model)
            sim = _num(row.get("Sim%"), sim)
            break

    # This is intentionally NOT a second probability source. The displayed
    # confidence remains the model probability; simulation is only a sanity
    # check because the simulator was conditioned on that same model.
    confidence = max(0.0, min(100.0, model))
    vetoes = []

    # A large model/simulation gap means the generated distribution is unstable.
    if abs(model - sim) > 8.0:
        vetoes.append("model/sim disagreement")

    consistency = _num((report.get("score_consistency") or {}).get("ft_vs_model_l1"))
    if consistency > 0.12:
        vetoes.append("score calibration mismatch")

    diagnostics = report.get("distribution_diagnostics") or {}

    # Score concentration/entropy are useful for exact-score markets, but they
    # are NOT valid reasons to reject broad markets such as HT/FT 1X2, DC, O/U,
    # or BTTS. Applying those gates to every market was producing false NO LOCKs.
    exact_score = section in {"FT_SCORE", "HT_SCORE", "SCORE", "EXACT_SCORE"}
    if exact_score:
        top1 = _num(diagnostics.get("top1_mass"))
        top3 = _num(diagnostics.get("top3_mass"))
        if top1 > 0.25 or top3 > 0.62:
            vetoes.append("score distribution concentration")
        entropy = _num(diagnostics.get("score_entropy"), 0.0)
        if entropy and entropy < 1.35:
            vetoes.append("low score entropy")

    # 80% remains the production lock threshold. This is now a genuine
    # probability threshold rather than an arbitrary shrinkage formula.
    secured = confidence >= 80.0 and not vetoes
    tip["status"] = "SECURED LOCK" if secured else "NO LOCK"
    tip["verdict"] = "HARD YES" if secured else "REVIEW"
    tip["confidence"] = round(confidence, 1)
    tip["raw_model"] = round(model, 1)
    tip["raw_sim"] = round(sim, 1)
    tip["confidence_basis"] = "model probability; simulation used only as stability veto"
    if vetoes:
        tip["lock_veto"] = "; ".join(vetoes)
    else:
        tip.pop("lock_veto", None)

    payload["locked_tip"] = tip
    payload.setdefault("report", {})["lock_quality"] = {
        "version": "lock-calibration-v2",
        "confidence": round(confidence, 1),
        "model_only": True,
        "simulation_agreement_counted_as_independent": False,
        "simulation_used_as_stability_veto": True,
        "secured": secured,
        "vetoes": vetoes,
    }
    return True


def main() -> None:
    if not SIMS_DIR.exists():
        print("No sims directory; nothing to calibrate")
        return

    changed = 0
    by_file = {}
    for path in sorted(SIMS_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"skip {path.name}: invalid JSON: {exc}")
            continue
        if calibrate(payload):
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            changed += 1
            by_file[path.name] = payload.get("locked_tip") or {}

    index_path = SIMS_DIR / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for item in index.get("sims") or []:
                tip = by_file.get(str(item.get("file") or ""))
                if tip is None:
                    continue
                item["locked_status"] = tip.get("status")
                item["locked_section"] = tip.get("section")
                item["locked_selection"] = tip.get("selection")
                item["locked_model"] = tip.get("model")
                item["locked_sim"] = tip.get("sim")
                item["locked_verdict"] = tip.get("verdict")
                item["locked_confidence"] = tip.get("confidence")
            index["lock_calibration"] = {
                "version": "lock-calibration-v2",
                "model_only": True,
                "simulation_agreement_counted_as_independent": False,
                "simulation_used_as_stability_veto": True,
                "updated_reports": changed,
            }
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        except Exception as exc:
            raise SystemExit(f"Could not update sims/index.json: {exc}") from exc

    print(f"Lock calibration: updated {changed} simulation reports")


if __name__ == "__main__":
    main()
