#!/usr/bin/env python3
"""Apply conservative, non-circular lock calibration to generated sim reports."""
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


def _calibrate(raw_pct: float, section: str) -> float:
    p = max(0.0, min(100.0, _num(raw_pct))) / 100.0
    baseline = 1.0 / 3.0 if str(section).upper() == "FT" else 0.5
    # The industrial simulator is explicitly conditioned on the model targets.
    # Model/sim agreement therefore cannot be counted as independent evidence.
    return (baseline + 0.62 * (p - baseline)) * 100.0


def calibrate(payload: dict) -> bool:
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

    confidence = _calibrate(model, section)
    vetoes = []
    if abs(model - sim) > 6.0:
        vetoes.append("model/sim disagreement")

    consistency = _num((report.get("score_consistency") or {}).get("ft_vs_model_l1"))
    if consistency > 0.12:
        vetoes.append("score calibration mismatch")

    diagnostics = report.get("distribution_diagnostics") or {}
    top1 = _num(diagnostics.get("top1_mass")) * 100.0
    top3 = _num(diagnostics.get("top3_mass")) * 100.0
    if not top1:
        top = report.get("top3_ft") or []
        if top and isinstance(top[0], dict):
            top1 = _num(top[0].get("pct"))
        if top:
            top3 = sum(_num(x.get("pct")) for x in top[:3] if isinstance(x, dict))
    if top1 > 25.0 or top3 > 62.0:
        vetoes.append("score distribution concentration")

    entropy = _num(diagnostics.get("score_entropy"), 0.0)
    if entropy and entropy < 1.35:
        vetoes.append("low score entropy")

    # 80% is intentionally difficult to reach after the 0.62 shrink. This
    # prevents a model-conditioned simulation from certifying itself.
    secured = confidence >= 80.0 and not vetoes
    tip["status"] = "SECURED LOCK" if secured else "NO LOCK"
    tip["verdict"] = "HARD YES" if secured else "REVIEW"
    tip["confidence"] = round(confidence, 1)
    tip["raw_model"] = round(model, 1)
    tip["raw_sim"] = round(sim, 1)
    tip["confidence_basis"] = "model-only; simulation agreement excluded"
    if vetoes:
        tip["lock_veto"] = "; ".join(vetoes)
    else:
        tip.pop("lock_veto", None)
    payload["locked_tip"] = tip
    payload.setdefault("report", {})["lock_quality"] = {
        "version": "lock-calibration-v1",
        "confidence": round(confidence, 1),
        "model_only": True,
        "simulation_agreement_counted_as_independent": False,
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

    # Keep the batch index consistent with the individual reports; the admin
    # publisher can consume either representation.
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
                "version": "lock-calibration-v1",
                "model_only": True,
                "simulation_agreement_counted_as_independent": False,
                "updated_reports": changed,
            }
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        except Exception as exc:
            raise SystemExit(f"Could not update sims/index.json: {exc}") from exc

    print(f"Lock calibration: updated {changed} simulation reports")


if __name__ == "__main__":
    main()
