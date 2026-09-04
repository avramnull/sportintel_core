#!/usr/bin/env python3
"""Calibrate locks WITHOUT destroying fixed HT/FT correct-score products."""
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


def _is_fixed_cs_product(payload: dict) -> bool:
    """Primary product is one fixed FT CS + one fixed HT CS from precision-v5."""
    meta = payload.get("metadata") or {}
    if meta.get("primary") == "ht_ft_score_matrix":
        return True
    tip = payload.get("locked_tip") or {}
    st = str(tip.get("status") or "").upper()
    if st in {"FIXED CS", "TOP CS", "FT CS", "—"}:
        return True
    rep = payload.get("report") or {}
    if rep.get("fixed_ft_cs") or rep.get("fixed_ht_cs"):
        return True
    # Clean Africa matrix docs
    if (payload.get("region") or "").lower() == "africa" and rep.get("top_ft") and not payload.get("table"):
        return True
    return False


def calibrate(payload: dict) -> bool:
    tip = payload.get("locked_tip")
    report = payload.get("report") or {}
    rows = payload.get("table") or []
    if not isinstance(tip, dict):
        return False

    # DO NOT rewrite fixed correct-score products into NO LOCK.
    if _is_fixed_cs_product(payload):
        ft = report.get("fixed_ft_cs") or {}
        ht = report.get("fixed_ht_cs") or {}
        if not ft and report.get("top_ft"):
            top = report["top_ft"][0]
            if isinstance(top, (list, tuple)) and len(top) >= 2:
                n = int(report.get("n") or 150000)
                ft = {"score": str(top[0]), "count": int(top[1]), "pct": round(100.0 * int(top[1]) / max(n, 1), 2)}
                report["fixed_ft_cs"] = ft
            elif isinstance(top, dict):
                ft = top
                report["fixed_ft_cs"] = ft
        if not ht and report.get("top_ht"):
            top = report["top_ht"][0]
            if isinstance(top, (list, tuple)) and len(top) >= 2:
                n = int(report.get("n") or 150000)
                ht = {"score": str(top[0]), "count": int(top[1]), "pct": round(100.0 * int(top[1]) / max(n, 1), 2)}
                report["fixed_ht_cs"] = ht
            elif isinstance(top, dict):
                ht = top
                report["fixed_ht_cs"] = ht
        payload["report"] = report
        payload["locked_tip"] = {
            "status": "FIXED CS",
            "section": "FT CS",
            "selection": (ft or {}).get("score") or tip.get("selection") or "—",
            "model": None,
            "sim": (ft or {}).get("pct"),
            "verdict": "TOP CS",
            "ht_cs": (ht or {}).get("score"),
            "ht_cs_pct": (ht or {}).get("pct"),
        }
        payload.setdefault("metadata", {})["primary"] = "ht_ft_score_matrix"
        return True

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

    confidence = max(0.0, min(100.0, model))
    vetoes = []
    if abs(model - sim) > 8.0:
        vetoes.append("model/sim disagreement")
    consistency = _num((report.get("score_consistency") or {}).get("ft_vs_model_l1"))
    if consistency > 0.12:
        vetoes.append("score calibration mismatch")

    diagnostics = report.get("distribution_diagnostics") or {}
    exact_score = section in {"FT_SCORE", "HT_SCORE", "SCORE", "EXACT_SCORE", "FT CS", "HT CS"}
    if exact_score:
        top1 = _num(diagnostics.get("top1_mass"))
        top3 = _num(diagnostics.get("top3_mass"))
        if top1 > 0.25 or top3 > 0.62:
            vetoes.append("score distribution concentration")
        entropy = _num(diagnostics.get("score_entropy"), 0.0)
        if entropy and entropy < 1.35:
            vetoes.append("low score entropy")

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
        "version": "lock-calibration-v3",
        "confidence": round(confidence, 1),
        "model_only": True,
        "fixed_cs_preserved": False,
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
        if path.name in {"index.json", "index_africa.json"}:
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
                if tip.get("status") == "FIXED CS":
                    item["fixed_ft_cs"] = tip.get("selection")
                    item["fixed_ht_cs"] = tip.get("ht_cs")
            index["lock_calibration"] = {
                "version": "lock-calibration-v3",
                "fixed_cs_preserved": True,
                "updated_reports": changed,
            }
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        except Exception as exc:
            raise SystemExit(f"Could not update sims/index.json: {exc}") from exc

    print(f"Lock calibration: updated {changed} simulation reports (fixed CS preserved)")


if __name__ == "__main__":
    main()
