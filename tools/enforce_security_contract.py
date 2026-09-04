#!/usr/bin/env python3
"""Post-process published simulation JSON into the core security contract.

Africa precision reports whose primary product is the HT/FT score matrix are
left intact — locked_tip is not rewritten to CORE_BLOCK noise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from security_core import enforce_payload

ROOT = Path(__file__).resolve().parents[1]
SIMS = ROOT / "daily_football_data" / "sims"


def _ensure_africa_ht_cs(payload: dict) -> None:
    rep = payload.get("report") or {}
    if rep.get("ht_clean_sheet"):
        return
    x = rep.get("ht_xg") or {}
    if not x:
        return
    try:
        rep["ht_clean_sheet"] = {
            "home": float(math.exp(-max(0.0, float(x.get("away", 0.0))))),
            "away": float(math.exp(-max(0.0, float(x.get("home", 0.0))))),
        }
    except (TypeError, ValueError):
        pass
    payload["report"] = rep


def _is_clean_africa_matrix(payload: dict) -> bool:
    meta = payload.get("metadata") or {}
    if meta.get("primary") == "ht_ft_score_matrix":
        return True
    if (payload.get("region") or "").lower() == "africa":
        rep = payload.get("report") or {}
        if rep.get("top_ft") and rep.get("top_ht"):
            return True
    return False


def main() -> int:
    if not SIMS.exists():
        print("security contract: no sims directory")
        return 0
    changed = 0
    for path in sorted(SIMS.glob("*.json")):
        if path.name in {"index.json", "index_africa.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        _ensure_africa_ht_cs(payload)
        before = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if _is_clean_africa_matrix(payload):
            # Keep precise HT/FT matrix; do not inject CORE_BLOCK lock stories
            tip = payload.get("locked_tip") or {}
            if not tip.get("selection") or tip.get("selection") == "—":
                top = (payload.get("report") or {}).get("top_ft") or []
                if top:
                    score = top[0][0] if isinstance(top[0], (list, tuple)) else top[0].get("score")
                    payload["locked_tip"] = {
                        "status": "—",
                        "section": "FT CS",
                        "selection": score,
                        "model": None,
                        "sim": None,
                        "verdict": "TOP CS",
                    }
            payload.setdefault("metadata", {})["primary"] = "ht_ft_score_matrix"
        else:
            payload = enforce_payload(payload)
        after = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if after != before:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            changed += 1

    idx_path = SIMS / "index.json"
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            entries = []
            for item in idx.get("sims") or []:
                f = item.get("file")
                if not f:
                    entries.append(item)
                    continue
                p = SIMS / str(f)
                if not p.exists():
                    entries.append(item)
                    continue
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    tip = payload.get("locked_tip") or {}
                    rep = payload.get("report") or {}
                    item = dict(item)
                    item.update({
                        "locked_status": tip.get("status"),
                        "locked_section": tip.get("section"),
                        "locked_selection": tip.get("selection"),
                        "locked_model": tip.get("model"),
                        "locked_sim": tip.get("sim"),
                        "locked_verdict": tip.get("verdict"),
                        "cs_home": (rep.get("clean_sheet") or {}).get("home"),
                        "cs_away": (rep.get("clean_sheet") or {}).get("away"),
                        "ht_cs_home": (rep.get("ht_clean_sheet") or {}).get("home"),
                        "ht_cs_away": (rep.get("ht_clean_sheet") or {}).get("away"),
                        "top_cs": (rep.get("top_ft") or [[None]])[0][0] if rep.get("top_ft") else None,
                    })
                except (OSError, json.JSONDecodeError):
                    pass
                entries.append(item)
            idx["sims"] = entries
            idx_path.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

    print(f"security contract: normalized {changed} reports (Africa matrices preserved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
