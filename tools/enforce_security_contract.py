#!/usr/bin/env python3
"""Post-process published simulation JSON into the core security contract."""
from __future__ import annotations

import json
import math
from pathlib import Path

from security_core import enforce_payload

ROOT = Path(__file__).resolve().parents[1]
SIMS = ROOT / "daily_football_data" / "sims"


def _ensure_africa_ht_cs(payload: dict) -> None:
    """Africa reports from older engines may lack HT-CS; derive a conservative
    Poisson clean-sheet probability from the simulated HT expected goals.
    Newer engines can overwrite this with their direct simulation value.
    """
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
        payload = enforce_payload(payload)
        after = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if after != before:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            changed += 1

    # Rebuild index entries from the authoritative per-match files without
    # deleting unrelated metadata. This prevents stale locked-tip fields.
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
                        "security": (payload.get("metadata") or {}).get("security"),
                        "cs_home": (rep.get("clean_sheet") or {}).get("home"),
                        "cs_away": (rep.get("clean_sheet") or {}).get("away"),
                        "ht_cs_home": (rep.get("ht_clean_sheet") or {}).get("home"),
                        "ht_cs_away": (rep.get("ht_clean_sheet") or {}).get("away"),
                    })
                except (OSError, json.JSONDecodeError):
                    pass
                entries.append(item)
            idx["sims"] = entries
            idx["security_contract"] = "core-lock-v1"
            idx_path.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

    print(f"security contract: normalized {changed} reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
