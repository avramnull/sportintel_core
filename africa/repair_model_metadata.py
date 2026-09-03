#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "football_models" / "africa"


def main():
    fixed = 0
    errors = []
    for p in MODELS.glob("team_*/preprocessors/feature_stats.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d.get("features"), list) and isinstance(d.get("mean"), list) and isinstance(d.get("std"), list):
                feats = d["features"]
                if not (len(feats) == len(d["mean"]) == len(d["std"])):
                    raise ValueError(f"feature/stat length mismatch: {len(feats)}/{len(d['mean'])}/{len(d['std'])}")
                d["mean"] = {str(k): float(v) for k, v in zip(feats, d["mean"])}
                d["std"] = {str(k): float(v) for k, v in zip(feats, d["std"])}
                p.write_text(json.dumps(d, indent=2), encoding="utf-8")
                fixed += 1
        except Exception as exc:
            errors.append(f"{p}: {exc}")
    if errors:
        for e in errors:
            print(f"ERROR {e}")
        raise SystemExit(f"Africa metadata repair failed for {len(errors)} model scopes")
    print(f"Africa team-model metadata repaired: {fixed}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
