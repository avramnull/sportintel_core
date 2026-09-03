#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "football_models" / "africa"


def main():
    fixed = 0
    for p in MODELS.glob("country_*/preprocessors/feature_stats.json"):
        try: d=json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        if isinstance(d.get("features"), list) and isinstance(d.get("mean"), list) and isinstance(d.get("std"), list):
            feats=d["features"]; d["mean"]={str(k):float(v) for k,v in zip(feats,d["mean"])}; d["std"]={str(k):float(v) for k,v in zip(feats,d["std"])}
            p.write_text(json.dumps(d,indent=2),encoding="utf-8"); fixed+=1
    print(f"Africa model metadata repaired: {fixed}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
