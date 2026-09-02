#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_train() -> None:
    p = ROOT / "train.py"
    s = p.read_text(encoding="utf-8")
    old = '''    split = int(len(run_df) * (1 - CONFIG["val_fraction"]))\n    split = max(30, min(split, len(run_df) - 20))\n    train_df, val_df = run_df.iloc[:split], run_df.iloc[split:]\n'''
    new = '''    # Keep the normal validation fraction for large teams, but guarantee the\n    # minimum validation rows required by prepare_xy. This prevents legitimate\n    # fixture teams with 25+ historical matches from aborting the whole run.\n    requested_val = int(len(run_df) * CONFIG["val_fraction"])\n    min_val = 25\n    val_size = max(min_val, requested_val)\n    split = len(run_df) - val_size\n    if split < 30:\n        log(f"    SKIP (insufficient rows for train/validation split: {len(run_df)})")\n        return run_name, {"team": focus, "skipped": True, "n": len(run_df), "reason": "insufficient_train_validation_rows"}\n    train_df, val_df = run_df.iloc[:split], run_df.iloc[split:]\n'''
    if old in s:
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        print("patched train.py validation split")
    elif "min_val = 25" in s and "insufficient_train_validation_rows" in s:
        print("train.py validation split already normalized")
    else:
        raise RuntimeError("train.py validation split anchor not found")


def validate_live_score() -> None:
    p = ROOT / "africa/live_score_api_fixtures.py"
    s = p.read_text(encoding="utf-8")
    required = ["def _next_page", "parse_qs", "LIVE_SCORE_MAX_REQUESTS", "def fetch_africa_fixtures"]
    missing = [x for x in required if x not in s]
    if missing:
        raise RuntimeError(f"Live-score client missing required quota-safe features: {missing}")
    print("Live-score client validation: OK")


def main() -> None:
    patch_train()
    validate_live_score()


if __name__ == "__main__":
    main()
