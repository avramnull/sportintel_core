#!/usr/bin/env python3
"""
Production health checks for sportintel_core.

  python -m tools.smoke_test --quick   # CI gate (imports + parquet + mappings)
  python -m tools.smoke_test           # full: + market-only sim on a dummy fixture
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_imports() -> None:
    import pandas  # noqa: F401
    import numpy  # noqa: F401
    import pyarrow  # noqa: F401
    import requests  # noqa: F401
    import sklearn  # noqa: F401
    import team_mapping  # noqa: F401
    import si_config  # noqa: F401
    print("[ok] core imports")


def check_optional_backends() -> None:
    for name in ("xgboost", "lightgbm", "catboost", "torch", "tensorflow"):
        try:
            __import__(name)
            print(f"[ok] optional backend: {name}")
        except Exception as e:
            print(f"[warn] optional backend missing: {name} ({e.__class__.__name__})")


def check_parquet(quick: bool) -> None:
    from si_config import PARQUET_PATH
    if not PARQUET_PATH.exists():
        raise SystemExit(f"[fail] missing parquet: {PARQUET_PATH}")
    head = PARQUET_PATH.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        raise SystemExit("[fail] parquet is still a Git LFS pointer — run git lfs pull")
    size = PARQUET_PATH.stat().st_size
    if size < 2_000_000:
        raise SystemExit(f"[fail] parquet too small ({size} bytes)")
    print(f"[ok] parquet present ({size / 1e6:.1f} MB)")
    if quick:
        return
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_PATH)
    names = set(pf.schema_arrow.names)
    required = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - names
    if missing:
        raise SystemExit(f"[fail] parquet missing columns: {sorted(missing)}")
    print(f"[ok] parquet schema ({pf.metadata.num_rows:,} rows, {len(names)} cols)")


def check_mappings() -> None:
    from pathlib import Path
    import team_mapping
    t2 = team_mapping.load_team2id()
    if len(t2) < 100:
        raise SystemExit(f"[fail] team2id too small ({len(t2)})")
    aliases = team_mapping.load_aliases()
    canon, status = team_mapping.normalize_team("Man United", aliases, t2)
    if "United" not in canon and "Man" not in canon:
        print(f"[warn] unexpected Man United mapping: {canon!r} ({status})")
    else:
        print(f"[ok] team mapping ({len(t2)} teams, sample={canon}/{status})")


def check_season() -> None:
    from si_config import season_code, season_label, infer_season
    code, label = infer_season()
    print(f"[ok] inferred season {code} ({label}); active={season_code()} / {season_label()}")


def check_market_only_sim() -> None:
    import sim as sim_mod
    cfg = {
        "league": "E0",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": "2099-01-01",
        "odds_home": 2.10,
        "odds_draw": 3.40,
        "odds_away": 3.50,
        "n_simulations": 200,
        "seed": 7,
        "odds_blend": 0.30,
    }
    report = sim_mod.run_one_match(cfg, quiet=True, allow_market_only=True)
    if not isinstance(report, dict) or "report" not in report:
        raise SystemExit("[fail] run_one_match did not return expected report dict")
    tip = report.get("locked_tip") or {}
    print(f"[ok] market-only sim (locked={tip.get('status')}, models={report.get('resolved', {}).get('models')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="CI gate only")
    args = ap.parse_args()
    print("=" * 56)
    print("SPORTINTEL SMOKE TEST", "--quick" if args.quick else "--full")
    print("=" * 56)
    check_imports()
    check_optional_backends()
    check_season()
    check_parquet(quick=args.quick)
    check_mappings()
    if not args.quick:
        check_market_only_sim()
    print("=" * 56)
    print("ALL CHECKS PASSED")
    print("=" * 56)


if __name__ == "__main__":
    main()
