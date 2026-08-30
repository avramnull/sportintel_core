#!/usr/bin/env python3
"""
Daily end-to-end pipeline:

  1. Fetch latest results
  2. Upsert parquet
  3. Fetch fixtures.csv
  4. Scan fixtures → accurate team mapping + train focus list
  5. Batch train focus teams (DAILY_LIGHT, capped)
  6. Batch sim.py over fixtures (odds + models)
  7. Optionally publish sims → sportintel Sim Lab (if SPORTINTEL_PUSH=1)

Env:
  FOOTBALL_SEASON, FOOTBALL_SEASON_LABEL
  MAX_TRAIN_TEAMS (default 12), MIN_TEAM_MATCHES (40)
  N_SIMULATIONS (8000), ODDS_BLEND (0.30)
  SKIP_TRAIN=1  SKIP_SIM=1  SPORTINTEL_PUSH=1
  SPORTINTEL_REPO_URL / GITHUB_TOKEN for admin push
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "daily_football_data"
os.chdir(ROOT)


def run(cmd, env=None, check=True):
    print("\n>>>", " ".join(cmd), flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e)
    if check and r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def ensure_parquet():
    pq = ROOT / "master_football_data.parquet"
    if not pq.exists():
        raise SystemExit(f"Missing {pq}")
    head = pq.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs") or pq.stat().st_size < 2000:
        raise SystemExit(
            "master_football_data.parquet is a Git LFS pointer — run: git lfs pull\n"
            "Training/sim cannot run without the real historical parquet."
        )
    print(f"Parquet OK: {pq.stat().st_size/1e6:.1f} MB")


def main():
    ensure_parquet()

    started = datetime.now(timezone.utc).isoformat()
    print("=" * 64)
    print("DAILY PIPELINE", started)
    print("=" * 64)

    season = os.environ.get("FOOTBALL_SEASON", "2627")
    label = os.environ.get("FOOTBALL_SEASON_LABEL", "2026/2027")

    # 1. Latest results
    run([sys.executable, "scraper.py"], env={"FOOTBALL_SEASON": season})

    # 2. Parquet upsert
    run([sys.executable, "daily_update_parquet.py"], env={
        "FOOTBALL_SEASON": season,
        "FOOTBALL_SEASON_LABEL": label,
    })

    # 3. Fixtures
    run([sys.executable, "scraper_fixtures.py"])

    # 4. Team scan + mapping
    run([sys.executable, "scan_fixture_teams.py"])

    focus_path = SAVE / "train_focus_teams.json"
    focus = []
    if focus_path.exists():
        focus = json.loads(focus_path.read_text()).get("focus_teams") or []
    print(f"Focus teams for train: {focus}")

    # 5. Batch train
    if os.environ.get("SKIP_TRAIN", "").strip() in ("1", "true", "yes"):
        print("SKIP_TRAIN set — skipping training")
    elif not focus:
        print("No focus teams — skipping training")
    else:
        run([sys.executable, "train.py"], env={
            "FOCUS_TEAMS": ",".join(focus),
            "DAILY_LIGHT": os.environ.get("DAILY_LIGHT", "1"),
            "MIN_TEAM_MATCHES": os.environ.get("MIN_TEAM_MATCHES", "40"),
            "MAX_BOOST_ROUNDS": os.environ.get("MAX_BOOST_ROUNDS", "1000"),
        })

    # 6. Live batch sim (sim.py engine)
    if os.environ.get("SKIP_SIM", "").strip() in ("1", "true", "yes"):
        print("SKIP_SIM set — skipping simulations")
    else:
        run([sys.executable, "run_fixture_sims.py"], env={
            "N_SIMULATIONS": os.environ.get("N_SIMULATIONS", "8000"),
            "ODDS_BLEND": os.environ.get("ODDS_BLEND", "0.30"),
        })

    # 7. Publish to admin Sim Lab
    if os.environ.get("SPORTINTEL_PUSH", "").strip() in ("1", "true", "yes"):
        run([sys.executable, "publish_sims_to_admin.py"])
    else:
        print("SPORTINTEL_PUSH not set — sims stay in daily_football_data/sims/")

    summary = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "focus_teams": focus,
        "sims_index": str(SAVE / "sims" / "index.json"),
    }
    (SAVE / "last_pipeline_run.json").write_text(json.dumps(summary, indent=2))
    print("=" * 64)
    print("PIPELINE DONE", summary["finished"])
    print("=" * 64)


if __name__ == "__main__":
    main()
