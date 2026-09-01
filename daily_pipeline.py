#!/usr/bin/env python3
"""Daily end-to-end results → data → fixture-driven training → simulation → publish pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from si_config import (  # noqa: E402
    SAVE_DIR, PARQUET_PATH, season_code, season_label, MAX_TRAIN_TEAMS,
    MIN_TEAM_MATCHES, N_SIMULATIONS, ODDS_BLEND, SKIP_TRAIN, SKIP_SIM,
    SPORTINTEL_PUSH, TODAY_ONLY,
)
from si_logging import get_logger, log_step  # noqa: E402

log = get_logger("pipeline")


def run(cmd, env=None, check=True):
    log.info("exec: %s", " ".join(cmd))
    e = os.environ.copy()
    if env:
        e.update({k: str(v) for k, v in env.items() if v is not None})
    result = subprocess.run(cmd, env=e)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def ensure_parquet():
    if not PARQUET_PATH.exists():
        raise SystemExit(f"Missing {PARQUET_PATH}")
    head = PARQUET_PATH.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs") or PARQUET_PATH.stat().st_size < 2_000_000:
        raise SystemExit("master_football_data.parquet is a Git LFS pointer or too small")
    log.info("parquet OK: %.1f MB", PARQUET_PATH.stat().st_size / 1e6)


def load_focus():
    focus_path = SAVE_DIR / "train_focus_teams.json"
    if not focus_path.exists():
        raise SystemExit(f"Missing {focus_path} after fixture-team scan")
    try:
        payload = json.loads(focus_path.read_text(encoding="utf-8"))
        focus = payload.get("focus_teams") or []
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid training focus file {focus_path}: {exc}") from exc
    if not isinstance(focus, list) or any(not isinstance(x, str) or not x.strip() for x in focus):
        raise SystemExit(f"Invalid focus_teams in {focus_path}")
    return [x.strip() for x in focus]


def main():
    ensure_parquet()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    sc, sl = season_code(), season_label()
    log.info("DAILY PIPELINE %s | season=%s (%s)", started, sc, sl)
    log.info("flags: MAX_TRAIN_TEAMS=%s SKIP_TRAIN=%s SKIP_SIM=%s PUSH=%s TODAY_ONLY=%s",
             MAX_TRAIN_TEAMS, SKIP_TRAIN, SKIP_SIM, SPORTINTEL_PUSH, TODAY_ONLY)

    log_step(log, "scrape_results", "fetching Latest_Results.csv")
    rc = run([sys.executable, "scraper.py"], env={"FOOTBALL_SEASON": sc}, check=False)
    if rc != 0:
        log.warning("scraper.py exited %s — continuing only if a usable local results file exists", rc)

    # A failed data upsert is not safe to ignore: training must never proceed
    # against an unexpectedly stale/corrupt master dataset.
    log_step(log, "upsert_parquet", "merging results into master parquet")
    run([sys.executable, "daily_update_parquet.py"], env={
        "FOOTBALL_SEASON": sc,
        "FOOTBALL_SEASON_LABEL": sl,
    })

    log_step(log, "scrape_fixtures", "fetching fixtures.csv")
    rc = run([sys.executable, "scraper_fixtures.py"], check=False)
    if rc != 0:
        log.warning("scraper_fixtures.py exited %s — continuing only if usable local fixtures exist", rc)

    log_step(log, "scan_teams", "building fixture-driven training focus list")
    run([sys.executable, "scan_fixture_teams.py"], env={
        "TODAY_ONLY": "1" if TODAY_ONLY else "0",
        # 0 means ALL unique teams from today's fixtures; no artificial cap.
        "MAX_TRAIN_TEAMS": str(MAX_TRAIN_TEAMS),
        "MIN_TEAM_MATCHES": str(MIN_TEAM_MATCHES),
        "TRAIN_DIVS": os.environ.get("TRAIN_DIVS", ""),
    })
    focus = load_focus()
    log.info("focus teams for train (%d): %s", len(focus), focus)

    if SKIP_TRAIN:
        log.info("SKIP_TRAIN set — skipping training")
    elif not focus:
        log.info("No fixtures/eligible teams today — skipping training")
    else:
        log_step(log, "train", f"training {len(focus)} fixture teams")
        run([sys.executable, "train.py"], env={
            "FOCUS_TEAMS": ",".join(focus),
            "DAILY_LIGHT": os.environ.get("DAILY_LIGHT", "1"),
            "MIN_TEAM_MATCHES": str(MIN_TEAM_MATCHES),
            "MAX_BOOST_ROUNDS": os.environ.get("MAX_BOOST_ROUNDS", "900"),
            "NN_EPOCHS": os.environ.get("NN_EPOCHS", "45"),
            "DAILY_TARGETS": os.environ.get("DAILY_TARGETS", "ft_result,over25,btts,ht_result"),
            "USE_PYTORCH": os.environ.get("USE_PYTORCH", "1"),
            "USE_TENSORFLOW": os.environ.get("USE_TENSORFLOW", "1"),
            "USE_ADABOOST": os.environ.get("USE_ADABOOST", "0"),
            "USE_RANDOM_FOREST": os.environ.get("USE_RANDOM_FOREST", "1"),
            "TRAIN_WORKERS": os.environ.get("TRAIN_WORKERS", "3"),
        })

    if os.environ.get("ENABLE_LIVE_STANDINGS", "").strip().lower() in ("1", "true", "yes") and os.environ.get("SKIP_STANDINGS", "").strip().lower() not in ("1", "true", "yes"):
        log_step(log, "standings", "live tables for unique EUR leagues on board")
        run([sys.executable, "fetch_day_standings.py", "--region", "eur", "--merge"])
    else:
        log.info("live standings DISABLED")

    if SKIP_SIM:
        log.info("SKIP_SIM set — skipping simulations")
    else:
        log_step(log, "sim", "batch simulating fixtures")
        run([sys.executable, "run_fixture_sims.py"], env={
            "N_SIMULATIONS": str(N_SIMULATIONS),
            "ODDS_BLEND": str(ODDS_BLEND),
            "TODAY_ONLY": "1" if TODAY_ONLY else "0",
            "STANDINGS_STRENGTH": os.environ.get("STANDINGS_STRENGTH", "0.55"),
        })

    if os.environ.get("AFRICA_SEGMENT", "").strip().lower() in ("1", "true", "yes"):
        log_step(log, "africa", "fixtures → train board teams → Sim Lab reports")
        run([sys.executable, "-m", "africa.daily_africa_segment"])
    else:
        log.info("AFRICA_SEGMENT not set — skip Africa board")

    if SPORTINTEL_PUSH:
        log_step(log, "publish", "pushing sims to sportintel admin")
        run([sys.executable, "publish_sims_to_admin.py"])
    else:
        log.info("SPORTINTEL_PUSH not set — sims stay local")

    summary = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "season_code": sc,
        "season_label": sl,
        "focus_teams": focus,
        "sims_index": str(SAVE_DIR / "sims" / "index.json"),
    }
    (SAVE_DIR / "last_pipeline_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("PIPELINE DONE %s", summary["finished"])


if __name__ == "__main__":
    main()
