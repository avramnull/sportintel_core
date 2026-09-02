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


def patch_fixture_sim_engine() -> None:
    """Make INDUSTRIAL_SIM=1 actually select industrial-v3."""
    p = ROOT / "run_fixture_sims.py"
    s = p.read_text(encoding="utf-8")
    if "from industrial_sim_engine import simulate_industrial" not in s:
        s = s.replace(
            "from hard_simulation import simulate_hard\n",
            "from hard_simulation import simulate_hard\nfrom industrial_sim_engine import simulate_industrial\n",
            1,
        )

    old = '''    hard = simulate_hard(\n        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),\n        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),\n        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),\n        N_SIM, seed, standings_prior=rep.get("standings_prior"),\n        hardness={"attack_cv":HARD_ATTACK_CV,"defense_cv":HARD_DEFENSE_CV,"shared_cv":HARD_SHARED_CV})\n'''
    new = '''    args = (\n        vals(rep.get("ft_model"), {"H":1/3,"D":1/3,"A":1/3}),\n        vals(rep.get("ht_model"), {"H":.30,"D":.40,"A":.30}),\n        rep.get("over25_model", .50), rep.get("btts_model", .50), rep.get("ht_over15_model", .30),\n        N_SIM, seed,\n    )\n    if os.environ.get("INDUSTRIAL_SIM", "1").strip().lower() in ("1", "true", "yes", "on"):\n        hard = simulate_industrial(*args, standings_prior=rep.get("standings_prior"))\n    else:\n        hard = simulate_hard(*args, standings_prior=rep.get("standings_prior"),\n                             hardness={"attack_cv":HARD_ATTACK_CV,"defense_cv":HARD_DEFENSE_CV,"shared_cv":HARD_SHARED_CV})\n'''
    if old in s:
        s = s.replace(old, new, 1)
        print("patched run_fixture_sims to select industrial-v3")
    elif "simulate_industrial(*args" not in s:
        raise RuntimeError("run_fixture_sims engine call anchor not found")

    old_index = '"simulation_engine":"hard-v2" if HARD_SIM else "sim-v1"'
    new_index = '"simulation_engine":("industrial-v3" if os.environ.get("INDUSTRIAL_SIM", "1").strip().lower() in ("1", "true", "yes", "on") else "hard-v2") if HARD_SIM else "sim-v1"'
    if old_index in s:
        s = s.replace(old_index, new_index, 1)
    p.write_text(s, encoding="utf-8")


def patch_pipeline_publish_order() -> None:
    """Calibrate generated locks after both EUR and Africa sims, before publish."""
    p = ROOT / "daily_pipeline.py"
    s = p.read_text(encoding="utf-8")
    marker = '''    if SPORTINTEL_PUSH:\n        log_step(log, "publish", "pushing sims to sportintel admin")\n'''
    insertion = '''    log_step(log, "lock_calibration", "removing circular model/simulation confidence and applying lock quality gates")\n    run([sys.executable, "-m", "tools.calibrate_locks"])\n\n'''
    if "tools.calibrate_locks" not in s:
        if marker not in s:
            raise RuntimeError("daily_pipeline publish anchor not found")
        s = s.replace(marker, insertion + marker, 1)
        p.write_text(s, encoding="utf-8")
        print("patched daily_pipeline lock calibration before publish")
    else:
        print("daily_pipeline lock calibration already normalized")


def main() -> None:
    patch_train()
    validate_live_score()
    patch_fixture_sim_engine()
    patch_pipeline_publish_order()


if __name__ == "__main__":
    main()
