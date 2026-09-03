#!/usr/bin/env python3
"""Fixture-driven Africa training: only teams on today's fixture board get scopes."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from .team_mapping import ensure_ids, load_aliases, resolve
from .train_africa import FEATURE_ID, FEATURE_NUM, TARGETS, engineer, load_africa, train_scope

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "football_models" / "africa"


def log(msg: str) -> None:
    print(f"[africa-team] {msg}", flush=True)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:100]


def _requested_teams() -> list[str]:
    raw = os.environ.get("FOCUS_TEAMS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _purge_stale_scopes() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for p in MODEL_ROOT.iterdir():
        if p.is_dir() and (p.name.startswith("country_") or p.name in {"GLOBAL", "teams"}):
            shutil.rmtree(p, ignore_errors=False)
    log("purged stale pooled Africa scopes")


def _write_fallback_scope(team: str, reason: str) -> None:
    """Write a neutral, non-predictive scope so one weak/missing team cannot crash the board."""
    run = MODEL_ROOT / f"team_{_slug(team)}"
    pre = run / "preprocessors"
    models = run / "models"
    pre.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    feats = FEATURE_NUM + FEATURE_ID
    stats = {"features": feats, "mean": [0.0] * len(feats), "std": [1.0] * len(feats), "classes": ["H", "D", "A"]}
    (pre / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    reg = {
        "scope": f"team_{_slug(team)}",
        "team": team,
        "fallback": True,
        "fallback_reason": str(reason)[:500],
        "n": 0,
        "n_train": 0,
        "n_val": 0,
        "features": feats,
        "targets": {t: {} for t in TARGETS},
        "countries": [],
    }
    (run / "registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    log(f"created harmless fallback for {team}: {reason}")


def main() -> int:
    requested = _requested_teams()
    if not requested:
        raise SystemExit("FOCUS_TEAMS is empty; refusing non-fixture Africa training")

    master = Path(os.environ.get("AFRICA_PARQUET", str(ROOT / "master_africa_football.parquet")))
    if not master.exists():
        master = master.with_suffix(".csv")
    if not master.exists():
        raise SystemExit(f"Africa master not found: {master}")

    _purge_stale_scopes()
    raw = load_africa(master)
    ids = ensure_ids(list(raw["HomeTeam"].astype(str)) + list(raw["AwayTeam"].astype(str)))
    aliases = load_aliases()
    raw["HomeTeam"] = raw["HomeTeam"].astype(str).map(lambda x: resolve(x, ids, aliases)[0])
    raw["AwayTeam"] = raw["AwayTeam"].astype(str).map(lambda x: resolve(x, ids, aliases)[0])
    raw = raw.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    engineered = engineer(raw)

    canonical: list[str] = []
    for name in requested:
        canon, _ = resolve(name, ids, aliases)
        if canon not in ids:
            # The fixture is still valid; retain its canonical spelling and use a neutral fallback.
            canon = str(name).strip()
        if canon not in canonical:
            canonical.append(canon)

    # IMPORTANT: canonical is derived exclusively from FOCUS_TEAMS, which is populated from
    # today's fixture report. Never expand this list to all historical teams/countries.
    min_matches = int(os.environ.get("MIN_TEAM_MATCHES", "25"))
    trained: list[str] = []
    for team in canonical:
        mask = (engineered["HomeTeam"] == team) | (engineered["AwayTeam"] == team)
        team_df = engineered.loc[mask].copy()
        if len(team_df) < min_matches:
            _write_fallback_scope(team, f"insufficient historical support: {len(team_df)} matches (<{min_matches})")
            trained.append(team)
            continue
        scope = f"team_{_slug(team)}"
        log(f"training {team}: matches={len(team_df)} -> {scope}")
        try:
            train_scope(team_df, scope, MODEL_ROOT)
            trained.append(team)
        except Exception as exc:
            log(f"training {team} failed safely: {exc}")
            _write_fallback_scope(team, f"training failure: {exc}")
            trained.append(team)

    log(f"training complete: {len(trained)} fixture teams only; no pooled fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
