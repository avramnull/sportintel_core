#!/usr/bin/env python3
"""Fixture-driven Africa training: one isolated model scope per canonical team."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from .team_mapping import ensure_ids, load_aliases, resolve
from .train_africa import engineer, load_africa, train_scope

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "football_models" / "africa"


def log(msg: str) -> None:
    print(f"[africa-team] {msg}", flush=True)


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:100]


def _truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _requested_teams() -> list[str]:
    raw = os.environ.get("FOCUS_TEAMS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _purge_stale_scopes() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for p in MODEL_ROOT.iterdir():
        if p.is_dir() and (p.name.startswith("country_") or p.name in {"GLOBAL", "teams"}):
            shutil.rmtree(p, ignore_errors=False)
    log("purged stale pooled Africa scopes")


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
            raise SystemExit(f"Africa fixture team has no persistent mapping: {name!r} -> {canon!r}")
        if canon not in canonical:
            canonical.append(canon)

    min_matches = int(os.environ.get("MIN_TEAM_MATCHES", "25"))
    trained = []
    for team in canonical:
        mask = (engineered["HomeTeam"] == team) | (engineered["AwayTeam"] == team)
        team_df = engineered.loc[mask].copy()
        if len(team_df) < min_matches:
            log(f"skip team_{_slug(team)}: only {len(team_df)} matches (<{min_matches})")
            continue
        scope = f"team_{_slug(team)}"
        log(f"training {team}: matches={len(team_df)} -> {scope}")
        train_scope(team_df, scope, MODEL_ROOT)
        trained.append(team)

    if set(trained) != set(canonical):
        missing = [t for t in canonical if t not in trained]
        raise SystemExit(f"Africa fixture-team training incomplete: {missing}")
    log(f"training complete: {len(trained)} fixture teams; no pooled fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
