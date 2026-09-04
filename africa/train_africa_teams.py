#!/usr/bin/env python3
"""Fixture-driven Africa training: only teams on today's fixture board get scopes.

- Resolve Live-score names onto historical master names (strict).
- Do NOT write empty-target "harmless" models that later present as market-only.
  Missing history → skip scope; sim uses league prior from precision engine.
- Softer chronological split so 40–80 match teams can still train FT/O25/BTTS.
"""
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

    # Re-resolve fixture names against the full historical name set
    hist_names = sorted(set(raw["HomeTeam"].astype(str)) | set(raw["AwayTeam"].astype(str)))
    hist_ids = {n: i for i, n in enumerate(hist_names)}

    canonical: list[str] = []
    unresolved: list[str] = []
    for name in requested:
        canon, how = resolve(name, hist_ids, aliases)
        if how == "raw":
            unresolved.append(name)
            log(f"UNRESOLVED fixture name (no history match): {name!r}")
            continue
        if canon not in canonical:
            canonical.append(canon)
            if how != "exact":
                log(f"mapped fixture {name!r} -> {canon!r} via {how}")

    min_matches = int(os.environ.get("MIN_TEAM_MATCHES", "25"))
    # Africa often has 40–90 rows; require enough for a minimal chronological split.
    min_split_rows = int(os.environ.get("AFRICA_MIN_SPLIT_ROWS", "40"))
    trained: list[str] = []
    skipped: list[str] = []

    for team in canonical:
        mask = (engineered["HomeTeam"] == team) | (engineered["AwayTeam"] == team)
        team_df = engineered.loc[mask].copy()
        n = len(team_df)
        if n < min_matches:
            log(f"SKIP {team}: only {n} matches (<{min_matches}) — no empty fallback model")
            skipped.append(team)
            continue
        if n < min_split_rows:
            log(f"SKIP {team}: {n} matches but <{min_split_rows} for train/val split — no empty fallback")
            skipped.append(team)
            continue
        scope = f"team_{_slug(team)}"
        log(f"training {team}: matches={n} -> {scope}")
        try:
            train_scope(team_df, scope, MODEL_ROOT)
            trained.append(team)
        except Exception as exc:
            log(f"training {team} FAILED: {exc} — no empty fallback")
            skipped.append(team)

    log(
        f"training complete: trained={len(trained)} skipped={len(skipped)} "
        f"unresolved={len(unresolved)} (no empty-target models written)"
    )
    if unresolved:
        log("unresolved names: " + ", ".join(unresolved[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
