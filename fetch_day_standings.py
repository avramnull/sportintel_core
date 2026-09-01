#!/usr/bin/env python3
"""
Probe API-Football for live league standings of focused leagues that play on a day.

Modes:
  --region africa   use africa fixtures JSON (league_id already present)
  --region eur      use football-data fixtures.csv Div → league id map
  --date YYYY-MM-DD optional (default today UTC)

Writes:
  daily_football_data/standings_day_{date}.json

Env:
  API_FOOTBALL_KEY
  API_FOOTBALL_MAX_STANDINGS (default 12)
  STANDINGS_STRENGTH (default 0.55) — stored for sim consumers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from api_football_client import (  # noqa: E402
    ApiFootballClient,
    FDC_DIV_TO_LEAGUE,
    api_season_year,
    parse_standings_table,
)
from si_logging import get_logger  # noqa: E402

log = get_logger("standings")
SAVE = ROOT / "daily_football_data"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def leagues_from_africa_fixtures(path: Path) -> dict:
    """league_id → meta"""
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for fx in doc.get("fixtures") or []:
        lid = fx.get("league_id")
        if not lid:
            continue
        lid = int(lid)
        out.setdefault(lid, {
            "league_id": lid,
            "league": fx.get("league"),
            "country": fx.get("country"),
            "region": "Africa",
            "n_fixtures": 0,
        })
        out[lid]["n_fixtures"] += 1
    return out


def leagues_from_eur_fixtures(path: Path, day: str) -> dict:
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if "Date" not in df.columns or "Div" not in df.columns:
        return {}
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    target = pd.Timestamp(day)
    df = df[df["Date"].dt.date == target.date()]
    out = {}
    for div, g in df.groupby(df["Div"].astype(str).str.strip()):
        lid = FDC_DIV_TO_LEAGUE.get(div)
        if not lid:
            continue
        out[lid] = {
            "league_id": lid,
            "div": div,
            "league": div,
            "region": "EUR",
            "n_fixtures": int(len(g)),
        }
    return out


def fetch_tables(client: ApiFootballClient, leagues: dict, season: int) -> dict:
    tables = {}
    # prioritize leagues with more fixtures that day
    ordered = sorted(leagues.values(), key=lambda x: -x.get("n_fixtures", 0))
    for meta in ordered:
        lid = int(meta["league_id"])
        if client._standings_calls >= client.max_standings:
            log.info("standings cap reached (%s) — remaining leagues skipped", client.max_standings)
            break
        try:
            payload = client.standings(lid, season=season)
            rows = parse_standings_table(payload)
            tables[str(lid)] = {
                "meta": meta,
                "season": season,
                "n_rows": len(rows),
                "rows": rows,
            }
            log.info(
                "standings league=%s (%s) rows=%s fixtures_today=%s",
                lid, meta.get("league") or meta.get("div"), len(rows), meta.get("n_fixtures"),
            )
        except Exception as e:
            log.error("standings fail league=%s: %s", lid, e)
    return tables


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch live standings for leagues playing on a day")
    p.add_argument("--region", choices=("africa", "eur", "both"), default="both")
    p.add_argument("--date", default=os.environ.get("FIXTURE_DATE") or _today())
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    client = ApiFootballClient()
    if not client.available:
        log.error("API_FOOTBALL_KEY not set")
        return 1

    day = args.date
    season = api_season_year()
    leagues = {}

    if args.region in ("africa", "both"):
        af = SAVE / "africa_fixtures_today.json"
        if af.exists():
            leagues.update(leagues_from_africa_fixtures(af))
        else:
            log.info("no africa fixtures file at %s", af)

    if args.region in ("eur", "both"):
        for cand in (SAVE / "fixtures_latest.csv", SAVE / f"fixtures_{day}.csv"):
            if cand.exists():
                leagues.update(leagues_from_eur_fixtures(cand, day))
                break
        else:
            log.info("no EUR fixtures CSV found under %s", SAVE)

    if not leagues:
        log.info("no focused leagues for %s — nothing to fetch", day)
        out = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "season": season,
            "leagues": {},
            "tables": {},
            "strength": float(os.environ.get("STANDINGS_STRENGTH", "0.55")),
            "client": client.stats(),
        }
    else:
        tables = fetch_tables(client, leagues, season)
        out = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "season": season,
            "leagues": {str(k): v for k, v in leagues.items()},
            "tables": tables,
            "strength": float(os.environ.get("STANDINGS_STRENGTH", "0.55")),
            "client": client.stats(),
        }

    out_path = Path(args.out) if args.out else SAVE / f"standings_day_{day}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%s leagues, %s tables)", out_path, len(out.get("leagues") or {}), len(out.get("tables") or {}))
    # also stable latest pointer
    (SAVE / "standings_latest.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
