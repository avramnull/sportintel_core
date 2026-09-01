#!/usr/bin/env python3
"""
Live league standings for focused leagues that have fixtures on a given day.

Design (responsible API use):
  • Deduplicate by league_id first — six Premier League fixtures → ONE standings call.
  • That one call returns the FULL table (all ~20 clubs), covering every board team.
  • Never one request per team or per fixture.
  • Disk cache (18h) so re-runs cost zero quota for the same league/season.
  • Merge into standings_latest.json so EUR + Africa tables coexist.

Modes:
  --region africa | eur | both
  --date YYYY-MM-DD   (default: today UTC)
  --merge / --no-merge  (default: merge into existing standings_latest)

Env:
  API_FOOTBALL_KEY
  API_FOOTBALL_MAX_STANDINGS  0 = no network cap (default); else max live standings pulls
  STANDINGS_STRENGTH          stored for sim consumers (default 0.55)
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

import si_config  # noqa: F401,E402 — loads .env (API_FOOTBALL_KEY)

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
    """
    Unique league_id → meta.
    Tracks fixture teams so logs show e.g. "6 board teams covered by 1 full table".
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict = {}
    for fx in doc.get("fixtures") or []:
        lid = fx.get("league_id")
        if not lid:
            continue
        lid = int(lid)
        meta = out.setdefault(lid, {
            "league_id": lid,
            "league": fx.get("league"),
            "country": fx.get("country"),
            "region": "Africa",
            "n_fixtures": 0,
            "board_teams": [],
        })
        meta["n_fixtures"] += 1
        for k in ("home", "away"):
            t = (fx.get(k) or "").strip()
            if t and t not in meta["board_teams"]:
                meta["board_teams"].append(t)
    return out


def leagues_from_eur_fixtures(path: Path, day: str) -> dict:
    """Unique API league_id from football-data Div codes with fixtures that day."""
    import pandas as pd

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if "Date" not in df.columns or "Div" not in df.columns:
        return {}
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    target = pd.Timestamp(day)
    df = df[df["Date"].dt.date == target.date()]
    out: dict = {}
    for div, g in df.groupby(df["Div"].astype(str).str.strip()):
        lid = FDC_DIV_TO_LEAGUE.get(div)
        if not lid:
            log.info("skip unmapped Div=%s (%s fixtures)", div, len(g))
            continue
        teams = sorted(
            set(g["HomeTeam"].astype(str).str.strip())
            | set(g["AwayTeam"].astype(str).str.strip())
        )
        # Same league_id may appear from one Div only in FDC map
        if lid in out:
            out[lid]["n_fixtures"] += int(len(g))
            for t in teams:
                if t not in out[lid]["board_teams"]:
                    out[lid]["board_teams"].append(t)
            continue
        out[lid] = {
            "league_id": lid,
            "div": div,
            "league": div,
            "region": "EUR",
            "n_fixtures": int(len(g)),
            "board_teams": teams,
        }
    return out


def fetch_tables(client: ApiFootballClient, leagues: dict, season: int) -> dict:
    """
    Exactly one standings request per unique league_id.
    Full table rows are stored; fixture teams are only used for logging coverage.
    """
    tables = {}
    ordered = sorted(
        leagues.values(),
        key=lambda x: (-int(x.get("n_fixtures") or 0), str(x.get("league") or "")),
    )
    for meta in ordered:
        lid = int(meta["league_id"])
        board_n = len(meta.get("board_teams") or [])
        try:
            payload = client.standings(lid, season=season)
            if payload.get("_skipped"):
                log.info(
                    "skipped league=%s (%s) — network standings budget exhausted "
                    "(board_teams=%s would have been covered by 1 full table)",
                    lid, meta.get("league") or meta.get("div"), board_n,
                )
                continue
            rows = parse_standings_table(payload)
            src = "cache" if payload.get("_from_cache") else "network"
            tables[str(lid)] = {
                "meta": {
                    **{k: v for k, v in meta.items() if k != "board_teams"},
                    "n_board_teams": board_n,
                    "board_teams": meta.get("board_teams") or [],
                },
                "season": season,
                "n_rows": len(rows),  # full table size (e.g. 20 for PL)
                "source": src,
                "rows": rows,
            }
            log.info(
                "standings league=%s (%s) full_table=%s rows | board_fixtures=%s "
                "board_teams=%s covered_by=1_call source=%s",
                lid,
                meta.get("league") or meta.get("div"),
                len(rows),
                meta.get("n_fixtures"),
                board_n,
                src,
            )
        except Exception as e:
            log.error("standings fail league=%s: %s", lid, e)
    return tables


def load_existing_bundle(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def merge_tables(existing: dict, new_tables: dict, day: str, season: int) -> dict:
    """
    Union of league tables. Newer fetch wins for the same league_id.
    Preserves EUR tables when Africa runs second (and vice versa).
    """
    base = dict(existing or {})
    if not base:
        base = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "season": season,
            "leagues": {},
            "tables": {},
            "strength": float(os.environ.get("STANDINGS_STRENGTH", "0.55")),
        }
    tables = dict(base.get("tables") or {})
    tables.update(new_tables or {})
    leagues = dict(base.get("leagues") or {})
    for lid, block in (new_tables or {}).items():
        meta = (block.get("meta") or {}).copy()
        meta.pop("board_teams", None)  # keep leagues map compact; teams stay under tables
        leagues[str(lid)] = meta
    base["tables"] = tables
    base["leagues"] = leagues
    base["fetched_at"] = datetime.now(timezone.utc).isoformat()
    base["date"] = day or base.get("date")
    base["season"] = season
    base["strength"] = float(os.environ.get("STANDINGS_STRENGTH", base.get("strength") or 0.55))
    base["n_leagues"] = len(tables)
    base["n_full_table_rows"] = sum(int(b.get("n_rows") or 0) for b in tables.values())
    return base


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fetch full league tables once per unique league with fixtures today"
    )
    p.add_argument("--region", choices=("africa", "eur", "both"), default="both")
    p.add_argument("--date", default=os.environ.get("FIXTURE_DATE") or _today())
    p.add_argument("--out", default="")
    p.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        default=True,
        help="Merge into standings_latest.json (default)",
    )
    p.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="Replace standings_latest instead of merging",
    )
    args = p.parse_args(argv)

    client = ApiFootballClient()
    if not client.available:
        log.error("API_FOOTBALL_KEY not set")
        return 1

    day = args.date
    season = api_season_year()
    leagues: dict = {}

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

    log.info(
        "unique leagues to cover: %s (one full-table call each, never per team)",
        len(leagues),
    )
    for lid, meta in sorted(leagues.items(), key=lambda kv: -kv[1].get("n_fixtures", 0)):
        log.info(
            "  league_id=%s %s | fixtures=%s board_teams=%s",
            lid,
            meta.get("league") or meta.get("div"),
            meta.get("n_fixtures"),
            len(meta.get("board_teams") or []),
        )

    if not leagues:
        log.info("no focused leagues for %s — nothing to fetch", day)
        new_tables = {}
    else:
        new_tables = fetch_tables(client, leagues, season)

    latest_path = SAVE / "standings_latest.json"
    if args.merge:
        existing = load_existing_bundle(latest_path)
        # same calendar day keeps union; new day starts clean then merge new
        if existing.get("date") and existing.get("date") != day:
            log.info(
                "standings_latest is for %s — starting fresh bundle for %s",
                existing.get("date"),
                day,
            )
            existing = {}
        out = merge_tables(existing, new_tables, day, season)
    else:
        out = merge_tables({}, new_tables, day, season)

    out["client"] = client.stats()
    out["policy"] = {
        "unit": "unique_league_id",
        "table_scope": "full_league",
        "per_team_requests": False,
        "merge": bool(args.merge),
        "region": args.region,
    }

    out_path = Path(args.out) if args.out else SAVE / f"standings_day_{day}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2, ensure_ascii=False)
    out_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")

    stats = client.stats()
    log.info(
        "wrote %s | leagues=%s full_rows=%s | network_standings=%s cache_hits=%s max=%s",
        out_path,
        out.get("n_leagues"),
        out.get("n_full_table_rows"),
        stats.get("standings_network"),
        stats.get("standings_cache_hits"),
        stats.get("max_standings_network"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
