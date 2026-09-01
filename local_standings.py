#!/usr/bin/env python3
"""
Live league tables from free football-data.co.uk season results.

Why: API-Football free plan only exposes standings seasons 2022–2024 — useless
for current-season priors. football-data publishes full current-season CSVs
(e.g. mmz4281/2627/E0.csv) with every played match → we build the table ourselves.

One table per Div / league. Same row schema as API-Football parse_standings_table
so standings_prior.py works unchanged.
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "daily_football_data"
USER_AGENT = "sportintel-core/1.0 (+https://github.com/avramnull/sportintel_core)"

# Div codes we map to API league ids (same as api_football_client.FDC_DIV_TO_LEAGUE)
from api_football_client import FDC_DIV_TO_LEAGUE  # noqa: E402
from si_config import season_code  # noqa: E402


def _season_candidates() -> List[str]:
    """Current + previous season codes (football-data may lag early in a campaign)."""
    primary = season_code()
    # also previous
    try:
        y = int(primary[:2])
        prev = f"{(y - 1) % 100:02d}{y % 100:02d}"
    except Exception:
        prev = None
    out = [primary]
    if prev and prev not in out:
        out.append(prev)
    return out


def fetch_div_results(div: str, season: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Download one Div full-season CSV from football-data.co.uk."""
    seasons = [season] if season else _season_candidates()
    for sc in seasons:
        if not sc:
            continue
        url = f"https://www.football-data.co.uk/mmz4281/{sc}/{div}.csv"
        try:
            r = requests.get(
                url,
                timeout=40,
                headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
            )
            if r.status_code != 200 or len(r.content) < 80:
                continue
            if b"<html" in r.content[:200].lower():
                continue
            # cache
            cache = SAVE / "fdc_season"
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / f"{sc}_{div}.csv"
            path.write_bytes(r.content)
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            if "HomeTeam" not in df.columns or "FTHG" not in df.columns:
                continue
            df = df.copy()
            df["_season_code"] = sc
            df["_div"] = div
            return df
        except Exception:
            continue
    return None


def load_results_latest() -> Optional[pd.DataFrame]:
    """Multi-div latest results already scraped by daily pipeline."""
    for cand in (SAVE / "results_latest.csv",):
        if cand.exists() and cand.stat().st_size > 100:
            try:
                df = pd.read_csv(cand, encoding="utf-8", on_bad_lines="skip")
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                return df
            except Exception:
                pass
    files = sorted(SAVE.glob("results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in files:
        if cand.stat().st_size > 100:
            try:
                df = pd.read_csv(cand, encoding="utf-8", on_bad_lines="skip")
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                return df
            except Exception:
                pass
    return None


def _normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    need = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    out = df.copy()
    out["HomeTeam"] = out["HomeTeam"].astype(str).str.strip()
    out["AwayTeam"] = out["AwayTeam"].astype(str).str.strip()
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out = out.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = out[(out["FTHG"] >= 0) & (out["FTAG"] >= 0)]
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
        out = out.sort_values("Date")
    if "FTR" not in out.columns:
        out["FTR"] = out.apply(
            lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTHG"] < r["FTAG"] else "D"),
            axis=1,
        )
    else:
        out["FTR"] = out["FTR"].astype(str).str.upper().str.strip()
    return out


def table_from_matches(df: pd.DataFrame, *, league_id=None, league_name=None, season=None) -> List[dict]:
    """
    Build full league table rows from played matches.
    Schema aligned with parse_standings_table / standings_prior.
    """
    df = _normalize_matches(df)
    if df.empty:
        return []

    stats: Dict[str, dict] = {}
    form: Dict[str, List[str]] = defaultdict(list)

    def ensure(team: str):
        if team not in stats:
            stats[team] = {
                "team": team,
                "played": 0,
                "won": 0,
                "draw": 0,
                "lost": 0,
                "gf": 0,
                "ga": 0,
                "points": 0,
            }

    for _, r in df.iterrows():
        h, a = str(r["HomeTeam"]), str(r["AwayTeam"])
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        ftr = str(r["FTR"]).upper()
        ensure(h)
        ensure(a)
        stats[h]["played"] += 1
        stats[a]["played"] += 1
        stats[h]["gf"] += hg
        stats[h]["ga"] += ag
        stats[a]["gf"] += ag
        stats[a]["ga"] += hg
        if ftr == "H":
            stats[h]["won"] += 1
            stats[a]["lost"] += 1
            stats[h]["points"] += 3
            form[h].append("W")
            form[a].append("L")
        elif ftr == "A":
            stats[a]["won"] += 1
            stats[h]["lost"] += 1
            stats[a]["points"] += 3
            form[a].append("W")
            form[h].append("L")
        else:
            stats[h]["draw"] += 1
            stats[a]["draw"] += 1
            stats[h]["points"] += 1
            stats[a]["points"] += 1
            form[h].append("D")
            form[a].append("D")

    rows = []
    for team, s in stats.items():
        s = dict(s)
        s["gd"] = s["gf"] - s["ga"]
        s["form"] = "".join(form[team][-5:])
        s["team_id"] = None
        s["league_id"] = league_id
        s["league"] = league_name
        s["season"] = season
        s["description"] = None
        rows.append(s)

    rows.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"], x["team"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def build_eur_tables_for_divs(divs: List[str]) -> dict:
    """
    div code → {meta, rows, source, season, n_rows}
    Prefer full-season Div CSV; merge in results_latest for same Div if newer.
    """
    out = {}
    latest = load_results_latest()
    for div in sorted(set(str(d).strip() for d in divs if d)):
        lid = FDC_DIV_TO_LEAGUE.get(div)
        season_df = fetch_div_results(div)
        frames = []
        season_code_used = None
        if season_df is not None and len(season_df):
            frames.append(season_df)
            season_code_used = str(season_df["_season_code"].iloc[0]) if "_season_code" in season_df.columns else None
        if latest is not None and "Div" in latest.columns:
            sub = latest[latest["Div"].astype(str).str.strip() == div]
            if len(sub):
                frames.append(sub)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        # dedupe by date+home+away
        if "Date" in merged.columns:
            merged["Date"] = pd.to_datetime(merged["Date"], dayfirst=True, errors="coerce")
            merged = merged.drop_duplicates(
                subset=["Date", "HomeTeam", "AwayTeam"], keep="last"
            )
        rows = table_from_matches(
            merged,
            league_id=lid,
            league_name=div,
            season=season_code_used,
        )
        if not rows:
            continue
        key = str(lid) if lid else f"div:{div}"
        out[key] = {
            "meta": {
                "league_id": lid,
                "div": div,
                "league": div,
                "region": "EUR",
                "source": "football-data.co.uk local table",
            },
            "season": season_code_used,
            "n_rows": len(rows),
            "source": "local_fdc",
            "rows": rows,
        }
    return out


def build_africa_tables_from_parquet(fixtures: list) -> dict:
    """
    For Africa board: group fixtures by country/league and build tables from
    master_africa_football history filtered to recent seasons.
    """
    path = ROOT / "master_africa_football.parquet"
    if not path.exists() or path.stat().st_size < 1000:
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    # expect HomeTeam, AwayTeam, FTHG, FTAG, Country or League
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None
    c_home = col("HomeTeam", "home", "home_team")
    c_away = col("AwayTeam", "away", "away_team")
    c_hg = col("FTHG", "hg", "home_goals")
    c_ag = col("FTAG", "ag", "away_goals")
    c_country = col("Country", "country")
    c_date = col("Date", "date", "match_date")
    if not all([c_home, c_away, c_hg, c_ag]):
        return {}

    # recent window: last ~2 years if dates exist
    if c_date:
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=800)
        df = df[df[c_date].isna() | (df[c_date] >= cutoff)]

    out = {}
    # group fixtures by country
    by_country: Dict[str, list] = defaultdict(list)
    for fx in fixtures or []:
        c = (fx.get("country") or "").strip()
        if c:
            by_country[c].append(fx)

    for country, fxs in by_country.items():
        if c_country:
            sub = df[df[c_country].astype(str).str.strip().str.lower() == country.lower()]
        else:
            sub = df
        if len(sub) < 10:
            continue
        matches = pd.DataFrame({
            "HomeTeam": sub[c_home],
            "AwayTeam": sub[c_away],
            "FTHG": sub[c_hg],
            "FTAG": sub[c_ag],
            "Date": sub[c_date] if c_date else None,
        })
        lid = fxs[0].get("league_id")
        league = fxs[0].get("league") or country
        try:
            lid_key = str(int(lid)) if lid is not None else f"africa:{country}"
            lid_int = int(lid) if lid is not None else None
        except Exception:
            lid_key = f"africa:{country}"
            lid_int = None
        rows = table_from_matches(
            matches,
            league_id=lid_int,
            league_name=str(league),
            season="africa-rolling",
        )
        if not rows:
            continue
        out[lid_key] = {
            "meta": {
                "league_id": lid_int,
                "league": league,
                "country": country,
                "region": "Africa",
                "source": "master_africa_football local table",
            },
            "season": "africa-rolling",
            "n_rows": len(rows),
            "source": "local_africa",
            "rows": rows,
        }
    return out
