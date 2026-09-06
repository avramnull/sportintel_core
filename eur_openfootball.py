#!/usr/bin/env python3
"""EUR results + fixtures from openfootball, replacing football-data.co.uk.

football-data.co.uk has gone down. openfootball (github.com/openfootball/*)
publishes the same kind of per-league, per-season match data as plain text,
auto-updated weekly, and — critically for fixture collection — the current
season's file lists upcoming matches with no score yet right alongside
completed ones, in the same file.

This module is deliberately scoped to leagues actually confirmed available
and auto-updating at write time (see LEAGUE_SOURCES). Div codes not listed
there (E2, E3, EC, SP2, I2, N1, P1, T1, G1, SC0-3, ...) are not covered by
this source; they either need a different source or fall back to whatever
historical data already exists locally. Do not silently invent a mapping
for an unconfirmed league — an entry here is a promise the file exists and
updates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

import requests

RAW_BASE = "https://raw.githubusercontent.com/{repo}/master/{path}"


@dataclass(frozen=True)
class LeagueSource:
    div: str            # football-data.co.uk-style code used elsewhere in this repo
    repo: str            # e.g. "openfootball/england"
    league_name: str
    country: str


# Confirmed present and auto-updating as of the current season at write time.
# Extend only after checking the file actually exists for the target season —
# see tools in this module's __main__ for a quick existence check.
LEAGUE_SOURCES: dict[str, LeagueSource] = {
    "E0": LeagueSource("E0", "openfootball/england", "Premier League", "England"),
    "E1": LeagueSource("E1", "openfootball/england", "Championship", "England"),
    "SP1": LeagueSource("SP1", "openfootball/espana", "La Liga", "Spain"),
    "I1": LeagueSource("I1", "openfootball/italy", "Serie A", "Italy"),
    "D1": LeagueSource("D1", "openfootball/deutschland", "Bundesliga", "Germany"),
    "D2": LeagueSource("D2", "openfootball/deutschland", "2. Bundesliga", "Germany"),
    "B1": LeagueSource("B1", "openfootball/belgium", "Pro League", "Belgium"),
}

# File path within the season folder, per div. Confirmed by direct listing.
_FILE_BY_DIV: dict[str, str] = {
    "E0": "1-premierleague.txt",
    "E1": "2-championship.txt",
    "SP1": "1-liga.txt",
    "I1": "1-seriea.txt",
    "D1": "1-bundesliga.txt",
    "D2": "2-bundesliga2.txt",
    "B1": "be1.txt",
}


def current_season_label(today: Optional[date] = None) -> str:
    """European club seasons run Jul-Jun; openfootball folders are YYYY-YY."""
    d = today or date.today()
    start_year = d.year if d.month >= 7 else d.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_league_text(div: str, season: Optional[str] = None, timeout: float = 30.0) -> str:
    src = LEAGUE_SOURCES[div]
    fname = _FILE_BY_DIV[div]
    season = season or current_season_label()
    url = RAW_BASE.format(repo=src.repo, path=f"{season}/{fname}")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# --- parsing -----------------------------------------------------------
# Openfootball match lines. Score is OPTIONAL — an upcoming, unplayed fixture
# is a completely valid line with just a kickoff time, home team, "v", and
# away team. This is the one thing africa/parse_football_txt.py's regex does
# not support (it requires a score), which is exactly the case that matters
# for fixture collection.
_MATCH_RE = re.compile(
    r"^\s*(?:(?P<time>\d{1,2}:\d{2})\s+)?"
    r"(?P<home>.+?)\s+v\s+(?P<away>.+?)"
    r"(?:\s+(?P<hg>\d+)\s*[-\u2013:]\s*(?P<ag>\d+)"
    r"(?:\s*\((?P<hh>\d+)\s*[-\u2013:]\s*(?P<ah>\d+)\))?)?"
    r"\s*$"
)
_DATE_FULL_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)?\s*"
    r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
    r"(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s*$",
    re.I,
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MATCHDAY_RE = re.compile(r"^\s*[\u25aa\u2022*-]?\s*(?:Matchday|Round|Week)\s+(?P<n>\d+)", re.I)


@dataclass
class ParsedMatch:
    date: Optional[date]
    time: str
    home: str
    away: str
    fthg: Optional[int]
    ftag: Optional[int]
    hthg: Optional[int]
    htag: Optional[int]
    matchday: Optional[int]
    played: bool


def _clean_team(name: str) -> str:
    return re.sub(r"\s{2,}", " ", name).strip()


def parse_league_text(text: str, season: Optional[str] = None) -> Iterator[ParsedMatch]:
    """Yield every match line, played or not. Carries the current date and
    matchday number forward across lines the way the source file is laid
    out (a date/matchday header applies to every match line beneath it
    until the next header)."""
    season = season or current_season_label()
    year_hint = int(season.split("-")[0])
    cur_date: Optional[date] = None
    cur_matchday: Optional[int] = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        md = _MATCHDAY_RE.match(line)
        if md:
            cur_matchday = int(md.group("n"))
            continue
        dm = _DATE_FULL_RE.match(line)
        if dm:
            mon = _MONTHS[dm.group("mon").lower()[:3]]
            day = int(dm.group("day"))
            yr = int(dm.group("year")) if dm.group("year") else (year_hint if mon >= 7 else year_hint + 1)
            try:
                cur_date = date(yr, mon, day)
            except ValueError:
                cur_date = None
            continue
        mm = _MATCH_RE.match(line)
        if not mm or " v " not in f" {line} ":
            continue
        home = _clean_team(mm.group("home"))
        away = _clean_team(mm.group("away"))
        if not home or not away:
            continue
        hg = mm.group("hg")
        yield ParsedMatch(
            date=cur_date,
            time=mm.group("time") or "",
            home=home,
            away=away,
            fthg=int(hg) if hg is not None else None,
            ftag=int(mm.group("ag")) if mm.group("ag") is not None else None,
            hthg=int(mm.group("hh")) if mm.group("hh") is not None else None,
            htag=int(mm.group("ah")) if mm.group("ah") is not None else None,
            matchday=cur_matchday,
            played=hg is not None,
        )


# --- public entry points ------------------------------------------------

def fetch_results(div: str, season: Optional[str] = None) -> list[dict]:
    """Completed matches only, in master_football_data.parquet's column
    shape (Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG)."""
    text = fetch_league_text(div, season)
    rows = []
    for m in parse_league_text(text, season):
        if not m.played or m.date is None:
            continue
        ftr = "H" if m.fthg > m.ftag else ("A" if m.fthg < m.ftag else "D")
        rows.append({
            "Div": div, "Date": m.date.isoformat(), "HomeTeam": m.home, "AwayTeam": m.away,
            "FTHG": m.fthg, "FTAG": m.ftag, "FTR": ftr,
            "HTHG": m.hthg, "HTAG": m.htag,
        })
    return rows


def fetch_upcoming_fixtures(div: str, season: Optional[str] = None, within_days: int = 14) -> list[dict]:
    """Unplayed fixtures within the next `within_days` days, in
    fixtures_latest.csv's column shape (Div, Date, Time, HomeTeam, AwayTeam).
    No odds — run_fixture_sims.py treats a missing/neutral odds row as
    pure-model (odds_blend=0), not as a reason to drop the fixture."""
    text = fetch_league_text(div, season)
    today = date.today()
    rows = []
    for m in parse_league_text(text, season):
        if m.played or m.date is None:
            continue
        delta = (m.date - today).days
        if delta < -1 or delta > within_days:
            continue
        rows.append({
            "Div": div, "Date": m.date.strftime("%d/%m/%Y"), "Time": m.time,
            "HomeTeam": m.home, "AwayTeam": m.away,
        })
    return rows


def fetch_all_results(season: Optional[str] = None) -> list[dict]:
    out = []
    for div in LEAGUE_SOURCES:
        try:
            out.extend(fetch_results(div, season))
        except Exception as e:
            print(f"[eur_openfootball] {div} results failed: {e}", flush=True)
    return out


def fetch_all_upcoming_fixtures(season: Optional[str] = None, within_days: int = 14) -> list[dict]:
    out = []
    for div in LEAGUE_SOURCES:
        try:
            out.extend(fetch_upcoming_fixtures(div, season, within_days))
        except Exception as e:
            print(f"[eur_openfootball] {div} fixtures failed: {e}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        for div, src in LEAGUE_SOURCES.items():
            try:
                text = fetch_league_text(div)
                matches = list(parse_league_text(text))
                played = sum(1 for m in matches if m.played)
                upcoming = sum(1 for m in matches if not m.played)
                print(f"{div:4s} {src.repo:30s} OK  played={played:4d} upcoming={upcoming:4d}")
            except Exception as e:
                print(f"{div:4s} {src.repo:30s} FAILED: {e}")
    else:
        results = fetch_all_results()
        fixtures = fetch_all_upcoming_fixtures()
        print(f"results: {len(results)}  upcoming fixtures: {len(fixtures)}")
