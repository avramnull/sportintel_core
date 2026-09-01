#!/usr/bin/env python3
"""
Parse openfootball Football.TXT match files into normalized row dicts.

Handles variants seen in africa/:
  15:00  Home Team  v  Away Team  2-1
  19:00  Home Team  v  Away Team  2-1 (1-0)
  Mon Sep 18 2023
  08.12.
  28.04.
"""
from __future__ import annotations

import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# League code from filename: 2023-24_ng1.txt → ng1
FNAME_RE = re.compile(r"(?P<season>\d{4}-\d{2}|\d{4})_(?P<code>[a-z0-9]+)\.txt$", re.I)
# Match line: optional time, home, v, away, score, optional HT
MATCH_RE = re.compile(
    r"^\s*(?:(?P<time>\d{1,2}:\d{2})\s+)?"
    r"(?P<home>.+?)\s+v\s+(?P<away>.+?)\s+"
    r"(?P<hg>\d+)\s*[-–:]\s*(?P<ag>\d+)"
    r"(?:\s*\((?P<hh>\d+)\s*[-–:]\s*(?P<ah>\d+)\))?"
    r"\s*$",
    re.I,
)
# Dates
DATE_FULL = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)?\s*"
    r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
    r"(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s*$",
    re.I,
)
DATE_DM = re.compile(r"^\s*(?P<day>\d{1,2})\.(?P<mon>\d{1,2})\.?(?:\s+(?P<year>\d{4}))?\s*$")
HEADER_TITLE = re.compile(r"^=\s*(?P<title>.+?)\s*$")
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

COUNTRY_FROM_DIR = {
    "nigeria": "Nigeria", "egypt": "Egypt", "morocco": "Morocco",
    "algeria": "Algeria", "ghana": "Ghana", "kenya": "Kenya",
    "south-africa": "South Africa", "tunisia": "Tunisia",
    "senegal": "Senegal", "uganda": "Uganda", "tanzania": "Tanzania",
    "zambia": "Zambia", "ivory-coast": "Ivory Coast", "cameroon": "Cameroon",
    "ethiopia": "Ethiopia", "angola": "Angola", "congo": "Congo",
    "champions-league": "CAF", "confederation-cup": "CAF",
}

DIV_LABEL = {
    "ng1": "NPFL", "ng2": "NNL",
    "eg1": "Egyptian Premier League",
    "ma1": "Botola Pro 1",
    "dz1": "Algeria Ligue 1",
    "gh1": "Ghana Premier League",
    "ke1": "Kenyan Premier League",
    "za1": "South African Premiership",
    "tn1": "Tunisia Ligue 1",
    "sn1": "Senegal Ligue 1",
    "ug1": "Uganda Premier League",
    "tz1": "Tanzania Premier League",
    "zm1": "Zambia Super League",
    "cafcl": "CAF Champions League",
    "cafcc": "CAF Confederation Cup",
}


def _season_label_from_fname(season_token: str) -> str:
    if re.match(r"^\d{4}-\d{2}$", season_token):
        y1 = int(season_token[:4])
        y2 = int(season_token[5:7])
        y2_full = (y1 // 100) * 100 + y2
        if y2_full < y1:
            y2_full += 100
        return f"{y1}/{y2_full}"
    if re.match(r"^\d{4}$", season_token):
        return season_token
    return season_token


def _parse_date(line: str, default_year: Optional[int], season_start_year: Optional[int]) -> Optional[date]:
    m = DATE_FULL.match(line)
    if m:
        mon = MONTHS[m.group("mon")[:3].lower()]
        day = int(m.group("day"))
        year = int(m.group("year")) if m.group("year") else default_year
        if year is None and season_start_year:
            # Heuristic: months Jan–Jun belong to season_start+1 if season spans years
            year = season_start_year + (1 if mon <= 6 else 0)
        if year:
            try:
                return date(year, mon, day)
            except ValueError:
                return None
    m = DATE_DM.match(line)
    if m:
        day, mon = int(m.group("day")), int(m.group("mon"))
        year = int(m.group("year")) if m.group("year") else default_year
        if year is None and season_start_year:
            year = season_start_year + (1 if mon <= 6 else 0)
        if year and 1 <= mon <= 12:
            try:
                return date(year, mon, day)
            except ValueError:
                return None
    return None


def parse_file(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    m = FNAME_RE.search(path.name)
    season_token = m.group("season") if m else ""
    league_code = (m.group("code") if m else path.stem).lower()
    season_label = _season_label_from_fname(season_token)
    season_start_year = None
    if re.match(r"^\d{4}", season_token):
        season_start_year = int(season_token[:4])

    country_key = path.parent.name.lower()
    country = COUNTRY_FROM_DIR.get(country_key, country_key.replace("-", " ").title())
    league_name = DIV_LABEL.get(league_code, league_code.upper())

    title = ""
    for line in lines[:5]:
        hm = HEADER_TITLE.match(line)
        if hm:
            title = hm.group("title").strip()
            if "|" in title:
                # "Egypt | Premiership 2023/24"
                parts = [p.strip() for p in title.split("|")]
                if parts:
                    country = parts[0] or country
                if len(parts) > 1:
                    league_name = re.sub(r"\s+\d{4}.*$", "", parts[1]).strip() or league_name
            break

    rows: List[dict] = []
    current_date: Optional[date] = None
    default_year = season_start_year
    stage = ""

    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.strip().startswith("= "):
            continue
        # Stage / matchday headers
        if line.lstrip().startswith("▪"):
            stage = re.sub(r"^[▪\s]+", "", line).strip()
            continue

        d = _parse_date(line.strip(), default_year, season_start_year)
        if d:
            current_date = d
            default_year = d.year
            continue

        mm = MATCH_RE.match(line)
        if not mm or current_date is None:
            continue
        home = re.sub(r"\s+", " ", mm.group("home")).strip()
        away = re.sub(r"\s+", " ", mm.group("away")).strip()
        if not home or not away or home.lower() == away.lower():
            continue
        hg, ag = int(mm.group("hg")), int(mm.group("ag"))
        hh = int(mm.group("hh")) if mm.group("hh") is not None else None
        ah = int(mm.group("ah")) if mm.group("ah") is not None else None
        if hg > ag:
            ftr = "H"
        elif hg < ag:
            ftr = "A"
        else:
            ftr = "D"
        rows.append({
            "Source": "openfootball",
            "SourceFile": path.name,
            "Country": country,
            "League": league_name,
            "Div": league_code.upper(),
            "Season": season_label,
            "Stage": stage,
            "Date": current_date.isoformat(),
            "Time": mm.group("time") or "",
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": ftr,
            "HTHG": hh,
            "HTAG": ah,
        })
    return rows


def iter_africa_txt(root: Path) -> Iterator[Path]:
    africa = root / "africa" if (root / "africa").is_dir() else root
    yield from sorted(africa.rglob("*.txt"))


def parse_openfootball_tree(root: Path) -> List[dict]:
    all_rows: List[dict] = []
    for path in iter_africa_txt(root):
        try:
            all_rows.extend(parse_file(path))
        except Exception as e:
            print(f"[warn] openfootball parse {path}: {e}")
    return all_rows
