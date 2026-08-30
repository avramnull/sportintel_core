#!/usr/bin/env python3
"""Accurate team + league mapping for football-data.co.uk strings.

Canonical names follow historical FDC UK spellings in master parquet / team2id.
Aliases only bridge known variants — never invent fuzzy wrong clubs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent
MAP_DIR = ROOT / "football_models" / "mappings"

# football-data.co.uk Div → display name
LEAGUE_MAP: Dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Süper Lig",
    "G1": "Super League Greece",
}

# Variants → FDC canonical (as used in historical CSVs / team2id)
TEAM_ALIASES: Dict[str, str] = {
    # England
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "manchester utd": "Manchester United",
    "man utd": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "tottenham hotspurs": "Tottenham",
    "nottm forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "nott forest": "Nott'm Forest",
    "nott_m_forest": "Nott'm Forest",
    "wolves": "Wolverhampton",
    "wolverhampton": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
    "west brom": "West Brom",
    "west bromwich albion": "West Brom",
    "sheffield utd": "Sheffield United",
    "sheff utd": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "sheff wed": "Sheffield Weds",
    "qpr": "QPR",
    "queens park rangers": "QPR",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "leicester city": "Leicester",
    "leeds united": "Leeds",
    "leicester": "Leicester",
    "nottingham": "Nott'm Forest",
    "birmingham city": "Birmingham",
    "blackburn rovers": "Blackburn",
    "bolton wanderers": "Bolton",
    "charlton athletic": "Charlton",
    "huddersfield town": "Huddersfield",
    "ipswich town": "Ipswich",
    "norwich city": "Norwich",
    "stoke city": "Stoke",
    "swansea city": "Swansea",
    "wigan athletic": "Wigan",
    "cardiff city": "Cardiff",
    "hull city": "Hull",
    "derby county": "Derby",
    "preston north end": "Preston",
    "peterborough united": "Peterboro",
    "peterborough": "Peterboro",
    "milton keynes dons": "MK Dons",
    "mk dons": "MK Dons",
    "crawley": "Crawley Town",
    "afc wimbledon": "AFC Wimbledon",
    # Spain
    "ath madrid": "Atletico Madrid",
    "atletico madrid": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "atleti": "Atletico Madrid",
    "ath bilbao": "Athletic Bilbao",
    "athletic bilbao": "Athletic Bilbao",
    "athletic club": "Athletic Bilbao",
    "celta": "Celta Vigo",
    "celta vigo": "Celta Vigo",
    "rc celta": "Celta Vigo",
    "sociedad": "Real Sociedad",
    "real sociedad": "Real Sociedad",
    "betis": "Real Betis",
    "real betis": "Real Betis",
    "espanyol": "Espanol",
    "rcd espanyol": "Espanol",
    "valencia cf": "Valencia",
    "villarreal cf": "Villarreal",
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "real madrid": "Real Madrid",
    # Germany
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "bayern": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "bvb": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "borussia mönchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "gladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "ein frankfurt": "Ein Frankfurt",
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "fc koln": "FC Koln",
    "fc köln": "FC Koln",
    "cologne": "FC Koln",
    "1. fc koln": "FC Koln",
    "1. fc köln": "FC Koln",
    "schalke": "Schalke 04",
    "fc schalke 04": "Schalke 04",
    "werder bremen": "Werder Bremen",
    "mainz 05": "Mainz",
    "fsv mainz 05": "Mainz",
    "union berlin": "Union Berlin",
    "1. fc union berlin": "Union Berlin",
    "bayer leverkusen": "Leverkusen",
    "leverkusen": "Leverkusen",
    "hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim",
    "freiburg": "Freiburg",
    "sc freiburg": "Freiburg",
    "wolfsburg": "Wolfsburg",
    "vfl wolfsburg": "Wolfsburg",
    "stuttgart": "Stuttgart",
    "vfb stuttgart": "Stuttgart",
    "augsburg": "Augsburg",
    "fc augsburg": "Augsburg",
    "hamburger sv": "Hamburg",
    "hsv": "Hamburg",
    # Italy
    "inter": "Inter",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "as roma": "Roma",
    "as roma": "Roma",
    "ssc napoli": "Napoli",
    "juventus": "Juventus",
    "juve": "Juventus",
    # France
    "paris sg": "Paris SG",
    "psg": "Paris SG",
    "paris saint-germain": "Paris SG",
    "paris saint germain": "Paris SG",
    "olympique lyonnais": "Lyon",
    "olympique marseille": "Marseille",
    "om": "Marseille",
    "ol": "Lyon",
    "st etienne": "St Etienne",
    "saint-etienne": "St Etienne",
    "saint etienne": "St Etienne",
    # Netherlands
    "psv": "PSV Eindhoven",
    "psv eindhoven": "PSV Eindhoven",
    "ajax": "Ajax",
    "afc ajax": "Ajax",
    "feyenoord": "Feyenoord",
    "az": "AZ Alkmaar",
    "az alkmaar": "AZ Alkmaar",
    "fc utrecht": "Utrecht",
    "utrecht": "Utrecht",
    "fc twente": "Twente",
    "sc heerenveen": "Heerenveen",
    "willem ii": "Willem II",
    # Belgium
    "club brugge": "Club Brugge",
    "club brugge kv": "Club Brugge",
    "cercle brugge": "Cercle Brugge",
    "standard liege": "Standard",
    "standard liège": "Standard",
    "standard": "Standard",
    "anderlecht": "Anderlecht",
    "rsc anderlecht": "Anderlecht",
    "genk": "Genk",
    "krc genk": "Genk",
    "gent": "Gent",
    "kaa gent": "Gent",
    "union sg": "St. Gilloise",
    "union saint-gilloise": "St. Gilloise",
    "st gilloise": "St. Gilloise",
    "st. gilloise": "St. Gilloise",
    "oh leuven": "Oud-Heverlee Leuven",
    "oud heverlee leuven": "Oud-Heverlee Leuven",
    "raal la louviere": "RAAL La Louviere",
    "la louviere": "RAAL La Louviere",
    "st truiden": "St Truiden",
    "sint-truiden": "St Truiden",
    "lommel": "Lommel SK",
    "lommel sk": "Lommel SK",
    # Portugal
    "sporting": "Sp Lisbon",
    "sporting cp": "Sp Lisbon",
    "sporting lisbon": "Sp Lisbon",
    "sp lisbon": "Sp Lisbon",
    "porto": "Porto",
    "fc porto": "Porto",
    "benfica": "Benfica",
    "sl benfica": "Benfica",
    # Scotland
    "rangers": "Rangers",
    "glasgow rangers": "Rangers",
    "celtic": "Celtic",
    "celtic fc": "Celtic",
    "hearts": "Hearts",
    "heart of midlothian": "Hearts",
    "hibernian": "Hibernian",
    "hibs": "Hibernian",
    # Turkey
    "galatasaray": "Galatasaray",
    "fenerbahce": "Fenerbahce",
    "besiktas": "Besiktas",
    "trabzonspor": "Trabzonspor",
}


def clean_name(name: str) -> str:
    s = str(name or "").replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_team2id() -> Dict[str, int]:
    p = MAP_DIR / "team2id.json"
    if not p.exists():
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(p.read_text()).items()}
    except Exception:
        return {}


def load_aliases() -> Dict[str, str]:
    aliases = dict(TEAM_ALIASES)
    p = MAP_DIR / "team_aliases.json"
    if p.exists():
        try:
            for k, v in json.loads(p.read_text()).items():
                aliases[str(k).lower().strip()] = str(v).strip()
        except Exception:
            pass
    return aliases


def _ci_index(names) -> Dict[str, str]:
    idx = {}
    for n in names:
        idx[n.lower()] = n
    return idx


def normalize_team(name: str, aliases: Optional[Dict[str, str]] = None,
                   known: Optional[Dict[str, int]] = None) -> Tuple[str, str]:
    """
    Returns (canonical_name, status)
    status: exact | alias | case | known | raw
    """
    raw = clean_name(name)
    if not raw or raw.lower() in ("nan", "none", "-"):
        return raw, "empty"

    aliases = aliases if aliases is not None else load_aliases()
    known = known if known is not None else load_team2id()
    known_ci = _ci_index(known.keys()) if known else {}

    # 1) Exact as stored in team2id
    if raw in known:
        return raw, "exact"

    key = raw.lower()

    # 2) Alias → canonical
    if key in aliases:
        canon = aliases[key]
        if not known or canon in known or canon.lower() in known_ci:
            if known and canon not in known and canon.lower() in known_ci:
                canon = known_ci[canon.lower()]
            return canon, "alias"
        return canon, "alias"

    # 3) Case-insensitive match to known
    if key in known_ci:
        return known_ci[key], "case"

    # 4) Keep FDC spelling as-is (do NOT title-case — breaks M'gladbach, Nott'm Forest)
    return raw, "raw"


def resolve_league(div_or_name: str) -> Tuple[str, str]:
    """Return (div_code, league_display_name)."""
    s = clean_name(div_or_name)
    if not s:
        return "", "Unknown"
    # already a div code
    if s in LEAGUE_MAP:
        return s, LEAGUE_MAP[s]
    # reverse lookup by display name
    low = s.lower()
    for code, name in LEAGUE_MAP.items():
        if name.lower() == low:
            return code, name
    # partial
    for code, name in LEAGUE_MAP.items():
        if low in name.lower() or name.lower() in low:
            return code, name
    return s, s


def slug(name: str) -> str:
    s = clean_name(name).lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug_s = "".join(out).strip("_")
    while "__" in slug_s:
        slug_s = slug_s.replace("__", "_")
    return slug_s or "team"


def write_maps(team2id: Dict[str, int]) -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    (MAP_DIR / "team2id.json").write_text(json.dumps(team2id, indent=2, ensure_ascii=False))
    (MAP_DIR / "id2team.json").write_text(
        json.dumps({str(i): t for t, i in team2id.items()}, indent=2, ensure_ascii=False)
    )
    (MAP_DIR / "league_map.json").write_text(json.dumps(LEAGUE_MAP, indent=2, ensure_ascii=False))
    (MAP_DIR / "team_aliases.json").write_text(json.dumps(TEAM_ALIASES, indent=2, ensure_ascii=False))



# Post-load corrections so aliases always point at team2id keys
ALIAS_CORRECTIONS = {
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "manchester utd": "Manchester United",
    "man utd": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "ath madrid": "Atletico Madrid",
    "atletico madrid": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "atleti": "Atletico Madrid",
    "ath bilbao": "Athletic Bilbao",
    "athletic bilbao": "Athletic Bilbao",
    "athletic club": "Athletic Bilbao",
    "celta": "Celta Vigo",
    "celta vigo": "Celta Vigo",
    "rc celta": "Celta Vigo",
    "betis": "Real Betis",
    "real betis": "Real Betis",
    "wolves": "Wolverhampton",
    "wolverhampton": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
    "sociedad": "Real Sociedad",
    "real sociedad": "Real Sociedad",
}
TEAM_ALIASES.update(ALIAS_CORRECTIONS)
