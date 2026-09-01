#!/usr/bin/env python3
"""
Parse RSSSF Africa domestic result pages (HTML) into normalized rows.

RSSSF pages mix final tables and round-by-round results like:
  Round 1
  [Sep 30]
  Bayelsa United          5-3 Akwa United

We only extract score lines; tables are ignored.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

RSSSF_AFRICA_INDEX = "http://www.rsssf.com/results-afr.html"
USER_AGENT = "sportintel-core/africa-etl (+https://github.com/avramnull/sportintel_core)"

# First-level domestic focus (extend as needed)
PRIORITY_COUNTRIES = {
    "nigeria": "Nigeria",
    "ghana": "Ghana",
    "egypt": "Egypt",
    "morocco": "Morocco",
    "algeria": "Algeria",
    "south africa": "South Africa",
    "kenya": "Kenya",
    "senegal": "Senegal",
    "tunisia": "Tunisia",
    "uganda": "Uganda",
    "tanzania": "Tanzania",
    "zambia": "Zambia",
    "ivory coast": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "cameroon": "Cameroon",
}

SCORE_LINE = re.compile(
    r"^(?P<home>.+?)\s+(?P<hg>\d+)\s*[-–]\s*(?P<ag>\d+)\s+(?P<away>.+?)\s*$"
)
DATE_BRACKET = re.compile(
    r"\[(?P<body>[^\]]+)\]"
)
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    return s


def _strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p>", "\n", html)
    html = re.sub(r"(?is)</tr>", "\n", html)
    html = re.sub(r"(?is)</h[1-6]>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&")
    html = re.sub(r"[ \t]+", " ", html)
    return html


def _parse_rsssf_date(body: str, season_year: Optional[int]) -> Optional[date]:
    body = body.strip()
    # e.g. Sep 30  or  Sep 30, 2023  or  30 Sep 2023
    m = re.search(
        r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
        r"(?P<day>\d{1,2})(?:\s*,?\s*(?P<year>\d{4}))?",
        body,
        re.I,
    )
    if not m:
        m = re.search(
            r"(?P<day>\d{1,2})\s+"
            r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
            r"(?:\s*(?P<year>\d{4}))?",
            body,
            re.I,
        )
    if not m:
        return None
    mon = MONTHS[m.group("mon")[:3].lower()]
    day = int(m.group("day"))
    year = int(m.group("year")) if m.group("year") else season_year
    if year is None:
        return None
    # season spanning: if season_year is start year and month early, +1
    if m.group("year") is None and season_year and mon <= 6:
        year = season_year + 1
    try:
        return date(year, mon, day)
    except ValueError:
        return None


def parse_rsssf_html(
    html: str,
    *,
    country: str,
    season_label: str,
    source_url: str,
    league: str = "Domestic",
    div: str = "AFR1",
) -> List[dict]:
    text = _strip_tags(html)
    season_year = None
    m = re.search(r"(20\d{2})", season_label)
    if m:
        season_year = int(m.group(1))

    rows: List[dict] = []
    current_date: Optional[date] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        dm = DATE_BRACKET.search(line)
        if dm:
            d = _parse_rsssf_date(dm.group("body"), season_year)
            if d:
                current_date = d
            # date line may also contain a match after the bracket — rare
            line = DATE_BRACKET.sub(" ", line).strip()
            if not line:
                continue
        sm = SCORE_LINE.match(line)
        if not sm or current_date is None:
            continue
        home = re.sub(r"\s+", " ", sm.group("home")).strip()
        away = re.sub(r"\s+", " ", sm.group("away")).strip()
        # filter table-like garbage
        if len(home) < 2 or len(away) < 2:
            continue
        if re.search(r"^\d+\.", home) or "Final Table" in home:
            continue
        hg, ag = int(sm.group("hg")), int(sm.group("ag"))
        if hg > 20 or ag > 20:
            continue
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        rows.append({
            "Source": "rsssf",
            "SourceFile": source_url,
            "Country": country,
            "League": league,
            "Div": div,
            "Season": season_label,
            "Stage": "",
            "Date": current_date.isoformat(),
            "Time": "",
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": ftr,
            "HTHG": None,
            "HTAG": None,
        })
    return rows


def discover_first_level_links(index_html: str, base_url: str = RSSSF_AFRICA_INDEX) -> List[Tuple[str, str, str]]:
    """
    Return list of (country, season_hint, url) for first-level table links.
    Heuristic: anchor text contains 'First Level' near a country heading.
    """
    out: List[Tuple[str, str, str]] = []
    # Split by country-ish list items is hard; collect all tablesa/tablesb links with season years
    for m in re.finditer(
        r'href=["\']([^"\']+?)["\'][^>]*>([^<]*First Level[^<]*)<',
        index_html,
        re.I,
    ):
        href, label = m.group(1), m.group(2)
        if "wom" in href.lower() or "women" in label.lower():
            continue
        url = urljoin(base_url, href.split("#")[0])
        # season from filename alg2026.html → 2025/26 heuristic
        sm = re.search(r"(19|20)(\d{2})", Path(url).name)
        season_hint = ""
        if sm:
            y = int(sm.group(1) + sm.group(2))
            # RSSSF often uses end year in filename
            season_hint = f"{y-1}/{y}"
        # country: walk backward in html for last country name before this match
        pos = m.start()
        window = index_html[max(0, pos - 800):pos]
        country = "Unknown"
        for key, nice in PRIORITY_COUNTRIES.items():
            if re.search(re.escape(key), window, re.I):
                country = nice
        if country == "Unknown":
            continue
        out.append((country, season_hint, url))
    # dedupe by url
    seen = set()
    uniq = []
    for item in out:
        if item[2] in seen:
            continue
        seen.add(item[2])
        uniq.append(item)
    return uniq


def fetch_rsssf_priority(
    session: Optional[requests.Session] = None,
    max_pages: int = 80,
) -> List[dict]:
    s = session or _session()
    print(f"[rsssf] GET {RSSSF_AFRICA_INDEX}")
    r = s.get(RSSSF_AFRICA_INDEX, timeout=60)
    r.raise_for_status()
    links = discover_first_level_links(r.text)
    print(f"[rsssf] discovered {len(links)} first-level links (priority countries)")
    rows: List[dict] = []
    for i, (country, season, url) in enumerate(links[:max_pages]):
        try:
            rr = s.get(url, timeout=45)
            if rr.status_code != 200:
                print(f"[rsssf] skip {url} HTTP {rr.status_code}")
                continue
            part = parse_rsssf_html(
                rr.text,
                country=country,
                season_label=season or "unknown",
                source_url=url,
                league=f"{country} First Level",
                div="AFR1",
            )
            print(f"[rsssf] {country} {season or '?'} → {len(part)} matches ({url})")
            rows.extend(part)
        except Exception as e:
            print(f"[rsssf] fail {url}: {e}")
    return rows
