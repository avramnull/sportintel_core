#!/usr/bin/env python3
"""
RSSSF Africa domestic scraper — levels 1 & 2 only (premiere / deuxieme).

Base: https://www.rsssf.org/
Examples:
  https://www.rsssf.org/tablesa/alg2026.html#premiere
  https://www.rsssf.org/tablesa/alg2026.html#deuxieme
  https://www.rsssf.org/tablesa/ango2027.html#girabola
"""
from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

RSSSF_BASE = "https://www.rsssf.org/"
RSSSF_AFRICA_INDEX = urljoin(RSSSF_BASE, "results-afr.html")
USER_AGENT = "sportintel-core/africa-etl (+https://github.com/avramnull/sportintel_core; research)"
DEFAULT_DELAY = 0.35

# Level-1 / level-2 section anchors & labels (case-insensitive)
LEVEL1_KEYS = re.compile(
    r"premiere|premier|premiership|ligue\s*1|liga\s*1|girabola|npfl|"
    r"first\s*level|division\s*1|super\s*league|botola\s*pro|fkf\s*premier|"
    r"lig\s*1|serie\s*a\b|top\s*league",
    re.I,
)
LEVEL2_KEYS = re.compile(
    r"deuxieme|deuxième|second\s*level|ligue\s*2|liga\s*2|division\s*2|"
    r"nnl|championship|botola\s*2|super\s*league|1st\s*division",
    re.I,
)
SKIP_SECTION = re.compile(
    r"troisieme|troisième|quatrieme|quatrième|third|fourth|women|wom\b|"
    r"fémin|cup|coupe|super\s*cup|play.?off",
    re.I,
)

COUNTRY_YEAR_PREFIXES: Dict[str, List[str]] = {
    "Nigeria": ["tablesn/nig"],
    "Ghana": ["tablesg/gha", "tablesg/ghana"],
    "Egypt": ["tablese/egy", "tablese/egypt"],
    "Morocco": ["tablesm/mor", "tablesm/moroc"],
    "Algeria": ["tablesa/alg"],
    "South Africa": ["tabless/saf", "tabless/safr"],
    "Kenya": ["tablesk/ken"],
    "Senegal": ["tabless/sen"],
    "Tunisia": ["tablest/tun"],
    "Uganda": ["tablesu/uga"],
    "Tanzania": ["tablest/tan", "tablest/tanz"],
    "Zambia": ["tablesz/zam"],
    "Ivory Coast": ["tablesc/civ", "tablesi/ivo"],
    "Cameroon": ["tablesc/cam"],
    "Angola": ["tablesa/ango", "tablesa/ang"],
    "Ethiopia": ["tablese/eth"],
    "Mali": ["tablesm/mali"],
    "Burkina Faso": ["tablesb/burkf"],
    "Benin": ["tablesb/benin"],
    "Botswana": ["tablesb/bots"],
    "Rwanda": ["tablesr/rwa"],
    "Mozambique": ["tablesm/moz"],
    "Sudan": ["tabless/sud"],
    "Libya": ["tablesl/lib"],
    "Guinea": ["tablesg/gui"],
    "Gabon": ["tablesg/gab"],
    "Zimbabwe": ["tablesz/zim"],
    "Congo": ["tablesc/congo"],
    "DR Congo": ["tablesc/congok", "tablesc/drc"],
}

SCORE_LINE = re.compile(
    r"^(?P<home>.+?)\s+(?P<hg>\d+)\s*[-–]\s*(?P<ag>\d+)\s+(?P<away>.+?)\s*$"
)
DATE_BRACKET = re.compile(r"\[(?P<body>[^\]]+)\]")
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    return s


def _strip_keep_sections(html: str) -> str:
    """Strip tags but keep <a name=...> as section markers."""
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<a\s+name=[\"']([^\"']+)[\"'][^>]*>", r"\n§§SECTION:\1\n", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|tr|h[1-6]|div|li|pre)>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&ndash;", "-")
    html = re.sub(r"[ \t]+", " ", html)
    return html


def _parse_date(body: str, season_year: Optional[int]) -> Optional[date]:
    body = body.strip()
    m = re.search(
        r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
        r"(?P<day>\d{1,2})(?:\s*,?\s*(?P<year>\d{4}))?",
        body, re.I,
    )
    if not m:
        m = re.search(
            r"(?P<day>\d{1,2})\s+"
            r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
            r"(?:\s*(?P<year>\d{4}))?",
            body, re.I,
        )
    if not m:
        return None
    mon = MONTHS[m.group("mon")[:3].lower()]
    day = int(m.group("day"))
    year = int(m.group("year")) if m.group("year") else season_year
    if year is None:
        return None
    if m.group("year") is None and season_year and mon <= 6:
        year = season_year + 1
    try:
        return date(year, mon, day)
    except ValueError:
        return None


def _level_from_section(section: str) -> Optional[str]:
    """Return '1', '2', or None (skip)."""
    if not section:
        return "1"  # default top of page often L1
    if SKIP_SECTION.search(section) and not LEVEL1_KEYS.search(section) and not LEVEL2_KEYS.search(section):
        return None
    if LEVEL2_KEYS.search(section):
        return "2"
    if LEVEL1_KEYS.search(section):
        return "1"
    # unknown section after an L1/L2 — keep if we're already in L1/L2 context
    return None


def parse_rsssf_html(
    html: str,
    *,
    country: str,
    season_label: str,
    source_url: str,
) -> List[dict]:
    text = _strip_keep_sections(html)
    season_year = None
    m = re.search(r"(20\d{2}|19\d{2})", season_label or source_url or "")
    if m:
        season_year = int(m.group(1))
        if season_label and not re.search(r"\d{4}/\d", season_label):
            season_label = f"{season_year - 1}/{season_year}"

    rows: List[dict] = []
    current_date: Optional[date] = None
    section = "premiere"
    level: Optional[str] = "1"  # pages usually start with top flight

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("§§SECTION:"):
            section = line.split(":", 1)[1].strip()
            lv = _level_from_section(section)
            # Sub-groups (e.g. 2ce under deuxieme) inherit parent L1/L2
            if lv is None and level in ("1", "2") and not SKIP_SECTION.search(section):
                pass  # keep level
            else:
                level = lv
            continue
        if re.match(r"^(Round|Matchday|Week)\b", line, re.I):
            # stay in current level
            continue
        if re.search(r"\b(First|Second|Third|Fourth)\s+Level\b|\bPremier\b|\bLigue\s*[123]\b", line, re.I):
            section = line
            level = _level_from_section(line)
            continue

        if level not in ("1", "2"):
            # still scan dates so we don't lose calendar if section flips later
            dm = DATE_BRACKET.search(line)
            if dm:
                d = _parse_date(dm.group("body"), season_year)
                if d:
                    current_date = d
            continue

        dm = DATE_BRACKET.search(line)
        if dm:
            d = _parse_date(dm.group("body"), season_year)
            if d:
                current_date = d
            line = DATE_BRACKET.sub(" ", line).strip()
            if not line:
                continue

        sm = SCORE_LINE.match(line)
        if not sm or current_date is None:
            continue
        home = re.sub(r"\s+", " ", sm.group("home")).strip()
        away = re.sub(r"\s+", " ", sm.group("away")).strip()
        if len(home) < 2 or len(away) < 2:
            continue
        if re.match(r"^\d+\.", home) or re.match(r"^\d+\.", away):
            continue
        if "Final Table" in home or "NB:" in home:
            continue
        nums = re.findall(r"\b\d+\b", line)
        if len(nums) > 4:
            continue
        if not re.search(r"[A-Za-zÀ-ÿ]", home) or not re.search(r"[A-Za-zÀ-ÿ]", away):
            continue
        hg, ag = int(sm.group("hg")), int(sm.group("ag"))
        if hg > 15 or ag > 15:
            continue
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        rows.append({
            "Source": "rsssf",
            "SourceFile": source_url,
            "Country": country,
            "League": f"{country} Level {level}",
            "Div": f"AFR{level}",
            "Season": season_label or "unknown",
            "Stage": section[:80],
            "Date": current_date.isoformat(),
            "Time": "",
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": ftr,
            "HTHG": None,
            "HTAG": None,
            "Level": level,
        })
    return rows


def _country_from_path(path: str, surrounding: str = "") -> str:
    low = (path + " " + surrounding).lower()
    for name, prefixes in COUNTRY_YEAR_PREFIXES.items():
        for pref in prefixes:
            key = pref.split("/")[-1]
            if key and key in low:
                return name
    for name in COUNTRY_YEAR_PREFIXES:
        if name.lower() in low:
            return name
    return "Unknown"


def discover_index_links(index_html: str) -> List[Tuple[str, str, str]]:
    """(country, season_hint, url) — pages that mention first/second level."""
    out: List[Tuple[str, str, str]] = []
    for m in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>([^<]*)<', index_html, re.I):
        href, label = m.group(1), m.group(2).strip()
        if "tables" not in href.lower():
            continue
        # keep first + second level links; skip women / third+
        if re.search(r"wom|third|fourth|troisi|quatri", href + label, re.I):
            continue
        if not re.search(
            r"first|second|premier|premiere|deuxieme|girabola|ligue|liga|super",
            href + " " + label,
            re.I,
        ):
            # still allow generic season pages (contain results for L1/L2)
            if not re.search(r"tables[a-z]/\w+\d{4}", href, re.I):
                continue
        url = urljoin(RSSSF_AFRICA_INDEX, href.split("#")[0])
        if ".html" not in url:
            continue
        sm = re.search(r"(19|20)(\d{2})", Path(url).name)
        season = f"{int(sm.group(1)+sm.group(2))-1}/{sm.group(1)+sm.group(2)}" if sm else ""
        window = index_html[max(0, m.start() - 900):m.start() + 120]
        country = _country_from_path(url, window + " " + label)
        out.append((country, season, url))
    seen: Set[str] = set()
    uniq = []
    for item in out:
        if item[2] in seen:
            continue
        seen.add(item[2])
        uniq.append(item)
    return uniq


def probe_year_urls(
    session: requests.Session,
    years: Iterable[int],
    delay: float = DEFAULT_DELAY,
) -> List[Tuple[str, str, str]]:
    found: List[Tuple[str, str, str]] = []
    for country, prefixes in COUNTRY_YEAR_PREFIXES.items():
        working = None
        for y in sorted(years, reverse=True):
            prefs = [working] if working else prefixes
            hit = False
            for pref in prefs:
                if not pref:
                    continue
                url = urljoin(RSSSF_BASE, f"{pref}{y}.html")
                try:
                    rr = session.get(url, timeout=20)
                    if rr.status_code == 200 and len(rr.content) > 800:
                        found.append((country, f"{y-1}/{y}", url))
                        working = pref
                        hit = True
                        print(f"[rsssf] found {country} {y}")
                        break
                except Exception:
                    pass
                time.sleep(delay * 0.2)
            if not hit and working is None and y < max(years) - 2:
                break
    return found


def fetch_rsssf_levels_1_2(
    *,
    year_start: int = 2014,
    year_end: int = 2027,
    delay: float = DEFAULT_DELAY,
    max_pages: Optional[int] = None,
    probe_years: bool = True,
) -> List[dict]:
    s = _session()
    print(f"[rsssf] GET {RSSSF_AFRICA_INDEX}")
    r = s.get(RSSSF_AFRICA_INDEX, timeout=60)
    r.raise_for_status()
    links = discover_index_links(r.text)
    print(f"[rsssf] index links: {len(links)}")
    if probe_years:
        print(f"[rsssf] probing years {year_start}–{year_end}")
        probed = probe_year_urls(s, range(year_start, year_end + 1), delay=delay)
        seen = {u for _, _, u in links}
        for item in probed:
            if item[2] not in seen:
                links.append(item)
                seen.add(item[2])
        print(f"[rsssf] after probe: {len(links)} pages")
    if max_pages is not None:
        links = links[:max_pages]

    rows: List[dict] = []
    for i, (country, season, url) in enumerate(links, 1):
        try:
            time.sleep(delay)
            rr = s.get(url, timeout=45)
            if rr.status_code != 200:
                print(f"[rsssf] ({i}/{len(links)}) HTTP {rr.status_code} {url}")
                continue
            part = parse_rsssf_html(
                rr.text,
                country=country if country != "Unknown" else _country_from_path(url),
                season_label=season or "unknown",
                source_url=url,
            )
            print(f"[rsssf] ({i}/{len(links)}) {country} {season} → {len(part)}  {url}")
            rows.extend(part)
        except Exception as e:
            print(f"[rsssf] fail {url}: {e}")
    print(f"[rsssf] total L1+L2 matches: {len(rows)}")
    return rows


# aliases for older imports
def fetch_all_rsssf(**kwargs) -> List[dict]:
    return fetch_rsssf_levels_1_2(**kwargs)


def fetch_rsssf_priority(session=None, max_pages: int = 80) -> List[dict]:
    return fetch_rsssf_levels_1_2(max_pages=max_pages, probe_years=True)
