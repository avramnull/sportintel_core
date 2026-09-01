#!/usr/bin/env python3
"""
Full RSSSF Africa domestic scraper — all levels (1st/2nd/3rd/…), men + optional women.

Strategy
--------
1. Parse results-afr.html for every tables*/… season page (all levels).
2. Expand priority countries by probing year files (e.g. tablesn/nig2015.html …).
3. Parse every page for Round/date brackets + "Home  n-m  Away" score lines.
4. Infer Div level from nearest section heading (Premier / First / Second / …).

Respectful: single Session, small delay, identifiable User-Agent.
"""
from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

RSSSF_AFRICA_INDEX = "http://www.rsssf.com/results-afr.html"
USER_AGENT = "sportintel-core/africa-etl (+https://github.com/avramnull/sportintel_core; research)"
DEFAULT_DELAY = 0.35

# (country_name, url_prefix without year)  e.g. tablesn/nig + 2024 + .html
COUNTRY_YEAR_PREFIXES: Dict[str, List[str]] = {
    "Nigeria": ["tablesn/nig"],
    "Ghana": ["tablesg/gha", "tablesg/ghana"],
    "Egypt": ["tablese/egy", "tablese/egypt"],
    "Morocco": ["tablesm/mor", "tablesm/moroc"],
    "Algeria": ["tablesa/alg"],
    "South Africa": ["tabless/saf", "tabless/safr", "tabless/southafr"],
    "Kenya": ["tablesk/ken", "tablesk/kenya"],
    "Senegal": ["tabless/sen", "tabless/sene"],
    "Tunisia": ["tablest/tun", "tablest/tuni"],
    "Uganda": ["tablesu/uga", "tablesu/ugan"],
    "Tanzania": ["tablest/tan", "tablest/tanz"],
    "Zambia": ["tablesz/zam", "tablesz/zamb"],
    "Ivory Coast": ["tablesc/civ", "tablesi/ivo", "tablesc/cote"],
    "Cameroon": ["tablesc/cam", "tablesc/camer"],
    "Angola": ["tablesa/ango", "tablesa/ang"],
    "Ethiopia": ["tablese/eth", "tablese/ethi"],
    "Congo-Kinshasa": ["tablesc/congok", "tablesc/drc", "tablesc/zaire"],
    "Congo-Brazzaville": ["tablesc/congob", "tablesc/congo"],
    "Mali": ["tablesm/mali", "tablesm/mal"],
    "Burkina Faso": ["tablesb/burkf", "tablesb/burk"],
    "Benin": ["tablesb/benin", "tablesb/ben"],
    "Botswana": ["tablesb/bots"],
    "Rwanda": ["tablesr/rwa", "tablesr/rwan"],
    "Mozambique": ["tablesm/moz"],
    "Sudan": ["tabless/sud"],
    "Libya": ["tablesl/lib", "tablesl/liby"],
    "Guinea": ["tablesg/gui"],
    "Gabon": ["tablesg/gab"],
    "Namibia": ["tablesn/nam"],
    "Zimbabwe": ["tablesz/zim", "tablesz/zimb"],
}

SCORE_LINE = re.compile(
    r"^(?P<home>.+?)\s+(?P<hg>\d+)\s*[-–]\s*(?P<ag>\d+)\s+(?P<away>.+?)\s*$"
)
DATE_BRACKET = re.compile(r"\[(?P<body>[^\]]+)\]")
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
LEVEL_PATTERNS = [
    (re.compile(r"premier|premiership|ligue\s*1\b|liga\s*1\b|npfl|first\s*level|division\s*1\b|super\s*league|botola\s*pro\s*1|girabola", re.I), "1"),
    (re.compile(r"second\s*level|division\s*2\b|ligue\s*2\b|liga\s*2\b|nnl|championship|botola\s*2", re.I), "2"),
    (re.compile(r"third\s*level|division\s*3\b|ligue\s*3\b", re.I), "3"),
    (re.compile(r"fourth\s*level|division\s*4\b", re.I), "4"),
    (re.compile(r"cup|coupe|federation\s*cup|fa\s*cup", re.I), "CUP"),
    (re.compile(r"women|wom\b|fémin", re.I), "W"),
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    return s


def _strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|tr|h[1-6]|div|li)>", "\n", html)
    html = re.sub(r"(?is)<a\s+name=[\"']([^\"']+)[\"'][^>]*>", r"\n§§SECTION:\1\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&ndash;", "-")
    html = re.sub(r"[ \t]+", " ", html)
    return html


def _parse_rsssf_date(body: str, season_year: Optional[int]) -> Optional[date]:
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


def _infer_level(section: str, page_hint: str = "") -> str:
    blob = f"{section} {page_hint}"
    for rx, code in LEVEL_PATTERNS:
        if rx.search(blob):
            return code
    return "1"


def parse_rsssf_html(
    html: str,
    *,
    country: str,
    season_label: str,
    source_url: str,
    default_level: str = "1",
) -> List[dict]:
    text = _strip_tags(html)
    season_year = None
    m = re.search(r"(20\d{2}|19\d{2})", season_label or "")
    if m:
        season_year = int(m.group(1))
        # filename often end-year → treat as season end
        if season_year and not re.search(r"\d{4}/\d{2,4}", season_label or ""):
            season_label = f"{season_year - 1}/{season_year}"

    rows: List[dict] = []
    current_date: Optional[date] = None
    section = ""
    level = default_level

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("§§SECTION:"):
            section = line.split(":", 1)[1].strip()
            level = _infer_level(section, source_url)
            continue
        # Heading-like lines
        if re.match(r"^(Round|Matchday|Week)\b", line, re.I):
            section = line
            continue
        if re.search(r"\b(First|Second|Third|Fourth)\s+Level\b|\bPremier\b|\bLigue\s*[123]\b", line, re.I):
            section = line
            level = _infer_level(line, source_url)
            continue

        dm = DATE_BRACKET.search(line)
        if dm:
            d = _parse_rsssf_date(dm.group("body"), season_year)
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
        # Standings rows look like: "1.MC Alger 30 20 5 5 41-18 65"
        if re.match(r"^\d+\.", home) or re.match(r"^\d+\.", away):
            continue
        if "Final Table" in home or "NB:" in home or "Champion" in away:
            continue
        if re.match(r"^[\d\.\s\-]+$", home) or re.match(r"^[\d\.\s\-]+$", away):
            continue
        # Reject table residue: too many bare integers on the line
        nums = re.findall(r"\b\d+\b", line)
        if len(nums) > 4:  # score uses 2; allow [aet] noise
            continue
        # Team names must contain a letter
        if not re.search(r"[A-Za-zÀ-ÿ]", home) or not re.search(r"[A-Za-zÀ-ÿ]", away):
            continue
        hg, ag = int(sm.group("hg")), int(sm.group("ag"))
        if hg > 15 or ag > 15:
            continue
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        div = f"AFR{level}" if level not in ("CUP", "W") else f"AFR{level}"
        league = section or f"{country} level {level}"
        rows.append({
            "Source": "rsssf",
            "SourceFile": source_url,
            "Country": country,
            "League": league[:120],
            "Div": div,
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
    # fallback keywords
    for name in COUNTRY_YEAR_PREFIXES:
        if name.lower() in low:
            return name
    return "Unknown"


def discover_index_links(index_html: str, base_url: str = RSSSF_AFRICA_INDEX) -> List[Tuple[str, str, str, str]]:
    """
    Returns list of (country, season_hint, url, level_hint).
    Includes all levels (not only first).
    """
    out: List[Tuple[str, str, str, str]] = []
    for m in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>([^<]*)<', index_html, re.I):
        href, label = m.group(1), m.group(2).strip()
        if "tables" not in href.lower():
            continue
        if href.lower().startswith("http") and "rsssf.com" not in href.lower():
            continue
        url = urljoin(base_url, href.split("#")[0])
        if not url.endswith(".html") and ".html" not in url:
            continue
        # season from filename
        sm = re.search(r"(19|20)(\d{2})", Path(url).name)
        season_hint = ""
        if sm:
            y = int(sm.group(1) + sm.group(2))
            season_hint = f"{y - 1}/{y}"
        pos = m.start()
        window = index_html[max(0, pos - 1000):pos + 200]
        country = _country_from_path(url, window + " " + label)
        level = _infer_level(label + " " + href, url)
        if country == "Unknown":
            # try window country list items
            for name in COUNTRY_YEAR_PREFIXES:
                if re.search(re.escape(name), window, re.I):
                    country = name
                    break
        out.append((country, season_hint, url, level))

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
    years: Iterable[int] = range(2005, 2027),
    delay: float = DEFAULT_DELAY,
) -> List[Tuple[str, str, str, str]]:
    """Probe year pages; lock onto first working URL prefix per country."""
    found: List[Tuple[str, str, str, str]] = []
    for country, prefixes in COUNTRY_YEAR_PREFIXES.items():
        working_pref = None
        for y in sorted(years, reverse=True):  # recent first
            prefs = [working_pref] if working_pref else prefixes
            hit = False
            for pref in prefs:
                if not pref:
                    continue
                path = f"{pref}{y}.html"
                url = urljoin("http://www.rsssf.com/", path)
                try:
                    rr = session.get(url, timeout=12)
                    if rr.status_code == 200 and len(rr.content) > 800:
                        found.append((country, f"{y - 1}/{y}", url, "1"))
                        working_pref = pref
                        hit = True
                        print(f"[rsssf] found {country} {y}: {url}")
                        break
                except Exception:
                    pass
                time.sleep(delay * 0.25)
            if not hit and working_pref is None and y < max(years) - 3:
                # no prefix works for this country recently — skip rest
                break
    return found


def fetch_all_rsssf(
    *,
    include_women: bool = False,
    probe_years: bool = True,
    year_start: int = 2008,
    year_end: int = 2026,
    delay: float = DEFAULT_DELAY,
    max_pages: Optional[int] = None,
) -> List[dict]:
    s = _session()
    print(f"[rsssf] GET index {RSSSF_AFRICA_INDEX}")
    r = s.get(RSSSF_AFRICA_INDEX, timeout=60)
    r.raise_for_status()
    links = discover_index_links(r.text)
    print(f"[rsssf] index links: {len(links)}")

    if probe_years:
        print(f"[rsssf] probing year pages {year_start}–{year_end}…")
        probed = probe_year_urls(s, years=range(year_start, year_end + 1), delay=delay)
        print(f"[rsssf] probed hits: {len(probed)}")
        # merge
        seen = {u for _, _, u, _ in links}
        for item in probed:
            if item[2] not in seen:
                links.append(item)
                seen.add(item[2])

    if not include_women:
        links = [x for x in links if x[3] != "W" and "wom" not in x[2].lower()]

    if max_pages is not None:
        links = links[:max_pages]

    print(f"[rsssf] pages to fetch: {len(links)}")
    rows: List[dict] = []
    for i, (country, season, url, level) in enumerate(links, 1):
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
                default_level=level or "1",
            )
            print(f"[rsssf] ({i}/{len(links)}) {country} {season} → {len(part)}  {url}")
            rows.extend(part)
        except Exception as e:
            print(f"[rsssf] fail {url}: {e}")
    print(f"[rsssf] total raw matches: {len(rows)}")
    return rows


# backward-compatible alias
def fetch_rsssf_priority(session=None, max_pages: int = 80) -> List[dict]:
    return fetch_all_rsssf(probe_years=True, max_pages=max_pages)
