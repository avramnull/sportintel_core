#!/usr/bin/env python3
"""
Portable API-Football (api-sports v3) client — quota-aware.

Endpoints used:
  GET /fixtures?date=YYYY-MM-DD
  GET /standings?league={id}&season={year}

Env:
  API_FOOTBALL_KEY   required
  API_FOOTBALL_BASE  optional (default https://v3.football.api-sports.io)
  API_FOOTBALL_CACHE_DIR  optional disk cache (default daily_football_data/api_football_cache)
  API_FOOTBALL_MAX_STANDINGS  max standings league pulls per process (default 12)
  API_FOOTBALL_MIN_INTERVAL_SEC  polite delay between requests (default 0.35)

Never logs the key. Safe to import from EUR and Africa pipelines.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = "https://v3.football.api-sports.io"
DEFAULT_CACHE = ROOT / "daily_football_data" / "api_football_cache"

# football-data.co.uk Div → API-Football league id (top European boards)
FDC_DIV_TO_LEAGUE: Dict[str, int] = {
    "E0": 39,   # Premier League
    "E1": 40,   # Championship
    "E2": 41,   # League One
    "E3": 42,   # League Two
    "EC": 43,   # National League
    "SC0": 179, # Premiership
    "SC1": 180, # Championship
    "D1": 78,   # Bundesliga
    "D2": 79,   # 2. Bundesliga
    "SP1": 140, # La Liga
    "SP2": 141, # Segunda
    "I1": 135,  # Serie A
    "I2": 136,  # Serie B
    "F1": 61,   # Ligue 1
    "F2": 62,   # Ligue 2
    "N1": 88,   # Eredivisie
    "B1": 144,  # Jupiler Pro League
    "P1": 94,   # Primeira Liga
    "T1": 203,  # Super Lig
    "G1": 197,  # Super League Greece
}

# Season year for API-Football = start calendar year of the campaign
def api_season_year(as_of: Optional[datetime] = None) -> int:
    d = (as_of or datetime.now(timezone.utc)).date()
    return d.year if d.month >= 7 else d.year - 1


class ApiFootballClient:
    def __init__(
        self,
        key: Optional[str] = None,
        *,
        base: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_standings: Optional[int] = None,
        min_interval: Optional[float] = None,
    ):
        self.key = (key or os.environ.get("API_FOOTBALL_KEY") or "").strip()
        self.base = (base or os.environ.get("API_FOOTBALL_BASE") or DEFAULT_BASE).rstrip("/")
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("API_FOOTBALL_CACHE_DIR")
            or DEFAULT_CACHE
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_standings = int(
            max_standings
            if max_standings is not None
            else os.environ.get("API_FOOTBALL_MAX_STANDINGS", "12")
        )
        self.min_interval = float(
            min_interval
            if min_interval is not None
            else os.environ.get("API_FOOTBALL_MIN_INTERVAL_SEC", "0.35")
        )
        self._last_req = 0.0
        self._standings_calls = 0
        self._req_count = 0

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _headers(self) -> dict:
        return {"x-apisports-key": self.key, "Accept": "application/json"}

    def _throttle(self):
        gap = self.min_interval - (time.time() - self._last_req)
        if gap > 0:
            time.sleep(gap)

    def get(self, path: str, params: Optional[dict] = None, *, use_cache: bool = True, cache_ttl_h: float = 18.0) -> dict:
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY not set")
        params = dict(params or {})
        cache_key = path.strip("/").replace("/", "_") + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        cache_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in cache_key)[:180]
        cache_path = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_path.exists():
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_h <= cache_ttl_h:
                try:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        self._throttle()
        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url, params=params, headers=self._headers(), timeout=35)
        self._last_req = time.time()
        self._req_count += 1
        if r.status_code == 429:
            # brief backoff once
            time.sleep(2.5)
            self._throttle()
            r = requests.get(url, params=params, headers=self._headers(), timeout=35)
            self._last_req = time.time()
            self._req_count += 1
        r.raise_for_status()
        payload = r.json()
        if use_cache:
            try:
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
            except Exception:
                pass
        return payload

    def fixtures_by_date(self, day: str) -> dict:
        """One request: all fixtures for YYYY-MM-DD."""
        return self.get("fixtures", {"date": day}, use_cache=True, cache_ttl_h=6.0)

    def standings(self, league_id: int, season: Optional[int] = None) -> dict:
        if self._standings_calls >= self.max_standings:
            return {"response": [], "errors": {"quota": "max_standings reached"}, "_skipped": True}
        season = season or api_season_year()
        payload = self.get(
            "standings",
            {"league": int(league_id), "season": int(season)},
            use_cache=True,
            cache_ttl_h=18.0,
        )
        # only count network-ish usage when not purely from long-lived cache empty skip
        if not payload.get("_skipped"):
            self._standings_calls += 1
        return payload

    def stats(self) -> dict:
        return {
            "requests": self._req_count,
            "standings_calls": self._standings_calls,
            "max_standings": self.max_standings,
            "cache_dir": str(self.cache_dir),
        }


def parse_standings_table(payload: dict) -> List[Dict[str, Any]]:
    """
    Flatten API-Football standings response into team rows:
      team_id, team, rank, points, played, won, draw, lost, gf, ga, gd, form, league_id, league, season
    """
    rows: List[Dict[str, Any]] = []
    for block in payload.get("response") or []:
        league = block.get("league") or {}
        league_id = league.get("id")
        league_name = league.get("name")
        season = league.get("season")
        groups = league.get("standings") or []
        # standings is list of groups (each group is list of team rows)
        for group in groups:
            if not isinstance(group, list):
                continue
            for t in group:
                team = t.get("team") or {}
                allm = t.get("all") or {}
                goals = allm.get("goals") or {}
                rows.append({
                    "team_id": team.get("id"),
                    "team": team.get("name") or "",
                    "rank": t.get("rank"),
                    "points": t.get("points"),
                    "played": allm.get("played"),
                    "won": allm.get("win"),
                    "draw": allm.get("draw"),
                    "lost": allm.get("lose"),
                    "gf": goals.get("for"),
                    "ga": goals.get("against"),
                    "gd": t.get("goalsDiff"),
                    "form": t.get("form") or "",
                    "league_id": league_id,
                    "league": league_name,
                    "season": season,
                    "description": t.get("description"),
                })
    return rows


def team_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize team name → row (first occurrence)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        name = (r.get("team") or "").strip()
        if not name:
            continue
        key = _norm(name)
        out.setdefault(key, r)
        # also raw
        out.setdefault(name.lower(), r)
    return out


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    for a, b in [
        ("á", "a"), ("à", "a"), ("â", "a"), ("ã", "a"),
        ("é", "e"), ("è", "e"), ("ê", "e"),
        ("í", "i"), ("ì", "i"),
        ("ó", "o"), ("ò", "o"), ("ô", "o"), ("õ", "o"),
        ("ú", "u"), ("ù", "u"),
        ("ç", "c"), ("ñ", "n"),
        ("º", ""), ("ª", ""),
    ]:
        s = s.replace(a, b)
    keep = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
    return " ".join("".join(keep).split())
