#!/usr/bin/env python3
"""
Portable API-Football (api-sports v3) client — quota-aware.

Endpoints used:
  GET /fixtures?date=YYYY-MM-DD     (1 call for the whole day board)
  GET /standings?league={id}&season={year}
      → returns the FULL league table once (e.g. all ~20 Premier League sides).
      One league_id = one request, no matter how many fixture teams share that league.

Env:
  API_FOOTBALL_KEY   required
  API_FOOTBALL_BASE  optional (default https://v3.football.api-sports.io)
  API_FOOTBALL_CACHE_DIR  optional (default daily_football_data/api_football_cache)
  API_FOOTBALL_MAX_STANDINGS
      max *network* standings pulls this process may perform.
      0 = no cap (fetch every unique league that has fixtures today).
      Default 0 — one full table per unique league is already minimal.
  API_FOOTBALL_MIN_INTERVAL_SEC  polite delay between live requests (default 0.40)

Never logs the key. Safe to import from EUR and Africa pipelines.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def api_season_year(as_of: Optional[datetime] = None) -> int:
    """API-Football season = campaign start calendar year (Jul–Jun)."""
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
        # 0 = unlimited unique leagues (still one request per league, never per team)
        raw_max = (
            max_standings
            if max_standings is not None
            else os.environ.get("API_FOOTBALL_MAX_STANDINGS", "0")
        )
        try:
            self.max_standings = int(raw_max)
        except (TypeError, ValueError):
            self.max_standings = 0
        self.min_interval = float(
            min_interval
            if min_interval is not None
            else os.environ.get("API_FOOTBALL_MIN_INTERVAL_SEC", "0.40")
        )
        self._last_req = 0.0
        self._network_standings = 0  # only live HTTP standings calls
        self._cache_standings = 0
        self._req_count = 0
        self._standings_seen: set = set()  # league_id already resolved this process

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _headers(self) -> dict:
        return {"x-apisports-key": self.key, "Accept": "application/json"}

    def _throttle(self):
        gap = self.min_interval - (time.time() - self._last_req)
        if gap > 0:
            time.sleep(gap)

    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        *,
        use_cache: bool = True,
        cache_ttl_h: float = 18.0,
    ) -> Tuple[dict, bool]:
        """
        Returns (payload, from_cache).
        from_cache=True means no network quota was spent.
        """
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY not set")
        params = dict(params or {})
        cache_key = path.strip("/").replace("/", "_") + "_" + "_".join(
            f"{k}-{v}" for k, v in sorted(params.items())
        )
        cache_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in cache_key)[:180]
        cache_path = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_path.exists():
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_h <= cache_ttl_h:
                try:
                    return json.loads(cache_path.read_text(encoding="utf-8")), True
                except Exception:
                    pass

        self._throttle()
        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url, params=params, headers=self._headers(), timeout=35)
        self._last_req = time.time()
        self._req_count += 1
        if r.status_code == 429:
            time.sleep(3.0)
            self._throttle()
            r = requests.get(url, params=params, headers=self._headers(), timeout=35)
            self._last_req = time.time()
            self._req_count += 1
        r.raise_for_status()
        payload = r.json()
        # API-Football often returns HTTP 200 with errors: {access: "..."} when suspended/quota
        errs = payload.get("errors")
        if errs:
            # Don't cache hard failures
            msg = errs if isinstance(errs, str) else json.dumps(errs)
            raise RuntimeError(f"API-Football error: {msg}")
        if use_cache:
            try:
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
            except Exception:
                pass
        return payload, False

    def fixtures_by_date(self, day: str) -> dict:
        """One request: all fixtures for YYYY-MM-DD."""
        payload, _ = self.get("fixtures", {"date": day}, use_cache=True, cache_ttl_h=6.0)
        return payload

    def standings(self, league_id: int, season: Optional[int] = None) -> dict:
        """
        Full league table for one league_id (covers every club in that division).

        Dedupes in-process: second call for the same league_id returns the same
        payload without another network hit. Cache hits never burn the network budget.
        """
        lid = int(league_id)
        season = int(season or api_season_year())
        dedupe_key = (lid, season)

        # Soft network budget: only live HTTP standings count (0 = unlimited)
        if (
            self.max_standings > 0
            and self._network_standings >= self.max_standings
            and dedupe_key not in self._standings_seen
        ):
            return {
                "response": [],
                "errors": {"quota": "max network standings reached"},
                "_skipped": True,
                "_from_cache": False,
            }

        payload, from_cache = self.get(
            "standings",
            {"league": lid, "season": season},
            use_cache=True,
            cache_ttl_h=18.0,
        )
        payload = dict(payload)
        payload["_from_cache"] = from_cache
        if from_cache:
            self._cache_standings += 1
        else:
            self._network_standings += 1
        self._standings_seen.add(dedupe_key)
        return payload

    def stats(self) -> dict:
        return {
            "requests_network": self._req_count,
            "standings_network": self._network_standings,
            "standings_cache_hits": self._cache_standings,
            "standings_unique_leagues": len(self._standings_seen),
            "max_standings_network": self.max_standings,  # 0 = unlimited
            "cache_dir": str(self.cache_dir),
        }


def parse_standings_table(payload: dict) -> List[Dict[str, Any]]:
    """
    Flatten API-Football standings into full-table team rows.
    One payload = entire league (all ranks), not a subset of fixture teams.
    """
    rows: List[Dict[str, Any]] = []
    if payload.get("_skipped"):
        return rows
    for block in payload.get("response") or []:
        league = block.get("league") or {}
        league_id = league.get("id")
        league_name = league.get("name")
        season = league.get("season")
        groups = league.get("standings") or []
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
