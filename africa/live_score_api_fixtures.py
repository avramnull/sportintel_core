#!/usr/bin/env python3
"""Quota-safe Live-score API client for African fixtures."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

AFRICA_COUNTRIES = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
    "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "Congo DR",
    "DR Congo", "Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire", "Djibouti", "Egypt",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
    "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
    "Niger", "Nigeria", "Rwanda", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
    "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
    "Zambia", "Zimbabwe",
}
AFRICA_COUNTRIES_NORM = {x.strip().casefold() for x in AFRICA_COUNTRIES}
BASE = "https://livescore-api.com/api-client/fixtures/list.json"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("LIVE_SCORE_CACHE_DIR", str(ROOT / "daily_football_data" / "live_score_cache")))
CACHE_TTL = int(os.environ.get("LIVE_SCORE_CACHE_TTL_SEC", "21600"))
MAX_REQUESTS = int(os.environ.get("LIVE_SCORE_MAX_REQUESTS", "20"))
MIN_INTERVAL = float(os.environ.get("LIVE_SCORE_MIN_INTERVAL_SEC", "0.5"))


def _cache_path(day: str) -> Path:
    return CACHE_DIR / f"africa_{day}.json"


def _load_cache(day: str):
    p = _cache_path(day)
    if not p.exists() or time.time() - p.stat().st_mtime > CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_africa(row: dict[str, Any]) -> bool:
    country = row.get("country") or {}
    federation = row.get("federation") or {}
    if isinstance(country, dict):
        country_name = str(country.get("name") or "").strip().casefold()
    else:
        country_name = str(country).strip().casefold()
    federation_name = str(federation.get("name") if isinstance(federation, dict) else federation).strip().upper()
    return country_name in AFRICA_COUNTRIES_NORM or federation_name == "CAF"


def _next_page(value: Any, current: int) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        page_values = parse_qs(urlparse(text).query).get("page")
        if page_values:
            return int(page_values[0])
    except (TypeError, ValueError):
        pass
    return current + 1


def _pages(key: str, secret: str, day: str) -> tuple[list[dict[str, Any]], int]:
    fixtures: list[dict[str, Any]] = []
    page = 1
    seen_pages: set[int] = set()
    requests_used = 0
    last_request = 0.0
    with requests.Session() as session:
        while page not in seen_pages:
            if requests_used >= MAX_REQUESTS:
                raise RuntimeError(f"Live-score request budget exhausted ({MAX_REQUESTS} pages/day)")
            seen_pages.add(page)
            wait = MIN_INTERVAL - (time.time() - last_request)
            if wait > 0:
                time.sleep(wait)
            r = session.get(
                BASE,
                params={"date": day, "key": key, "secret": secret, "page": page},
                timeout=45,
            )
            last_request = time.time()
            requests_used += 1
            r.raise_for_status()
            payload = r.json()
            if payload.get("success") is False:
                raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))
            data = payload.get("data") or {}
            batch = data.get("fixtures") or []
            fixtures.extend(x for x in batch if isinstance(x, dict))
            nxt = _next_page(data.get("next_page"), page)
            if nxt is None:
                break
            if nxt == page:
                raise RuntimeError(f"Live-score API returned repeated page {page}")
            page = nxt
            if page > 1000:
                raise RuntimeError("Live-score API pagination exceeded safety limit")
    return fixtures, requests_used


def _normalize(raw: list[dict[str, Any]], day: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for x in raw:
        if not _is_africa(x):
            continue
        fid = x.get("id") or x.get("fixture_id")
        try:
            fixture_id = int(fid)
        except (TypeError, ValueError):
            continue
        if fixture_id in seen:
            continue
        seen.add(fixture_id)
        country = x.get("country") or {}
        federation = x.get("federation") or {}
        competition = x.get("competition") or {}
        home = x.get("home") or {}
        away = x.get("away") or {}
        odds = (x.get("odds") or {}).get("pre") or {}
        out.append({
            "fixture_id": fixture_id,
            "date": str(x.get("date") or day),
            "timestamp": None,
            "status": "NS",
            "elapsed": None,
            "country": str(country.get("name") or federation.get("name") or "CAF") if isinstance(country, dict) else str(country),
            "league_id": competition.get("id") if isinstance(competition, dict) else None,
            "league": competition.get("name") if isinstance(competition, dict) else None,
            "season": None,
            "round": x.get("round"),
            "home": home.get("name") if isinstance(home, dict) else None,
            "away": away.get("name") if isinstance(away, dict) else None,
            "home_id": home.get("id") if isinstance(home, dict) else None,
            "away_id": away.get("id") if isinstance(away, dict) else None,
            "goals_home": None,
            "goals_away": None,
            "ht_home": None,
            "ht_away": None,
            "odds_h": odds.get("1") if isinstance(odds, dict) else None,
            "odds_d": odds.get("X") if isinstance(odds, dict) else None,
            "odds_a": odds.get("2") if isinstance(odds, dict) else None,
            "odds_book": "live-score-api",
            "source": "live-score-api",
        })
    return out


def fetch_africa_fixtures(day: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cached = _load_cache(day)
    if cached is not None:
        return cached.get("fixtures", []), cached.get("meta", {})

    key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()
    secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET are not configured")

    raw, requests_used = _pages(key, secret, day)
    out = _normalize(raw, day)
    meta = {
        "provider": "live-score-api",
        "date": day,
        "raw_fixtures": len(raw),
        "africa_fixtures": len(out),
        "requests_used": requests_used,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(day).write_text(json.dumps({"fixtures": out, "meta": meta}, ensure_ascii=False), encoding="utf-8")
    return out, meta
