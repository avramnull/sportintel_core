#!/usr/bin/env python3
"""Live-score API fallback for today's African fixtures.

Credentials are supplied only through environment variables:
  LIVE_SCORE_API_KEY
  LIVE_SCORE_API_SECRET

The provider supports calendar fixtures with a date filter and paginates at
30 fixtures per response. We follow pagination and keep only African domestic
fixtures plus CAF federation fixtures, returning the same normalized shape used
by the Africa pipeline.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

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
BASE = "https://livescore-api.com/api-client/fixtures/list.json"


def _is_africa(row: dict[str, Any]) -> bool:
    country = row.get("country") or {}
    federation = row.get("federation") or {}
    country_name = str(country.get("name") or "").strip()
    federation_name = str(federation.get("name") or "").strip().upper()
    return country_name in AFRICA_COUNTRIES or federation_name == "CAF"


def _pages(key: str, secret: str, day: str) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    page = 1
    seen_pages: set[int] = set()
    with requests.Session() as session:
        while page not in seen_pages:
            seen_pages.add(page)
            r = session.get(
                BASE,
                params={"date": day, "key": key, "secret": secret, "page": page},
                timeout=45,
            )
            r.raise_for_status()
            payload = r.json()
            if payload.get("success") is False:
                raise RuntimeError(str(payload.get("error") or "Live-score API returned success=false"))
            data = payload.get("data") or {}
            batch = data.get("fixtures") or []
            fixtures.extend(x for x in batch if isinstance(x, dict))
            nxt = data.get("next_page")
            if not nxt:
                break
            try:
                # The API returns a URL containing the next page number. We only
                # advance the integer page; credentials stay in our own params.
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(str(nxt)).query)
                page = int(qs.get("page", [page + 1])[0])
            except Exception:
                page += 1
            if page > 100:
                raise RuntimeError("Live-score API pagination exceeded safety limit")
    return fixtures


def fetch_africa_fixtures(day: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = os.environ.get("LIVE_SCORE_API_KEY", "").strip()
    secret = os.environ.get("LIVE_SCORE_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("LIVE_SCORE_API_KEY/LIVE_SCORE_API_SECRET are not configured")

    raw = _pages(key, secret, day)
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for x in raw:
        if not _is_africa(x):
            continue
        fid = x.get("id")
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
            "date": str(x.get("date") or ""),
            "timestamp": None,
            "status": "NS",
            "elapsed": None,
            "country": str(country.get("name") or federation.get("name") or "CAF"),
            "league_id": competition.get("id"),
            "league": competition.get("name"),
            "season": None,
            "round": x.get("round"),
            "home": home.get("name"),
            "away": away.get("name"),
            "home_id": home.get("id"),
            "away_id": away.get("id"),
            "goals_home": None,
            "goals_away": None,
            "ht_home": None,
            "ht_away": None,
            "odds_h": odds.get("1"),
            "odds_d": odds.get("X"),
            "odds_a": odds.get("2"),
            "odds_book": "live-score-api",
            "source": "live-score-api",
        })

    meta = {
        "provider": "live-score-api",
        "date": day,
        "raw_fixtures": len(raw),
        "africa_fixtures": len(out),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return out, meta
