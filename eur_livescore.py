#!/usr/bin/env python3
"""EUR results + upcoming fixtures from the API-Football livescore API.

Preferred over eur_openfootball.py when an API-Football key is configured:
one /fixtures?date=... call covers every league for that day (not just the
ones openfootball happens to have a current-season file for), it separates
finished from not-yet-played matches by real match status rather than by
"does this line have a score", and it updates same-day rather than weekly.

Uses the existing ApiFootballClient (api_football_client.py) for its
caching, key rotation, and quota handling — this module only adds the
fixture-list parsing and Div-code mapping on top.

Finished-match status codes: FT, AET, PEN (a completed, official result).
Not-yet-played: NS, TBD, PST (still worth listing as upcoming; a postponed
match is still on the board, just not confirmed for its original slot).
Anything else (in-progress: 1H, HT, 2H, ET, BT, P, LIVE; abandoned/voided:
CANC, ABD, AWD, WO) is skipped for both categories — a live pipeline run
should not treat a match with the whistle mid-way as either a result or a
clean upcoming fixture.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from api_football_client import ApiFootballClient, FDC_DIV_TO_LEAGUE

_FINISHED = {"FT", "AET", "PEN"}
_UPCOMING = {"NS", "TBD", "PST"}

_LEAGUE_TO_DIV = {v: k for k, v in FDC_DIV_TO_LEAGUE.items()}


def _iter_fixture_objects(client: ApiFootballClient, day: str):
    payload = client.fixtures_by_date(day)
    for item in payload.get("response") or []:
        league = (item.get("league") or {}).get("id")
        div = _LEAGUE_TO_DIV.get(league)
        if div is None:
            continue
        yield div, item


def fetch_results(client: ApiFootballClient, days_back: int = 3, today: Optional[date] = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    rows = []
    for offset in range(days_back, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        for div, item in _iter_fixture_objects(client, day):
            status = ((item.get("fixture") or {}).get("status") or {}).get("short")
            if status not in _FINISHED:
                continue
            goals = item.get("goals") or {}
            hg, ag = goals.get("home"), goals.get("away")
            if hg is None or ag is None:
                continue
            teams = item.get("teams") or {}
            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            if not home or not away:
                continue
            ht = ((item.get("score") or {}).get("halftime") or {})
            fixture_date_iso = ((item.get("fixture") or {}).get("date") or "")[:10] or day
            try:
                fixture_date = datetime.strptime(fixture_date_iso, "%Y-%m-%d").strftime("%d/%m/%Y")
            except ValueError:
                fixture_date = datetime.strptime(day, "%Y-%m-%d").strftime("%d/%m/%Y")
            rows.append({
                "Div": div, "Date": fixture_date, "HomeTeam": home, "AwayTeam": away,
                "FTHG": int(hg), "FTAG": int(ag),
                "FTR": "H" if hg > ag else ("A" if hg < ag else "D"),
                "HTHG": ht.get("home"), "HTAG": ht.get("away"),
            })
    return rows


def fetch_upcoming_fixtures(client: ApiFootballClient, days_forward: int = 14, today: Optional[date] = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    rows = []
    for offset in range(0, days_forward + 1):
        day = (today + timedelta(days=offset)).isoformat()
        for div, item in _iter_fixture_objects(client, day):
            status = ((item.get("fixture") or {}).get("status") or {}).get("short")
            if status not in _UPCOMING:
                continue
            teams = item.get("teams") or {}
            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            if not home or not away:
                continue
            raw_date = (item.get("fixture") or {}).get("date") or ""
            try:
                dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                date_str, time_str = dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M")
            except Exception:
                date_str, time_str = day, ""
            rows.append({"Div": div, "Date": date_str, "Time": time_str, "HomeTeam": home, "AwayTeam": away})
    return rows


if __name__ == "__main__":
    import sys
    client = ApiFootballClient(purpose="fixtures")
    if not client.available:
        print("no API_FOOTBALL_BACKUP_KEY / API_FOOTBALL_KEY configured", file=sys.stderr)
        raise SystemExit(1)
    results = fetch_results(client)
    fixtures = fetch_upcoming_fixtures(client)
    print(f"results: {len(results)}  upcoming fixtures: {len(fixtures)}  {client.stats()}")
