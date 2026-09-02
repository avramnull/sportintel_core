#!/usr/bin/env python3
"""Fetch today's African fixtures from Live-score API only.

The Africa segment intentionally has one fixture provider. Requests are
quota-efficient: the shared Live-score client caches responses, deduplicates
pages/fixtures, and handles pagination without per-fixture calls.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fetch(day: str) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_score_api_fixtures import fetch_africa_fixtures
    return fetch_africa_fixtures(day)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        sys.path.insert(0, str(ROOT))
        import si_config  # noqa: F401 — loads local .env when present
    except Exception:
        pass

    day = os.environ.get("FIXTURE_DATE", "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(os.environ.get(
        "AFRICA_FIXTURES_OUT",
        str(ROOT / "daily_football_data" / "africa_fixtures_today.json"),
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")

    try:
        rows, meta = _fetch(day)
        doc = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "provider": "live-score-api",
            "total_world": meta.get("raw_fixtures"),
            "total_africa": len(rows),
            "fixtures": rows,
            "ok": True,
            "schema_version": "africa-fixture-v2",
        }
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(rows, csv_path)
        print(f"[ok] Africa fixtures {day}: {len(rows)} via Live-score API")
        for row in rows:
            print(f"  {row.get('date','')} | {row.get('country','')} {row.get('league','')} | {row.get('home','')} vs {row.get('away','')} | {row.get('status','')}")
        print(f"[ok] provider=live-score-api wrote {out}")
        return 0
    except Exception as exc:
        error = f"Live-score API Africa fixture fetch failed: {exc}"
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        doc = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "provider": "live-score-api",
            "total_world": None,
            "total_africa": None,
            "fixtures": [],
            "api_errors": [error],
            "ok": False,
            "schema_version": "africa-fixture-v2",
        }
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
