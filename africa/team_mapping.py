#!/usr/bin/env python3
"""Strict Africa team identity layer.

No fuzzy nearest-team matching is permitted.
A name resolves only by:
  - exact canonical text
  - explicit alias
  - case-insensitive exact match
  - unique normalized spelling

Persistent IDs are append-only so new clubs cannot renumber existing model features.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "football_models" / "africa" / "mappings"
MAP_PATH = MAP_DIR / "team2id.json"
ALIAS_PATH = MAP_DIR / "team_aliases.json"


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").replace("\ufeff", "").strip())


def normalized(name: str) -> str:
    s = unicodedata.normalize("NFKD", clean(name)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(fc|sc|ac|afc|cf|club|football|football club)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_ids() -> Dict[str, int]:
    if not MAP_PATH.exists():
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(MAP_PATH.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def load_aliases() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if ALIAS_PATH.exists():
        try:
            out.update(
                {str(k).lower().strip(): str(v).strip() for k, v in json.loads(ALIAS_PATH.read_text(encoding="utf-8")).items()}
            )
        except Exception:
            pass
    return out


def resolve(name: str, known=None, aliases=None) -> Tuple[str, str]:
    raw = clean(name)
    if not raw:
        return raw, "empty"
    known = known if known is not None else load_ids()
    aliases = aliases if aliases is not None else load_aliases()
    ci = {k.lower(): k for k in known}

    if raw in known:
        return raw, "exact"
    if raw.lower() in aliases:
        c = aliases[raw.lower()]
        return (ci[c.lower()], "alias") if c.lower() in ci else (c, "alias")
    if raw.lower() in ci:
        return ci[raw.lower()], "case"

    norm = normalized(raw)
    matches = [k for k in known if normalized(k) == norm]
    if len(matches) == 1:
        return matches[0], "normalized"
    return raw, "raw"


def ensure_ids(names: Iterable[str]) -> Dict[str, int]:
    ids = load_ids()
    aliases = load_aliases()
    next_id = max(ids.values(), default=-1) + 1

    # Collapse only exact normalized duplicates. Never nearest-neighbour or partial-token guessing.
    groups: Dict[str, list] = {}
    for raw in names:
        raw = clean(raw)
        if not raw:
            continue
        c = aliases.get(raw.lower(), raw)
        groups.setdefault(normalized(c), []).append(c)

    for key, vals in groups.items():
        existing = [k for k in ids if normalized(k) == key]
        canon = existing[0] if len(existing) == 1 else min(vals, key=lambda x: (len(x), x))
        if canon not in ids:
            ids[canon] = next_id
            next_id += 1

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(ids, indent=2, ensure_ascii=False), encoding="utf-8")
    if not ALIAS_PATH.exists():
        ALIAS_PATH.write_text("{}", encoding="utf-8")
    return ids
