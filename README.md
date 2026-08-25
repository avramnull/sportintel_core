# System-001 (Clean Rebuild)

Football results master dataset built from [football-data.co.uk](https://www.football-data.co.uk/).

## What was fixed

| Issue in original | Fix |
|---|---|
| 122 odds columns stored as `large_string` | All odds are `float64` / `double` |
| 1,047 rows with null team names (old Greek data) | Dropped during clean build |
| Hard-coded season only in places | `FOOTBALL_SEASON` env var |
| No validation | Builder enforces non-null teams, non-negative scores, consistent FTR |
| Empty README | This file |

## Master parquet

- **Rows:** ~217k (1993/94 → present)
- **Columns:** 109 (identity, scores, stats, 1X2 / O-U / AH odds, derived targets)
- **Compression:** zstd
- **Types:** timestamps, float64 odds, int64 flags — no mixed string odds

### Core columns
`Season`, `SeasonCode`, `Div`, `Country`, `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, plus HT stats, shots, cards, bookmaker odds, and derived labels (`HomeWin`, `Over2_5`, `BTTS`, implied/fair probs, calendar features).

## Usage

```bash
pip install -r requirements.txt

# Rebuild master from downloaded historical CSVs (already in data/raw/)
python scripts/build_master_parquet.py

# Daily update
python scraper.py
python daily_update_parquet.py
```

Environment:
- `FOOTBALL_SEASON` (default `2627`)
- `FOOTBALL_SEASON_LABEL` (default `2026/2027`)

## Layout

```
├── master_football_data.parquet   # clean master
├── scraper.py                     # fetch Latest_Results.csv
├── daily_update_parquet.py        # upsert into master
├── scripts/build_master_parquet.py
├── data/raw/{season}/{div}.csv    # historical source CSVs
├── daily_football_data/           # daily snapshots
└── .github/workflows/daily_run.yaml
```

## Data quality guarantees

- Zero null `HomeTeam` / `AwayTeam`
- Zero negative scores
- FTR always consistent with FTHG/FTAG
- Deterministic match key: `Div|YYYY-MM-DD|HomeTeam|AwayTeam`
- Upserts replace prior rows for the same key
