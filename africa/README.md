# Africa domestic football ETL

Build a clean **`master_africa_football.parquet`** from free public sources:

1. **[openfootball/world](https://github.com/openfootball/world)** (`africa/`) — primary  
2. **[RSSSF Africa domestic](http://www.rsssf.com/results-afr.html)** — optional fill-in  

## Quick start

```bash
# 1) Clone openfootball world (one-time / refresh)
mkdir -p data/africa_raw
git clone --depth 1 https://github.com/openfootball/world.git data/africa_raw/openfootball_world

# 2) Build master (openfootball only)
python -m africa.build_master_africa_parquet \
  --openfootball-root data/africa_raw/openfootball_world \
  --out master_africa_football.parquet

# 3) Optional: also pull RSSSF first-level pages (slower, network)
python -m africa.build_master_africa_parquet \
  --openfootball-root data/africa_raw/openfootball_world \
  --fetch-rsssf \
  --rsssf-max-pages 60 \
  --out master_africa_football.parquet
```

## Output schema (core columns)

| Column | Notes |
|--------|--------|
| `Source` | `openfootball` or `rsssf` |
| `Country`, `League`, `Div`, `Season` | League identity |
| `Date`, `Time`, `HomeTeam`, `AwayTeam` | Fixture |
| `FTHG`, `FTAG`, `FTR` | Full-time result |
| `HTHG`, `HTAG` | Half-time when present in Football.TXT |
| `TotalGoals`, `Over2_5`, `BTTS`, `HomeWin`, … | Targets |
| `HomeTeamId`, `AwayTeamId` | Stable within Africa corpus |
| `HomeFormPts_5/10`, `EloHome`, `EloDiff`, … | Leak-free features |

Dedupe key: `Div|YYYY-MM-DD|home|away` (openfootball preferred over RSSSF).

## Coverage (openfootball snapshot)

Typically **~14k** matches across Nigeria (largest), Ghana, Egypt, Morocco, Algeria, Kenya, Uganda, Tanzania, Zambia, South Africa, CAF CL, etc.  
Nigeria NPFL/NNL dominates historical depth.

## Design notes

- **No book odds** in free Africa dumps — train result / O2.5 / BTTS without market blend until odds exist.
- Keep **separate** from European `master_football_data.parquet` so null odds do not pollute EU models.
- Team map written to `football_models/mappings/africa_team2id.json`.

## Next (optional)

- Wire `train.py` with `AFRICA=1` + this parquet path  
- Daily scraper refresh of openfootball clone in CI  
