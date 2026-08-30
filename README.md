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
| Absolute `/workspaces/...` paths in model registries | Relative paths from repo root |
| Missing `.gitignore` (pycache, temps) | Added |
| `build_master_parquet.py` ROOT pointed at wrong dir | Fixed to repo root |

## Master parquet

- **Rows:** ~233k (1993/94 → present)
- **Columns:** 109 (identity, scores, stats, 1X2 / O-U / AH odds, derived targets)
- **Compression:** zstd
- **Types:** timestamps, float64 odds, int64 flags — no mixed string odds

### Core columns
`Season`, `SeasonCode`, `Div`, `Country`, `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, plus HT stats, shots, cards, bookmaker odds, and derived labels (`HomeWin`, `Over2_5`, `BTTS`, implied/fair probs, calendar features).

## Usage

```bash
pip install -r requirements.txt

# Rebuild master from downloaded historical CSVs (place under data/raw/{season}/{div}.csv)
python football_models/scripts/build_master_parquet.py

# Daily update
python scraper.py
python daily_update_parquet.py

# Simulation (default Valencia vs Betis; needs sklearn + model libs)
python sim.py
```

Environment:
- `FOOTBALL_SEASON` (default `2627`)
- `FOOTBALL_SEASON_LABEL` (default `2026/2027`)

## Layout

```
├── master_football_data.parquet   # clean master
├── scraper.py                     # fetch Latest_Results.csv
├── daily_update_parquet.py        # upsert into master
├── sim.py                         # match simulation
├── train.py                       # model training (heavy deps)
├── football_models/
│   ├── model_registry.json
│   ├── mappings/
│   ├── scripts/build_master_parquet.py
│   └── teams/{valencia,real_betis}/...
├── daily_football_data/           # daily snapshots
└── .github/workflows/daily_run.yaml
```

## Data quality guarantees

- Zero null `HomeTeam` / `AwayTeam`
- Zero negative scores
- FTR always consistent with FTHG/FTAG
- Deterministic match key: `Div|YYYY-MM-DD|HomeTeam|AwayTeam`
- Upserts replace prior rows for the same key

## Daily pipeline (GitHub Actions)

1. `scraper.py` — latest results CSV  
2. `scraper_fixtures.py` — upcoming `fixtures.csv` from football-data.co.uk  
3. `daily_update_parquet.py` — upsert results into master parquet  
4. `run_fixture_sims.py` — pure-raw Monte Carlo sims for every fixture with odds  
   - Outputs `daily_football_data/picks_for_admin.json` (shape matches sportintel `picks` table)  
   - Outputs `fixture_sims_YYYY-MM-DD.json` (full reports)  
   - Outputs `fixtures_teams.json` (teams seen — feed these into `train.py` focus_teams for targeted retrain)

Large binaries (parquet, CatBoost `.cbm`, pickles) are tracked with **Git LFS**.

## Daily pipeline (recommended)

Orchestrator: `python daily_pipeline.py`

| Step | Script | What it does |
|------|--------|----------------|
| 1 | `scraper.py` | Latest results CSV |
| 2 | `daily_update_parquet.py` | Upsert into master parquet |
| 3 | `scraper_fixtures.py` | Upcoming `fixtures.csv` (odds included) |
| 4 | `scan_fixture_teams.py` | Map fixture team names → canonical; build `train_focus_teams.json` |
| 5 | `train.py` | Batch train focus teams (`FOCUS_TEAMS`, `DAILY_LIGHT=1`) |
| 6 | `run_fixture_sims.py` | **`sim.run_one_match`** for every fixture (odds + models) → `sims/*.json` |
| 7 | `publish_sims_to_admin.py` | Push `data/sims/` to [sportintel](https://github.com/avramnull/sportintel) Sim Lab |

GitHub Action: `.github/workflows/daily_run.yaml` (schedule + manual).

**Secrets (admin push):** set repo secret `SPORTINTEL_PUSH_TOKEN` (PAT with `repo` on `avramnull/sportintel`). Falls back to `GITHUB_TOKEN` if it can write that repo.

**Caps:** `MAX_TRAIN_TEAMS=8` (default in CI), priority leagues first, teams without models preferred.
