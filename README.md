# sportintel_core — System-001 (Industrial Rebuild)

Football results master dataset + daily train/sim pipeline built from
[football-data.co.uk](https://www.football-data.co.uk/).

Feeds the **sportintel** admin Sim Lab (`data/sims/`) every day via GitHub Actions.

## Architecture

```
scraper.py                → Latest_Results.csv
daily_update_parquet.py   → upsert into master_football_data.parquet
scraper_fixtures.py       → fixtures_latest.csv (odds included)
scan_fixture_teams.py     → train_focus_teams.json (canonical mapping)
train.py                  → per-team models (XGB/LGB/CatBoost/RF/Torch/TF)
run_fixture_sims.py       → sim.run_one_match for every fixture with odds
publish_sims_to_admin.py  → push daily_football_data/sims → sportintel
```

Orchestrator: `python daily_pipeline.py`

Shared config / logging:
- `si_config.py` — season inference, paths, caps, env
- `si_logging.py` — structured console (optional JSON via `LOG_JSON=1`)
- `tools/smoke_test.py` — CI health gate

## What was fixed (industrial pass)

| Issue | Fix |
| --- | --- |
| CI `MAX_TRAIN_TEAMS=0` disabled training | Default **12**; workflow_dispatch override |
| Hard-coded season `2627` | Auto-infer from calendar (`si_config.infer_season`) |
| Silent `pip … \|\| true` for boosters | Boosters are **hard** deps in CI; NNs remain optional |
| LFS pointer / tiny parquet | Fail-fast size + pointer checks in workflow and pipeline |
| `TODAY_ONLY` dropped slate near UTC midnight | ±1 day safety window when exact day is empty |
| Bare publish / token handling | Central `GITHUB_TOKEN` resolution; no token echo |
| No health checks | `python -m tools.smoke_test [--quick]` |
| Weak User-Agent on scrapers | Explicit sportintel UA + stable `*_latest.csv` copies |
| Empty / sparse README | This document |

## Master parquet

- **Rows:** ~233k (1993/94 → present)
- **Columns:** 109 (identity, scores, stats, 1X2 / O-U / AH odds, derived targets)
- **Compression:** zstd
- **Types:** timestamps, float64 odds, int64 flags — no mixed string odds

### Core columns

`Season`, `SeasonCode`, `Div`, `Country`, `Date`, `HomeTeam`, `AwayTeam`,
`FTHG`, `FTAG`, `FTR`, plus HT stats, shots, cards, bookmaker odds, and derived
labels (`HomeWin`, `Over2_5`, `BTTS`, implied/fair probs, calendar features).

## Quick start

```bash
pip install -r requirements.txt
# Optional full stack:
# pip install xgboost lightgbm catboost torch tensorflow

# Health check (needs real parquet via git lfs pull)
python -m tools.smoke_test --quick

# Rebuild master from historical CSVs under data/raw/{season}/{div}.csv
python football_models/scripts/build_master_parquet.py

# One-shot daily run
python daily_pipeline.py

# Simulation only (default odds blend 0.30)
python run_fixture_sims.py
```

### Environment (selected)

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOOTBALL_SEASON` | auto | e.g. `2627` |
| `FOOTBALL_SEASON_LABEL` | auto | e.g. `2026/2027` |
| `MAX_TRAIN_TEAMS` | `12` | Cap focus list (`0` = skip train) |
| `MIN_TEAM_MATCHES` | `25` | History threshold for focus |
| `N_SIMULATIONS` | `3000` | Monte Carlo draws per fixture |
| `ODDS_BLEND` | `0.30` | Model↔market blend weight |
| `TODAY_ONLY` | `1` | Restrict scan/sim to run-day window |
| `SKIP_TRAIN` / `SKIP_SIM` | `0` | Bypass steps |
| `SPORTINTEL_PUSH` | `0` | Publish sims to admin repo |
| `SPORTINTEL_REPO` | `avramnull/sportintel` | Admin target |
| `GITHUB_TOKEN` | — | PAT with `repo` on admin |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `0` | Logging |

## Data quality guarantees

- Zero null `HomeTeam` / `AwayTeam`
- Zero negative scores
- `FTR` always consistent with `FTHG`/`FTAG`
- Deterministic match key: `Div\|YYYY-MM-DD\|HomeTeam\|AwayTeam`
- Upserts replace prior rows for the same key
- H/D/A class order pinned in train + sim (never sklearn-sorted)

## Daily pipeline (GitHub Actions)

Workflow: `.github/workflows/daily_run.yaml` (schedule + manual).

1. LFS pull + parquet size gate  
2. Install core + **required** boosters  
3. `python -m tools.smoke_test --quick`  
4. `python daily_pipeline.py`  
5. Commit models / daily artifacts back to `main`

**Secrets:** set repo secret `SPORTINTEL_PUSH_TOKEN` (PAT with `repo` on
`avramnull/sportintel`). Falls back to `GITHUB_TOKEN` if it can write that repo.

Large binaries (parquet, CatBoost `.cbm`, pickles) are tracked with **Git LFS**.

## Layout

```
├── si_config.py / si_logging.py
├── tools/smoke_test.py
├── master_football_data.parquet
├── scraper.py / scraper_fixtures.py
├── daily_update_parquet.py
├── scan_fixture_teams.py
├── train.py / sim.py / run_fixture_sims.py
├── publish_sims_to_admin.py
├── daily_pipeline.py
├── team_mapping.py
├── football_models/
│   ├── model_registry.json
│   ├── mappings/
│   ├── scripts/build_master_parquet.py
│   └── teams/...
├── daily_football_data/
└── .github/workflows/daily_run.yaml
```
