# Africa domestic football — ETL, train, sim

Industrial path for African leagues using **free** public sources.

## 1. Build master parquet

```bash
mkdir -p data/africa_raw
git clone --depth 1 https://github.com/openfootball/world.git data/africa_raw/openfootball_world

# Openfootball only
python -m africa.build_master_africa_parquet \
  --openfootball-root data/africa_raw/openfootball_world \
  --out master_africa_football.parquet

# + full RSSSF (all levels, year probe 2014–2026)
python -m africa.build_master_africa_parquet \
  --openfootball-root data/africa_raw/openfootball_world \
  --fetch-rsssf \
  --rsssf-year-start 2014 \
  --rsssf-year-end 2026 \
  --out master_africa_football.parquet
```

Sources:
- **openfootball/world** `africa/` — clean Football.TXT (primary)
- **RSSSF** `results-afr.html` + year pages — all levels (1st/2nd/3rd/cup), not only top flight

Features: form, Elo, calendar, team IDs, O2.5 / BTTS targets.  
Separate from European `master_football_data.parquet` (no free African odds).

## 2. Train

```bash
# Requires xgboost / lightgbm / catboost / sklearn as available
AFRICA_PARQUET=master_africa_football.parquet \
FOCUS_COUNTRIES=Nigeria,Ghana,Egypt,Morocco,Kenya,Algeria \
MIN_TEAM_MATCHES=40 \
MAX_BOOST_ROUNDS=800 \
python -m africa.train_africa
```

Writes `football_models/africa/GLOBAL/` and `football_models/africa/country_*/`  
Targets: `ft_result`, `over25`, `btts` (no odds features).

## 3. Simulate

```bash
python -m africa.sim_africa \
  --home "Kano Pillars FC" \
  --away "Enyimba FC" \
  --country Nigeria

# batch
python -m africa.sim_africa --fixtures africa_fixtures.csv --out africa_sim.json
```

Confidence-weighted ensemble + IPF score grid (FT / O2.5 / BTTS aligned).

## Modules

| File | Role |
|------|------|
| `parse_football_txt.py` | openfootball parser |
| `parse_rsssf.py` | full RSSSF scraper (all levels + year probe) |
| `build_master_africa_parquet.py` | merge, dedupe, features |
| `train_africa.py` | Africa model training |
| `sim_africa.py` | Africa match simulation |

## Notes

- RSSSF HTML is messy; standings rows are filtered out. Lower divisions and cups are kept when scorelines parse cleanly.
- Network timeouts to rsssf.com can happen from some hosts — re-run `--fetch-rsssf` locally.
- Openfootball alone already yields ~14k matches (Nigeria-heavy).

## Today's fixtures (API-Football, 1 request)

Free plan is ~100 req/day. This path uses **exactly one** call:

```bash
export API_FOOTBALL_KEY=your_key_here
python -m africa.fetch_today_fixtures
# optional: FIXTURE_DATE=2026-09-01
```

Writes:
- `daily_football_data/africa_fixtures_today.json`
- `daily_football_data/africa_fixtures_today.csv`

Filters world fixtures for that date down to African countries client-side (Kenya, Egypt, Uganda, …).
Do **not** loop per-league or pull multi-season history on the free key.



## Live standings (hardcore sim)

After today's fixtures are fetched, unique `league_id`s on the board are probed:

```bash
API_FOOTBALL_KEY=xxx python fetch_day_standings.py --region africa
# or EUR Div map:
API_FOOTBALL_KEY=xxx python fetch_day_standings.py --region eur
```

Standings priors adjust Dixon–Coles λ and mild FT / O2.5 / BTTS tilts inside
`sim_africa.simulate_match` and EUR `sim.run_one_match` (`standings_prior` key).

Quota: one fixtures call + ≤ `API_FOOTBALL_MAX_STANDINGS` (default 12) standings calls,
cached under `daily_football_data/api_football_cache/`.
