#!/usr/bin/env python3
"""Upsert latest results CSV into master_football_data.parquet. Clean schema, float odds only."""
from __future__ import annotations
import gc, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "master_football_data.parquet"
SAVE_DIR = ROOT / "daily_football_data"
SEASON_CODE = os.environ.get("FOOTBALL_SEASON", "2627")
SEASON_LABEL = os.environ.get("FOOTBALL_SEASON_LABEL", "2026/2027")
COUNTRY_MAP = {
    "E0":"England","E1":"England","E2":"England","E3":"England","EC":"England",
    "SC0":"Scotland","SC1":"Scotland","SC2":"Scotland","SC3":"Scotland",
    "D1":"Germany","D2":"Germany","I1":"Italy","I2":"Italy","SP1":"Spain","SP2":"Spain",
    "F1":"France","F2":"France","N1":"Netherlands","B1":"Belgium","P1":"Portugal","T1":"Turkey","G1":"Greece",
}

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def make_key(df):
    return (df["Div"].astype(str) + "|" + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d") + "|"
            + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str))

def pick_latest_csv():
    stable = SAVE_DIR / "results_latest.csv"
    if stable.exists() and stable.stat().st_size > 100:
        return stable
    files = sorted(SAVE_DIR.glob("results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No results_*.csv in {SAVE_DIR}")
    return files[0]

def prepare(csv_path, schema):
    raw = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    log(f"CSV {csv_path.name}: {len(raw)} rows, {len(raw.columns)} cols")
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    for c in ("HomeTeam","AwayTeam","Div"):
        raw[c] = raw[c].astype(str).str.strip()
        raw.loc[raw[c].isin(["","nan","None"]), c] = np.nan
    raw = raw.dropna(subset=["Date","HomeTeam","AwayTeam","Div"])
    raw["FTHG"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    raw["FTAG"] = pd.to_numeric(raw["FTAG"], errors="coerce")
    raw = raw.dropna(subset=["FTHG","FTAG"])
    raw = raw[(raw["FTHG"]>=0)&(raw["FTAG"]>=0)]
    if raw.empty:
        return pd.DataFrame()
    computed = np.where(raw["FTHG"]>raw["FTAG"],"H", np.where(raw["FTHG"]<raw["FTAG"],"A","D"))
    if "FTR" in raw.columns:
        ftr = raw["FTR"].astype(str).str.strip().str.upper()
        raw["FTR"] = np.where(ftr.isin(["H","D","A"]), ftr, computed)
    else:
        raw["FTR"] = computed
    n = len(raw)
    cols = {}
    cols["Season"] = SEASON_LABEL
    cols["SeasonCode"] = SEASON_CODE
    cols["Div"] = raw["Div"].values
    cols["Country"] = raw["Div"].map(COUNTRY_MAP).fillna("Unknown").values
    cols["Date"] = raw["Date"].values
    cols["HomeTeam"] = raw["HomeTeam"].values
    cols["AwayTeam"] = raw["AwayTeam"].values
    cols["FTHG"] = raw["FTHG"].astype("float64").values
    cols["FTAG"] = raw["FTAG"].astype("float64").values
    cols["FTR"] = raw["FTR"].values
    cols["SourceFile"] = csv_path.name
    for name in ("Time","HTR","Referee"):
        if name in raw.columns:
            cols[name] = raw[name].astype(str).values
    for field in schema:
        if field.name in cols:
            continue
        if pa.types.is_floating(field.type) and field.name in raw.columns:
            cols[field.name] = pd.to_numeric(raw[field.name], errors="coerce").astype("float64").values
    cols["TotalGoals"] = cols["FTHG"] + cols["FTAG"]
    cols["GoalDiff"] = cols["FTHG"] - cols["FTAG"]
    cols["Over2_5"] = (cols["TotalGoals"] > 2.5).astype("int64")
    cols["BTTS"] = ((cols["FTHG"]>0)&(cols["FTAG"]>0)).astype("int64")
    cols["HomeWin"] = (raw["FTR"].values=="H").astype("int64")
    cols["Draw"] = (raw["FTR"].values=="D").astype("int64")
    cols["AwayWin"] = (raw["FTR"].values=="A").astype("int64")
    cols["HomePoints"] = raw["FTR"].map({"H":3,"D":1,"A":0}).fillna(0).astype("int64").values
    cols["AwayPoints"] = raw["FTR"].map({"H":0,"D":1,"A":3}).fillna(0).astype("int64").values
    if "AvgH" in cols:
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["ImpProbH"] = 1.0 / np.clip(np.asarray(cols["AvgH"], dtype=float), 1.01, None)
            cols["ImpProbD"] = 1.0 / np.clip(np.asarray(cols.get("AvgD", np.nan), dtype=float), 1.01, None)
            cols["ImpProbA"] = 1.0 / np.clip(np.asarray(cols.get("AvgA", np.nan), dtype=float), 1.01, None)
            s = cols["ImpProbH"]+cols["ImpProbD"]+cols["ImpProbA"]
            cols["FairProbH"] = cols["ImpProbH"]/s
            cols["FairProbD"] = cols["ImpProbD"]/s
            cols["FairProbA"] = cols["ImpProbA"]/s
    dates = pd.to_datetime(cols["Date"])
    cols["Year"] = dates.year.astype("int64").values
    cols["Month"] = dates.month.astype("int64").values
    cols["DayOfWeek"] = dates.dayofweek.astype("int64").values
    cols["IsWeekend"] = (cols["DayOfWeek"]>=5).astype("int64")
    data = {}
    for field in schema:
        name = field.name
        val = cols.get(name)
        if val is None:
            if pa.types.is_floating(field.type): data[name] = np.full(n, np.nan)
            elif pa.types.is_integer(field.type): data[name] = np.zeros(n, dtype="int64")
            else: data[name] = np.array([None]*n, dtype=object)
        else:
            data[name] = val
    df = pd.DataFrame(data)
    df["_key"] = make_key(df)
    log(f"Prepared {len(df)} clean rows")
    return df

def merge(parquet_path, res):
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    n_old = pf.metadata.num_rows
    update_keys = set(res["_key"])
    log(f"Master before: {n_old:,} | upsert keys: {len(update_keys)}")
    tmp = parquet_path.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    dropped = 0
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        keys = make_key(df)
        mask = ~keys.isin(update_keys)
        dropped += int((~mask).sum())
        df = df.loc[mask]
        if len(df)==0: continue
        for f in schema:
            if f.name not in df.columns:
                if pa.types.is_floating(f.type): df[f.name] = np.nan
                elif pa.types.is_integer(f.type): df[f.name] = 0
                else: df[f.name] = None
        table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
        writer.write_table(table)
        del df, table
        gc.collect()
    res2 = res.drop(columns=["_key"])
    for f in schema:
        if f.name not in res2.columns:
            if pa.types.is_floating(f.type): res2[f.name] = np.nan
            elif pa.types.is_integer(f.type): res2[f.name] = 0
            else: res2[f.name] = None
    table = pa.Table.from_pandas(res2[schema.names], schema=schema, preserve_index=False)
    writer.write_table(table)
    writer.close()
    tmp.replace(parquet_path)  # atomic: never leaves parquet_path missing mid-write
    n_new = pq.ParquetFile(parquet_path).metadata.num_rows
    log(f"Dropped overlap: {dropped} | Appended: {len(res)} | Master after: {n_new:,}")
    log(f"OK -> {parquet_path} ({parquet_path.stat().st_size/1e6:.1f} MB)")

def main():
    log("="*56)
    log("DAILY PARQUET UPDATE")
    log("="*56)
    if not PARQUET_PATH.exists():
        log(f"ERROR: {PARQUET_PATH} missing")
        sys.exit(1)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(PARQUET_PATH).schema_arrow
    try:
        csv_path = pick_latest_csv()
        log(f"Using {csv_path}")
    except FileNotFoundError:
        log("No local CSV — run scraper.py first")
        sys.exit(1)
    res = prepare(csv_path, schema)
    if res.empty:
        log("No rows to merge")
        sys.exit(0)
    merge(PARQUET_PATH, res)
    log("DONE")

if __name__ == "__main__":
    main()
