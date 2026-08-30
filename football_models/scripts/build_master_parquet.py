#!/usr/bin/env python3
"""Memory-efficient clean master parquet builder. All odds float64. Zero null teams."""
from __future__ import annotations
import gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]  # repo root
RAW_DIR = ROOT / "data" / "raw"
OUT_PARQUET = ROOT / "master_football_data.parquet"

DIVS = ["E0","E1","E2","E3","EC","SC0","SC1","SC2","SC3","D1","D2","I1","I2","SP1","SP2","F1","F2","N1","B1","P1","T1","G1"]
COUNTRY_MAP = {
    "E0":"England","E1":"England","E2":"England","E3":"England","EC":"England",
    "SC0":"Scotland","SC1":"Scotland","SC2":"Scotland","SC3":"Scotland",
    "D1":"Germany","D2":"Germany","I1":"Italy","I2":"Italy","SP1":"Spain","SP2":"Spain",
    "F1":"France","F2":"France","N1":"Netherlands","B1":"Belgium","P1":"Portugal","T1":"Turkey","G1":"Greece",
}

FLOAT_COLS = [
    "FTHG","FTAG","HTHG","HTAG","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR","HxG","AxG",
    "B365H","B365D","B365A","PSH","PSD","PSA","WHH","WHD","WHA","BWH","BWD","BWA","IWH","IWD","IWA","VCH","VCD","VCA",
    "MaxH","MaxD","MaxA","AvgH","AvgD","AvgA","B365>2.5","B365<2.5","Max>2.5","Max<2.5","Avg>2.5","Avg<2.5",
    "AHh","B365AHH","B365AHA","MaxAHH","MaxAHA","AvgAHH","AvgAHA",
    "B365CH","B365CD","B365CA","MaxCH","MaxCD","MaxCA","AvgCH","AvgCD","AvgCA",
    "B365C>2.5","B365C<2.5","MaxC>2.5","MaxC<2.5","AvgC>2.5","AvgC<2.5",
    "AHCh","B365CAHH","B365CAHA","MaxCAHH","MaxCAHA","AvgCAHH","AvgCAHA",
    "TotalGoals","GoalDiff","HTGoalDiff","ImpProbH","ImpProbD","ImpProbA","FairProbH","FairProbD","FairProbA",
]
INT_COLS = ["HomeWin","Draw","AwayWin","HomePoints","AwayPoints","Over2_5","BTTS","Year","Month","DayOfWeek","IsWeekend"]
ALL_COLS = [
    "Season","SeasonCode","Div","Country","Date","Time","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
    "HTHG","HTAG","HTR","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR","HxG","AxG",
    "B365H","B365D","B365A","PSH","PSD","PSA","WHH","WHD","WHA","BWH","BWD","BWA","IWH","IWD","IWA","VCH","VCD","VCA",
    "MaxH","MaxD","MaxA","AvgH","AvgD","AvgA","B365>2.5","B365<2.5","Max>2.5","Max<2.5","Avg>2.5","Avg<2.5",
    "AHh","B365AHH","B365AHA","MaxAHH","MaxAHA","AvgAHH","AvgAHA",
    "B365CH","B365CD","B365CA","MaxCH","MaxCD","MaxCA","AvgCH","AvgCD","AvgCA",
    "B365C>2.5","B365C<2.5","MaxC>2.5","MaxC<2.5","AvgC>2.5","AvgC<2.5",
    "AHCh","B365CAHH","B365CAHA","MaxCAHH","MaxCAHA","AvgCAHH","AvgCAHA",
    "TotalGoals","GoalDiff","HTGoalDiff","HomeWin","Draw","AwayWin","HomePoints","AwayPoints","Over2_5","BTTS",
    "ImpProbH","ImpProbD","ImpProbA","FairProbH","FairProbD","FairProbA",
    "Referee","Year","Month","DayOfWeek","IsWeekend","SourceFile",
]

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)

def code_to_label(code):
    if code == "9900": return "1999/2000"
    a, b = int(code[:2]), int(code[2:])
    ya = (1900 if a >= 90 else 2000) + a
    yb = (1900 if b >= 90 else 2000) + b
    if yb < ya: yb += 100
    return f"{ya}/{yb}"

def season_codes():
    codes = [f"{y:02d}{(y+1)%100:02d}" for y in range(93, 100)]
    codes += [f"{y:02d}{(y+1)%100:02d}" for y in range(0, 27)]
    return codes

def clean_file(path, season_code):
    try:
        raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)
    except Exception:
        try:
            raw = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
        except Exception:
            return pd.DataFrame()
    if raw.empty or "Div" not in raw.columns or "Date" not in raw.columns:
        return pd.DataFrame()
    raw = raw.dropna(axis=1, how="all")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    raw.columns = [str(c).strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    for col in ("HomeTeam", "AwayTeam", "Div"):
        if col not in raw.columns: return pd.DataFrame()
        raw[col] = raw[col].astype(str).str.strip()
        raw.loc[raw[col].isin(["", "nan", "None", "NaN"]), col] = np.nan
    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    if "FTHG" not in raw.columns or "FTAG" not in raw.columns:
        return pd.DataFrame()
    raw["FTHG"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    raw["FTAG"] = pd.to_numeric(raw["FTAG"], errors="coerce")
    raw = raw.dropna(subset=["FTHG", "FTAG"])
    raw = raw[(raw["FTHG"] >= 0) & (raw["FTAG"] >= 0) & (raw["FTHG"] < 30) & (raw["FTAG"] < 30)]
    if raw.empty: return pd.DataFrame()
    computed = np.where(raw["FTHG"] > raw["FTAG"], "H", np.where(raw["FTHG"] < raw["FTAG"], "A", "D"))
    if "FTR" in raw.columns:
        ftr = raw["FTR"].astype(str).str.strip().str.upper()
        raw["FTR"] = np.where(ftr.isin(["H", "D", "A"]), ftr, computed)
    else:
        raw["FTR"] = computed
    n = len(raw)
    out = {
        "Season": code_to_label(season_code),
        "SeasonCode": season_code,
        "Div": raw["Div"].values,
        "Country": raw["Div"].map(COUNTRY_MAP).fillna("Unknown").values,
        "Date": raw["Date"].values,
        "Time": raw["Time"].astype(str).values if "Time" in raw.columns else np.array([None]*n, dtype=object),
        "HomeTeam": raw["HomeTeam"].values,
        "AwayTeam": raw["AwayTeam"].values,
        "FTHG": raw["FTHG"].astype("float64").values,
        "FTAG": raw["FTAG"].astype("float64").values,
        "FTR": raw["FTR"].values,
        "HTR": raw["HTR"].astype(str).values if "HTR" in raw.columns else np.array([None]*n, dtype=object),
        "Referee": raw["Referee"].astype(str).values if "Referee" in raw.columns else np.array([None]*n, dtype=object),
        "SourceFile": f"{season_code}/{path.name}",
    }
    for col in ["HTHG","HTAG","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR","HxG","AxG",
                "B365H","B365D","B365A","PSH","PSD","PSA","WHH","WHD","WHA","BWH","BWD","BWA","IWH","IWD","IWA","VCH","VCD","VCA",
                "MaxH","MaxD","MaxA","AvgH","AvgD","AvgA","B365>2.5","B365<2.5","Max>2.5","Max<2.5","Avg>2.5","Avg<2.5",
                "AHh","B365AHH","B365AHA","MaxAHH","MaxAHA","AvgAHH","AvgAHA",
                "B365CH","B365CD","B365CA","MaxCH","MaxCD","MaxCA","AvgCH","AvgCD","AvgCA",
                "B365C>2.5","B365C<2.5","MaxC>2.5","MaxC<2.5","AvgC>2.5","AvgC<2.5",
                "AHCh","B365CAHH","B365CAHA","MaxCAHH","MaxCAHA","AvgCAHH","AvgCAHA"]:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce").astype("float64").values
        else:
            out[col] = np.full(n, np.nan)
    out["TotalGoals"] = out["FTHG"] + out["FTAG"]
    out["GoalDiff"] = out["FTHG"] - out["FTAG"]
    out["HTGoalDiff"] = out["HTHG"] - out["HTAG"]
    out["Over2_5"] = (out["TotalGoals"] > 2.5).astype("int64")
    out["BTTS"] = ((out["FTHG"] > 0) & (out["FTAG"] > 0)).astype("int64")
    out["HomeWin"] = (raw["FTR"].values == "H").astype("int64")
    out["Draw"] = (raw["FTR"].values == "D").astype("int64")
    out["AwayWin"] = (raw["FTR"].values == "A").astype("int64")
    out["HomePoints"] = raw["FTR"].map({"H":3,"D":1,"A":0}).fillna(0).astype("int64").values
    out["AwayPoints"] = raw["FTR"].map({"H":0,"D":1,"A":3}).fillna(0).astype("int64").values
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ImpProbH"] = 1.0 / np.clip(out["AvgH"], 1.01, None)
        out["ImpProbD"] = 1.0 / np.clip(out["AvgD"], 1.01, None)
        out["ImpProbA"] = 1.0 / np.clip(out["AvgA"], 1.01, None)
        s = out["ImpProbH"] + out["ImpProbD"] + out["ImpProbA"]
        out["FairProbH"] = out["ImpProbH"] / s
        out["FairProbD"] = out["ImpProbD"] / s
        out["FairProbA"] = out["ImpProbA"] / s
    dates = pd.to_datetime(out["Date"])
    out["Year"] = dates.year.astype("int64").values
    out["Month"] = dates.month.astype("int64").values
    out["DayOfWeek"] = dates.dayofweek.astype("int64").values
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype("int64")
    return pd.DataFrame({c: out[c] for c in ALL_COLS})

def make_schema():
    fields = []
    for c in ALL_COLS:
        if c == "Date":
            fields.append(pa.field(c, pa.timestamp("us")))
        elif c in INT_COLS:
            fields.append(pa.field(c, pa.int64()))
        elif c in FLOAT_COLS:
            fields.append(pa.field(c, pa.float64()))
        else:
            fields.append(pa.field(c, pa.large_string()))
    return pa.schema(fields)

def main():
    log("=" * 60)
    log("BUILD CLEAN MASTER PARQUET (streaming)")
    log("=" * 60)
    schema = make_schema()
    tmp = OUT_PARQUET.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    total = 0
    files_ok = 0
    seen = set()
    for season in season_codes():
        season_rows = 0
        for div in DIVS:
            path = RAW_DIR / season / f"{div}.csv"
            if not path.exists(): continue
            df = clean_file(path, season)
            if df.empty: continue
            keys = (df["Div"].astype(str) + "|" + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d") + "|"
                    + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str))
            mask = ~keys.isin(seen)
            df = df.loc[mask]
            if df.empty: continue
            seen.update(keys[mask].tolist())
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            writer.write_table(table)
            n = len(df)
            total += n
            season_rows += n
            files_ok += 1
            del df, table
        if season_rows:
            log(f"{season} ({code_to_label(season)}): +{season_rows:,} rows")
        gc.collect()
    writer.close()
    tmp.replace(OUT_PARQUET)
    log(f"Files: {files_ok} | Rows: {total:,} | Keys: {len(seen):,}")
    log(f"Wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.2f} MB)")
    log("DONE")

if __name__ == "__main__":
    main()
