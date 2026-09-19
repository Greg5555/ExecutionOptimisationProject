from __future__ import division
import warnings

from fontTools.subset import subset

warnings.filterwarnings('ignore')


import os
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import pandas as pd
import numpy as np
from pathlib import Path
np.seterr(all="ignore")



# def hhmmss_to_seconds(x):
#     try:
#         if pd.isna(x): return np.nan
#         h, m, s = str(x).split(':'); return int(h)*3600 + int(m)*60 + float(s)
#     except Exception:
#         return np.nan
#
#
# def add_calendar_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
#     """Ajoute minute-of-day, sin/cos, jour de semaine (dow)."""
#     d = df.copy()
#     d[ts_col] = pd.to_datetime(d[ts_col], errors='coerce')
#     minute = d[ts_col].dt.hour * 60 + d[ts_col].dt.minute
#     d['minute_of_day'] = minute
#     d['minute_sin'] = np.sin(2*np.pi*minute/1440.0)
#     d['minute_cos'] = np.cos(2*np.pi*minute/1440.0)
#     d['dow'] = d[ts_col].dt.dayofweek
#     return d



# ---------- Chargement ----------
def extract_name_spread(spread):
    try:
        parts = spread.split('-')
        o1=parts[0][:-4]
        o2=parts[1][:-4]
        return o1+o2
    except Exception:
        return "UNK"


def process_file(filepath):
    try:
        df_ML = pd.read_csv(filepath)
        df_ML["spread_name"] = df_ML["spread_id"].apply(extract_name_spread)
        has_nan = df_ML.isna().values.any()

        df_ML = df_ML.replace([np.inf, -np.inf], np.nan)
        df_ML = df_ML.apply(lambda col: col.fillna(col.mean()) if col.dtype.kind in "fc" else col)

        df_ML = df[['#DATETIME','minute_of_day','minute_sin','minute_cos','dow','time2closing','Commo','unique_Contract','BEST_ASIZ1','Mid','BEST_BSIZ1','BA_spread_%','lastBD_settlementPX_spread','Today_settlementPX_spread']].copy()
        df_ML["ts"] = pd.to_datetime(df_ML["#DATETIME"])
        df_ML["date"] = df_ML["ts"].dt.date
        df_ML["spread_id"] = df_ML.get("unique_Contract", Path(filepath).stem)
        df_ML["commodity"] = df_ML.get("Commo", "UNK")
        df_ML["t2c_min"] = pd.to_timedelta(df_ML["time2closing"]).dt.total_seconds() / 60.0
        df_ML["liquidity_avg"] = 0.5 * (df_ML["BEST_ASIZ1"] + df_ML["BEST_BSIZ1"])
        df_ML["liquidity_log"] = np.log1p(df_ML["liquidity_avg"])
        df_ML['BA_spread_%']=df_ML['BA_spread_%'].apply(lambda x: abs(x))
        df_ML['mid_minus_prev_close'] = df_ML['Mid'] - df_ML['lastBD_settlementPX_spread']
        df_ML.dropna(how='any', inplace=True)
        df_ML['mid_over_prev_close'] = df_ML['mid_minus_prev_close'] / df_ML['lastBD_settlementPX_spread']
        df_ML.dropna(how='any', inplace=True)
        df_ML.drop(columns=['time2closing','#DATETIME','Commo','unique_Contract','BEST_ASIZ1','BEST_BSIZ1'], inplace=True)
        df_ML.sort_values(["spread_id", "ts"]).reset_index(drop=True)


        out_path = Path(r".\ML_tables_processed")
        os.makedirs(out_path, exist_ok=True)
        output_path = os.path.join(out_path, os.path.basename(filepath))
        df_ML.to_csv(output_path, index=False)

        return f"Success: {os.path.basename(filepath)} | saved: {output_path} & {ml_out_path}"

    except Exception as e:
        print(e)
        return f"Error {os.path.basename(filepath)}: {str(e)}"


if __name__ == "__main__":

    folder_path = r".\ML_tables_processed"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    print(f"Found {len(csv_files)} CSV files to process")

    # Process files in parallel
    n_processes = max(1, cpu_count() - 2)  # Leave 2 cores free
    print(f"Using {n_processes} processes")

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_file, csv_files), total=len(csv_files)))

    for r in results:
        print(r)
