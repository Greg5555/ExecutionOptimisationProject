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


List_spreadname= ['CLFCLH', 'CLGCLH', 'CLGCLM', 'CLHCLJ', 'CLHCLK', 'CLHCLM', 'CLHCLN', 'CLHCLZ', 'CLJCLN', 'CLJCLK', 'CLJCLQ', 'CLJCLZ', 'CLKCLM', 'CLKCLN', 'CLKCLQ', 'CLKCLU', 'CLMCLU', 'CLMCLN', 'CLMCLV', 'CLMCLM', 'CLNCLV', 'CLNCLU', 'CLNCLQ', 'CLNCLX', 'CLNCLM', 'CLQCLU', 'CLQCLX', 'CLQCLM', 'CLQCLZ', 'CLUCLF', 'CLUCLZ', 'CLUCLV', 'CLUCLX', 'CLUCLM', 'CLVCLF', 'CLVCLX', 'CLVCLG', 'CLVCLM', 'CLXCLG', 'CLXCLF', 'CLXCLZ', 'CLXCLH', 'CLXCLM', 'CLZCLH', 'CLZCLF', 'CLZCLJ', 'FCFFCH', 'FCHFCK', 'FCHFCJ', 'FCHFCQ', 'FCJFCK', 'FCJFCQ', 'FCKFCQ', 'FCQFCV', 'FCQFCU', 'FCQFCX', 'FCUFCV', 'FCUFCF', 'FCVFCX', 'FCXFCF', 'FCXFCH', 'NGFNGH', 'NGGNGH', 'NGGNGK', 'NGHNGK', 'NGHNGJ', 'NGHNGM', 'NGHNGU', 'NGJNGK', 'NGJNGN', 'NGKNGM', 'NGKNGN', 'NGKNGQ', 'NGMNGU', 'NGMNGN', 'NGNNGQ', 'NGNNGV', 'NGNNGU', 'NGQNGX', 'NGQNGU', 'NGUNGH', 'NGUNGZ', 'NGUNGV', 'NGUNGX', 'NGVNGF', 'NGVNGX', 'NGXNGG', 'NGXNGF', 'NGXNGZ', 'NGZNGF', 'LHGLHM', 'LHJLHQ', 'LHJLHN', 'LHJLHM', 'LHMLHN', 'LHMLHQ', 'LHMLHV', 'LHNLHV', 'LHNLHQ', 'LHQLHZ', 'LHQLHV', 'LHVLHG', 'LHVLHZ', 'LHVLHJ', 'LHZLHN', 'LHZLHJ', 'LHZLHG', 'LHZLHM', 'LCGLCJ', 'LCJLCQ', 'LCJLCM', 'LCMLCV', 'LCMLCQ', 'LCQLCZ', 'LCQLCG', 'LCQLCV', 'LCVLCG', 'LCVLCZ', 'LCZLCJ', 'LCZLCG', 'QSFQSG', 'QSFQSJ', 'QSFQSH', 'QSGQSH', 'QSGQSK', 'QSHQSJ', 'QSHQSK', 'QSHQSM', 'QSJQSK', 'QSJQSN', 'QSKQSM', 'QSKQSQ', 'QSKQSN', 'QSKQSX', 'QSMQSN', 'QSMQSU', 'QSNQSQ', 'QSNQSV', 'QSNQSX', 'QSNQSU', 'QSQQSU', 'QSQQSX', 'QSUQSV', 'QSUQSZ', 'QSUQSX', 'QSVQSF', 'QSVQSX', 'QSXQSH', 'QSXQSG', 'QSXQSZ', 'QSXQSF', 'QSZQSF', 'QSZQSH', 'KWHKWN', 'KWHKWK', 'KWKKWU', 'KWKKWN', 'KWNKWZ', 'KWNKWU', 'KWNKWH', 'KWUKWH', 'KWUKWZ', 'KWUKWK', 'KWZKWK', 'KWZKWH', 'KWZKWN', 'KCHKCU', 'KCHKCN', 'KCHKCK', 'KCKKCN', 'KCKKCU', 'KCNKCU', 'KCNKCZ', 'KCUKCZ', 'KCZKCK', 'KCZKCH', 'KCZKCN', 'GCGGCJ', 'GCGGCM', 'GCJGCQ', 'GCJGCM', 'GCMGCQ', 'GCMGCZ', 'GCQGCG', 'GCQGCZ', 'GCZGCJ', 'GCZGCG', 'SFSH', 'SHSK', 'SHSN', 'SKSN', 'SKSX', 'SNSF', 'SNSX', 'SXSH', 'SXSF', 'SXSK', 'SBHSBK', 'SBHSBN', 'SBHSBV', 'SBKSBV', 'SBKSBN', 'SBNSBK', 'SBNSBV', 'SBVSBN', 'SBVSBH', 'SIHSIK', 'SIHSIN', 'SIKSIU', 'SIKSIN', 'SINSIU', 'SINSIZ', 'SIUSIH', 'SIUSIZ', 'SIZSIK', 'SIZSIH', 'CTHCTK', 'CTHCTN', 'CTKCTN', 'CTKCTZ', 'CTNCTZ', 'CTZCTK', 'CTZCTN', 'CTZCTH', 'COFCOK', 'COFCOH', 'COFCOG', 'COHCOJ', 'COHCOK', 'COJCOK', 'COKCOM', 'COKCOU', 'COKCON', 'COMCON', 'CONCOX', 'CONCOQ', 'CONCOU', 'COQCOU', 'COUCOF', 'COUCOV', 'COUCOX', 'COVCOX', 'COXCOZ', 'COXCOF', 'COZCOF', 'CHCK', 'CHCN', 'CHCU', 'CKCU', 'CKCN', 'CNCN', 'CNCU', 'CNCZ', 'CUCZ', 'CZCH', 'CZCN', 'CCHCCK', 'CCHCCN', 'CCKCCU', 'CCKCCN', 'CCNCCU', 'CCUCCZ', 'CCZCCH', 'WHWK', 'WHWN', 'WKWU', 'WKWN', 'WKWZ', 'WNWU', 'WNWZ', 'WNWH', 'WUWZ', 'WUWK', 'WZWH', 'WZWN', 'BOFBOH', 'BOHBOK', 'BOKBON', 'BONBOZ', 'BOZBOF', 'SMFSMH', 'SMHSMK', 'SMKSMN', 'SMNSMZ', 'SMZSMF']

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


        out_path = Path(r"X:\EQD\New_SGI\Scripts\CTY Project\Intraday_Data\Files4strat\ML_tables\ML_tables_processed")
        os.makedirs(out_path, exist_ok=True)
        output_path = os.path.join(out_path, os.path.basename(filepath))
        df_ML.to_csv(output_path, index=False)

        return f"Success: {os.path.basename(filepath)} | saved: {output_path} & {ml_out_path}"

    except Exception as e:
        print(e)
        return f"Error {os.path.basename(filepath)}: {str(e)}"


if __name__ == "__main__":

    folder_path = r"X:\EQD\New_SGI\Scripts\CTY Project\Intraday_Data\Files4strat\ML_tables\ML_tables_processed"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    print(f"Found {len(csv_files)} CSV files to process")

    # Process files in parallel
    n_processes = max(1, cpu_count() - 2)  # Leave 2 cores free
    print(f"Using {n_processes} processes")

    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_file, csv_files), total=len(csv_files)))

    for r in results:
        print(r)
