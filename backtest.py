from __future__ import division
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import glob
import pandas as pd
import numpy as np
import joblib
from custom_transformers import RareCategoryGrouper
from custom_transformers import cast_float32
import matplotlib.pyplot as plt

#############################

HISTO_Z_SPREAD = ".\Z_HistoLevels.csv"
DATA_DIR = r".\ML_tables_processed"
MODEL_DIR = Path(r".\Pred_Models")
TARGET = "Today_settlementPX_spread"

CAT_COLS = ["spread_id", "commodity"]
NUM_COLS = [
    "t2c_min","bid","mid","ask","prev_close", "ba_spread_pct",
    "liquidity_avg", "liquidity_log",
    "mid_minus_prev_close", "mid_over_prev_close",
    "dow", "minute_of_day", "minute_sin", "minute_cos"
]

RAW_KEEP = [
    "ts", "date", "spread_id", "commodity",
    "Mid", "BA_spread_%", "lastBD_settlementPX_spread",
    "t2c_min", "liquidity_avg", "liquidity_log",
    "mid_minus_prev_close", "mid_over_prev_close",
    "minute_of_day", "minute_sin", "minute_cos", "dow",
    TARGET
]

############################

def load_all_csv(folder: str) -> pd.DataFrame:
    files = glob.glob(str(Path(folder) / "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        cols = [c for c in RAW_KEEP if c in df.columns]
        df = df[cols].copy()
        df = df.rename(columns={
            "Mid": "mid",
            "BA_spread_%": "ba_spread_pct",
            "lastBD_settlementPX_spread": "prev_close",
        })
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {folder}")
    out = pd.concat(dfs, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["spread_id", "ts"]).reset_index(drop=True)


def backtest_spread(model_path: str, spread: str, nb_contract : int, nb_exec:int, nb_obs:int, BoS : str):

    pipe = joblib.load(model_path)
    df = load_all_csv(DATA_DIR)
    df = df.sort_values(["spread_id", "ts"]).copy()
    mid_col = "mid" if "mid" in df.columns else ("Mid" if "Mid" in df.columns else None)
    s = df[mid_col].astype(float)
    settle = df["Today_settlementPX_spread"].astype(float)

    df["ask"] = df["mid"] + (df["ba_spread_pct"] / 200) * df["mid"]
    df["bid"] = df["mid"] - (df["ba_spread_pct"] / 200) * df["mid"]

    X = df[CAT_COLS + NUM_COLS]
    y_true = df[TARGET]

    y_pred = pipe.predict(X)
    df["predicted_" + TARGET] = y_pred

    g = df.groupby("spread_id", group_keys=False)
    sd = (g[mid_col]
          .rolling(window=120, min_periods=30)
          .std(ddof=1)
          .reset_index(level=0, drop=True))
    sd_safe = sd.replace(0, np.nan)
    z_set = (s - settle) / sd_safe
    z_set = pd.Series(z_set, index=df.index).replace([np.inf, -np.inf], np.nan)
    df["z_set"] = z_set

    z_pred = (s - y_pred) / sd_safe
    z_pred = pd.Series(z_pred, index=df.index).replace([np.inf, -np.inf], np.nan)
    df["z_pred"] = z_pred
    df = df.dropna()

    df_spread = df[df["spread_id"] == spread].copy()
    df_spread = df_spread.sort_values("ts").reset_index(drop=True)

    #add histo data
    histo_df = pd.read_csv(HISTO_Z_SPREAD)
    histo_df = histo_df[histo_df["spread_id"] == spread]
    histo_df.drop(columns=["nb_obs"], inplace=True)
    histo_cols = [c for c in histo_df.columns if c == "minute_of_day" or c.startswith("z_MidvsSet_histo_")]
    histo_df = histo_df[histo_cols].rename(columns={"minute_of_day": "minute_of_day"})
    df_spread=df_spread.merge(histo_df, on="minute_of_day", how="left")

    qty_total_signal =0
    qty_total_last5 = 0
    exec_bc_intrasignal = 0
    exec_fivelastmin = 0
    dates = df_spread["date"].unique()
    df_pnl = pd.DataFrame(columns=["date","avg_exec", "avg_bench", "avg_twap"])
    np.random.seed(42)
    for dt in dates:
        df_spread_dt = df_spread[df_spread["date"] == dt]
        min_of_obs = df_spread_dt[df_spread_dt["t2c_min"]>5]['t2c_min'].unique()
        if len(min_of_obs) < nb_obs:
            continue
        selected_obs = np.random.choice(min_of_obs, size=nb_obs, replace=False)
        df_sel = df_spread_dt[df_spread_dt["t2c_min"].isin(selected_obs)][[
            "date","t2c_min","bid","mid","ask",TARGET, "z_set","predicted_" + TARGET, "z_pred","z_MidvsSet_histo_q15","z_MidvsSet_histo_q25",
            "z_MidvsSet_histo_q40","z_MidvsSet_histo_q60", "z_MidvsSet_histo_q75", "z_MidvsSet_histo_q85"
        ]].copy()
        df_sel.sort_values("t2c_min",ascending=False, inplace=True)
        df_sel.reset_index(drop=True, inplace=True)

        df_sel[["Exec_px","Bench_px","qty_exec","nom_exec","bench_exec"]] = np.nan

        qty_ex=0
        nb_exec_updated = nb_exec
        for i in range(len(df_sel)):
            z_pred_i = df_sel.loc[df_sel.index[i], "z_pred"]
            z_diff_i_q15 = z_pred_i - df_sel.loc[df_sel.index[i], "z_MidvsSet_histo_q15"]
            z_diff_i_q25 = z_pred_i - df_sel.loc[df_sel.index[i], "z_MidvsSet_histo_q25"]
            z_diff_i_q40 = z_pred_i - df_sel.loc[df_sel.index[i], "z_MidvsSet_histo_q40"]
            z_diff_i_q60 = z_pred_i - df_sel.loc[df_sel.index[i], "z_MidvsSet_histo_q60"]
            z_diff_i_q75 = z_pred_i - df_sel.loc[df_sel.index[i], "z_MidvsSet_histo_q75"]
            z_diff_i_q85 = z_pred_i - df_sel.loc[df_sel.index[i], "z_MidvsSet_histo_q85"]


            if BoS == "B" and i < len(df_sel)-1 and z_pred_i < 0 and nb_exec_updated != 0 and qty_ex < nb_contract:
                if z_diff_i_q15 < 0:
                    df_sel.loc[df_sel.index[i], 'Exec_px'] = df_sel.loc[df_sel.index[i], "ask"]
                    df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                    df_sel.loc[df_sel.index[i], 'qty_exec'] = -int(0.3*nb_contract) if int(0.3*nb_contract) <= (nb_contract-qty_ex) else -int(nb_contract-qty_ex)
                    qty_ex = qty_ex + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])
                    df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Exec_px']
                    df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Bench_px']
                    nb_exec_updated=nb_exec_updated-1
                    exec_bc_intrasignal = exec_bc_intrasignal+1
                    qty_total_signal=qty_total_signal+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

                elif z_diff_i_q25 < 0:
                    df_sel.loc[df_sel.index[i], 'Exec_px'] = df_sel.loc[df_sel.index[i], "ask"]
                    df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                    df_sel.loc[df_sel.index[i], 'qty_exec'] = -int(0.2 * nb_contract) if int(0.2*nb_contract) <= (nb_contract-qty_ex) else -int(nb_contract-qty_ex)
                    qty_ex = qty_ex + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])
                    df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec'] * 1000 * df_sel.loc[df_sel.index[i], 'Exec_px']
                    df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec'] * 1000 * df_sel.loc[df_sel.index[i], 'Bench_px']
                    nb_exec_updated = nb_exec_updated - 1
                    exec_bc_intrasignal = exec_bc_intrasignal + 1
                    qty_total_signal=qty_total_signal+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

                elif z_diff_i_q40 < 0:
                    df_sel.loc[df_sel.index[i], 'Exec_px'] = df_sel.loc[df_sel.index[i], "ask"]
                    df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                    df_sel.loc[df_sel.index[i], 'qty_exec'] = -int(0.1 * nb_contract) if int(0.1*nb_contract) <= (nb_contract-qty_ex) else -int(nb_contract-qty_ex)
                    qty_ex = qty_ex + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])
                    df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec'] * 1000 * df_sel.loc[df_sel.index[i], 'Exec_px']
                    df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec'] * 1000 * df_sel.loc[df_sel.index[i], 'Bench_px']
                    nb_exec_updated = nb_exec_updated - 1
                    exec_bc_intrasignal = exec_bc_intrasignal + 1
                    qty_total_signal=qty_total_signal+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

                else :
                    df_sel.loc[df_sel.index[i], ['Exec_px', 'Bench_px', 'qty_exec', 'nom_exec', 'bench_exec']] = [np.nan, np.nan, 0, 0, 0]

            elif BoS == "B" and qty_ex < nb_contract and i == len(df_sel)-1:
                df_sel.loc[df_sel.index[i], 'Exec_px'] = df_spread_dt[df_spread_dt["t2c_min"]<=5]["ask"].mean()
                df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                df_sel.loc[df_sel.index[i], 'qty_exec']= -int(nb_contract-qty_ex)
                df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Exec_px']
                df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Bench_px']
                exec_fivelastmin=exec_fivelastmin+1
                qty_total_last5 = qty_total_last5+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

            elif BoS == "S" and i < len(df_sel)-1 and z_pred_i > 0 and nb_exec_updated != 0 and qty_ex < nb_contract:
                if z_diff_i_q85>0:
                    df_sel.loc[df_sel.index[i], 'Exec_px'] = df_sel.loc[df_sel.index[i], "bid"]
                    df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                    df_sel.loc[df_sel.index[i], 'qty_exec'] = int(0.3*nb_contract) if int(0.3*nb_contract) <= (nb_contract-qty_ex) else int(nb_contract-qty_ex)
                    qty_ex = qty_ex + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])
                    df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Exec_px']
                    df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Bench_px']
                    nb_exec_updated=nb_exec_updated-1
                    exec_bc_intrasignal = exec_bc_intrasignal+1
                    qty_total_signal=qty_total_signal+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

                elif z_diff_i_q75>0:
                    df_sel.loc[df_sel.index[i], 'Exec_px'] = df_sel.loc[df_sel.index[i], "bid"]
                    df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                    df_sel.loc[df_sel.index[i], 'qty_exec'] = int(0.2*nb_contract) if int(0.2*nb_contract) <= (nb_contract-qty_ex) else int(nb_contract-qty_ex)
                    qty_ex = qty_ex + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])
                    df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Exec_px']
                    df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Bench_px']
                    nb_exec_updated=nb_exec_updated-1
                    exec_bc_intrasignal = exec_bc_intrasignal+1
                    qty_total_signal=qty_total_signal+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

                elif z_diff_i_q60>0:
                    df_sel.loc[df_sel.index[i], 'Exec_px'] = df_sel.loc[df_sel.index[i], "bid"]
                    df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                    df_sel.loc[df_sel.index[i], 'qty_exec'] = int(0.1*nb_contract) if int(0.1*nb_contract) <= (nb_contract-qty_ex) else int(nb_contract-qty_ex)
                    qty_ex = qty_ex + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])
                    df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Exec_px']
                    df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Bench_px']
                    nb_exec_updated=nb_exec_updated-1
                    exec_bc_intrasignal = exec_bc_intrasignal+1
                    qty_total_signal=qty_total_signal+abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

                else :
                    df_sel.loc[df_sel.index[i], ['Exec_px', 'Bench_px', 'qty_exec', 'nom_exec', 'bench_exec']] = [np.nan, np.nan, 0, 0, 0]

            elif BoS == "S" and qty_ex < nb_contract and i == len(df_sel)-1:
                df_sel.loc[df_sel.index[i], 'Exec_px'] = df_spread_dt[df_spread_dt["t2c_min"]<=5]["bid"].mean()
                df_sel.loc[df_sel.index[i], 'Bench_px'] = df_sel.loc[df_sel.index[i], TARGET]
                df_sel.loc[df_sel.index[i], 'qty_exec']= int(nb_contract-qty_ex)
                df_sel.loc[df_sel.index[i], 'nom_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Exec_px']
                df_sel.loc[df_sel.index[i], 'bench_exec'] = df_sel.loc[df_sel.index[i], 'qty_exec']*1000*df_sel.loc[df_sel.index[i], 'Bench_px']
                exec_fivelastmin = exec_fivelastmin+1
                qty_total_last5 = qty_total_last5 + abs(df_sel.loc[df_sel.index[i], 'qty_exec'])

            else:
                df_sel.loc[df_sel.index[i], ['Exec_px','Bench_px','qty_exec','nom_exec','bench_exec']] = [np.nan, np.nan, 0, 0, 0]

        avg_twap = df_spread_dt[df_spread_dt["t2c_min"]<=5]["bid"].mean() if BoS == "S" else df_spread_dt[df_spread_dt["t2c_min"]<=5]["ask"].mean()
        avg_exec = df_sel['nom_exec'].sum()/(df_sel['qty_exec'].sum()*1000) if df_sel['qty_exec'].sum() != 0 else 0.0
        avg_bench = df_sel['bench_exec'].sum()/(df_sel['qty_exec'].sum()*1000) if df_sel['qty_exec'].sum() != 0 else 0.0
        if (avg_exec == 0.0) or (abs((avg_exec/avg_bench)-1) > 0.15) or (avg_twap == 0.0) or (avg_twap == np.nan):
            continue
        daily_df = pd.DataFrame({"date": dt, "avg_exec": [avg_exec], "avg_bench": [avg_bench],"avg_twap": [avg_twap]})
        df_pnl = pd.concat([df_pnl, daily_df], ignore_index=True)
        print(f"PnL for Date {dt} = {avg_exec-avg_bench:.2f} | Avg Exec: {avg_exec:.2f} | Avg Bench: {avg_bench:.2f} | Avg TWAP: {avg_twap:.2f}")

    df_pnl["pnl"] = df_pnl["avg_exec"] - df_pnl["avg_bench"]
    df_pnl["pnl_twap"] = df_pnl["avg_twap"] - df_pnl["avg_bench"]


    return df_pnl, qty_total_signal, qty_total_last5



if __name__ == "__main__":

    spread_list = ["BOFBOH","BOHBOK","BOKBON","BONBOZ","BOZBOF"]
    commo = "BO"
    way = "B" # "B" for Buy the spread when signal, "S" for Sell the spread when signal

    model_dir = MODEL_DIR
    list_models = sorted(str(p) for p in model_dir.glob("*.joblib"))

    OUTPUT_DIR = Path(rf".\Backtest_results\{commo}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for model in list_models:
        model_name = Path(model).stem

        plt.figure(figsize=(14, 6))

        for spr in spread_list:
            output = backtest_spread(model, spread=spr, nb_contract=50, nb_exec=5, nb_obs=10, BoS=way)

            df_pnl = output[0]
            print(f'[{model_name}] qty exec bc signal = {output[1]} | qty exec last 5min = {output[2]}')

            df_pnl["pnl"].cumsum().reset_index(drop=True).plot(label='PnL '+ spr)

        plt.title(f"Intraday spread execution strategy {way} for commo {commo} — Cumulated PnL for model : {model_name}")

        plt.xlabel("Observations")
        plt.ylabel("Cumulated PnL vs Settlement")
        plt.grid(True)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        save_path = OUTPUT_DIR / f"backtest_{commo}_{model_name}_{way}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Graph saved : {save_path}")


    print('End of backtest')
