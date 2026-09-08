from __future__ import annotations
from __future__ import division

import joblib
import pandas as pd
from custom_transformers import RareCategoryGrouper
from custom_transformers import cast_float32

import warnings
warnings.filterwarnings('ignore')
import sys, os
import inspect
import math
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import joblib
from pandas import to_datetime
from datetime import datetime, timedelta
sys.path.append('Z:\EQD\SGI\SGIPython\Pycharm')
_path_folder = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))
from FIND.Holiday.holiday_function import get_holiday
from pandas.tseries.offsets import CustomBusinessDay

nyse_holiday = get_holiday('NYSE')
index_holiday = get_holiday('NYSE')
index_holiday.append(pd.to_datetime('2012-10-29'))  # sandy
index_holiday.append(pd.to_datetime('2012-10-30'))  # sandy
index_holiday.append(pd.to_datetime('2018-12-05'))  # George W Bush
index_CustomBusinessDay = CustomBusinessDay(holidays=index_holiday)

import carambar as bbg
from pytz import timezone
from bloomberg_client.local_bloomberg_client import LocalBloombergClient as LBC
bbg2 = LBC()



############################################################
# Configuration / constantes
##############################################################
DEFAULT_MODEL_PATH = r"X:\EQD\New_SGI\Scripts\CTY Project\Pred_Models\GradHistBoost_model1.joblib"

NY_TZ = ZoneInfo("America/New_York")
utc_tz = timezone('UTC')
local_tz = timezone('America/New_York')

HISTO_Z_DATA = r"X:\EQD\New_SGI\Scripts\CTY Project\Stats\HistoStats\Z_HistoLevels.csv"

CAT_COLS = ["spread_id", "commodity"]
NUM_COLS = [
    "t2c_min", "mid", "prev_close", "ba_spread_pct",
    "liquidity_avg", "liquidity_log",
    "mid_minus_prev_close", "mid_over_prev_close",
    "dow", "minute_of_day", "minute_sin", "minute_cos"
]
FEATS = CAT_COLS + NUM_COLS

closing = r"X:\EQD\New_SGI\Scripts\CTY Project\Intraday_Data\ExpiByContract_closeCommo\close_commo.xlsx"
df_closing = pd.read_excel(closing)
dic_closing = dict(zip(df_closing['commo'], df_closing['close']))

###################FUNCTIONS#################################

def get_ticker_bbg(spread_id: str) -> str:
    #spread id to bbg ticker
    try:
        fut1 = spread_id.split(sep="-")[0]
        fut2 = spread_id.split(sep="-")[1]
        ticker = fut1[:-4] + fut1[-1] + fut2[:-4] + fut2[-1] + " Comdty"
        return ticker
    except Exception as e:
        raise ValueError(f"Invalid spread_id format: {spread_id}. Expected format like 'BOF2024-BOH2024'.") from e

def get_px_bbg(ticker: str) -> float:
    try:
        px = bbg.BDP(ticker, "PX_LAST")
        return px.iloc[0] if hasattr(px, "iloc") else float(px)
    except Exception as e:
        raise ValueError(f"Error while getting BBG mid price for ticker {ticker}: {str(e)}") from e

def compute_intraday_RealizedVol(ticker):
    global utc_tz
    global local_tz
    now = pd.to_datetime(datetime.now())

    try:
        START_DATE = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second) - timedelta(hours=3)
        END_DATE = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
        start_time = pd.Timestamp(START_DATE, tzinfo=local_tz)
        end_time = pd.Timestamp(END_DATE, tzinfo=local_tz)
        time_range = pd.date_range(start_time, end_time, freq='1min').tz_convert(utc_tz).tz_convert(None)
        temp_data = bbg2.bdit(ticker, start_time, end_time, event_type_list=['BID', 'ASK'])
        bid_snaps = []
        ask_snaps = []
        if len(temp_data) > 0:
            bid_prices = temp_data.loc[temp_data.type == 'BID']
            ask_prices = temp_data.loc[temp_data.type == 'ASK']
            bid_times = pd.DatetimeIndex(bid_prices.time.values)
            bid_values = bid_prices.value.values
            ask_times = pd.DatetimeIndex(ask_prices.time.values)
            ask_values = ask_prices.value.values
            for i in range(0, len(time_range)):
                temp_bids = bid_values[bid_times <= time_range[i]]
                temp_asks = ask_values[ask_times <= time_range[i]]
                try:
                    bid_snaps.append(temp_bids[-1])
                except:
                    bid_snaps.append(0)
                try:
                    ask_snaps.append(temp_asks[-1])
                except:
                    ask_snaps.append(0)
            mid_snaps = list((bid_snaps[i]+ask_snaps[i])/2 for i in range(len(bid_snaps)))
            std_mid_roll120 = np.std(mid_snaps,ddof=1)
        else :
            raise ValueError(f"No intraday data found for ticker {ticker} in the last 3 hours.")
        return round(std_mid_roll120, 4)
    except Exception as e:
        raise ValueError(f"Error computing intraday RV for {ticker}: {e}") from e


def get_prev_closeBBG(ticker: str) -> float:
    try:
        dt_end = to_datetime(datetime.now().strftime('%Y-%m-%d'))
        dt_tm1 = dt_end - index_CustomBusinessDay
        px = bbg.BDH([ticker], ['PX_LAST'], dt_tm1)
        return px.iloc[0] if hasattr(px, "iloc") else float(px)
    except Exception as e:
        raise ValueError(f"Error fetching BBG previous close for ticker {ticker}: {str(e)}") from e


def get_spread_BA_BBG(ticker: str) -> float:
    try:
        data = bbg.BDP([ticker], ['BID', 'ASK'])
        bid_raw = data['BID']
        ask_raw = data['ASK']
        bid = bid_raw.iloc[0] if hasattr(bid_raw, "iloc") else float(bid_raw)
        ask = ask_raw.iloc[0] if hasattr(ask_raw, "iloc") else float(ask_raw)
        mid = (bid + ask) / 2.0
        return round((ask - bid) / mid * 100.0,4)
    except Exception as e:
        raise ValueError(f"Error fetching BBG bid-ask spread for ticker {ticker}: {str(e)}") from e


def get_average_liquidity_BBG(ticker: str) -> float:
    try:
        data = bbg.BDP([ticker], ['BID_SIZE', 'ASK_SIZE'])
        bid_size = data['BID_SIZE']
        ask_size = data['ASK_SIZE']
        return (bid_size + ask_size) / 2.0
    except Exception as e:
        raise ValueError(f"Error fetching liquidity from BBG for ticker {ticker}: {str(e)}") from e


def get_histo_z_levels(spread_id: str, minute_of_day: int) -> dict:
    #Get historical z-score level for a spread id and a minute of the day
    df = pd.read_csv(HISTO_Z_DATA)
    row = df[(df["spread_id"] == spread_id) & (df["minute_of_day"] == minute_of_day)]
    if row.empty:
        raise ValueError(f"No historical z-score levels found for spread_id={spread_id} and minute_of_day={minute_of_day}.")
    out = dict()
    for col in row.columns:
        if col.startswith("z_MidvsSet_histo_q"):
            out[col] = float(row[col].values[0])
    return out


def compute_time_features(dt: datetime) -> dict:
    dow = dt.weekday()
    minute_of_day = dt.hour * 60 + dt.minute
    angle = 2 * math.pi * (minute_of_day / 1440.0)
    minute_sin = math.sin(angle)
    minute_cos = math.cos(angle)
    return {
        "dow": dow,
        "minute_of_day": minute_of_day,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
    }


def get_spread_id_from_spread_name(spread_name: str) -> str:
    try:
        fut1 = spread_name.split(sep="-")[0]
        fut2 = spread_name.split(sep="-")[1]
        spread_id = fut1[:-4] + fut2[:-4]
        return spread_id
    except Exception as e:
        raise ValueError(f"Invalid spread_name format: {spread_name}. Expected format like 'BON2026-BOZ2026'.") from e


def compute_derived_features(spread_name: str, commo: str, min_of_day) -> dict:

    close = dic_closing.get(commo, None)
    if close is None:
        raise ValueError(f"No closing price found for commodity {commo} in the closing dictionary.")
    minute_of_day_close = close.hour * 60 + close.minute
    if minute_of_day_close < min_of_day :
        raise ValueError(f"Current time is after the closing time for commodity {commo}.")
    else :
        time2close = minute_of_day_close - min_of_day

    #BBG data
    ticker_BBG = get_ticker_bbg(spread_name)
    prev_close = get_prev_closeBBG(ticker_BBG)
    prev_close = prev_close.iloc[0] if hasattr(prev_close, "iloc") else float(prev_close)
    mid = get_px_bbg(ticker_BBG)
    mid = mid.iloc[0] if hasattr(mid, "iloc") else float(mid)

    mid_minus_prev_close = float(mid) - float(prev_close)
    if float(prev_close) == 0:
        mid_over_prev_close = np.nan
    else:
        mid_over_prev_close = float(mid) / float(prev_close) - 1.0

    ba_spread = get_spread_BA_BBG(ticker_BBG)
    avg_liquidity = get_average_liquidity_BBG(ticker_BBG)
    liquidity_log = float(np.log10(avg_liquidity)) if avg_liquidity > 0 else np.nan
    spread_id = get_spread_id_from_spread_name(spread_name)

    return {
        "mid": mid,
        "mid_minus_prev_close": mid_minus_prev_close,
        "mid_over_prev_close": mid_over_prev_close,
        "liquidity_log": liquidity_log,
        "spread_id": spread_id,
        "ba_spread_pct": ba_spread,
        "liquidity_avg": avg_liquidity,
        "prev_close": prev_close,
        "t2c_min": time2close
    }


##################LIVE MONITOR WINDOW################################################

SIGNAL_COLORS = {
    "STRONG BUY":  {"bg": "#006400", "fg": "#ffffff"},   # dark green
    "SOLID BUY":   {"bg": "#228B22", "fg": "#ffffff"},   # medium green
    "BUY":         {"bg": "#90EE90", "fg": "#000000"},   # light green
    "STRONG SELL": {"bg": "#8B0000", "fg": "#ffffff"},   # dark red
    "SOLID SELL":  {"bg": "#CC0000", "fg": "#ffffff"},   # medium red
    "SELL":        {"bg": "#FF6666", "fg": "#000000"},   # light red
    "NO SIGNAL":   {"bg": "#D3D3D3", "fg": "#000000"},   # grey
    "ERROR":       {"bg": "#FF8C00", "fg": "#ffffff"},   # orange
}


class LiveMonitorWindow(tk.Toplevel):

    UPDATE_INTERVAL_MS = 30_000  # 30s

    def __init__(self, parent, pipe, spread_name: str, commodity: str, buyorsell: str, model_name: str = ""):
        super().__init__(parent)
        self.pipe        = pipe
        self.spread_name = spread_name
        self.commodity   = commodity
        self.buyorsell   = buyorsell
        self.model_name  = model_name
        self._running    = True
        self._after_id   = None

        self.title(f"Live Monitor — {spread_name} ({buyorsell})")
        self.geometry("460x380")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh()

    ###UI
    def _build_ui(self):
        # Header info
        frm_info = ttk.Frame(self)
        frm_info.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(frm_info, text=f"Spread : {self.spread_name}",
                  font=("Calibri", 12, "bold")).pack(side="left", padx=8)
        ttk.Label(frm_info, text=f"Commodity : {self.commodity}",
                  font=("Calibri", 12)).pack(side="left", padx=8)
        ttk.Label(frm_info, text=f"Side : {self.buyorsell}",
                  font=("Calibri", 12, "bold")).pack(side="left", padx=8)
        ttk.Label(frm_info, text=f"Model : {self.model_name}",
                  font=("Calibri", 10, "italic"), foreground="#555555").pack(side="left", padx=8)

        # Signal banner
        self.lbl_signal = tk.Label(self, text="Computing…", font=("Calibri", 27, "bold"),
                                   bg="#D3D3D3", fg="#000000", relief="raised", pady=27)
        self.lbl_signal.pack(fill="x", padx=17, pady=13)

        # Last update + countdown
        frm_status = ttk.Frame(self)
        frm_status.pack(fill="x", padx=10)
        self.lbl_update = ttk.Label(frm_status, text="Last update: —", font=("Calibri", 9))
        self.lbl_update.pack(side="left")
        self.lbl_next   = ttk.Label(frm_status, text="", font=("Calibri", 9))
        self.lbl_next.pack(side="right")

        # Details log
        frm_log = ttk.LabelFrame(self, text="Details")
        frm_log.pack(fill="both", expand=True, padx=9, pady=5)
        self.txt_log = tk.Text(frm_log, height=6, wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(frm_log, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)

        # Refresh button
        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(frm_btn, text="🔄 Force refresh now", command=self._force_refresh).pack(side="left", padx=8)
        ttk.Button(frm_btn, text="🛑 Stop monitoring",   command=self._on_close).pack(side="left", padx=8)

    #helpers
    def _log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def _clear_log(self):
        self.txt_log.delete("1.0", "end")

    def _set_signal_banner(self, signal_key: str, text: str):
        colors = SIGNAL_COLORS.get(signal_key, SIGNAL_COLORS["NO SIGNAL"])
        self.lbl_signal.config(text=text, bg=colors["bg"], fg=colors["fg"])

    #core logic
    def _compute_signal(self) -> tuple[str, str, dict]:
        now_ny      = datetime.now(tz=NY_TZ)
        time_feats  = compute_time_features(now_ny)
        ticker_bbg  = get_ticker_bbg(self.spread_name)

        derived = compute_derived_features(
            spread_name=self.spread_name,
            commo=self.commodity,
            min_of_day=time_feats["minute_of_day"]
        )

        row = {
            "spread_id":           derived["spread_id"],
            "commodity":           self.commodity,
            "t2c_min":             derived["t2c_min"],
            "mid":                 derived["mid"],
            "prev_close":          derived["prev_close"],
            "ba_spread_pct":       derived["ba_spread_pct"],
            "liquidity_avg":       derived["liquidity_avg"],
            "liquidity_log":       derived["liquidity_log"],
            "mid_minus_prev_close":derived["mid_minus_prev_close"],
            "mid_over_prev_close": derived["mid_over_prev_close"],
            "dow":                 time_feats["dow"],
            "minute_of_day":       time_feats["minute_of_day"],
            "minute_sin":          time_feats["minute_sin"],
            "minute_cos":          time_feats["minute_cos"],
        }
        X    = pd.DataFrame([row], columns=FEATS)
        pred = float(self.pipe.predict(X)[0])

        rv = compute_intraday_RealizedVol(ticker_bbg)
        z_spread = (row["mid"] - pred) / rv if (rv is not None and rv != 0) else np.nan

        histo_z  = get_histo_z_levels(
            spread_id=derived["spread_id"],
            minute_of_day=time_feats["minute_of_day"]
        )

        # Signal logic
        if self.buyorsell == "BUY":
            if z_spread < histo_z.get("z_MidvsSet_histo_q15"):
                sk, st = "STRONG BUY", "STRONG BUY"
            elif z_spread < histo_z.get("z_MidvsSet_histo_q25"):
                sk, st = "SOLID BUY",  "SOLID BUY"
            elif z_spread < histo_z.get("z_MidvsSet_histo_q40"):
                sk, st = "BUY",        "BUY"
            else:
                sk, st = "NO SIGNAL",  "NO BUYING SIGNAL"
        else:
            if z_spread > histo_z.get("z_MidvsSet_histo_q85"):
                sk, st = "STRONG SELL","STRONG SELL"
            elif z_spread > histo_z.get("z_MidvsSet_histo_q75"):
                sk, st = "SOLID SELL", "SOLID SELL"
            elif z_spread > histo_z.get("z_MidvsSet_histo_q60"):
                sk, st = "SELL",       "SELL"
            else:
                sk, st = "NO SIGNAL",  "NO SELLING SIGNAL"

        details = {
            "time_ny":    now_ny.strftime("%Y-%m-%d %H:%M:%S"),
            "mid":        derived["mid"],
            "pred":       pred,
            "z_spread":   z_spread,
            "rv":         rv,
            "t2c_min":    derived["t2c_min"],
            "prev_close": derived["prev_close"],
            "ba_spread":  derived["ba_spread_pct"],
            "liquidity":  derived["liquidity_avg"],
            "histo_z":    histo_z,
        }
        return sk, st, details

    def _refresh(self):
        if not self._running:
            return

        def task():
            try:
                sk, st, details = self._compute_signal()

                def update_ui():
                    self._clear_log()
                    now_str = details["time_ny"]
                    self._set_signal_banner(sk, st)
                    self.lbl_update.config(text=f"Last update: {now_str}")
                    self.lbl_next.config(text="Next update in ~30s")

                    self._log(f"{'='*50}")
                    self._log(f"  Timestamp (NY)   : {now_str}")
                    self._log(f"  Spread           : {self.spread_name}  ({self.buyorsell})")
                    self._log(f"  Mid              : {details['mid']:.4f}")
                    self._log(f"  Predicted Set.   : {details['pred']:.4f}")
                    self._log(f"  Intraday RV      : {details['rv']:.4f}" if details['rv'] is not None else "  Intraday RV      : N/A")
                    self._log(f"  Z-Spread         : {details['z_spread']:.4f}")
                    self._log(f"  t2c (min)        : {details['t2c_min']}")
                    self._log(f"  Prev close       : {details['prev_close']:.4f}")
                    self._log(f"  B/A spread %     : {details['ba_spread']:.4f}")
                    self._log(f"  Avg liquidity    : {details['liquidity']:.1f}")
                    self._log(f"{'='*50}")
                    self._log("  Histo Z quantiles :")
                    for k, v in details["histo_z"].items():
                        self._log(f"    {k}: {v:.4f}")
                    self._log(f"{'='*50}")
                    self._log(f"  ➤ SIGNAL : {st}")
                    self._log(f"{'='*50}\n")

                self.after(0, update_ui)

            except Exception as e:
                err_msg = str(e)
                def show_err(msg=err_msg):
                    self._set_signal_banner("ERROR", f"⚠️ ERROR — {msg[:60]}")
                    self._log(f"[ERROR] {msg}")
                self.after(0, show_err)

            finally:
                if self._running:
                    self._after_id = self.after(self.UPDATE_INTERVAL_MS, self._refresh)

        threading.Thread(target=task, daemon=True).start()

    def _force_refresh(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.lbl_next.config(text="Refreshing…")
        self._refresh()

    def _on_close(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
        self.destroy()


##################MAIN APP################################################

class PredictApp(tk.Tk):

    MODEL_DIR = r"X:\EQD\New_SGI\Scripts\CTY Project\Pred_Models"

    def __init__(self):
        super().__init__()
        self.title("CTY Spread — Live Signal")
        self.geometry("460x380")
        self.resizable(False, False)
        self.pipe = None
        self._build_ui()

    #UI
    def _build_ui(self):
        hdr = tk.Frame(self, bg="#1a3c6e", pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📊 CTY Spread Live Signal",
                 font=("Calibri", 18, "bold"), bg="#1a3c6e", fg="white").pack()

        #Parameters frame
        frm = ttk.LabelFrame(self, text="Parameters", padding=14)
        frm.pack(fill="both", expand=True, padx=16, pady=9)
        frm.columnconfigure(1, weight=1)

        row = 0

        # Model selection
        ttk.Label(frm, text="Model :").grid(row=row, column=0, sticky="w", pady=5)
        frm_model = ttk.Frame(frm)
        frm_model.grid(row=row, column=1, sticky="ew", pady=5)

        # Dropdown avec les .joblib disponibles
        self.model_var = tk.StringVar()
        model_files = self._get_model_list()
        self.cmb_model = ttk.Combobox(frm_model, textvariable=self.model_var,
                                      values=model_files, state="readonly", width=34)
        if model_files:
            self.cmb_model.current(0)
        self.cmb_model.pack(side="left", fill="x", expand=True)
        ttk.Button(frm_model, text="📂 Browse…", command=self._browse_model).pack(side="left", padx=(6, 0))
        row += 1

        # Model status label
        self.lbl_model_status = ttk.Label(frm, text="⚠ Model not loaded", foreground="red",
                                           font=("Calibri", 9, "italic"))
        self.lbl_model_status.grid(row=row, column=1, sticky="w")
        ttk.Button(frm, text="✔ Load model", command=self._load_model).grid(
            row=row, column=0, sticky="w", pady=(0, 8))
        row += 1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                      sticky="ew", pady=8)
        row += 1

        # Spread
        ttk.Label(frm, text="Spread (e.g. BON2026-BOZ2026) :").grid(
            row=row, column=0, sticky="w", pady=5)
        self.ent_spread = ttk.Entry(frm, width=28)
        self.ent_spread.grid(row=row, column=1, sticky="ew", pady=5)
        row += 1

        # Commodity
        ttk.Label(frm, text="Commodity :").grid(row=row, column=0, sticky="w", pady=5)
        self.ent_commodity = ttk.Entry(frm, width=28)
        self.ent_commodity.grid(row=row, column=1, sticky="ew", pady=5)
        row += 1

        # BUY / SELL
        ttk.Label(frm, text="Side :").grid(row=row, column=0, sticky="w", pady=5)
        self.side_var = tk.StringVar(value="BUY")
        frm_side = ttk.Frame(frm)
        frm_side.grid(row=row, column=1, sticky="w", pady=5)
        ttk.Radiobutton(frm_side, text="BUY",  variable=self.side_var, value="BUY").pack(side="left", padx=(0, 16))
        ttk.Radiobutton(frm_side, text="SELL", variable=self.side_var, value="SELL").pack(side="left")
        row += 1

        #Launch button
        btn_launch = tk.Button(self, text="🚀  Start Live Monitoring",
                               font=("Calibri", 16, "bold"),
                               bg="#1a3c6e", fg="white", relief="raised",
                               pady=10, cursor="hand2",
                               command=self._on_launch)
        btn_launch.pack(fill="x", padx=16, pady=(0, 6))

        #Status bar
        self.lbl_status = ttk.Label(self, text="Ready.", font=("Calibri", 9),
                                    relief="sunken", anchor="w")
        self.lbl_status.pack(fill="x", side="bottom")

    #helpers
    def _get_model_list(self) -> list[str]:
        try:
            return [f for f in os.listdir(self.MODEL_DIR) if f.endswith(".joblib")]
        except Exception:
            return []

    def _browse_model(self):
        path = filedialog.askopenfilename(
            initialdir=self.MODEL_DIR,
            title="Select a model",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.model_var.set(os.path.basename(path))
            self._custom_model_path = path
            self.lbl_model_status.config(text="⚠ Model not loaded — click 'Load model'",
                                         foreground="orange")

    def _get_model_path(self) -> str:
        name = self.model_var.get()
        if not name:
            raise ValueError("No model selected.")
        if hasattr(self, "_custom_model_path") and os.path.basename(self._custom_model_path) == name:
            return self._custom_model_path
        return os.path.join(self.MODEL_DIR, name)

    def _load_model(self):
        try:
            path = self._get_model_path()
            self.lbl_status.config(text=f"Loading {os.path.basename(path)}…")
            self.update_idletasks()
            self.pipe = joblib.load(path)
            self.lbl_model_status.config(
                text=f"✔ Loaded : {os.path.basename(path)}", foreground="green")
            self.lbl_status.config(text="Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model load error", str(e))
            self.lbl_model_status.config(text="✘ Load failed", foreground="red")
            self.lbl_status.config(text="Error loading model.")

    def _on_launch(self):
        if self.pipe is None:
            messagebox.showwarning("Model", "Please load a model first.")
            return

        spread = self.ent_spread.get().strip()
        commodity = self.ent_commodity.get().strip()
        side = self.side_var.get()

        if not spread:
            messagebox.showwarning("Input", "Please enter a spread (e.g. BON2026-BOZ2026).")
            return
        if not commodity:
            messagebox.showwarning("Input", "Please enter a commodity.")
            return

        try:
            get_ticker_bbg(spread)
        except ValueError as e:
            messagebox.showerror("Spread format error", str(e))
            return

        self.lbl_status.config(text=f"Opening live monitor for {spread} ({side})…")
        LiveMonitorWindow(
            parent=self,
            pipe=self.pipe,
            spread_name=spread,
            commodity=commodity,
            buyorsell=side,
            model_name=os.path.basename(self._get_model_path()),
        )


#########MAIN#########

if __name__ == "__main__":
    app = PredictApp()
    app.mainloop()