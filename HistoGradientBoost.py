from __future__ import division
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple
from custom_transformers import RareCategoryGrouper
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

import joblib

DATA_DIR = r"X:\EQD\New_SGI\Scripts\CTY Project\Intraday_Data\Files4strat\ML_tables\ML_tables_processed"
MODEL_PATH = r"X:\EQD\New_SGI\Scripts\CTY Project\Pred_Models\GradHistBoost_model4.joblib"
TARGET = "Today_settlementPX_spread"

CAT_COLS = ["spread_id", "commodity"]
NUM_COLS = [
    "t2c_min", "mid", "prev_close", "ba_spread_pct",
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


def cast_float32(X):
    return X.astype(np.float32)


def load_all_csv(folder: str) -> pd.DataFrame:
    files = glob.glob(str(Path(folder) / "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        cols = [c for c in RAW_KEEP if c in df.columns]
        df = df[cols].copy()
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {folder}")
    out = pd.concat(dfs, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["spread_id", "ts"]).reset_index(drop=True)

def build_lite_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "ts": df["ts"],
        "date": pd.to_datetime(df["date"]).dt.date,
        "spread_id": df["spread_id"].astype(str),
        "commodity": df["commodity"].astype(str) if "commodity" in df.columns else "UNK",

        "t2c_min": df["t2c_min"].astype(float) if "t2c_min" in df.columns else np.nan,
        "mid": df["Mid"].astype(float),
        "prev_close": df["lastBD_settlementPX_spread"].astype(float),
        "ba_spread_pct": df["BA_spread_%"].astype(float),

        "liquidity_avg": df["liquidity_avg"].astype(float) if "liquidity_avg" in df.columns else np.nan,
        "liquidity_log": df["liquidity_log"].astype(float) if "liquidity_log" in df.columns else np.nan,
    })


    out["dow"] = df["dow"].astype(int)
    out["minute_of_day"] = df["minute_of_day"].astype(int)
    out["minute_sin"] = df["minute_sin"].astype(float)
    out["minute_cos"] = df["minute_cos"].astype(float)

    out["mid_minus_prev_close"] = df["mid_minus_prev_close"].astype(float)
    out["mid_over_prev_close"] = df["mid_over_prev_close"].astype(float)

    out[TARGET] = df[TARGET].astype(float)

    return out


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    #diminuish columns cardinality : cat freq >= min_freq // if > (max_categ - 1), we keep top (max_categ - 1)

    def __init__(self, max_categories=255, min_freq=20, other_token="__OTHER__"):
        self.max_categories = max_categories
        self.min_freq = min_freq
        self.other_token = other_token
        self.kept_per_col_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        n_cols = X.shape[1]
        self.kept_per_col_ = []
        for j in range(n_cols):
            col = X[:, j]
            cnt = Counter(col)
            keep = [v for v, f in cnt.most_common() if f >= self.min_freq]
            if len(keep) > self.max_categories - 1:
                keep = keep[: self.max_categories - 1]
            self.kept_per_col_.append(set(keep))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).copy()
        for j, keep in enumerate(self.kept_per_col_):
            col = X[:, j]
            mask = ~np.isin(col, list(keep))
            if mask.any():
                col = col.copy()
                col[mask] = self.other_token
                X[:, j] = col
        return X


EXTREME_COLS = ["ba_spread_pct", "mid_over_prev_close"]

def compute_cutoffs_on_train(tr: pd.DataFrame, cols, q=0.99):
    #on train set compute (0,q) threshold
    cutoffs = {}
    for c in cols:
        v = tr[c].values
        v = v[np.isfinite(v)]
        if v.size == 0:
            cutoffs[c] = (-np.inf, np.inf)
        else:
            lo, hi = np.quantile(v, [0, q])
            cutoffs[c] = (lo, hi)
    return cutoffs

def apply_cutoffs_clip(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    #cliping data
    out = df.copy()
    for c, (lo, hi) in cutoffs.items():
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def make_pipe(feature_cols: List[str]) -> Pipeline:
    cat = [c for c in feature_cols if c in CAT_COLS]
    num = [c for c in feature_cols if c not in cat]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(max_categories=255, min_freq=20, other_token="__OTHER__")),
        ("ord", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64
        )),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num),
            ("cat", cat_pipe, cat),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    to_float32 = FunctionTransformer(cast_float32, accept_sparse=False)

    categorical_mask = [False]*len(num) + [True]*len(cat)

    model = HistGradientBoostingRegressor(
        loss="squared_error", max_depth=6, learning_rate=0.05, max_iter=500, random_state=42,
        categorical_features=categorical_mask
    )

    return Pipeline([
        ("pre", pre),
        ("to32", to_float32),
        ("model", model)
    ])


def walk_forward_eval(df_lite: pd.DataFrame, train_days: int = 55, test_days: int = 3,spreads: List[str] = None) -> Tuple[float, float, float]:
    # Split by date
    if spreads:
        df_lite = df_lite[df_lite["spread_id"].isin(spreads)].copy()
        print(f"Dataset réduit à {df_lite.shape[0]} lignes pour l'entrainement.")
    df_lite = df_lite.sort_values("ts").reset_index(drop=True)
    dates = sorted(df_lite["date"].unique())
    feats = CAT_COLS + NUM_COLS

    maes, rmses, R2s = [], [], []
    for i in range(train_days, len(dates) - test_days):
        tr_dates = dates[i-train_days:i]
        te_dates = dates[i:i+test_days]
        tr = df_lite[df_lite["date"].isin(tr_dates)]
        te = df_lite[df_lite["date"].isin(te_dates)]

        cutoffs = compute_cutoffs_on_train(tr, EXTREME_COLS, q=0.99)
        tr = apply_cutoffs_clip(tr, cutoffs)
        te = apply_cutoffs_clip(te, cutoffs)

        X_tr, y_tr = tr[feats], tr[TARGET]
        X_te, y_te = te[feats], te[TARGET]

        pipe = make_pipe(feats)

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        mae = mean_absolute_error(y_te, pred)
        rmse = mean_squared_error(y_te, pred, squared=False)
        R2 = r2_score(y_te, pred)
        maes.append(mae); rmses.append(rmse);R2s.append(R2)

    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(R2s))

def main():
    raw = load_all_csv(DATA_DIR)
    df = build_lite_frame(raw)

    print("Taille dataset:", df.shape)
    mae, rmse, R2 = walk_forward_eval(df, train_days=55, test_days=3,spreads=["CTHCTK", "CTKCTN", "GCMGCQ", "COXCOF", "KCNKCU", "CLFCLH", "LHZLHG", "KCHKCK", "GCZGCG", "SIZSIH", "CTZCTH", "SBKSBN", "SBHSBK", "KCZKCH", "SINSIU", "CLUCLX", "GCGGCJ", "CTNCTZ", "SHSK", "QSHQSK", "LCVLCZ", "WUWZ", "COKCON"])
    print(f"For train_days = 55 and test_days = 3, Walk-forward (moyenne) — MAE: {mae:.5f} | RMSE: {rmse:.5f} | R2: {R2:.5f}")

    feats = CAT_COLS + NUM_COLS
    pipe = make_pipe(feats)
    pipe.fit(df[feats], df[TARGET])
    joblib.dump(pipe, MODEL_PATH)
    print("Modèle sauvegardé →", MODEL_PATH)

if __name__ == "__main__":
    main()