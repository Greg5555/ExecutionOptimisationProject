from __future__ import division
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import glob
import re
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin, clone

from tqdm.auto import tqdm
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


DATA_DIR = r".\ML_tables_processed"
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
    import scipy.sparse as sp
    if sp.issparse(X):
        return X.astype(np.float32)
    return np.asarray(X, dtype=np.float32)


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

    out = df.copy()
    for c, (lo, hi) in cutoffs.items():
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out



def build_linear_pipe(loss="ridge", alpha=1.0, l1_ratio=0.5):

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(max_categories=255, min_freq=20, other_token="__OTHER__")),
        ("oh", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32
        )),
    ])

    pre = ColumnTransformer(
        [("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)],
        remainder="drop",
        sparse_threshold=0.3
    )

    if loss == "ridge":
        model = Ridge(alpha=alpha)
    elif loss == "lasso":
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor(
            loss="squared_error", penalty="l1",
            alpha=alpha, random_state=42, max_iter=2000, tol=1e-3
        )
    elif loss == "enet":
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor(
            loss="squared_error", penalty="elasticnet",
            alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000, tol=1e-3
        )
    else:
        raise ValueError("loss linéaire inconnu")

    if loss == "ridge":
        return Pipeline([("pre", pre), ("model", model)])
    else:
        to_float32 = FunctionTransformer(cast_float32, accept_sparse=True)
        return Pipeline([("pre", pre), ("to32", to_float32), ("model", model)])



def build_tree_pipe(kind="rf"):

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(max_categories=255, min_freq=20, other_token="__OTHER__")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)),
    ])
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ], remainder="drop", sparse_threshold=1.0)

    to_float32 = FunctionTransformer(cast_float32, accept_sparse=False)

    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            max_features="sqrt",
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError("modèle arbre inconnu")

    return Pipeline([("pre", pre), ("to32", to_float32),("model", model)])


def get_model_zoo():

    return {
        "Ridge_a0.1":     build_linear_pipe("ridge", alpha=0.1),
        "Lasso_a1e-3":    build_linear_pipe("lasso", alpha=1e-3),
        "ENet_a1e-3_r0.1":  build_linear_pipe("enet", alpha=1e-3, l1_ratio=0.1),
        "RF": build_tree_pipe("rf"),
    }



def walk_forward_eval_multi(df_lite: pd.DataFrame,
                            models: dict,
                            train_days: int = 50,
                            test_days: int = 3,
                            spreads: List[str] = None):
    if spreads:
        df_lite = df_lite[df_lite["spread_id"].isin(spreads)].copy()
        print(f"Dataset réduit à {df_lite.shape[0]} lignes pour l'entrainement.")
    df_lite = df_lite.sort_values("ts").reset_index(drop=True)
    dates = sorted(df_lite["date"].unique())
    feats = CAT_COLS + NUM_COLS

    results = {name: {"MAE": [], "RMSE": [], "R2": []} for name in models}


    outer_iters = range(train_days, len(dates) - test_days)
    for i in tqdm(outer_iters, desc="Walk-forward", unit="window"):

        tr_dates = dates[i - train_days:i]
        te_dates = dates[i:i + test_days]
        tr = df_lite[df_lite["date"].isin(tr_dates)]
        te = df_lite[df_lite["date"].isin(te_dates)]

        X_tr, y_tr = tr[feats], tr[TARGET]
        X_te, y_te = te[feats], te[TARGET]

        for name, pipe in tqdm(models.items(), leave=False, desc="Models", unit="model"):
            fitted = clone(pipe)
            fitted.fit(X_tr, y_tr)
            pred = fitted.predict(X_te)
            results[name]["MAE"].append(mean_absolute_error(y_te, pred))
            results[name]["RMSE"].append(np.sqrt(mean_squared_error(y_te, pred)))
            results[name]["R2"].append(r2_score(y_te, pred))
            del fitted

    out_rows = []
    for name, d in results.items():
        out_rows.append({
            "model": name,
            "MAE": float(np.mean(d["MAE"])),
            "RMSE": float(np.mean(d["RMSE"])),
            "R2": float(np.mean(d["R2"])),
        })
    return pd.DataFrame(out_rows).sort_values("R2", ascending=False).reset_index(drop=True)


MODEL_DIR = r".\Pred_Models"

def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)

def save_all_models(models: dict, X: pd.DataFrame, y: pd.Series, model_dir: str):
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for name, pipe in tqdm(models.items(), desc="Fit & Save all models", unit="model"):
        pipe.fit(X, y)
        clean = sanitize_name(name)
        out_path = Path(model_dir) / f"{clean}.joblib"
        joblib.dump(pipe, out_path)
        saved_paths[name] = str(out_path)
        del pipe
    return saved_paths



def main():
    raw = load_all_csv(DATA_DIR)
    df = build_lite_frame(raw)
    del raw
    logging.info(f"Taille dataset: {df.shape}")

    feats = CAT_COLS + NUM_COLS
    X_all, y_all = df[feats], df[TARGET]

    models = get_model_zoo()

    logging.info("Début évaluation walk-forward…")
    res = walk_forward_eval_multi(df, models, train_days=55, test_days=3,spreads=["CTHCTK", "CTKCTN", "GCMGCQ", "COXCOF","WZWH","GCGGCM","CONCOQ","KWZKWK","QSFQSH","KCUKCZ", "KCNKCU", "CLFCLH", "LHZLHG", "KCHKCK", "GCZGCG", "SIZSIH", "CTZCTH", "SBKSBN", "SBHSBK", "KCZKCH", "SINSIU", "CLUCLX", "GCGGCJ", "CTNCTZ", "SHSK", "QSHQSK", "LCVLCZ", "WUWZ", "COKCON"])
    print("\n=== Résultats walk-forward (moyennes) ===\n", res)
    del df

    logging.info("Fit & save de tous les modèles sur l'historique complet…")
    saved = save_all_models(models, X_all, y_all, MODEL_DIR)
    del X_all, y_all
    for name, pth in saved.items():
        logging.info(f"Modèle '{name}' sauvegardé → {pth}")


if __name__ == "__main__":
    main()
