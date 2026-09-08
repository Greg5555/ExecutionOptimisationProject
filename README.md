# Commodity Futures Spread Execution Optimization

## Overview

This repository contains the research code developed for my ENSAE Paris professional thesis on **data-driven execution of commodity futures calendar spreads**.

The objective is not to predict the outright direction of a commodity. Instead, the project addresses a practical execution problem:

> **Given an order to buy or sell a commodity calendar spread before the end-of-day settlement, can the execution schedule be improved relative to a simple time-based benchmark such as TWAP?**


The framework combines:

1. **Intraday market-data processing and feature engineering**
2. **End-of-day settlement price forecasting**
3. **Historical intraday z-score calibration**
4. **Signal-based execution rules**
5. **Backtesting against a late-session TWAP benchmark**
6. **A live monitoring prototype for real-time execution signals**

The code was developed in the context of a V.I.E. at **Société Générale, New York**, and accompanies the ENSAE Paris professional thesis.

---

## Research Idea

For a calendar spread between two futures contracts, let


![Calendar spread definition](assets/spread_definition.png)


denote the observed spread at time `t`.

The model estimates the end-of-day settlement spread:


![Settlement forecast](assets/settlement_forecast.png)


where `X_t` contains market, liquidity and time-of-day information.

The difference between the current market level and the predicted settlement is then normalized by an intraday volatility estimate:


![Standardized execution signal](assets/zscore_signal.png)


This standardized signal is compared with the **historical distribution of z-scores observed at the same minute of the trading day**.

The execution engine therefore does not rely on a single fixed threshold. Instead, it uses **time-dependent empirical quantiles**.

| Signal strength | Buy rule | Sell rule | Indicative execution fraction |
|---|---:|---:|---:|
| Strong | `z < q15(t)` | `z > q85(t)` | 30% |
| Medium | `z < q25(t)` | `z > q75(t)` | 20% |
| Moderate | `z < q40(t)` | `z > q60(t)` | 10% |
| No signal | otherwise | otherwise | 0% |

Any remaining quantity is forced near the end of the execution window.

In plain terms:

> **Execute more aggressively when the current spread looks statistically attractive relative to the model-implied settlement and to its historical intraday distribution.**

---

# Repository Structure

```text
.
├── build_features.py
├── build_models.py
├── HistoGradientBoost.py
├── build_z_histo.py
├── backtest.py
├── live_pred.py
├── CUSTOM_TRANSFO.py
│
├── data/
│   └── ...
├── models/
│   └── *.joblib
├── results/
│   └── ...
└── README.md
```

The folder names above are a recommended public GitHub structure. The original research scripts contain local/internal paths and need to be adapted before running outside the original environment.

---

# End-to-End Pipeline

```text
Raw intraday data
        │
        ▼
┌──────────────────────┐
│  Feature engineering │
│   build_features.py  │
└──────────────────────┘
        │
        ▼
Processed ML tables
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
┌──────────────────────┐     ┌──────────────────────┐
│ Settlement forecasting│     │ Historical z-scores  │
│    build_models.py    │     │   build_z_histo.py   │
│ HistoGradientBoost.py │     └──────────────────────┘
└──────────────────────┘               │
        │                               │
        └──────────────┬────────────────┘
                       ▼
              ┌───────────────────┐
              │ Execution signals │
              │    backtest.py    │
              └───────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Strategy P&L                TWAP benchmark
          │                         │
          └────────────┬────────────┘
                       ▼
              Performance analysis

Optional production prototype:
                       │
                       ▼
                 live_pred.py
```

---

# 1. Feature Engineering

## `build_features.py`

This script transforms the raw intraday spread data into the machine-learning dataset.

Main engineered variables:

| Feature | Description |
|---|---|
| `spread_id` | Identifier of the futures calendar spread |
| `commodity` | Commodity family |
| `t2c_min` | Minutes remaining until the commodity closing/settlement time |
| `Mid` / `mid` | Current spread mid-price |
| `lastBD_settlementPX_spread` / `prev_close` | Previous business-day settlement spread |
| `BA_spread_%` | Relative bid-ask spread |
| `liquidity_avg` | Average displayed bid/ask size |
| `liquidity_log` | Log-transformed liquidity measure |
| `mid_minus_prev_close` | Difference between the current mid and previous settlement |
| `mid_over_prev_close` | Relative deviation from previous settlement |
| `dow` | Day of week |
| `minute_of_day` | Intraday time index |
| `minute_sin`, `minute_cos` | Cyclical encoding of time of day |
| `Today_settlementPX_spread` | Prediction target |

The cyclical time variables are defined as:

![Cyclical sine time feature](assets/minute_sin.png)

![Cyclical cosine time feature](assets/minute_cos.png)

where `m_t` is the minute of the day.

The script also handles invalid values, derives spread identifiers, computes liquidity features and relative-price features, and writes processed CSV files used by the modelling pipeline.

---

# 2. Settlement Forecasting

## `build_models.py`

This script builds and compares several regression models for the end-of-day settlement spread.

### Target

```python
TARGET = "Today_settlementPX_spread"
```

The statistical problem is


![Forecasting target](assets/forecast_target.png)


where `S_T` is the final settlement spread of the trading day.

### Models

The model zoo includes:

- **Ridge regression**
- **Lasso regression**
- **Elastic Net**
- **Random Forest**

The linear models use median imputation, feature standardization and one-hot encoding of categorical variables. Tree-based models use numerical imputation and ordinal encoding of categorical variables. Rare categories are grouped before encoding.

### Walk-forward validation

Model evaluation is performed chronologically rather than using a random train/test split.

A typical configuration is:

```text
Training window : 55 trading days
Test window     : 3 trading days
```

For each rolling window:

1. train only on past observations;
2. predict the next test period;
3. compute out-of-sample metrics;
4. roll the window forward.

The main metrics are:


![Mean absolute error](assets/mae.png)


![Root mean squared error](assets/rmse.png)


and


![Coefficient of determination](assets/r2.png)


This time-series validation scheme is important because random cross-validation would allow future observations to contaminate the training sample.

---

# 3. Histogram Gradient Boosting

## `HistoGradientBoost.py`

This script implements a separate **Histogram Gradient Boosting Regressor**.

The model is configured with squared-error loss, maximum tree depth, learning rate and number of boosting iterations, then evaluated with the same type of walk-forward validation.

The trained pipeline can be serialized with `joblib` and later consumed by the backtest or live prototype.

---

# 4. Custom Transformers

## `CUSTOM_TRANSFO.py`

This module contains reusable preprocessing components.

### `RareCategoryGrouper`

High-cardinality categorical variables such as spread identifiers can create unstable or excessively large encodings.

`RareCategoryGrouper`:

1. counts category frequencies;
2. retains categories above a minimum frequency;
3. limits the number of retained categories;
4. maps remaining observations to `__OTHER__`.

### `cast_float32`

Casts numerical matrices to `float32` to reduce memory usage and ensure compatibility with some modelling pipelines.

---

# 5. Historical Intraday Z-Score Calibration

## `build_z_histo.py`

A central part of the execution strategy is the estimation of the historical distribution of the normalized distance between the current spread and its final settlement.

For each pair

```text
(spread_id, minute_of_day)
```

the script computes empirical quantiles of the historical z-score distribution:

```python
q = [0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85]
```

The resulting table contains:

```text
z_MidvsSet_histo_q15
z_MidvsSet_histo_q25
z_MidvsSet_histo_q40
z_MidvsSet_histo_q50
z_MidvsSet_histo_q60
z_MidvsSet_histo_q75
z_MidvsSet_histo_q85
```

Conceptually, for spread `s`, minute `m`, and quantile level `alpha`,


![Empirical intraday quantile](assets/empirical_quantile.png)


The motivation is that the distribution of the spread-to-settlement signal is strongly time dependent. The execution thresholds are therefore conditioned on the minute of day rather than assumed constant throughout the session.

---

# 6. Execution Backtest

## `backtest.py`

This script converts settlement forecasts into an execution policy.

For each observation, the model predicts


![Settlement forecast](assets/settlement_forecast.png)


The current spread is compared with that predicted terminal value through


![Standardized execution signal](assets/zscore_signal.png)


where `sigma_hat_t` is an intraday rolling volatility estimate.

The predicted z-score is then compared with the historical quantiles generated by `build_z_histo.py`.

## Buy Logic

For a spread that must be **bought**, the strategy looks for unusually low spread levels:

```text
z < q15   → execute 30%
z < q25   → execute 20%
z < q40   → execute 10%
otherwise → wait
```

## Sell Logic

For a spread that must be **sold**, the strategy looks for unusually high spread levels:

```text
z > q85   → execute 30%
z > q75   → execute 20%
z > q60   → execute 10%
otherwise → wait
```

## Execution Constraints

The backtest specifies:

- total quantity to execute;
- maximum number of signal-driven executions;
- number of intraday observations;
- buy or sell direction.

A representative configuration used in the research is:

```python
nb_contract = 50
nb_exec     = 5
nb_obs      = 10
```

Any remaining quantity is completed during the final minutes of the execution window. This guarantees full execution while allowing the signal to opportunistically accelerate the order earlier in the session.

---

# 7. Benchmark and P&L

The strategy is compared with a late-session TWAP-style benchmark.

For each trading day:

- `avg_exec` = average execution price generated by the strategy;
- `avg_bench` = settlement-price benchmark;
- `avg_twap` = average executable bid/ask price during the final minutes.

The code computes:

```python
pnl      = avg_exec - avg_bench
pnl_twap = avg_twap - avg_bench
```

with the economic sign interpreted according to the buy/sell direction.

Typical performance analysis includes:

- cumulative P&L;
- average daily P&L;
- median daily P&L;
- volatility of daily P&L;
- win rate;
- Sharpe ratio;
- maximum drawdown;
- best day;
- worst day.

---

# 8. Live Signal Prototype

## `live_pred.py`

`live_pred.py` is a graphical prototype showing how the trained model and historical thresholds can be used in real time.

The application:

1. loads a serialized `.joblib` model;
2. accepts a spread, commodity and BUY/SELL side;
3. retrieves current market information;
4. rebuilds the model features;
5. predicts the settlement spread;
6. estimates recent intraday volatility;
7. computes the live z-score;
8. retrieves the corresponding historical intraday quantiles;
9. displays a qualitative signal.

Possible outputs include:

```text
STRONG BUY
SOLID BUY
BUY
NO SIGNAL
SELL
SOLID SELL
STRONG SELL
```

The monitor refreshes automatically at regular intervals.

### Important

The live script relies on internal market-data infrastructure and Bloomberg-related modules that are not part of this public repository. It is therefore included primarily to document the architecture of the live prototype.

---

# 9. Data

The original research dataset consists of minute-level observations of commodity futures calendar spreads.

A processed observation contains information similar to:

```text
timestamp
date
spread identifier
commodity
mid price
previous settlement
current-day settlement
bid-ask spread
bid/ask liquidity
minutes to close
time-of-day features
relative-price features
```

Because the underlying market dataset is proprietary, **raw data are not distributed in this repository**.

A small anonymized or synthetic sample can be added to illustrate the expected schema.

---

# 10. Dependencies

Core open-source dependencies include:

```text
Python 3.x
numpy
pandas
scikit-learn
scipy
joblib
matplotlib
tqdm
pytz
```

The live application also uses:

```text
tkinter
zoneinfo
```

plus internal/proprietary market-data packages.

A minimal installation for the research and backtest components is:

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib tqdm pytz
```

---

# 11. Running the Research Pipeline

The scripts currently use project-specific directory constants. Before running them on another machine, replace paths such as:

```python
DATA_DIR = "..."
MODEL_DIR = "..."
HISTO_Z_SPREAD = "..."
```

with local paths.

A typical workflow is:

### Step 1 — Prepare features

```bash
python build_features.py
```

### Step 2 — Estimate historical intraday thresholds

```bash
python build_z_histo.py
```

### Step 3 — Train and compare models

```bash
python build_models.py
```

### Step 4 — Train Histogram Gradient Boosting

```bash
python HistoGradientBoost.py
```

### Step 5 — Run the execution backtest

```bash
python backtest.py
```

### Step 6 — Optional live prototype

```bash
python live_pred.py
```

The live step requires access to the original internal data infrastructure.

---

# 12. Reproducibility Notes

Several precautions are important when reproducing or extending the results.

### Time-series validation

Training and test sets should always be separated chronologically.

### Avoiding look-ahead bias

Any variable used at time `t` must be observable at time `t`.

In particular:

- preprocessing parameters should be estimated on the training period;
- historical quantile thresholds should only use information available before the test period;
- production backtests should ideally use predictions generated from rolling out-of-sample models rather than models refitted on the complete historical sample.

### Execution assumptions

The current backtest approximates executable prices using reconstructed bid and ask levels.

It does not fully model:

- queue position;
- partial fills;
- exchange fees;
- nonlinear market impact;
- order-book depth beyond displayed liquidity.

The results should therefore be interpreted as a **research execution-price backtest**, not as a complete exchange simulator.

---

# 13. Suggested Public GitHub Cleanup

The original research code was developed inside a banking environment.

Before making the repository public, remove or replace:

- internal network paths;
- usernames or workstation paths;
- proprietary package names when disclosure is restricted;
- Bloomberg credentials or configuration;
- internal server names;
- proprietary datasets;
- internal ticker mappings if confidential;
- saved production models trained on proprietary data.

A cleaner public structure could use a central configuration file:

```python
# config.py

DATA_DIR = "./data/processed"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
HISTO_Z_PATH = "./data/z_score_quantiles.csv"
```

This makes the repository portable and avoids exposing internal infrastructure.

---

# 14. Limitations and Extensions

Possible extensions include:

- strict rolling out-of-sample execution backtests;
- transaction-cost and market-impact modelling;
- bootstrap confidence intervals for execution improvement;
- Diebold-Mariano tests for forecast comparison;
- block-bootstrap inference for daily P&L;
- regime-dependent quantiles;
- volatility-scaled execution sizes;
- direct quantile regression;
- probabilistic settlement forecasts;
- online model updating;
- reinforcement-learning or optimal-control formulations;
- joint optimization of execution timing and quantity.

---

# 15. Academic Context

This project was developed as part of the **ENSAE Paris Specialized Master's professional thesis**.

**Author:** Grégoire Chauchot  
**Company:** Société Générale  
**Location:** New York  
**Professional supervisor:** Daniel Zelenski  
**Academic year:** 2024–2025  
**V.I.E. period:** August 2025 – January 2027

The academic report develops the statistical methodology, model validation, empirical results, limitations and relationship with the optimal-execution literature in substantially greater detail.

---

# Disclaimer

This repository is provided for **academic and research purposes only**.

It does not constitute investment advice, a trading recommendation, or a production trading system.

Any public version of this repository should contain only code, data and documentation that are authorized for external disclosure.

---

## Author

**Grégoire Chauchot**  
ENSAE Paris  
Professional thesis — Commodity Futures Spread Execution Optimization
