# MASI Monte Carlo Simulation (2026)

Simulation **Monte Carlo** du **MASI (Moroccan All Shares Index)** afin de générer des **scénarios probabilistes** (trajectoires possibles) sur l’année 2026, à partir d’un modèle stochastique standard en finance : **Geometric Brownian Motion (GBM)**.

> 🎯 Objectif : quantifier l’incertitude (risque) et produire une **distribution de futurs possibles**, plutôt que prédire une valeur unique.

---

## Table of contents
- [1. Data](#1-data)
- [2. Log returns](#2-log-returns)
- [3. Probability assumption](#3-probability-assumption)
- [4. Parameter estimation](#4-parameter-estimation)
- [5. Random shocks](#5-random-shocks)
- [6-Minimal code](#minimal-code)

---

## 1) Data

We use historical closing values of the MASI index.

**Data requirements**
- A `Date` column (datetime)
- A `Value` / `Close` column (numeric)
- Dates sorted in chronological order
- No duplicated dates
- Missing values handled

---

## 2) Log returns

From index values \( S_t \), we compute **log returns**:

\[
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
\]

Why log returns?
- Additive over time: \(\ln(S_T/S_0)=\sum_{t=1}^T r_t\)
- Consistent with log-normal price dynamics under GBM

---

## 3) Probability assumption

We assume log returns are i.i.d. (independent, identically distributed) and Gaussian:

\[
r_t \sim \mathcal{N}(\mu\,\Delta t,\;\sigma^2\,\Delta t)
\]

- \(\mu\): drift (average return)
- \(\sigma\): volatility (risk)
- \(\Delta t\): time step (e.g. \(1/252\) for daily)

> ⚠️ \(\mu\) and \(\sigma\) are **estimated** from historical data (fixed inputs). Randomness comes from simulated shocks.

---

## 4) Parameter estimation

For daily returns:

\[
\hat\mu_{daily}=\text{mean}(r_t),\qquad \hat\sigma_{daily}=\text{std}(r_t)
\]

Annualization (standard approximation):

\[
\hat\mu_{ann}=252\cdot \hat\mu_{daily}
\]
\[
\hat\sigma_{ann}=\sqrt{252}\cdot \hat\sigma_{daily}
\]

---

## 5) Random shocks

Monte Carlo uses standard normal shocks:

\[
Z_t^{(s)} \sim \mathcal{N}(0,1)
\]

where:
- \(t\) is the day index (1..T)
- \(s\) is the scenario index (1..N)



## 6) Minimal code
In code:

```python
Z = np.random.normal(0, 1, size=(N, T))
import numpy as np
import pandas as pd

# --- Load data ---
df = pd.read_csv("data/masi.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").drop_duplicates("Date")
df = df.set_index("Date")

S = df["Value"]  # adapt column name

# --- Log returns ---
r = np.log(S / S.shift(1)).dropna()

# --- Estimate parameters (daily) ---
mu_daily = r.mean()
sigma_daily = r.std()

# --- Annualize ---
mu = 252 * mu_daily
sigma = np.sqrt(252) * sigma_daily

# --- Simulate paths ---
S0 = S.iloc[-1]
N = 10000
T = 252
dt = 1/252

Z = np.random.normal(0, 1, size=(N, T))
increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z

log_paths = np.cumsum(increments, axis=1)
S_paths = S0 * np.exp(log_paths)

# --- Quantiles (per day) ---
q05 = np.percentile(S_paths, 5, axis=0)
q50 = np.percentile(S_paths, 50, axis=0)
q95 = np.percentile(S_paths, 95, axis=0)
