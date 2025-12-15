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
- [6. GBM model](#6-gbm-model)
- [7. Building simulated paths](#7-building-simulated-paths)
- [8. Quantiles and prediction band](#8-quantiles-and-prediction-band)
- [9. Risk metrics](#9-risk-metrics)
- [10. Limitations](#10-limitations)
- [Project structure](#project-structure)
- [Minimal code](#minimal-code)

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

In code:

```python
Z = np.random.normal(0, 1, size=(N, T))
