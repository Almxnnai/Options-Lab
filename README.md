# Options Lab

A live options analytics dashboard built with Python and Streamlit.

This project implements:

- Black–Scholes analytical pricing
- Monte Carlo simulation pricing
- Convergence diagnostics
- Full option Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility solver (numerical root-finding)
- Volatility smile using real market option chains
- Payoff & PnL visualization
- Live market data integration via Yahoo Finance

---

## Features

### 1. Pricing Engines
- Analytical Black–Scholes model
- Monte Carlo simulation
- Error convergence analysis

### 2. Risk Sensitivities
- Delta
- Gamma
- Vega
- Theta
- Rho

### 3. Implied Volatility Calibration
Numerically solves for volatility that matches a market option price.

### 4. Volatility Smile
Builds IV vs Strike curve using real option chain data.

### 5. Payoff & PnL Analysis
Visualizes option payoff and profit/loss at expiry.

---

## Mathematical Background

The Black–Scholes formula assumes:

- Lognormal asset price dynamics
- Constant volatility
- No arbitrage
- Continuous trading

Monte Carlo simulation models:

\[
S_T = S_0 \exp\left((r - \frac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\right)
\]

Implied volatility is solved using a bisection root-finding algorithm.

---

## Tech Stack

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib
- Streamlit
- yFinance API

---

## Deployment

This project is deployed using Streamlit Community Cloud.

---

## Purpose

Built as a quantitative finance portfolio project focused on derivatives analytics and volatility modeling. Feel free to use and experiment
