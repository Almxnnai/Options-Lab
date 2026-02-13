# OPTIONS LAB
# Black–Scholes + Monte Carlo + Greeks + IV + Convergence

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

from src.black_scholes import black_scholes_price, black_scholes_greeks
from src.monte_carlo import monte_carlo_price
from src.implied_vol import implied_volatility


# Helper: estimate annualized volatility from returns

def estimate_annualized_vol(prices: pd.Series) -> float:
    """
    Estimates annualized volatility using daily log returns.
    Assumes ~252 trading days per year.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_vol = log_returns.std()
    return float(daily_vol * np.sqrt(252))


# Options chain helpers (cached for speed)

@st.cache_data(ttl=300)
def get_expirations(ticker: str):
    """Fetch available option expirations for a ticker."""
    tk = yf.Ticker(ticker)
    return list(tk.options)

@st.cache_data(ttl=300)
def get_option_chain(ticker: str, expiry: str):
    """Fetch calls and puts DataFrames for a given expiry."""
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()

def pick_market_price(row: pd.Series, source: str = "mid"):
    """
    Choose a market price from an option chain row.
    - 'mid': (bid+ask)/2 if both available, otherwise lastPrice
    - 'last': lastPrice
    """
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last = float(row.get("lastPrice", np.nan))

    if source == "last":
        return last if np.isfinite(last) and last > 0 else np.nan

    # source == "mid"
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)

    return last if np.isfinite(last) and last > 0 else np.nan

# Page Configuration
st.set_page_config(page_title="Options Lab", layout="wide", page_icon="📈")

# Session State Initialization
# Inputs
st.session_state.setdefault("S", 100.0)
st.session_state.setdefault("sigma", 0.20)

# IV controls/results
st.session_state.setdefault("iv_market_price", 2.50)
st.session_state.setdefault("iv_type", "call")
st.session_state.setdefault("iv_result", None)
st.session_state.setdefault("iv_error", None)

# Store last computed model prices (for autofill)
st.session_state.setdefault("last_call_bs", None)
st.session_state.setdefault("last_put_bs", None)
st.session_state.setdefault("last_call_mc", None)
st.session_state.setdefault("last_put_mc", None)

# App Header
st.title("Options Lab 📊")
st.markdown("### Quantitative Options Analytics Dashboard")
st.write("Black–Scholes • Monte Carlo • Greeks • Implied Volatility • Volatility Smile")

# Layout
left, right = st.columns([1, 1])


# LEFT SIDE: Market Data

with left:
    st.header("Market Data")

    ticker = st.text_input(
        "Ticker",
        value="AAPL",
        help="Yahoo Finance ticker symbol (e.g., AAPL, TSLA, MSFT)."
    )

    if st.button("Fetch Last 5 Days"):
        data = yf.download(ticker, period="5d")
        st.write(data.tail())


# RIGHT SIDE: Option Pricing

with right:
    st.header("Option Pricing (European)")

    # Auto-fill from ticker
    
    st.subheader("Auto-fill from ticker (optional)")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Use latest price as S", help="Sets S to the latest Close price for the ticker."):
            hist = yf.download(ticker, period="5d")
            if len(hist) > 0 and "Close" in hist:
                last_close = float(hist["Close"].dropna().iloc[-1])
                st.session_state["S"] = last_close
            else:
                st.warning("Could not fetch price for this ticker.")

    with col_b:
        if st.button("Estimate σ from 1Y history", help="Estimates volatility using 1-year daily log returns (annualized)."):
            hist = yf.download(ticker, period="1y")
            if len(hist) > 0 and "Close" in hist:
                vol = estimate_annualized_vol(hist["Close"].dropna())
                st.session_state["sigma"] = vol
            else:
                st.warning("Could not fetch enough data to estimate volatility.")

    st.divider()


    # Inputs
    
    S = st.number_input(
        "Stock Price (S)",
        min_value=0.01,
        value=float(st.session_state["S"]),
        step=1.0,
        help="Current price of the underlying asset."
    )

    K = st.number_input(
        "Strike Price (K)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        help="Strike price of the option contract."
    )

    r = st.number_input(
        "Risk-free rate (r)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        help="Annual risk-free rate in decimals (0.05 = 5%)."
    )

    sigma = st.number_input(
        "Volatility (σ)",
        min_value=0.0001,
        value=float(st.session_state["sigma"]),
        step=0.01,
        help="Annualized volatility in decimals (0.20 = 20%)."
    )

    days = st.number_input(
        "Days to maturity",
        min_value=1,
        value=30,
        step=1,
        help="Time to expiry in days (converted to years internally)."
    )
    T = days / 365.0

    simulations = st.number_input(
        "Monte Carlo simulations",
        min_value=1000,
        value=100000,
        step=1000,
        help="Number of Monte Carlo paths. More paths → usually lower error but slower."
    )

    show_convergence = st.checkbox(
        "Show Monte Carlo convergence (Call)",
        value=True,
        help="Plots absolute pricing error |BS - MC| across different simulation sizes."
    )
    sim_grid = [1000, 3000, 5000, 10000, 20000, 50000, 100000, 200000]

    
    # Pricing
    
    if st.button("Price Option (BS + Monte Carlo)", help="Computes Call/Put prices using Black–Scholes and Monte Carlo."):
        # Black–Scholes
        call_bs = black_scholes_price(S, K, r, sigma, T, "call")
        put_bs = black_scholes_price(S, K, r, sigma, T, "put")

        # Monte Carlo
        call_mc = monte_carlo_price(S, K, r, sigma, T, "call", simulations)
        put_mc = monte_carlo_price(S, K, r, sigma, T, "put", simulations)

        # Save results for IV autofill
        st.session_state["last_call_bs"] = call_bs
        st.session_state["last_put_bs"] = put_bs
        st.session_state["last_call_mc"] = call_mc
        st.session_state["last_put_mc"] = put_mc

        st.success("Pricing complete ✅")

        col_call, col_put = st.columns(2)

        with col_call:
            st.subheader("📈 Call Option")
            st.metric("Black–Scholes (Call)", f"{call_bs:.4f}")
            st.metric("Monte Carlo (Call)", f"{call_mc:.4f}")
            st.metric("Absolute Error (Call)", f"{abs(call_bs - call_mc):.6f}")

        with col_put:
            st.subheader("📉 Put Option")
            st.metric("Black–Scholes (Put)", f"{put_bs:.4f}")
            st.metric("Monte Carlo (Put)", f"{put_mc:.4f}")
            st.metric("Absolute Error (Put)", f"{abs(put_bs - put_mc):.6f}")

        # Convergence plot + compact table (Call)
        if show_convergence:
            mc_prices = []
            abs_errors = []

            for n in sim_grid:
                mc = monte_carlo_price(S, K, r, sigma, T, "call", simulations=n)
                mc_prices.append(mc)
                abs_errors.append(abs(call_bs - mc))

            fig = plt.figure()
            plt.plot(sim_grid, abs_errors, marker="o")
            plt.xlabel("Number of simulations")
            plt.ylabel("Absolute Error |BS - MC| (Call)")
            plt.title("Monte Carlo Convergence (Call Option)")
            plt.xscale("log")
            st.pyplot(fig)

            conv_df = pd.DataFrame({
                "simulations": sim_grid,
                "mc_call_price": mc_prices,
                "abs_error": abs_errors
            })
            conv_df["mc_call_price"] = conv_df["mc_call_price"].round(4)
            conv_df["abs_error"] = conv_df["abs_error"].round(6)

            st.dataframe(
                conv_df,
                use_container_width=True,
                height=200,
                hide_index=True
            )

        # Greeks
        st.subheader("Greeks (Black–Scholes)")
        st.caption("Greeks measure how the option price changes when inputs change.")

        with st.expander("ℹ️ What are Greeks?"):
            st.markdown(
        """
        - **Delta (Δ):** How much the option price changes when **S** moves by 1.
        - **Gamma (Γ):** How much **Delta** changes when **S** moves by 1.
        - **Vega (ν):** How much the option price changes when **volatility σ** changes by 1 (i.e., +100%).  
        *(Rule of thumb: per +1% vol, divide Vega by 100.)*
        - **Theta (Θ):** How much the option price changes when **time passes** (time decay).  
        *(Θ is per year; per day ≈ Θ / 365.)*
        - **Rho (ρ):** How much the option price changes when **interest rate r** changes by 1 (i.e., +100%).  
        *(Per +1% rate, divide Rho by 100.)*
        """
        )
        
        call_greeks = black_scholes_greeks(S, K, r, sigma, T, "call")
        put_greeks = black_scholes_greeks(S, K, r, sigma, T, "put")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("### 📈 Call Greeks")
            for k, v in call_greeks.items():
                st.metric(k, f"{v:.6f}")

        with col_g2:
            st.markdown("### 📉 Put Greeks")
            for k, v in put_greeks.items():
                st.metric(k, f"{v:.6f}")

    # Implied Volatility + Autofill (always visible)

    st.divider()
    st.subheader("Implied Volatility (IV)")

    st.caption("IV is the volatility σ that makes the Black–Scholes price match a market option price.")

    # Autofill buttons (work only after pricing)
    st.markdown("**IV Autofill**")
    has_prices = (
        st.session_state["last_call_bs"] is not None and
        st.session_state["last_put_bs"] is not None and
        st.session_state["last_call_mc"] is not None and
        st.session_state["last_put_mc"] is not None
    )

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button("Use BS Call", disabled=not has_prices, help="Copies the latest Black–Scholes call price into the Market option price box."):
            st.session_state["iv_market_price"] = float(st.session_state["last_call_bs"])
            st.session_state["iv_type"] = "call"

    with b2:
        if st.button("Use BS Put", disabled=not has_prices, help="Copies the latest Black–Scholes put price into the Market option price box."):
            st.session_state["iv_market_price"] = float(st.session_state["last_put_bs"])
            st.session_state["iv_type"] = "put"

    with b3:
        if st.button("Use MC Call", disabled=not has_prices, help="Copies the latest Monte Carlo call price into the Market option price box."):
            st.session_state["iv_market_price"] = float(st.session_state["last_call_mc"])
            st.session_state["iv_type"] = "call"

    with b4:
        if st.button("Use MC Put", disabled=not has_prices, help="Copies the latest Monte Carlo put price into the Market option price box."):
            st.session_state["iv_market_price"] = float(st.session_state["last_put_mc"])
            st.session_state["iv_type"] = "put"

    # IV Form (prevents annoying rerun behavior)
    with st.form("iv_form", clear_on_submit=False):
        market_price = st.number_input(
            "Market option price",
            min_value=0.0001,
            value=float(st.session_state["iv_market_price"]),
            step=0.10,
            help="Observed option price in the market (or a model price if you're testing)."
        )

        iv_type = st.selectbox(
            "Option type",
            ["call", "put"],
            index=0 if st.session_state["iv_type"] == "call" else 1,
            help="Choose whether the market price is for a call or put."
        )

        submitted = st.form_submit_button("Solve IV")

    # Remember selections
    st.session_state["iv_market_price"] = float(market_price)
    st.session_state["iv_type"] = iv_type

    # Solve only when submitted
    if submitted:
        try:
            iv = implied_volatility(market_price, S, K, r, T, iv_type)
            st.session_state["iv_result"] = iv
            st.session_state["iv_error"] = None
        except Exception as e:
            st.session_state["iv_result"] = None
            st.session_state["iv_error"] = str(e)

    # Display persisted result
    if st.session_state["iv_error"]:
        st.error(st.session_state["iv_error"])

    if st.session_state["iv_result"] is not None:
        st.success("IV solved ✅")
        st.metric("Implied Volatility (σ)", f"{st.session_state['iv_result']:.4f}")
    
# Volatility Smile (IV vs Strike)

st.divider()
st.subheader("Volatility Smile (IV vs Strike)")
st.caption("Uses real option chain prices to compute implied volatility across strikes.")

# Fetch expirations safely
try:
    expirations = get_expirations(ticker)
except Exception:
    expirations = []

if not expirations:
    st.warning("No expirations found for this ticker (or Yahoo data unavailable). Try another ticker.")
else:
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        expiry = st.selectbox(
            "Expiry",
            expirations,
            help="Select an expiration date from Yahoo Finance options data."
        )

    with col_s2:
        smile_type = st.selectbox(
            "Option type",
            ["call", "put"],
            help="Build the smile using call or put contracts."
        )

    with col_s3:
        price_source = st.selectbox(
            "Price source",
            ["mid", "last"],
            help="mid uses (bid+ask)/2 when available; last uses last traded price."
        )

    strike_band = st.slider(
        "Strike range around S (percent)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Filters strikes to a band around the underlying price S. Example: 30% keeps strikes in [0.7S, 1.3S]."
    )

    max_points = st.number_input(
        "Max strikes to plot",
        min_value=10,
        max_value=200,
        value=60,
        step=5,
        help="Limits number of strikes for speed/cleaner chart."
    )

    if st.button("Build Volatility Smile", help="Downloads option chain and computes implied vol for each strike."):
        try:
            calls, puts = get_option_chain(ticker, expiry)
            df = calls if smile_type == "call" else puts

            # Keep only the columns we need, if they exist
            needed_cols = [c for c in ["strike", "bid", "ask", "lastPrice", "volume", "openInterest"] if c in df.columns]
            df = df[needed_cols].copy()

            # Filter strikes around S
            lowK = (1 - strike_band / 100) * S
            highK = (1 + strike_band / 100) * S
            df = df[(df["strike"] >= lowK) & (df["strike"] <= highK)].copy()

            # Market price per row
            df["market_price"] = df.apply(lambda row: pick_market_price(row, price_source), axis=1)
            df = df[np.isfinite(df["market_price"]) & (df["market_price"] > 0)].copy()

            if df.empty:
                st.warning("No valid option prices in the selected strike band.")
            else:
                # Sort by closeness to ATM and cap number of points
                df["atm_dist"] = (df["strike"] - S).abs()
                df = df.sort_values("atm_dist").head(int(max_points)).sort_values("strike")

                # Compute IV for each strike (skip rows that fail bracketing)
                ivs = []
                strikes = []
                prices = []

                for _, row in df.iterrows():
                    K_i = float(row["strike"])
                    mkt = float(row["market_price"])

                    try:
                        iv = implied_volatility(mkt, S, K_i, r, T, smile_type)
                        ivs.append(iv)
                        strikes.append(K_i)
                        prices.append(mkt)
                    except Exception:
                        # Not bracketed or bad data → skip point
                        continue

                if len(ivs) < 5:
                    st.warning("Too few valid IV points (market prices may be stale / wide spreads). Try 'last' price or widen strike band.")
                else:
                    # Plot smile
                    fig = plt.figure()
                    plt.plot(strikes, ivs, marker="o")
                    plt.xlabel("Strike (K)")
                    plt.ylabel("Implied Volatility (σ)")
                    plt.title(f"Volatility Smile — {ticker} {expiry} ({smile_type})")
                    st.pyplot(fig)

                    # Compact table
                    smile_df = pd.DataFrame({
                        "strike": strikes,
                        "market_price": prices,
                        "implied_vol": ivs
                    })
                    smile_df["market_price"] = smile_df["market_price"].round(4)
                    smile_df["implied_vol"] = smile_df["implied_vol"].round(4)

                    st.dataframe(
                        smile_df,
                        use_container_width=True,
                        height=220,
                        hide_index=True
                    )

        except Exception as e:
            st.error(f"Smile build failed: {e}")
    
    # Payoff & PnL Diagram

    st.divider()
    st.subheader("Payoff & PnL at Expiry")
    st.caption("Shows payoff and profit/loss at maturity across a range of underlying prices.")

    # Choose which premium to use for PnL
    premium_source = st.selectbox(
        "Premium source for PnL",
        ["Black–Scholes", "Monte Carlo", "Market (IV input)"],
        help="PnL = Payoff - Premium. Choose what premium you paid."
    )

    payoff_type = st.selectbox(
        "Option type for payoff",
        ["call", "put"],
        help="Choose whether to plot a call or put payoff/PnL."
    )

    # Range for underlying prices at expiry (S_T)
    range_pct = st.slider(
        "Price range around S (%)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Example: 50% means plot S_T from 0.5S to 1.5S."
    )

    points = st.number_input(
        "Number of points",
        min_value=50,
        max_value=500,
        value=200,
        step=10,
        help="More points makes the curve smoother."
    )

    # Determine premium based on selected source
    premium = None

    # Use latest computed model prices if available (from session_state)
    last_call_bs = st.session_state.get("last_call_bs")
    last_put_bs = st.session_state.get("last_put_bs")
    last_call_mc = st.session_state.get("last_call_mc")
    last_put_mc = st.session_state.get("last_put_mc")

    if premium_source == "Black–Scholes":
        if payoff_type == "call":
            premium = last_call_bs
        else:
            premium = last_put_bs

    elif premium_source == "Monte Carlo":
        if payoff_type == "call":
            premium = last_call_mc
        else:
            premium = last_put_mc

    # Market (IV input)
    else:
        premium = st.session_state.get("iv_market_price")

    if premium is None:
        st.warning("Price the option first (so premiums exist), or choose Market premium.")
    else:
        premium = float(premium)

        # Build S_T grid
        low = (1 - range_pct / 100) * S
        high = (1 + range_pct / 100) * S
        ST_grid = np.linspace(low, high, int(points))

        # Compute payoff at expiry
        if payoff_type == "call":
            payoff = np.maximum(ST_grid - K, 0.0)
        else:
            payoff = np.maximum(K - ST_grid, 0.0)

        # Profit/Loss at expiry (long option)
        pnl = payoff - premium

        # Plot payoff and pnl
        fig = plt.figure()
        plt.plot(ST_grid, payoff, label="Payoff at expiry")
        plt.plot(ST_grid, pnl, label="PnL at expiry")
        plt.axhline(0, linewidth=1)   # zero line
        plt.axvline(K, linewidth=1)   # strike line
        plt.xlabel("Underlying price at expiry (S_T)")
        plt.ylabel("Value")
        plt.title(f"{payoff_type.upper()} Payoff & PnL (Premium = {premium:.4f})")
        plt.legend()
        st.pyplot(fig)

        # Quick key points
        st.markdown("**Key points**")
        if payoff_type == "call":
            breakeven = K + premium
        else:
            breakeven = K - premium

        st.write(f"- Premium used: **{premium:.4f}**")
        st.write(f"- Break-even (approx): **{breakeven:.4f}**")
        st.write("- Payoff curve shows intrinsic value at expiry.")
        st.write("- PnL curve subtracts the premium (what you paid).")




