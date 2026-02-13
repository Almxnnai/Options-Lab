# black_scholes.py
# Pricing + Greeks for European options

import numpy as np
from scipy.stats import norm


# Black–Scholes Pricing

def black_scholes_price(S, K, r, sigma, T, option_type="call"):
    """
    Black–Scholes price for a European option.
    """

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    if option_type.lower() == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    raise ValueError("option_type must be 'call' or 'put'")


# Black–Scholes Greeks

def black_scholes_greeks(S, K, r, sigma, T, option_type="call"):
    """
    Returns Greeks: Delta, Gamma, Vega, Theta, Rho
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf = norm.pdf(d1)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (
            - (S * pdf * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            - (S * pdf * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)

    return {
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Vega": float(vega),
        "Theta": float(theta),
        "Rho": float(rho)
    }
