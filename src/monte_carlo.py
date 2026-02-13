# Monte_carlo.py
# Monte Carlo pricing for European options

import numpy as np


def monte_carlo_price(S, K, r, sigma, T, option_type="call", simulations=100000):
    """
    Monte Carlo pricing under Geometric Brownian Motion.
    """

    # Generate standard normal random numbers
    Z = np.random.standard_normal(simulations)

    # Simulate stock price at maturity
    ST = S * np.exp(
        (r - 0.5 * sigma**2) * T +
        sigma * np.sqrt(T) * Z
    )

    # Compute payoff
    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0)

    elif option_type.lower() == "put":
        payoff = np.maximum(K - ST, 0)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount back to today
    price = np.exp(-r * T) * np.mean(payoff)

    return float(price)
