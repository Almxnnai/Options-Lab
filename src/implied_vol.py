# ====================================================
# implied_vol.py
# Solve implied volatility using Black–Scholes
# ====================================================

from src.black_scholes import black_scholes_price


def implied_volatility(market_price, S, K, r, T, option_type="call", sigma_low=1e-6, sigma_high=5.0, tol=1e-6, max_iter=200):
    """
    Solves for implied volatility using bisection method.
    """

    if market_price <= 0:
        raise ValueError("market_price must be > 0")

    low = sigma_low
    high = sigma_high

    price_low = black_scholes_price(S, K, r, low, T, option_type)
    price_high = black_scholes_price(S, K, r, high, T, option_type)

    if not (price_low <= market_price <= price_high):
        raise ValueError(
            "Market price not bracketed. "
            "Try increasing sigma_high or check inputs."
        )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price_mid = black_scholes_price(S, K, r, mid, T, option_type)

        if abs(price_mid - market_price) < tol:
            return float(mid)

        if price_mid < market_price:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))
