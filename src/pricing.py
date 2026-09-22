"""
pricing.py

Forward and option pricing for FX, all quoted as USD per 1 unit of foreign
currency (S). r_d is the domestic (USD) rate, r_f the foreign rate, both
continuously compounded, T in years.

Forward -- covered interest rate parity:
    F = S * exp((r_d - r_f) * T)
A forward is NOT a forecast. It's the only price at which "convert now and
invest abroad" and "invest at home and convert later via the forward" pay
the same; any other price is a riskless arbitrage. So a high-rate currency
(MXN) trades at a forward DISCOUNT, a low-rate one (JPY) at a premium.

Options -- Garman-Kohlhagen (1983): Black-Scholes where the foreign currency
is an asset paying a continuous "dividend" equal to r_f.
"""

import numpy as np
from scipy.stats import norm


def forward_rate(spot, r_d, r_f, T):
    return spot * np.exp((r_d - r_f) * T)


def garman_kohlhagen(spot, strike, T, r_d, r_f, sigma, kind="put"):
    """Premium in USD per 1 unit of foreign currency, paid today."""
    spot, strike, sigma = np.asarray(spot, float), np.asarray(strike, float), np.asarray(sigma, float)
    d1 = (np.log(spot / strike) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return spot * np.exp(-r_f * T) * norm.cdf(d1) - strike * np.exp(-r_d * T) * norm.cdf(d2)
    if kind == "put":
        return strike * np.exp(-r_d * T) * norm.cdf(-d2) - spot * np.exp(-r_f * T) * norm.cdf(-d1)
    raise ValueError(f"kind must be 'put' or 'call', got {kind!r}")


if __name__ == "__main__":
    # EUR example: S=1.15, USD 4%, EUR 2%, 6 months, 8% vol, ATM-forward strike
    S, rd, rf, T, vol = 1.15, 0.04, 0.02, 0.5, 0.08
    F = forward_rate(S, rd, rf, T)
    put = garman_kohlhagen(S, F, T, rd, rf, vol, "put")
    call = garman_kohlhagen(S, F, T, rd, rf, vol, "call")
    print(f"Forward: {F:.5f}  (premium to spot, since USD rate > EUR rate)")
    print(f"ATM-forward put: {put:.5f} USD/EUR = {put / S:.2%} of spot")
    # At an ATM-forward strike, put-call parity says call == put.
    print(f"ATM-forward call: {call:.5f}  (should equal the put)")
