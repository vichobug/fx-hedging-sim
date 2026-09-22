# FX Exposure & Hedging Simulator

Monte Carlo simulation of a multinational's foreign-currency exposure across
five currencies, comparing hedging strategies -- unhedged, forwards, options,
and layered mixes -- on 10,000 correlated six-month FX paths calibrated to
16 years of real daily exchange rates.

**Scenario**: a US company expects ~$175M of foreign revenue in six months
(EUR 50M, GBP 25M, JPY 4B, CNY 200M, MXN 500M). The budget converts it at
today's spot rates. How much should it hedge, and with what?

## Project structure

- `scripts/fetch_fx_data.py` -- downloads daily FX and interest rates from FRED (no API key)
- `src/config.py` -- the company's exposures, horizon, simulation settings, strategies
- `src/fx_data.py` -- loads FRED data, normalizes every quote to USD per 1 foreign unit
- `src/pricing.py` -- forwards (covered interest rate parity) and options (Garman-Kohlhagen)
- `src/simulate.py` -- two engines: correlated GBM and block bootstrap of real history
- `src/strategies.py` -- USD received under any forward / option / open split
- `src/risk.py` -- VaR and CVaR (expected shortfall)
- `src/scenario.py` -- shared entry point tying market, simulation and strategies together
- `notebooks/run_simulation.py` -- main strategy comparison + charts
- `notebooks/sensitivity.py` -- hedge-ratio frontier, strike choice, diversification, model risk
- `tests/test_model.py` -- checks against closed-form results (put-call parity, IRP, etc.)

## Run it

```
pip install -r requirements.txt
python scripts/fetch_fx_data.py      # refresh data (a snapshot is already committed)
python notebooks/run_simulation.py
python notebooks/sensitivity.py
pytest
```

## Method

**Two simulation engines**, both returning joint outcomes for all five currencies:

1. **Correlated GBM** -- terminal returns drawn from a multivariate normal
   using the historical covariance matrix. The textbook model.
2. **Block bootstrap** -- each path is stitched from random ~1-month blocks
   of *real consecutive trading days*, all currencies from the same dates.
   Keeps fat tails, volatility clustering and crisis-time correlation
   spikes without assuming a distribution.

**Pricing**: forwards from covered interest rate parity, `F = S·exp((r_usd − r_fx)·T)`;
puts from Garman-Kohlhagen at the same historical vol the simulation uses,
with premiums carried to the horizon so every strategy is compared in
horizon dollars.

**Centering**: simulated spot is centered on today's spot (random walk, no
directional view). `config.CENTER = "forward"` switches to treating the
forward as an unbiased forecast instead.

## Results (data as of 2026-09-18, block bootstrap, 10,000 paths)

P&L vs. budget, $M:

| Strategy | Mean | CVaR 95% | Worst | 95th pct upside |
|---|---:|---:|---:|---:|
| Unhedged | +0.02 | 14.47 | −23.39 | +11.99 |
| 100% forwards | +0.89 | −0.89 (locked gain) | +0.89 | +0.89 |
| 100% ATM-forward puts | +0.40 | 3.20 | −3.20 | +8.29 |
| 100% puts struck at 95% of forward | +0.12 | 8.37 | −9.08 | +10.87 |
| Layered 50% fwd / 30% puts / 20% open | +0.50 | 4.23 | −6.15 | +5.88 |

What the numbers say:

- **Forwards remove all the risk and all the upside.** The gain they lock in
  is forward points, not a forecast: USD rates exceed most of these
  currencies' rates, so selling them forward is at a premium to spot
  (except MXN, where a 6.8% rate means hedging *costs* ~1.3%).
- **Puts cap the loss at the premium** (~$4M for full ATM cover) and keep
  roughly 70% of the unhedged upside.
- **Strike is a cost/protection dial**: moving from ATM to 90% of the
  forward cuts the premium from $4.0M to $0.2M, but the worst case goes
  from −$3.2M to −$15.7M. Cheaper puts mean self-insuring the first few %.
- **Diversification matters**: summed standalone CVaR is $21.1M vs. $14.5M
  for the portfolio -- 31% of the risk cancels across currencies. Hedging
  each currency to its own standalone risk over-hedges.
- **Fat tails wash out at six months**: GBM and bootstrap give nearly the
  same CVaR99 ($18.2M vs. $18.6M). Summing 126 daily returns averages
  away single-day extremes; fat tails would matter much more for a short
  horizon.

## Limitations

- 3-month rates stand in for the 6-month tenor, and the OECD foreign-rate
  series on FRED lag by months (EUR/GBP last print Jan 2026).
- Options are priced at historical, not implied, vol; real implied vol
  usually carries a premium over realized, making real options costlier.
- No bid-ask spreads, transaction costs, counterparty risk, or hedge accounting.
- CNY is a managed currency: low historical vol can hide the risk of a
  discrete policy devaluation (e.g. August 2015).
- Hedge ratios are uniform across currencies; a natural extension is to
  optimize them per currency given the correlation structure.
