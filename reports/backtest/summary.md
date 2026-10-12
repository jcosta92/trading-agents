# Backtest results

Generated 2026-10-12T14:35:45.650669+00:00 from commit `9d9d012`; data through 2026-10-12. Distinct configurations tested so far (for the Deflated Sharpe Ratio): 16.

Costs: x0 = no trading costs, x1 = modelled fees + spread + slippage + FX, x2 = those trading costs doubled. Decisions are identical at every level (made with x1 costs); perp funding is market data and applies unscaled at every level.

| Agent | Period | CAGR x0 | CAGR x1 | CAGR x2 | Sharpe x1 (90% CI) | Max DD x1 | Cost drag/yr | vs benchmark (excess CAGR) | Kill switch would fire in | DSR | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 Benchmark: S&P 500 UCITS ETF, buy and hold | 2011-01-03 to 2026-10-12 | 15.1% | 15.1% | 15.1% | 0.96 (0.52 to 1.38) | -33.7% | 0.01% | n/a | n/a | n/a | benchmark |
| B1 Benchmark: Bitcoin spot, buy and hold | 2018-03-01 to 2026-10-12 | 27.8% | 27.8% | 27.7% | 0.71 (0.08 to 1.38) | -73.6% | 0.01% | n/a | n/a | n/a | benchmark |
| A1 ETF trend: SMA 50/200 + 12-month momentum, ATR trailing stop (IBKR) | 2011-01-03 to 2026-10-12 | 2.1% | -12.3% | -14.8% | -2.35 (-2.77 to -1.94) | -87.5% | 11.92% | -27.5% | 0% of months | 0.00 | FAIL (Sharpe at 2x costs <= 0) |
| A1b Same ETF trend rules, Trading 212 cost model | 2011-01-03 to 2026-10-12 | 2.4% | 1.2% | -0.2% | 0.27 (-0.12 to 0.68) | -12.3% | 1.29% | -14.0% | 0% of months | 0.00 | FAIL (Sharpe at 2x costs <= 0) |
| A2 Stock factor ranking + SMA(200) filter, ATR trailing stop (US large caps, IBKR) | 2008-01-02 to 2026-10-12 | 2.4% | -5.9% | -8.8% | -0.18 (-0.52 to 0.16) | -73.0% | 7.99% | -17.5% | 0% of months | 0.00 | FAIL (Sharpe at 2x costs <= 0) |
| R1 Research only: UCITS factor ETFs (momentum/value/quality), equal weight | 2015-01-02 to 2026-10-12 | 12.7% | 12.5% | 12.5% | 0.79 (0.33 to 1.23) | -32.0% | 0.05% | -1.8% | n/a | 0.00 | WEAK (DSR 0.00 < 0.95) |
| A3 Crypto breakout: Donchian 20/10 with 2N stop (BTC/ETH perps, Kraken EEA, 1x) | 2018-03-01 to 2026-10-12 | 15.1% | 14.8% | 14.6% | 0.97 (0.43 to 1.51) | -16.2% | 0.26% | -13.0% | 0% of months | 0.00 | WEAK (DSR 0.00 < 0.95) |
| A7 RSI(2) mean reversion on the S&P 500 UCITS ETF (Trading 212) | 2011-01-03 to 2026-10-12 | 3.2% | 0.5% | -1.8% | 0.13 (-0.25 to 0.53) | -10.5% | 2.22% | -14.7% | 0% of months | 0.00 | FAIL (Sharpe at 2x costs <= 0) |
| A4 Funding carry: long spot / short perp when funding > 12%/yr (Bitvavo + Kraken) | 2020-01-01 to 2026-10-12 | 4.8% | 4.0% | 3.2% | 0.67 (0.05 to 1.32) | -15.8% | 0.72% | -39.4% | 0% of months | 0.00 | WEAK (DSR 0.00 < 0.95) |
| A5 Put credit spreads on XSP: VIX filter, 50% take-profit, 2x stop (IBKR) | 2008-01-02 to 2026-10-12 | 1.5% | -4.1% | -4.4% | -0.42 (-0.82 to 0.02) | -60.3% | 3.15% | -18.3% | 0% of months | 0.00 | FAIL (Sharpe at 2x costs <= 0) |

## Robustness variants (x1 costs, pre-registered; never used to pick the paper config)

| Agent | Variant | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| trend_etf | stop_2atr | -14.6% | -2.95 | -91.7% |
| factor_equity | holdings_20 | -5.3% | -0.25 | -66.4% |
| crypto_trend | donchian_55_20 | 12.2% | 0.74 | -20.8% |
| rsi2_meanrev | rsi_5 | 0.6% | 0.19 | -9.4% |
| funding_carry | entry_8pct | 4.4% | 0.41 | -27.9% |
| vrp_spreads | delta_10 | -0.2% | -0.24 | -5.7% |

## Caveats per agent

- **A2 Stock factor ranking + SMA(200) filter, ATR trailing stop (US large caps, IBKR)**: Free data has no point-in-time fundamentals, so the backtest can only test the momentum leg, on point-in-time S&P 500 membership (no sector exclusion and no price floor in the backtest, because only today's sectors and adjusted prices are available). Delisted names are often missing from Yahoo, which biases the backtest upward; coverage is reported. The UCITS factor-ETF proxy (factor_ucits_proxy) is the honest check on what a multi-factor tilt earned.
- **A5 Put credit spreads on XSP: VIX filter, 50% take-profit, 2x stop (IBKR)**: Option prices are SYNTHETIC (Black-Scholes on VIX with a fixed skew and a spread model). Results are indicative of the mechanics and cost drag, not evidence of an edge.
- **A6 LLM news: Claude-labelled headlines, 7% stop / 10% target / 5-day exit (IBKR)**: Not backtestable honestly: no point-in-time news archive, and the model was trained on the historical period. Forward (paper) test only.
