# Paper run status

Last update: 19 Nov 2026 11:51 UTC · status: **finished** · period: 2026-10-12 → 2026-11-12

| Agent | Equity | Return | vs benchmark | Max DD | Trading costs | Funding | Fills | Backtest gate |
|---|---|---|---|---|---|---|---|---|
| B0 Benchmark: S&P 500 UCITS ETF, buy and hold | €965.48 | -3.45% | – | -5.5% | €2.50 | €0.00 | 1 | benchmark |
| B1 Benchmark: Bitcoin spot, buy and hold | €785.58 | -21.44% | – | -24.7% | €2.79 | €0.00 | 1 | benchmark |
| A1 ETF trend: SMA 50/200 + 12-month momentum, ATR trailing stop (IBKR) | €975.69 | -2.43% | +1.02% | -2.3% | €9.42 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A1b Same ETF trend rules, Trading 212 cost model | €982.93 | -1.71% | +1.75% | -1.8% | €1.42 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A2 Stock factor ranking + SMA(200) filter, ATR trailing stop (US large caps, IBKR) | €903.72 | -9.63% | -6.18% | -13.0% | €20.91 | €0.00 | 24 | FAIL (Sharpe at 2x costs <= 0) |
| A3 Crypto breakout: Donchian 20/10 with 2N stop (BTC/ETH perps, Kraken EEA, 1x) | €1,000.00 | +0.00% | +21.44% | 0.0% | €0.00 | €0.00 | 0 | WEAK (DSR 0.00 < 0.95) |
| A7 RSI(2) mean reversion on the S&P 500 UCITS ETF (Trading 212) | €1,002.30 | +0.23% | +3.68% | -1.6% | €3.41 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A4 Funding carry: long spot / short perp when funding > 12%/yr (Bitvavo + Kraken) | €1,012.70 | +1.27% | +22.71% | -0.6% | €1.74 | −€3.97 | 2 | WEAK (DSR 0.00 < 0.95) |
| A5 Put credit spreads on XSP: VIX filter, 50% take-profit, 2x stop (IBKR) | €911.71 | -8.83% | -5.38% | -11.4% | €72.17 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A6 LLM news: Claude-labelled headlines, 7% stop / 10% target / 5-day exit (IBKR) | €933.15 | -6.68% | -3.23% | -9.9% | €10.56 | €0.00 | 21 | not backtested |

Simulated fills against public quotes; not investment advice. Full dashboard: docs/classic.html (GitHub Pages).
