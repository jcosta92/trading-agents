# Paper run status

Last update: 14 Oct 2026 17:37 UTC · status: **running** · period: 2026-10-12 → 2026-11-12

| Agent | Equity | Return | vs benchmark | Max DD | Trading costs | Funding | Fills | Backtest gate |
|---|---|---|---|---|---|---|---|---|
| B0 Benchmark: S&P 500 UCITS ETF, buy and hold | €994.68 | -0.53% | – | -0.4% | €2.50 | €0.00 | 1 | benchmark |
| B1 Benchmark: Bitcoin spot, buy and hold | €990.73 | -0.93% | – | -2.2% | €2.79 | €0.00 | 1 | benchmark |
| A1 ETF trend: SMA 50/200 + 12-month momentum, ATR trailing stop (IBKR) | €995.20 | -0.48% | +0.05% | -0.4% | €4.68 | €0.00 | 2 | FAIL (Sharpe at 2x costs <= 0) |
| A1b Same ETF trend rules, Trading 212 cost model | €999.14 | -0.09% | +0.45% | -0.2% | €0.68 | €0.00 | 2 | FAIL (Sharpe at 2x costs <= 0) |
| A2 Stock factor ranking + SMA(200) filter, ATR trailing stop (US large caps, IBKR) | €991.78 | -0.82% | -0.29% | -1.3% | €15.71 | €0.00 | 10 | FAIL (Sharpe at 2x costs <= 0) |
| A3 Crypto breakout: Donchian 20/10 with 2N stop (BTC/ETH perps, Kraken EEA, 1x) | €1,000.00 | +0.00% | +0.93% | 0.0% | €0.00 | €0.00 | 0 | WEAK (DSR 0.00 < 0.95) |
| A7 RSI(2) mean reversion on the S&P 500 UCITS ETF (Trading 212) | €1,000.00 | +0.00% | +0.53% | 0.0% | €0.00 | €0.00 | 0 | FAIL (Sharpe at 2x costs <= 0) |
| A4 Funding carry: long spot / short perp when funding > 12%/yr (Bitvavo + Kraken) | €1,000.00 | +0.00% | +0.93% | 0.0% | €0.00 | €0.00 | 0 | WEAK (DSR 0.00 < 0.95) |
| A5 Put credit spreads on XSP: VIX filter, 50% take-profit, 2x stop (IBKR) | €981.46 | -1.85% | -1.32% | -1.7% | €6.85 | €0.00 | 2 | FAIL (Sharpe at 2x costs <= 0) |
| A6 LLM news: Claude-labelled headlines, 7% stop / 10% target / 5-day exit (IBKR) | €981.84 | -1.82% | -1.28% | -3.5% | €1.16 | €0.00 | 2 | not backtested |

Simulated fills against public quotes; not investment advice. Full dashboard: docs/classic.html (GitHub Pages).
