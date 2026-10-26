# Paper run status

Last update: 26 Oct 2026 10:17 UTC · status: **running** · period: 2026-10-12 → 2026-11-12

| Agent | Equity | Return | vs benchmark | Max DD | Trading costs | Funding | Fills | Backtest gate |
|---|---|---|---|---|---|---|---|---|
| B0 Benchmark: S&P 500 UCITS ETF, buy and hold | €975.98 | -2.40% | – | -2.5% | €2.50 | €0.00 | 1 | benchmark |
| B1 Benchmark: Bitcoin spot, buy and hold | €968.09 | -3.19% | – | -6.0% | €2.79 | €0.00 | 1 | benchmark |
| A1 ETF trend: SMA 50/200 + 12-month momentum, ATR trailing stop (IBKR) | €987.48 | -1.25% | +1.15% | -1.2% | €4.68 | €0.00 | 2 | FAIL (Sharpe at 2x costs <= 0) |
| A1b Same ETF trend rules, Trading 212 cost model | €989.41 | -1.06% | +1.34% | -1.2% | €0.93 | €0.00 | 3 | FAIL (Sharpe at 2x costs <= 0) |
| A2 Stock factor ranking + SMA(200) filter, ATR trailing stop (US large caps, IBKR) | €990.56 | -0.94% | +1.46% | -3.9% | €15.71 | €0.00 | 10 | FAIL (Sharpe at 2x costs <= 0) |
| A3 Crypto breakout: Donchian 20/10 with 2N stop (BTC/ETH perps, Kraken EEA, 1x) | €1,000.00 | +0.00% | +3.19% | 0.0% | €0.00 | €0.00 | 0 | WEAK (DSR 0.00 < 0.95) |
| A7 RSI(2) mean reversion on the S&P 500 UCITS ETF (Trading 212) | €993.28 | -0.67% | +1.73% | -1.0% | €1.68 | €0.00 | 2 | FAIL (Sharpe at 2x costs <= 0) |
| A4 Funding carry: long spot / short perp when funding > 12%/yr (Bitvavo + Kraken) | €1,000.28 | +0.03% | +3.22% | -0.3% | €1.74 | −€0.92 | 2 | WEAK (DSR 0.00 < 0.95) |
| A5 Put credit spreads on XSP: VIX filter, 50% take-profit, 2x stop (IBKR) | €901.15 | -9.89% | -7.48% | -11.4% | €72.17 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A6 LLM news: Claude-labelled headlines, 7% stop / 10% target / 5-day exit (IBKR) | €969.66 | -3.03% | -0.63% | -5.7% | €3.64 | €0.00 | 7 | not backtested |

Simulated fills against public quotes; not investment advice. Full dashboard: docs/classic.html (GitHub Pages).
