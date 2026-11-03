# Paper run status

Last update: 03 Nov 2026 13:51 UTC · status: **running** · period: 2026-10-12 → 2026-11-12

| Agent | Equity | Return | vs benchmark | Max DD | Trading costs | Funding | Fills | Backtest gate |
|---|---|---|---|---|---|---|---|---|
| B0 Benchmark: S&P 500 UCITS ETF, buy and hold | €966.16 | -3.38% | – | -3.7% | €2.50 | €0.00 | 1 | benchmark |
| B1 Benchmark: Bitcoin spot, buy and hold | €933.93 | -6.61% | – | -9.5% | €2.79 | €0.00 | 1 | benchmark |
| A1 ETF trend: SMA 50/200 + 12-month momentum, ATR trailing stop (IBKR) | €975.69 | -2.43% | +0.95% | -2.3% | €9.42 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A1b Same ETF trend rules, Trading 212 cost model | €982.93 | -1.71% | +1.68% | -1.8% | €1.42 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A2 Stock factor ranking + SMA(200) filter, ATR trailing stop (US large caps, IBKR) | €938.65 | -6.14% | -2.75% | -7.5% | €18.34 | €0.00 | 17 | FAIL (Sharpe at 2x costs <= 0) |
| A3 Crypto breakout: Donchian 20/10 with 2N stop (BTC/ETH perps, Kraken EEA, 1x) | €1,000.00 | +0.00% | +6.61% | 0.0% | €0.00 | €0.00 | 0 | WEAK (DSR 0.00 < 0.95) |
| A7 RSI(2) mean reversion on the S&P 500 UCITS ETF (Trading 212) | €989.03 | -1.10% | +2.29% | -1.6% | €2.50 | €0.00 | 3 | FAIL (Sharpe at 2x costs <= 0) |
| A4 Funding carry: long spot / short perp when funding > 12%/yr (Bitvavo + Kraken) | €1,002.14 | +0.21% | +6.82% | -0.5% | €1.74 | −€2.28 | 2 | WEAK (DSR 0.00 < 0.95) |
| A5 Put credit spreads on XSP: VIX filter, 50% take-profit, 2x stop (IBKR) | €901.84 | -9.82% | -6.43% | -11.4% | €72.17 | €0.00 | 4 | FAIL (Sharpe at 2x costs <= 0) |
| A6 LLM news: Claude-labelled headlines, 7% stop / 10% target / 5-day exit (IBKR) | €1,015.28 | +1.53% | +4.91% | -5.7% | €6.61 | €0.00 | 13 | not backtested |

Simulated fills against public quotes; not investment advice. Full dashboard: docs/classic.html (GitHub Pages).
