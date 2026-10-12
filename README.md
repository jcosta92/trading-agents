# trading-agents

## The bot test (12 Oct - 31 Dec 2026): 910 bots, 163 strategy families

Every bot trades its own simulated **EUR 1,000** on real candles, buying and selling throughout the test; on
31 Dec every bot sells what it holds. A mid-point review runs on 12 Nov, the final review on 31 Dec.

- **Families, not bots.** A family is one published strategy: an open-source bot (Freqtrade, Jesse, QuantConnect,
  Hummingbot, Pionex/KuCoin bot types, ai-hedge-fund), a classic or TradingView system, a crypto or stock/ETF
  research paper, or a small machine-learning model. Each family runs its source's default rules, unchanged, on
  several markets (usually 8 coins or 10 ETFs). That is a robustness check, not parameter fishing: a family is judged
  by its median bot and by how many of its bots make money, never by its single best bot.
- **The original 17 bots** (6 most-used bot strategies, 5 less common ones, 6 classic systems) are in
  [`config/bots.yaml`](config/bots.yaml); the 146 other families, with source links and rules, in
  [`config/families.yaml`](config/families.yaml).
- **Luck.** With ~900 bots, dozens will beat the market by chance. The dashboard shows a luck yardstick (how far the
  best bot would sit above the average with no skill at all) and the reviews correct for multiple testing.
- **No look-ahead.** Every signal and weight function is checked by `tests/test_families.py`: its value at a candle
  must be identical whether computed with or without the later candles.
- **Simulation.** Candles from Bitvavo (Kraken fallback) and Yahoo are replayed in order; resting orders fill when a
  candle trades through them, market orders at the next candle's open. Bitvavo fees (0.15% / 0.25%) or IBKR fees and
  FX, plus spread and slippage, on every fill. Long only: a strategy's short signal means "go to cash".
- **Where things are.** The bots' state, every fill (`state/bots/fills.csv`), the status table (`reports/BOTS.md`)
  and the dashboard pages live on the branch **`bot-state`**, which holds a single commit replaced every 15 minutes,
  so ~900 ledgers do not bloat the repository. The GitHub Pages site loads the latest page from that branch:
  `index.html` (live) and `preview.html` (the same bots replayed over the month before the test).
- **Running it.** The `bots` workflow loops every 15 minutes for ~5h40m, then starts its next run (free for a public
  repository); a scheduled check restarts it if the chain ever breaks.

## The research agents (background)

Eight slower systematic agents plus two buy-and-hold benchmarks also paper-trade alongside, run hourly by the same
loop (dashboard `docs/classic.html`). Each was also backtested over many years (`docs/backtest.html`); none beat
its benchmark after costs.

## The agents and their rules

| Id | Buy when | Sell when | Protective orders | Instruments, venue |
|---|---|---|---|---|
| `bench_sp500` | once, at the start | never | – | S&P 500 UCITS ETF (VUAA, Xetra), IBKR |
| `bench_btc` | once, at the start | never | – | BTC/EUR spot, Bitvavo |
| `trend_etf` | close > SMA(50) > SMA(200), 12-month return > 0, and a new 20-day closing high | SMA(50) < SMA(200) | trailing stop: highest close − 3×ATR(20) | 5 UCITS ETFs/ETCs (world, EM, EUR govt bonds, gold, commodities), IBKR |
| `trend_etf_t212` | same | same | same | same ETFs, Trading 212 |
| `factor_equity` | monthly: top-10 by momentum + value + quality, and close > SMA(200) | monthly: drops out of the top 25% | trailing stop: highest close − 3×ATR(20) | S&P 500 stocks, IBKR (fractional) |
| `crypto_trend` | close > highest high of the previous 20 days (Donchian breakout) | close < lowest low of the previous 10 days | stop 2×ATR(20) below entry | BTC/ETH perps at ≤ 1x, Kraken EEA |
| `rsi2_meanrev` | close > SMA(200) and RSI(2) < 10 | close > SMA(5) (take-profit order at the 5-day average) | stop 3×ATR(10); time exit after 10 business days | S&P 500 UCITS ETF, Trading 212 |
| `funding_carry` | 7-day average funding > 12%/yr: buy spot, short the same amount of perp | 7-day average < 4%/yr, or the last 24 h negative | collateral rebalance if the perp account passes 2x | BTC/ETH: Bitvavo spot + Kraken perp |
| `vrp_spreads` | 13 ≤ VIX ≤ its 1-year 95th percentile: sell a ~16-delta XSP put, buy one 0.5% lower, 30–45 days | – | take-profit at 50% of the credit; stop at 2× the credit or the index at the short strike; time exit at 21 days to expiry | XSP options, IBKR |
| `llm_news` | a headline labelled material (≥ 2) and positive by Claude, with a positive 1-month return (max 3 names) | – | stop −7%, take-profit +10%, time exit after 5 business days | the 100 most liquid S&P 500 names, IBKR |
| `factor_ucits_proxy` | research only: equal-weight UCITS momentum/value/quality ETFs | quarterly rebalance | – | iShares MSCI World factor ETFs (backtest only) |

Exact rules, parameters and position sizing are in [`config/agents.yaml`](config/agents.yaml) (pre-registration
version 2). That file is committed before any real-data backtest runs. Changing a parameter creates a new trial,
recorded in `reports/trials.csv`, which raises the bar of the Deflated Sharpe Ratio. The paper run always uses the
`main` configuration, never the best-looking variant.

### How orders are generated

- **Indicator signals** (buy/sell) use completed daily bars only, so each agent decides once per day, on the first
  hourly cycle at least 30 minutes after its market opens. Crypto agents decide just after 00:00 UTC.
- **Protective orders** are attached to each entry. Their levels are set from the actual fill price, then:
  - in the paper run, every hourly cycle compares them with the live price while the market is open; a level that
    has been touched becomes a market order and fills at the bid (so a fast move can fill beyond the level, like a
    real stop order);
  - in backtests, each day's open/high/low bar is checked: a gap through a level fills at the open, otherwise at the
    level, and if a bar touches both the stop and the target, the stop is assumed to have come first. Time exits fill
    at the open of the day they fall due (the paper run sells at that day's first open cycle).
  - trailing stops and indicator targets move once a day from the last completed bar, and a stop never moves down.
  - protective orders are checked before anything else in a cycle, so a failed data download or a bad labels file
    can block new entries but never an exit.
- Orders that could not fill because no usable quote was available are retried on the next cycle for up to 6 hours.
  Retried entries still pass the risk filters; a retried exit is dropped once the position is gone.
- Cash instruments (ETFs, stocks, spot crypto) can never be sold short; a stale exit order is rejected.
- Stock splits listed by the data feed are applied to the position, its average price and its stop/target levels.
  If the price history implies a split the feed does not list yet, that holding's mark and stops are frozen for up
  to a day instead of booking a fake loss.

## Costs that are modelled (both in backtests and in the paper run)

Fee schedules live in [`config/venues.yaml`](config/venues.yaml) and were checked against the brokers' pages on 2026-10-10.

- **IBKR Ireland, Xetra (Tiered):** 0.05% with a EUR 1.25 minimum, plus the Xetra exchange fee (EUR 0.90 minimum),
  clearing and regulatory fees. That comes to about EUR 2.18 for any order under roughly EUR 1,800.
- **IBKR, US stocks:** USD 0.0035 per share with a USD 0.35 minimum, plus exchange, clearing, CAT and SEC fees.
  Converting EUR to USD costs 0.03%.
- **IBKR, XSP options:** USD 0.25 to 0.65 per contract (USD 1 minimum per order), plus OCC clearing, the CAT fee and
  a conservative exchange/ORF allowance. Fills cross the full bid-ask spread.
- **Trading 212:** no commission, a 0.15% FX fee and an extra 5 bp execution allowance.
- **Kraken perp margin:** the carry agent's spot coins sit at Bitvavo and do not back the short perp. When the perp
  account goes above 2x, the agent sells spot and buys back perp to return to 1x (paying fees on both). Below 1%
  collateral the position is liquidated, with an assumed 1% fee.
- **Bitvavo spot:** 0.25% taker. **Kraken EEA perps:** 0.05% taker, plus real funding.
- **Every order:** fills at the ask when buying and the bid when selling (or mid ± a fallback half-spread),
  plus slippage. Backtests are also re-run with every cost doubled (`x2`); decisions are identical at every level.

## How the evaluation works

1. **Backtest** (`backtest` workflow). Full history is pulled from Yahoo, the funding venues and point-in-time S&P 500
   membership. Every agent is then run at 0x, 1x and 2x costs, plus its pre-declared robustness variants. Metrics:
   CAGR, Sharpe with a bootstrap 90% CI, Sortino, max drawdown, Calmar, turnover, cost drag, excess return vs the
   benchmark, Probabilistic and Deflated Sharpe Ratio.
2. **Gate**:
   - **FAIL** if the Sharpe ratio at 2x costs is not positive, or the profit disappears once the best calendar year is
     removed.
   - **PASS** needs a Deflated Sharpe Ratio of at least 0.95.
   - Everything else is **WEAK**.

   All agents are paper-traded anyway, with their gate shown next to their results.
3. **Paper month** (`paper` workflow, every hour at :07). Kill switches:
   - a loss of 10–30% from the start (set per agent) flattens the account and halts the agent;
   - a daily loss limit blocks new entries for the rest of the day.

Every ledger is an append-only journal and is re-verified by replay on every load and save.

## The LLM agent without an API bill

1. The `news` workflow (weekdays 12:20 UTC, NYSE trading days only) writes `state/llm/inbox/<date>.json`: headlines
   since the previous inbox for the 100 most liquid S&P 500 names.
2. A scheduled Claude task (weekdays, about an hour later) reads the inbox, labels every headline following
   [`config/llm_prompt.md`](config/llm_prompt.md), checks the file with `python -m quantdesk.cli validate-labels`,
   and commits `state/llm/labels/<date>.json`.
3. The hourly paper cycle trades on the labels with the fixed rules above. No labels, or an invalid file: the agent
   does not open positions that day (its stops and exits keep working) and the dashboard says why.

## Repository layout

```
config/          agents.yaml (pre-registered rules), instruments.yaml, venues.yaml (fees), llm_prompt.md
src/quantdesk/   costs.py, ledger.py, execution.py, portfolio.py, risk.py, context.py (look-ahead guard)
  indicators.py  SMA, EMA, RSI, ATR, Donchian, ROC, MACD
  protect.py     stop-loss / take-profit / trailing-stop / time-exit orders
  agents/        one module per strategy - pure functions of (point-in-time data, ledger) -> orders
  backtest/      engine.py (daily event loop), metrics.py (PSR/DSR, bootstrap), runner.py (gate, trials)
  live/          runner.py (hourly paper cycle), quotes.py (Yahoo / ccxt / option chains)
  data/          Yahoo, ccxt funding, S&P 500 membership, synthetic option model
  llm.py         headline inbox, label validation
  reporting.py   dashboard + backtest HTML, STATUS.md
tests/           unit tests, look-ahead perturbation tests, offline end-to-end paper runs
state/           paper-run ledgers, equity.csv, fills.csv, news inbox/labels, logs (written by the workflows)
reports/         backtest results, trials.csv, STATUS.md
docs/            GitHub Pages site
```

## Running it yourself

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                   # tests (offline)
python -m quantdesk.cli selftest            # quick end-to-end run on synthetic data
python -m quantdesk.cli fetch-history       # needs internet: Yahoo, Kraken, GitHub
python -m quantdesk.cli backtest            # -> reports/backtest/
python -m quantdesk.cli paper --force       # one paper cycle -> state/, docs/
python -m quantdesk.cli news-inbox          # today's headline inbox -> state/llm/inbox/
python -m quantdesk.cli validate-labels     # check today's labels file
```

## GitHub setup (one-off)

1. **Settings → Pages:** deploy from branch `main`, folder `/docs`.
2. The paper run starts at the first hourly cycle after `paper.enabled` is set to `true` in `config/agents.yaml`,
   and ends 31 days later. No secrets are needed.

## Known limitations

- **Fills are simulated against public quotes.** Yahoo quotes are often 15 minutes delayed, and Kraken quotes are
  real-time. This is not a broker paper account; a broker account needs IB Gateway on an always-on machine with a
  daily login.
- **Yahoo bad-print filter.** A Yahoo quote more than 35% (stocks), 20% (ETFs) or 15% (indices) away from the previous
  daily close is ignored until a daily close confirms the move, so a genuine crash of that size is only acted on the
  next day. Kraken crypto quotes are not filtered.
- **Stops are checked hourly, not tick by tick.** GitHub can also delay or skip scheduled runs under load. A price that
  dips through a stop and recovers between two checks is not caught; a stop that is caught can fill well beyond its
  level. Backtests, which see the whole daily range, are the more pessimistic of the two on this point.
- **The `factor_equity` backtest flatters itself.** It tests only the momentum leg, and delisted S&P 500 members are
  mostly missing from Yahoo, so it carries survivorship bias. Coverage is reported in the backtest.
- **`vrp_spreads` backtests use modelled option prices.** These are Black-Scholes on VIX with a fixed skew and spread.
- **Funding history before each venue's data starts is charged at a 10%/yr fallback rate.** The report shows how many
  days that affected.
- **Backtests use total-return (dividend-adjusted) prices** with no withholding tax. The paper run credits real
  dividends net of 15% US withholding, so US-stock backtests are about 0.2%/yr optimistic on that count.
- **`llm_news` cannot be backtested honestly** (no point-in-time news archive, and the model has seen the past). It is a
  forward test only, and it depends on the scheduled labelling task running.
- **Regulation is moving.** ESMA has said crypto perpetuals are probably CFDs, which would cap leverage at 2:1; the
  crypto agents never exceed 1x.

Simulated results. Not investment, tax or legal advice.
