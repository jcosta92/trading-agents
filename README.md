# trading-agents

Seven systematic trading agents plus two buy-and-hold benchmarks. Each agent gets its own simulated
**EUR 1,000** account. Every agent is backtested over many years, then paper-traded for one month.
The same Python code runs both, and both apply the full cost model an EU retail investor based in
Portugal would actually pay.

The point is to find out, honestly, whether any of these strategies survives costs at this account size. It is not a
profit machine. The research behind the design concluded that no strategy credibly delivers "best profits" on
EUR 1,000 after costs, and that one month of trading cannot separate skill from luck. The month mainly checks that the
plumbing works: data feeds, orders, the cost model, kill switches and unattended operation. The multi-year backtest
gives the statistical evidence.

- **Dashboard:** `docs/index.html` (GitHub Pages), rebuilt after every paper cycle
- **Status table:** [`reports/STATUS.md`](reports/STATUS.md)
- **Backtest report:** `docs/backtest.html` and [`reports/backtest/summary.md`](reports/backtest/summary.md)

## The agents

| Id | Thesis | Instruments, venue (cost model) | Backtest |
|---|---|---|---|
| `bench_sp500` | Passive control | Vanguard S&P 500 UCITS ETF (VUAA, Xetra), IBKR | yes |
| `bench_btc` | Passive crypto control | BTC/EUR spot, Bitvavo fees | yes |
| `trend_etf` | Time-series momentum across asset classes, monthly | 5 UCITS ETFs/ETCs: world equity, EM equity, EUR govt bonds, gold, commodities. IBKR Tiered | yes, 2011+ (US-proxy spliced) |
| `trend_etf_t212` | Same signals, different broker | Same ETFs, Trading 212 (no commission, fractional) | yes |
| `factor_equity` | Momentum + value + quality tilt, 10 US large caps | S&P 500 stocks ex financials/real estate, IBKR fractional | momentum leg only (no free point-in-time fundamentals) |
| `crypto_trend` | BTC/ETH long or flat: above 100-day SMA and 30-day return > 0 | Kraken EEA perps at ≤1x, funding paid | yes, 2018+ |
| `funding_carry` | Long spot / short perp only when funding > 12%/yr | Bitvavo spot + Kraken EEA perp | yes, from the start of the funding history |
| `vrp_spreads` | Volatility risk premium via defined-risk XSP put credit spreads | XSP options, IBKR | **synthetic option prices**: indicative only |
| `llm_news` | LLM labels headlines; fixed rules trade them (max 3 names, 5-day hold, 7% stop) | S&P 500's 100 most liquid names, IBKR; LLM cost charged to the account | not backtestable (forward test only) |
| `factor_ucits_proxy` | Research only: what a UCITS multi-factor ETF portfolio really earned | iShares MSCI World momentum/value/quality ETFs | yes (not paper-traded) |

Every rule and parameter is in [`config/agents.yaml`](config/agents.yaml). That file was committed before any
real-data backtest ran. Changing a parameter creates a new trial, recorded in `reports/trials.csv`, which raises the
bar of the Deflated Sharpe Ratio. The paper run always uses the `main` configuration, never the best-looking
variant.

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
- **Bitvavo spot:** 0.25% taker. **Kraken EEA perps:** 0.05% taker, plus real hourly funding.
- **Every order:** fills at the ask when buying and the bid when selling (or mid ± a fallback half-spread),
  plus slippage. Backtests are also re-run with every cost doubled (`x2`).
- **LLM inference:** token usage × the model's price, charged to `llm_news`.

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
3. **Paper month** (`paper` workflow, every day at 14:50 and 15:20 UTC). Each agent decides once per day, using data
   up to yesterday's close. Orders fill against live quotes when the agent's market is open. Kill switches:
   - a loss of 10–30% from the start (set per agent) flattens the account and halts the agent;
   - a daily loss limit blocks new entries for the rest of the day.

Every ledger is an append-only journal and is re-verified by replay on every load and save.

## Repository layout

```
config/          agents.yaml (pre-registered rules), instruments.yaml, venues.yaml (fees)
src/quantdesk/   costs.py, ledger.py, execution.py, portfolio.py, risk.py, context.py (look-ahead guard)
  agents/        one module per strategy - pure functions of (point-in-time data, ledger) -> orders
  backtest/      engine.py (daily event loop), metrics.py (PSR/DSR, bootstrap), runner.py (gate, trials)
  live/          runner.py (paper cycle), quotes.py (Yahoo / ccxt / option chains)
  data/          Yahoo, ccxt funding, S&P 500 membership, synthetic option model
  reporting.py   dashboard + backtest HTML, STATUS.md
tests/           unit tests, look-ahead perturbation tests, offline end-to-end paper runs
state/           paper-run ledgers, equity.csv, fills.csv, run logs (written by the workflow)
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
```

## GitHub setup (one-off)

1. **Settings → Pages:** deploy from branch `main`, folder `/docs`.
2. **Settings → Secrets and variables → Actions:** add `ANTHROPIC_API_KEY` if you want `llm_news` to run. Without it
   that agent stays in cash and says so on the dashboard. Its spend is capped at USD 15 a month in the config.
3. The paper run starts at the first scheduled cycle after `paper.enabled` is set to `true` in `config/agents.yaml`,
   and ends 31 days later.

## Known limitations

- **Fills are simulated against public quotes.** Yahoo quotes are often 15 minutes delayed, and Kraken quotes are
  real-time. This is not an IBKR paper account. A broker paper account needs IB Gateway on an always-on machine with
  a daily login; that is the natural phase 2.
- **The `factor_equity` backtest flatters itself.** It tests only the momentum leg, and delisted S&P 500 members are
  mostly missing from Yahoo, so it carries survivorship bias. Coverage is reported in the backtest.
- **`vrp_spreads` backtests use modelled option prices.** These are Black-Scholes on VIX with a fixed skew and spread.
- **Funding history before each venue's data starts is charged at a 10%/yr fallback rate.** The report shows how many
  days that affected.
- **Backtests use total-return (dividend-adjusted) prices** with no withholding tax. The paper run credits real
  dividends net of 15% US withholding, so US-stock backtests are about 0.2%/yr optimistic on that count.
- **Daily data only.** Decisions use daily closes; the carry agent's perp collateral and liquidation checks run once
  a day, so an intraday spike that would have liquidated the perp can be missed.
- **Regulation is moving.** ESMA has said crypto perpetuals are probably CFDs, which would cap leverage at 2:1; the
  crypto agents never exceed 1x.

Simulated results. Not investment, tax or legal advice.
