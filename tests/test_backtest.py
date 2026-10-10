import numpy as np
import pandas as pd
import pytest

from quantdesk.agents import build_agent
from quantdesk.backtest.engine import run_backtest
from quantdesk.backtest.runner import run_all
from quantdesk.costs import CostModel
from quantdesk.data.store import HistoryStore
from quantdesk.registry import Registry


def test_flat_prices_zero_cost_preserve_equity(settings, instruments, venues):
    idx = pd.bdate_range("2015-01-01", "2018-01-01")
    store = HistoryStore()
    for k in ["WORLD_EQ", "EM_EQ", "EUR_GOVT", "GOLD", "COMMOD"]:
        s = pd.Series(100.0, index=idx) * (1 + 0.0001 * np.sin(np.arange(len(idx))))   # tiny wiggle, positive trend 0
        store.set_series("close", k, s)
        store.set_series("open", k, s)
    store.set_series("close", "EURUSD", pd.Series(1.1, index=idx))
    reg = Registry(instruments)
    cm = CostModel(venues, 0.0)
    agent = build_agent(settings.agents["trend_etf"], reg, cm)
    res = run_backtest(agent, store, reg, cm, start="2016-02-01", capital=1000)
    assert res.equity.iloc[-1] == pytest.approx(1000, rel=2e-3)
    assert res.diagnostics["ledger_verified"]


def test_costs_reduce_returns_monotonically(synth, settings, instruments, venues):
    store, extras = synth
    finals = []
    for stress in (0.0, 1.0, 3.0):
        reg = Registry(instruments)
        cm = CostModel(venues, stress)
        spec = settings.agents["crypto_trend"]
        agent = build_agent(spec, reg, cm)
        res = run_backtest(agent, store, reg, cm, start="2017-01-01", capital=1000)
        finals.append(res.equity.iloc[-1])
    assert finals[0] > finals[1] > finals[2]


def test_perp_funding_charged_to_longs(synth, settings, instruments, venues):
    store, extras = synth
    reg = Registry(instruments)
    cm = CostModel(venues, 1.0)
    agent = build_agent(settings.agents["crypto_trend"], reg, cm)
    res = run_backtest(agent, store, reg, cm, start="2017-01-01", capital=1000)
    assert res.costs.get("funding", 0) != 0
    assert res.diagnostics["fallback_funding_days"] == 0


def test_run_all_on_synthetic_writes_gate(tmp_path, synth, settings, instruments, venues):
    store, extras = synth
    for s in settings.agents.values():
        s.backtest_start = "2016-06-01"
        s.variants = {}
    settings.backtest["stress_levels"] = [1.0, 2.0]
    out = run_all(settings, store, instruments, venues, extras=extras, only=["trend_etf", "crypto_trend", "bench_sp500"],
                  out_dir=tmp_path / "bt", trials_path=tmp_path / "trials.csv")
    t = out["table"]
    assert set(t["agent"]) == {"trend_etf", "crypto_trend", "bench_sp500"}
    gates = t[(t["stress"] == 1.0)].set_index("agent")["gate"]
    assert gates["bench_sp500"] == "benchmark"
    assert gates["trend_etf"].split()[0] in {"PASS", "WEAK", "FAIL"}
    assert (tmp_path / "bt" / "summary.md").exists()
    assert t["ledger_verified"].all()


def test_combo_retried_as_a_whole(instruments, venues):
    from quantdesk.backtest.engine import _execute_pending
    from quantdesk.execution import Executor, OrderIntent, Quote
    from quantdesk.ledger import FxRates, Ledger

    class Q:
        def __init__(self, ok): self.ok = ok
        def quote(self, inst):
            if inst.id == "ETH_SPOT" and not self.ok:
                return None
            return Quote(2000.0)
        def fx(self): return FxRates(1.1)
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 1000.0)
    pending = [OrderIntent("ETH_SPOT", 0.2, "s", combo="c"), OrderIntent("ETH_PERP", -0.2, "p", combo="c")]
    fills, rej = [], []
    retry = _execute_pending(ex, led, pending, Q(False), "d1", fills, rej)
    assert {o.inst_id for o in retry} == {"ETH_SPOT", "ETH_PERP"} and not led.positions
    _execute_pending(ex, led, retry, Q(True), "d2", fills, rej)
    assert led.qty("ETH_SPOT") > 0 and led.qty("ETH_SPOT") == pytest.approx(-led.qty("ETH_PERP"))


def test_carry_collateral_rebalance(settings, instruments, venues):
    from quantdesk.agents import build_agent
    from quantdesk.context import Context
    from quantdesk.execution import Quote
    from quantdesk.ledger import FxRates, Ledger
    reg = Registry(instruments)
    cm = CostModel(venues, 1.0)
    agent = build_agent(settings.agents["funding_carry"], reg, cm)
    led = Ledger("funding_carry", 1000.0)
    fx = FxRates(1.1)
    led.trade("d", reg.get("BTC_SPOT"), 0.01, 50000.0, {}, fx, 50000.0)        # EUR 500 at Bitvavo
    led.trade("d", reg.get("BTC_PERP"), -0.01, 55000.0, {}, fx, 55000.0)       # short at Kraken
    idx = pd.date_range("2026-01-01", "2026-03-01")
    st = HistoryStore()
    st.set_series("funding", "BTC_PERP", pd.Series(0.0006, index=idx))       # ~22%/yr: stay in the trade
    st.set_series("funding", "ETH_PERP", pd.Series(0.0, index=idx))

    class Q:
        def quote(self, inst):
            return Quote({"BTC_SPOT": 120000.0, "BTC_PERP": 132000.0}.get(inst.id, 1.0))   # +140% rally
        def fx(self): return fx
    ctx = Context(pd.Timestamp("2026-03-01"), st, reg, Q(), "backtest")
    orders = agent.decide(ctx, led)
    assert orders and {o.inst_id for o in orders} == {"BTC_SPOT", "BTC_PERP"}
    spot = next(o for o in orders if o.inst_id == "BTC_SPOT")
    perp = next(o for o in orders if o.inst_id == "BTC_PERP")
    assert spot.qty < 0 and perp.qty == pytest.approx(-spot.qty)


def test_reissued_order_supersedes_retry():
    from quantdesk.backtest.engine import merge_pending
    from quantdesk.execution import OrderIntent
    retry = [OrderIntent("COMMOD", 4, "r"), OrderIntent("BTC_SPOT", 0.1, "s", combo="c"),
             OrderIntent("BTC_PERP", -0.1, "p", combo="c"), OrderIntent("GOLD", 2, "keep")]
    new = [OrderIntent("COMMOD", 4, "again"), OrderIntent("BTC_PERP", -0.1, "again")]
    out = merge_pending(retry, new)
    ids = [(o.inst_id, o.reason) for o in out]
    assert ("COMMOD", "r") not in ids and ("BTC_SPOT", "s") not in ids and ("GOLD", "keep") in ids
    assert sum(1 for o in out if o.inst_id == "COMMOD") == 1


def test_carry_closes_orphan_leg_after_liquidation(settings, instruments, venues):
    from quantdesk.agents import build_agent
    from quantdesk.context import Context
    from quantdesk.execution import Quote
    from quantdesk.ledger import FxRates, Ledger
    reg = Registry(instruments)
    agent = build_agent(settings.agents["funding_carry"], reg, CostModel(venues, 1.0))
    led = Ledger("funding_carry", 1000.0)
    fx = FxRates(1.1)
    led.trade("d", reg.get("ETH_SPOT"), 0.2, 2000.0, {}, fx, 2000.0)          # spot left after a liquidation
    idx = pd.date_range("2026-01-01", "2026-03-01")
    st = HistoryStore()
    st.set_series("funding", "ETH_PERP", pd.Series(0.0006, index=idx))
    st.set_series("funding", "BTC_PERP", pd.Series(0.0, index=idx))

    class Q:
        def quote(self, inst): return Quote(2000.0)
        def fx(self): return fx
    orders = agent.decide(Context(pd.Timestamp("2026-03-01"), st, reg, Q(), "backtest"), led)
    assert [(o.inst_id, o.qty) for o in orders] == [("ETH_SPOT", -0.2)]


def test_liquidation_fee_only_on_filled_notional(instruments, venues):
    from quantdesk import risk
    from quantdesk.execution import Executor
    from quantdesk.ledger import FxRates, Ledger
    reg = Registry(instruments)
    led = Ledger("t", 1000.0)
    fx = FxRates(1.1)
    led.trade("d", reg.get("BTC_PERP"), 0.02, 50000.0, {}, fx, 50000.0)
    state = {"liquidation_fee": 0.01, "notional_eur": 1000.0}

    class NoQuote:
        def quote(self, inst): return None
        def fx(self): return fx
    risk.liquidate(led, Executor(CostModel(venues, 1.0), reg), NoQuote(), fx, state, "d2")
    assert led.costs.get("liquidation", 0.0) == 0.0 and led.qty("BTC_PERP") == 0.02
