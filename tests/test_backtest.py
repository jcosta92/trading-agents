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
