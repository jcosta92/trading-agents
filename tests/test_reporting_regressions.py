"""Regressions found in the first real-data backtest run."""
import json

import numpy as np
import pandas as pd

from quantdesk.agents import build_agent
from quantdesk.context import Context
from quantdesk.costs import CostModel
from quantdesk.data.store import HistoryStore
from quantdesk.registry import Registry


def test_backtest_html_tolerates_funding_overlap_stats(tmp_path, settings, synth, instruments, venues):
    """data_meta.funding.<perp>.sources also holds 'overlap_days' (int) and a dict of means."""
    from quantdesk.backtest.runner import run_all
    from quantdesk.reporting import write_backtest_html
    store, extras = synth
    store.meta["funding"] = {"BTC_PERP": {"chosen": "binance_vision", "sources": {
        "binance_vision": {"events": 10, "first": "2020-01-01", "last": "2026-01-01"},
        "overlap_days": 360, "mean_daily_annualised": {"binance_vision": 0.03}}}}
    for s in settings.agents.values():
        s.variants = {}
        s.backtest_start = "2017-01-01"
    settings.backtest["stress_levels"] = [1.0]
    run_all(settings, store, instruments, venues, extras=extras, only=["bench_sp500", "crypto_trend"],
            out_dir=tmp_path / "bt", trials_path=tmp_path / "t.csv")
    write_backtest_html(tmp_path / "bt", settings, docs_dir=tmp_path / "docs")
    assert "binance_vision: 10 events" in (tmp_path / "bt" / "index.html").read_text()


def test_factor_coverage_counts_members_without_prices(settings, instruments, venues):
    idx = pd.bdate_range("2019-01-01", "2020-06-30")
    store = HistoryStore()
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(40)]
    for i, t in enumerate(names):
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        if i < 10:
            px[:] = np.nan                        # delisted: column exists but no prices
        store.set_series("close", t, px)
        store.set_series("volume", t, pd.Series(1e6, index=idx))
    membership = pd.DataFrame({"tickers": [names + ["GONE1", "GONE2"]]}, index=[pd.Timestamp("2018-01-01")])
    reg = Registry(instruments)
    agent = build_agent(settings.agents["factor_equity"], reg, CostModel(venues, 1.0))
    ctx = Context(pd.Timestamp("2020-06-30"), store, reg, None, "backtest",
                  extras={"membership": membership, "sectors": {}})
    agent.eligible(ctx)
    cov = ctx.extras["_diag"]["coverage"]
    assert cov["members"] == 42 and cov["with_data"] == 30
