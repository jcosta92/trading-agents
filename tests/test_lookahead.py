"""Look-ahead guard: an agent's decision at date t must not change when data after t changes."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.agents import build_agent
from quantdesk.backtest.engine import BacktestQuotes
from quantdesk.context import Context
from quantdesk.costs import CostModel
from quantdesk.data.options import ModelParams, SyntheticOptionModel
from quantdesk.data.store import HistoryStore
from quantdesk.ledger import Ledger
from quantdesk.registry import Registry

AGENTS = ["trend_etf", "factor_equity", "crypto_trend", "funding_carry", "vrp_spreads", "factor_ucits_proxy"]


def _perturb_future(store: HistoryStore, t: pd.Timestamp, seed=9) -> HistoryStore:
    rng = np.random.default_rng(seed)
    frames = {}
    for f, df in store.frames.items():
        df = df.copy()
        fut = df.index > t
        df.loc[fut] = df.loc[fut] * rng.uniform(0.5, 1.5, size=df.loc[fut].shape)
        frames[f] = df
    return HistoryStore(frames, store.meta)


def _decide(aid, store, extras, t, settings, instruments, venues):
    reg = Registry(instruments)
    cm = CostModel(venues, 1.0)
    agent = build_agent(settings.agents[aid], reg, cm)
    close = store.frame("close")
    om = None
    if "options" in agent.extra_needs():
        om = SyntheticOptionModel(store.close("XSP"), store.close("VIX"), store.close("TBILL"), ModelParams.from_agent(agent.p))
    q = BacktestQuotes(t, close, close.ffill(limit=7), store.close("EURUSD"), reg, om, exact_only=False)
    ctx = Context(t, store, reg, q, "backtest", extras=extras, options=om)
    led = Ledger(aid, 1000.0)
    return [(o.inst_id, round(o.qty, 8)) for o in agent.decide(ctx, led)]


@pytest.mark.parametrize("aid", AGENTS)
def test_decisions_ignore_future_data(aid, synth, settings, instruments, venues):
    store, extras = synth
    for t in pd.to_datetime(["2017-03-15", "2019-11-20", "2022-06-15"]):
        base = _decide(aid, store, extras, t, settings, instruments, venues)
        pert = _decide(aid, _perturb_future(store, t), extras, t, settings, instruments, venues)
        assert base == pert, f"{aid} decision at {t.date()} changed when future data changed"


def test_context_history_stops_at_asof(synth, instruments):
    store, _ = synth
    t = pd.Timestamp("2018-05-15")
    ctx = Context(t, store, Registry(instruments), None, "backtest")
    assert ctx.history("SP500").index.max() <= t
    assert ctx.panel(["SP500", "GOLD"]).index.max() <= t
    assert ctx.history("BTC_PERP", field="funding").index.max() <= t
