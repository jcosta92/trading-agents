"""Paper runner end-to-end with an offline data provider (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from quantdesk.data.options import ModelParams, SyntheticOptionModel
from quantdesk.data.store import HistoryStore
from quantdesk.live.quotes import LiveQuotes
from quantdesk.live.runner import PaperRunner
from quantdesk.synthetic import make_store

CCXT_MAP = {"BTC/EUR": "BTC_SPOT", "ETH/EUR": "ETH_SPOT", "BTC/USD:USD": "BTC_PERP", "ETH/USD:USD": "ETH_PERP"}


class FakeChains:
    def __init__(self, model, today):
        self.m, self.today, self.S, self.source = model, today, None, "fake"

    def chain_between(self, dte_min, dte_max, right="P"):
        return self.m.chain(self.today, dte_min, dte_max, right)

    def quote(self, expiry, strike, right):
        return self.m.quote(self.today, expiry, strike, right)


class FakeData:
    def __init__(self, store, extras, now, shock=None):
        self.store, self.extras, self.now = store, extras, pd.Timestamp(now)
        self.shock = shock or {}
        self.news_calls = 0

    def _last(self, key):
        s = self.store.close(key).loc[:self.now.tz_localize(None)]
        return float(s.iloc[-1]) * self.shock.get(key, 1.0)

    def core_store(self, instruments, ids, start):
        st = self.store.truncate(self.now.tz_localize(None).normalize())
        st.meta = {"resolved": {k: k for k in st.keys()}}
        return st

    def sp500_store(self, start):
        return self.store.truncate(self.now.tz_localize(None).normalize()), {"sectors": self.extras["sectors"]}

    def history(self, tickers, start):
        return self.store.truncate(self.now.tz_localize(None).normalize())

    def fundamentals(self, tickers):
        rng = np.random.default_rng(0)
        return pd.DataFrame({"enterpriseToEbitda": rng.uniform(5, 30, len(tickers)),
                             "returnOnEquity": rng.uniform(0, .4, len(tickers)),
                             "grossMargins": rng.uniform(.2, .7, len(tickers)),
                             "debtToEquity": rng.uniform(0, 200, len(tickers))}, index=tickers)

    def news(self, ticker):
        self.news_calls += 1
        return [{"id": f"{ticker}-1", "title": f"{ticker} raises guidance", "summary": "", "publisher": "X",
                 "published": (self.now - pd.Timedelta(hours=2)).isoformat(), "ticker": ticker}]

    def funding_events(self, inst):
        idx = pd.date_range(self.now - pd.Timedelta(days=5), self.now, freq="h")
        return pd.Series(0.00001, index=idx)

    def live_quotes(self, resolved, option_chains, fallback_eurusd=None):
        def yq(sym):
            key = "EURUSD" if sym == "EURUSD=X" else sym
            if not self.store.has(key):
                return None
            mid = self._last(key)
            return {"mid": mid, "bid": mid * 0.9998, "ask": mid * 1.0002, "ts": self.now.isoformat(), "source": "fake"}

        def cq(ex, sym):
            return yq(CCXT_MAP[sym])
        return LiveQuotes(resolved, option_chains, yahoo_fn=yq, ccxt_fn=cq, fallback_eurusd=fallback_eurusd)

    def option_chains(self, spot, rate):
        m = SyntheticOptionModel(self.store.close("XSP"), self.store.close("VIX"), self.store.close("TBILL"),
                                 ModelParams())
        return FakeChains(m, self.now.tz_localize(None).normalize())


@pytest.fixture(scope="module")
def paper_store():
    return make_store(start="2023-01-01", end="2026-11-30", n_stocks=40, seed=5)


def _runner(settings, instruments, venues, tmp_path, store, extras, now, open_fn=lambda c, n: True, shock=None, agents=None):
    return PaperRunner(settings, instruments, venues, state_dir=tmp_path, now=pd.Timestamp(now),
                       data=FakeData(store, extras, pd.Timestamp(now), shock), market_open_fn=open_fn, agents=agents)


def test_first_run_opens_accounts_and_trades(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z").run()
    assert not s["errors"], s["errors"]
    assert s["agents"]["bench_sp500"]["fills"], "benchmark should buy on the first open run"
    assert s["agents"]["bench_btc"]["fills"]
    assert set(s["agents"]) == {a for a, sp in settings.agents.items() if sp.paper}
    for aid, a in s["agents"].items():
        assert 900 < a["equity_eur"] < 1010, (aid, a["equity_eur"])
    led = json.loads((tmp_path / "ledgers" / "bench_sp500.json").read_text())
    assert led["positions"]
    eq = pd.read_csv(tmp_path / "equity.csv")
    assert len(eq) == len(s["agents"])


def test_second_run_same_day_does_not_double_trade(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z").run()
    s2 = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:45:00Z").run()
    assert not s2["errors"]
    for aid, a in s2["agents"].items():
        assert not a["fills"], f"{aid} traded twice on the same day: {a['fills']}"


def test_closed_market_means_no_equity_trades(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    open_fn = lambda cal, now: cal == "CRYPTO"
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-11T15:10:00Z", open_fn).run()
    assert not s["agents"]["bench_sp500"]["fills"]
    assert not s["agents"]["trend_etf"]["decided"]
    assert s["agents"]["bench_btc"]["fills"]


def test_kill_switch_flattens_and_halts(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z").run()
    led = json.loads((tmp_path / "ledgers" / "bench_btc.json").read_text())
    assert led["positions"]
    # force crypto_trend into a ~EUR 900 BTC position, then crash BTC/ETH by 60%
    from quantdesk.ledger import Ledger, FxRates
    from quantdesk.registry import Registry
    reg = Registry(instruments)
    ct = Ledger.from_dict(json.loads((tmp_path / "ledgers" / "crypto_trend.json").read_text()))
    px = float(store.close("BTC_PERP").loc[:"2026-10-12"].iloc[-1])
    target = 900 * 1.1 / px                                   # ~EUR 900 notional long
    ct.trade("2026-10-12T15:10:00+00:00", reg.get("BTC_PERP"), target - ct.qty("BTC_PERP"), px, {}, FxRates(1.1), px)
    (tmp_path / "ledgers" / "crypto_trend.json").write_text(json.dumps(ct.to_dict()))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-13T15:10:00Z",
                shock={"BTC_PERP": 0.4, "ETH_PERP": 0.4}).run()
    a = s["agents"]["crypto_trend"]
    assert a["halted"] and a["risk"]["action"] == "halt"
    assert not a["positions"], "kill switch must flatten"
    s2 = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-14T15:10:00Z").run()
    assert s2["agents"]["crypto_trend"]["halted"] and not s2["agents"]["crypto_trend"]["fills"]


def test_llm_agent_charges_inference(tmp_path, settings, instruments, venues, paper_store, monkeypatch):
    store, extras = paper_store
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    from quantdesk import llm

    def fake_classify(self, items):
        labels = [{"id": it["id"], "ticker": it["ticker"], "event_type": "guidance", "direction": 1,
                   "materiality": 3, "confidence": 0.8, "rationale": "x"} for it in items]
        return labels, {"input_tokens": 5000, "output_tokens": 1000, "cost_usd": 0.001, "model": "fake",
                        "prompt_version": "t"}
    monkeypatch.setattr(llm.NewsClassifier, "classify", fake_classify)
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z", agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not s["errors"], s["errors"]
    assert a["llm_calls"] >= 1
    assert a["costs_eur"]["inference"] > 0
    assert 0 < len(a["fills"]) <= 3
    assert (tmp_path / "llm").exists()


def test_paper_period_ends(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    settings.paper["length_days"] = 2
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z").run()
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-16T15:10:00Z").run()
    assert s["paper"]["status"] == "finished"
    assert not any(a.get("decided") for a in s["agents"].values())


def test_dashboard_renders(tmp_path, settings, instruments, venues, paper_store):
    from quantdesk.reporting import write_dashboard
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path / "state", store, extras, "2026-10-12T15:10:00Z").run()
    write_dashboard(tmp_path / "state", settings, tmp_path / "docs", tmp_path / "reports")
    html = (tmp_path / "docs" / "index.html").read_text()
    assert "Scoreboard" in html and "lineChart" in html
    assert (tmp_path / "reports" / "STATUS.md").read_text().startswith("# Paper run status")


def test_daily_loss_blocks_new_entries(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z").run()
    from quantdesk.ledger import Ledger, FxRates
    from quantdesk.registry import Registry
    reg = Registry(instruments)
    ct = Ledger.from_dict(json.loads((tmp_path / "ledgers" / "crypto_trend.json").read_text()))
    px = float(store.close("BTC_PERP").loc[:"2026-10-12"].iloc[-1])
    ct.trade("2026-10-12T16:00:00+00:00", reg.get("BTC_PERP"), 450 * 1.1 / px - ct.qty("BTC_PERP"), px, {}, FxRates(1.1), px)
    ct.meta["last_equity"] = {"date": "2026-10-12", "equity": 1000.0}
    (tmp_path / "ledgers" / "crypto_trend.json").write_text(json.dumps(ct.to_dict()))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-13T15:10:00Z",
                shock={"BTC_PERP": 0.6, "ETH_PERP": 0.6}).run()
    a = s["agents"]["crypto_trend"]
    assert a["risk"]["action"] == "block_entries"
    assert all(f["qty"] < 0 for f in a["fills"]), "no buys allowed after the daily loss limit"
