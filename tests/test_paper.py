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
    SPLITS: dict = {}                                        # inst id -> pd.Series of ratios (tests set it)

    def __init__(self, store, extras, now, shock=None):
        self.store, self.extras, self.now = store, extras, pd.Timestamp(now)
        self.shock = shock or {}
        self.news_calls = 0

    def splits(self, inst, symbol=None):
        return self.SPLITS.get(inst.id if inst is not None else symbol, pd.Series(dtype="float64"))

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

    def dividends(self, inst, symbol=None):
        if inst is not None and inst.id == "EUR_GOVT":
            return pd.Series([0.5], index=[pd.Timestamp("2026-10-20")])
        return pd.Series(dtype="float64")

    def live_quotes(self, resolved, option_chains, fallback_eurusd=None, reference=None):
        from quantdesk.config import load_instruments
        scale = {k: v.price_scale for k, v in load_instruments().items()}

        def yq(sym):
            key = "EURUSD" if sym == "EURUSD=X" else sym
            if not self.store.has(key):
                return None
            # the store holds scaled prices (XSP = S&P 500 / 10); a live feed returns the raw symbol price
            mid = self._last(key) / (scale.get(key) or 1.0)
            return {"mid": mid, "bid": mid * 0.9998, "ask": mid * 1.0002, "ts": self.now.isoformat(), "source": "fake"}

        def cq(ex, sym):
            return yq(CCXT_MAP[sym])
        return LiveQuotes(resolved, option_chains, yahoo_fn=yq, ccxt_fn=cq, fallback_eurusd=fallback_eurusd,
                          reference=reference)

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
    target = 950 * 1.1 / px                                   # ~EUR 950 notional long
    ct.trade("2026-10-12T15:10:00+00:00", reg.get("BTC_PERP"), target - ct.qty("BTC_PERP"), px, {}, FxRates(1.1), px)
    (tmp_path / "ledgers" / "crypto_trend.json").write_text(json.dumps(ct.to_dict()))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-13T15:10:00Z",
                shock={"BTC_PERP": 0.4, "ETH_PERP": 0.4}).run()
    a = s["agents"]["crypto_trend"]
    assert a["halted"] and a["risk"]["action"] == "halt"
    assert not a["positions"], "kill switch must flatten"
    s2 = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-14T15:10:00Z").run()
    assert s2["agents"]["crypto_trend"]["halted"] and not s2["agents"]["crypto_trend"]["fills"]


def _write_labels(state_dir, data, date, now, tickers, materiality=3, direction=1):
    from quantdesk import llm
    inbox = llm.build_inbox(tickers, data.news, pd.Timestamp(now))
    llm.inbox_path(state_dir, date).parent.mkdir(parents=True, exist_ok=True)
    llm.inbox_path(state_dir, date).write_text(json.dumps(inbox))
    labels = {"date": date, "prompt_version": llm.PROMPT_VERSION, "labelled_by": "test",
              "labels": [{"id": it["id"], "ticker": it["ticker"], "event_type": "guidance", "direction": direction,
                          "materiality": materiality, "confidence": 0.8, "rationale": "x"} for it in inbox["items"]]}
    llm.labels_path(state_dir, date).parent.mkdir(parents=True, exist_ok=True)
    llm.labels_path(state_dir, date).write_text(json.dumps(labels))
    return inbox, labels


def test_llm_agent_waits_for_labels_then_trades_with_protection(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    now = "2026-10-12T15:10:00Z"
    s = _runner(settings, instruments, venues, tmp_path, store, extras, now, agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not s["errors"], s["errors"]
    assert not a["fills"] and "no headline inbox" in a["llm_status"]
    led = json.loads((tmp_path / "ledgers" / "llm_news.json").read_text())
    assert led["meta"].get("last_decision") is None, "no labels: the day's decision must be retried next run"

    tickers = sorted(extras["sectors"])[:12]
    _write_labels(tmp_path, FakeData(store, extras, pd.Timestamp(now)), "2026-10-12", now, tickers)
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T16:10:00Z", agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not s["errors"], s["errors"]
    assert a["llm_status"].startswith("12 labels")
    assert 0 < len(a["fills"]) <= 3
    assert all(f["reason"].startswith("BUY signal: guidance news") for f in a["fills"])
    led = json.loads((tmp_path / "ledgers" / "llm_news.json").read_text())
    for f in a["fills"]:
        pr = led["meta"]["protect"][f["inst"]]
        assert pr["stop"] == pytest.approx(f["price"] * 0.93)
        assert pr["target"] == pytest.approx(f["price"] * 1.10)
        assert pr["spec"]["max_days"] == 5
    assert {r["inst"] for r in a["resting"]} == {f["inst"] for f in a["fills"]}


def test_invalid_labels_are_not_traded(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    now = "2026-10-12T15:10:00Z"
    tickers = sorted(extras["sectors"])[:5]
    inbox, labels = _write_labels(tmp_path, FakeData(store, extras, pd.Timestamp(now)), "2026-10-12", now, tickers)
    from quantdesk import llm
    labels["labels"] = labels["labels"][:-1]                  # one headline left unlabelled
    llm.labels_path(tmp_path, "2026-10-12").write_text(json.dumps(labels))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, now, agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not a["fills"] and a["llm_status"].startswith("labels file invalid")


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
                shock={"BTC_PERP": 0.7, "ETH_PERP": 0.7}).run()
    a = s["agents"]["crypto_trend"]
    assert a["risk"]["action"] == "block_entries"
    assert all(f["qty"] < 0 for f in a["fills"]), "no buys allowed after the daily loss limit"


def test_kill_switch_reissued_until_flat(tmp_path, settings, instruments, venues, paper_store, monkeypatch):
    """If the flatten order cannot fill (no quote), the next run must try again."""
    store, extras = paper_store
    from quantdesk.ledger import Ledger, FxRates
    from quantdesk.registry import Registry
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z", agents=["crypto_trend"]).run()
    reg = Registry(instruments)
    ct = Ledger.from_dict(json.loads((tmp_path / "ledgers" / "crypto_trend.json").read_text()))
    px = float(store.close("BTC_PERP").loc[:"2026-10-12"].iloc[-1])
    ct.trade("2026-10-12T16:00:00+00:00", reg.get("BTC_PERP"), 950 * 1.1 / px - ct.qty("BTC_PERP"), px, {}, FxRates(1.1), px)
    ct.halted, ct.halt_reason = True, "test"
    (tmp_path / "ledgers" / "crypto_trend.json").write_text(json.dumps(ct.to_dict()))
    r = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-13T15:10:00Z", agents=["crypto_trend"])
    orig = FakeData.live_quotes

    def no_btc(self, *a, **k):
        lq = orig(self, *a, **k)
        inner = lq._ccxt
        lq._ccxt = lambda ex, sym: None if sym.startswith("BTC") else inner(ex, sym)
        return lq
    monkeypatch.setattr(FakeData, "live_quotes", no_btc)
    s1 = r.run()
    assert s1["agents"]["crypto_trend"]["positions"], "flatten could not fill without a quote"
    monkeypatch.setattr(FakeData, "live_quotes", orig)
    s2 = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-14T15:10:00Z", agents=["crypto_trend"]).run()
    assert not s2["agents"]["crypto_trend"]["positions"], "kill switch must re-issue the flatten"


def test_bad_yahoo_print_is_ignored(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z", agents=["bench_sp500"]).run()
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-13T15:10:00Z",
                shock={"SP500": 0.05}, agents=["bench_sp500"]).run()     # a -95% Yahoo print
    a = s["agents"]["bench_sp500"]
    assert not a["halted"] and a["equity_eur"] > 900


def test_real_crypto_crash_is_not_filtered(tmp_path, settings, instruments, venues, paper_store):
    """Exchange (ccxt) quotes are trusted: a 60% crash must reach the books."""
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z", agents=["bench_btc"]).run()
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-13T15:10:00Z",
                shock={"BTC_SPOT": 0.4}, agents=["bench_btc"]).run()
    assert s["agents"]["bench_btc"]["equity_eur"] < 500


def _give_position(tmp_path, aid, instruments, iid, qty, px, ts="2026-10-12T15:10:00+00:00", protect_spec=None):
    from quantdesk import protect
    from quantdesk.execution import Fill
    from quantdesk.ledger import FxRates, Ledger
    from quantdesk.registry import Registry
    p = tmp_path / "ledgers" / f"{aid}.json"
    led = Ledger.from_dict(json.loads(p.read_text())) if p.exists() else Ledger(aid, 1000.0, ts=ts)
    led.trade(ts, Registry(instruments).get(iid), qty, px, {}, FxRates(1.1), px)
    if protect_spec:
        protect.register(led, [Fill(ts, aid, iid, qty, px, px, 0.0, "EUR", qty * px, "test", None, protect_spec)])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(led.to_dict()))
    return led


def test_dividends_credited_once(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    px = float(store.close("EUR_GOVT").loc[:"2026-10-12"].iloc[-1])
    _give_position(tmp_path, "trend_etf", instruments, "EUR_GOVT", 2, px)
    for d in ("2026-10-12T15:10", "2026-10-21T15:10", "2026-10-21T16:10", "2026-10-22T15:10"):
        _runner(settings, instruments, venues, tmp_path, store, extras, f"{d}:00Z", agents=["trend_etf"]).run()
    led = json.loads((tmp_path / "ledgers" / "trend_etf.json").read_text())
    incomes = [e for e in led["journal"] if e["type"] == "income"]
    if "EUR_GOVT" in led["positions"]:
        assert len(incomes) == 1
    else:                                       # sold by a rule before the ex-date
        assert len(incomes) <= 1


def test_hourly_stop_loss_fires_and_is_not_doubled(tmp_path, settings, instruments, venues, paper_store):
    """A resting stop is checked on every hourly run; when the live price falls through it the position
    is sold once, at the bid, with a STOP-LOSS reason, and the protection is removed."""
    store, extras = paper_store
    px = float(store.close("GOLD").loc[:"2026-10-12"].iloc[-1])
    led = _give_position(tmp_path, "trend_etf", instruments, "GOLD", 3, px,
                         protect_spec={"stop_pct": 0.03, "label": "test stop"})
    assert led.meta["protect"]["GOLD"]["stop"] == pytest.approx(px * 0.97)
    led.meta["last_decision"] = "2026-10-11"            # today's indicator decision already made
    (tmp_path / "ledgers" / "trend_etf.json").write_text(json.dumps(led.to_dict()))
    # an hour later, price unchanged: nothing happens, the stop is shown as resting
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T16:10:00Z", agents=["trend_etf"]).run()
    assert not s["errors"], s["errors"]
    assert not s["agents"]["trend_etf"]["fills"]
    assert [r["inst"] for r in s["agents"]["trend_etf"]["resting"]] == ["GOLD"]
    # next hour GOLD trades 5% lower: the stop is hit
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T17:10:00Z",
                shock={"GOLD": 0.95}, agents=["trend_etf"]).run()
    fills = s["agents"]["trend_etf"]["fills"]
    assert len(fills) == 1 and fills[0]["qty"] == pytest.approx(-3)
    assert fills[0]["reason"].startswith("STOP-LOSS hit")
    assert fills[0]["price"] < px * 0.95                       # sold at the bid, below the touched price
    led = json.loads((tmp_path / "ledgers" / "trend_etf.json").read_text())
    assert "GOLD" not in led["positions"] and "GOLD" not in led["meta"].get("protect", {})
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T18:10:00Z",
                shock={"GOLD": 0.95}, agents=["trend_etf"]).run()
    assert not s["agents"]["trend_etf"]["fills"], "a stop must not fire twice"


def test_protection_ignored_while_market_closed(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    px = float(store.close("GOLD").loc[:"2026-10-12"].iloc[-1])
    _give_position(tmp_path, "trend_etf", instruments, "GOLD", 3, px,
                   protect_spec={"stop_pct": 0.03, "label": "test stop"})
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T22:10:00Z",
                open_fn=lambda cal, now: cal == "CRYPTO", shock={"GOLD": 0.9}, agents=["trend_etf"]).run()
    assert not s["agents"]["trend_etf"]["fills"]


def test_no_quote_order_is_retried_next_run_then_expires(tmp_path, settings, instruments, venues, paper_store, monkeypatch):
    store, extras = paper_store
    orig = FakeData.live_quotes

    def stale_sp500(self, *a, **k):
        """SP500 quote is 3 hours old: usable as a mark, but not executable."""
        lq = orig(self, *a, **k)
        inner = lq._yahoo

        def yq(sym):
            d = inner(sym)
            if sym == "SP500" and d:
                d = {**d, "ts": (self.now - pd.Timedelta(hours=3)).isoformat()}
            return d
        lq._yahoo = yq
        return lq
    monkeypatch.setattr(FakeData, "live_quotes", stale_sp500)
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T09:10:00Z", agents=["bench_sp500"]).run()
    assert not s["agents"]["bench_sp500"]["fills"] and s["agents"]["bench_sp500"]["rejects"]
    led = json.loads((tmp_path / "ledgers" / "bench_sp500.json").read_text())
    assert len(led["meta"]["pending"]) == 1
    monkeypatch.setattr(FakeData, "live_quotes", orig)
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T10:10:00Z", agents=["bench_sp500"]).run()
    assert len(s["agents"]["bench_sp500"]["fills"]) == 1, "retried order fills (BuyAndHold re-decides too, but only one fill)"
    # a retried order older than the limit is dropped
    led = json.loads((tmp_path / "ledgers" / "bench_sp500.json").read_text())
    led["meta"]["pending"] = [{"inst_id": "SP500", "qty": -1.0, "reason": "old", "combo": None,
                               "created": "2026-10-12T01:00:00+00:00", "meta": {}}]
    (tmp_path / "ledgers" / "bench_sp500.json").write_text(json.dumps(led))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T11:10:00Z", agents=["bench_sp500"]).run()
    assert not s["agents"]["bench_sp500"]["fills"]


def _llm_position_with_stop(tmp_path, store, instruments, iid="S000", stop_frac=0.93):
    px = float(store.close(iid).loc[:"2026-10-12"].iloc[-1])
    from quantdesk.registry import Registry
    Registry(instruments)
    led = _give_position(tmp_path, "llm_news", instruments, iid, 2, px, ts="2026-10-09T15:10:00+00:00",
                         protect_spec={"stop_pct": 1 - stop_frac, "target_pct": 0.10, "max_days": 5, "label": "t"})
    return led, px


@pytest.fixture()
def stock_registry(monkeypatch):
    """_give_position needs the synthetic stock S000 in the registry."""
    from quantdesk.registry import Registry
    orig_get = Registry.get

    def get(self, iid):
        i = orig_get(self, iid)
        return i if i is not None or not iid.startswith("S0") else self.stock(iid)
    monkeypatch.setattr(Registry, "get", get)


def test_unreadable_labels_do_not_block_stops(tmp_path, settings, instruments, venues, paper_store, stock_registry):
    store, extras = paper_store
    _, px = _llm_position_with_stop(tmp_path, store, instruments)
    from quantdesk import llm
    for path in (llm.inbox_path(tmp_path, "2026-10-12"), llm.labels_path(tmp_path, "2026-10-12")):
        path.parent.mkdir(parents=True, exist_ok=True)
    llm.inbox_path(tmp_path, "2026-10-12").write_text(json.dumps({"date": "2026-10-12", "items": []}))
    llm.labels_path(tmp_path, "2026-10-12").write_text('{"date": "2026-10-12", "labels": [')     # truncated
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z",
                shock={"S000": 0.8}, agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not s["errors"], s["errors"]
    assert "unreadable" in a["llm_status"]
    assert [f["reason"][:14] for f in a["fills"]] == ["STOP-LOSS hit:"]


def test_decision_crash_does_not_block_stops(tmp_path, settings, instruments, venues, paper_store, stock_registry,
                                            monkeypatch):
    store, extras = paper_store
    _llm_position_with_stop(tmp_path, store, instruments)
    from quantdesk.agents.llmnews import LLMNews

    def boom(self, runner, ledger, asof):
        raise RuntimeError("panel download failed")
    monkeypatch.setattr(LLMNews, "prepare_live", boom)
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z",
                shock={"S000": 0.8}, agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert any("decision failed" in e for e in s["errors"])
    assert [f["reason"][:14] for f in a["fills"]] == ["STOP-LOSS hit:"]
    led = json.loads((tmp_path / "ledgers" / "llm_news.json").read_text())
    assert led["journal"] and not led["positions"]


def test_retried_entry_respects_risk_filters_and_paper_end(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T09:10:00Z", agents=["trend_etf"]).run()
    led = json.loads((tmp_path / "ledgers" / "trend_etf.json").read_text())
    led["meta"]["pending"] = [{"inst_id": "GOLD", "qty": 1.0, "reason": "retried BUY", "combo": None,
                               "created": "2026-10-12T09:10:00+00:00", "meta": {"reduce_only": False}}]
    (tmp_path / "ledgers" / "trend_etf.json").write_text(json.dumps(led))
    meta = json.loads((tmp_path / "paper.json").read_text())
    meta["end"] = "2026-10-12T10:00:00+00:00"                      # the paper period ends before the retry
    (tmp_path / "paper.json").write_text(json.dumps(meta))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T10:10:00Z", agents=["trend_etf"]).run()
    assert not [f for f in s["agents"]["trend_etf"]["fills"] if f["inst"] == "GOLD"]


def test_retried_perp_exit_cannot_open_a_short(tmp_path, settings, instruments, venues, paper_store):
    store, extras = paper_store
    _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T09:10:00Z", agents=["crypto_trend"]).run()
    led = json.loads((tmp_path / "ledgers" / "crypto_trend.json").read_text())
    assert "BTC_PERP" not in led["positions"]
    led["meta"]["pending"] = [{"inst_id": "BTC_PERP", "qty": -0.01, "reason": "SELL signal (retried)", "combo": None,
                               "created": "2026-10-12T09:10:00+00:00", "meta": {"reduce_only": True}}]
    (tmp_path / "ledgers" / "crypto_trend.json").write_text(json.dumps(led))
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T10:10:00Z", agents=["crypto_trend"]).run()
    assert "BTC_PERP" not in s["agents"]["crypto_trend"]["positions"] or \
        s["agents"]["crypto_trend"]["positions"]["BTC_PERP"]["qty"] > 0


def test_split_is_applied_not_booked_as_a_loss(tmp_path, settings, instruments, venues, paper_store, stock_registry,
                                              monkeypatch):
    store, extras = paper_store
    px = float(store.close("S000").loc[:"2026-10-12"].iloc[-1])
    # bought 2 shares at 4x today's price before a 4:1 split that goes ex today
    led = _give_position(tmp_path, "llm_news", instruments, "S000", 2, 4 * px, ts="2026-10-09T15:10:00+00:00",
                         protect_spec={"stop_pct": 0.07, "target_pct": 0.10, "max_days": 5, "label": "t"})
    eq0 = led.equity(__import__("quantdesk.ledger", fromlist=["FxRates"]).FxRates(1.1), {"S000": 4 * px})
    monkeypatch.setattr(FakeData, "SPLITS", {"S000": pd.Series([4.0], index=[pd.Timestamp("2026-10-12")])})
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z", agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not s["errors"], s["errors"]
    assert not a["fills"], "a split must not trigger the stop-loss"
    assert a["positions"]["S000"]["qty"] == pytest.approx(8)
    assert a["equity_eur"] == pytest.approx(eq0, rel=0.02)
    led = json.loads((tmp_path / "ledgers" / "llm_news.json").read_text())
    assert led["meta"]["protect"]["S000"]["stop"] == pytest.approx(px * 0.93, rel=1e-6)
    # the next run does not apply it again
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T16:10:00Z", agents=["llm_news"]).run()
    assert s["agents"]["llm_news"]["positions"]["S000"]["qty"] == pytest.approx(8)


def test_unconfirmed_split_freezes_mark_and_stops(tmp_path, settings, instruments, venues, paper_store, stock_registry):
    store, extras = paper_store
    px = float(store.close("S000").loc[:"2026-10-12"].iloc[-1])
    led = _give_position(tmp_path, "llm_news", instruments, "S000", 2, 4 * px, ts="2026-10-09T15:10:00+00:00",
                         protect_spec={"stop_pct": 0.07, "target_pct": 0.10, "max_days": 5, "label": "t"})
    s = _runner(settings, instruments, venues, tmp_path, store, extras, "2026-10-12T15:10:00Z", agents=["llm_news"]).run()
    a = s["agents"]["llm_news"]
    assert not a["fills"] and not a["halted"]
    assert any("not yet in the data feed" in n for n in a["notes"])
    assert a["positions"]["S000"]["mark"] == pytest.approx(4 * px)
