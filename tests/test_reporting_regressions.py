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


def test_yahoo_junk_book_is_ignored():
    """First paper cycle: APA ask 48.78 vs bid 43.57 (last ~48.7) made the fill 5.6% above mid."""
    from quantdesk.data.yahoo import usable_book
    assert not usable_book(43.57, 48.78, 48.70)          # 11% wide
    assert not usable_book(100.0, 100.2, 101.0)          # last outside the book
    assert usable_book(100.0, 100.04, 100.02)
    assert not usable_book(None, 100.0, 100.0)


def test_live_quote_falls_back_to_last_on_junk_book(monkeypatch):
    from quantdesk.data import yahoo

    class T:
        fast_info = {"last_price": 48.70}
        info = {"bid": 43.57, "ask": 48.78, "regularMarketTime": 1791817056}

    class YF:
        @staticmethod
        def Ticker(sym):
            return T()
    monkeypatch.setattr(yahoo, "_yf", lambda: YF)
    q = yahoo.live_quote("APA")
    assert q["bid"] is None and q["ask"] is None and q["mid"] == 48.70


def test_rss_news_fallback_parses_items():
    from quantdesk.data import yahoo
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>Yahoo</title>
    <item><title>Acme beats estimates, raises guidance</title><link>https://x/1</link><guid>g1</guid>
    <description>Acme reported...</description><pubDate>Mon, 12 Oct 2026 11:30:00 +0000</pubDate></item>
    <item><title>No date item</title><guid>g2</guid></item></channel></rss>"""
    out = yahoo.news_rss("ACME", fetch=lambda s: xml)
    assert len(out) == 1
    assert out[0]["id"] == "g1" and out[0]["published"].startswith("2026-10-12T11:30:00")


def test_news_inbox_fails_loudly_when_no_headlines(tmp_path, monkeypatch):
    import json as _json
    from quantdesk import cli
    from quantdesk.data import yahoo
    monkeypatch.setattr(yahoo, "news", lambda t: [])
    uni = tmp_path / "llm" / "universe.json"
    uni.parent.mkdir(parents=True)
    uni.write_text(_json.dumps({"month": "2026-10", "tickers": ["AAA", "BBB"]}))
    rc = cli.main(["news-inbox", "--state", str(tmp_path), "--now", "2026-10-12T12:13:00Z"])
    assert rc == 1 and not (tmp_path / "llm" / "inbox" / "2026-10-12.json").exists()
