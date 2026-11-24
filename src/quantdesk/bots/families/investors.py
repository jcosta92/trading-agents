"""Warren Buffett's strategies and other famous investors (added 24 Nov 2026, replayed from 12 Oct).

Four kinds of rule, all traded by EtfWeightBot (orders 5 minutes before the US close):

  inv_fixed_mix        a fixed allocation (lazy portfolios, Buffett's 90/10, single-ETF holds), bought at
                       the start and rebalanced yearly / quarterly
  inv_dated_portfolio  dated target portfolios frozen in config/investors/*.json: 13F clones (each
                       filing becomes tradable the session after its SEC filing date) and fundamental
                       stock screens computed only from data public before the start
  inv_xs_rank          price-only stock anomalies on the S&P 500, ranked at each month-end
  timing rules         Buffett's 'greedy when others are fearful', the Buffett indicator, and the
                       Berkshire buyback threshold (1.2 x book)

Every weight function returns an empty frame on days the bot should not trade (schedule: calendar),
so the bots hold their positions between rebalances instead of trading drift.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ...config import CONFIG_DIR
from ..generic import weights

SESSIONS_LEAD = pd.Timedelta(days=40)     # runner._sessions(t0, now) starts its calendar 40 days before t0


@lru_cache(maxsize=None)
def _book(name: str) -> dict:
    return json.loads((CONFIG_DIR / "investors" / name).read_text())


def _session_on_or_after(sessions, date) -> pd.Timestamp | None:
    idx = sessions.index
    later = idx[idx >= pd.Timestamp(date).normalize()]
    return later[0] if len(later) else None


def _row(panel, w: dict) -> pd.DataFrame:
    """A one-row weight frame for the panel's last date (an empty frame means: do not trade today)."""
    if not len(panel.index):
        return pd.DataFrame()
    s = pd.Series({k: float(v) for k, v in w.items() if k in panel.names and float(v) > 0}, dtype=float)
    return pd.DataFrame([s.reindex(panel.names).fillna(0.0)], index=[panel.index[-1]])


def _none() -> pd.DataFrame:
    return pd.DataFrame()


def _start_session(extra, p) -> pd.Timestamp | None:
    """The bot's first session: `start` if given, else the test start, recovered from the session
    calendar the runner hands the bot (it begins 40 days before the start)."""
    sess = extra["sessions"]
    if p.get("start"):
        return _session_on_or_after(sess, p["start"])
    return _session_on_or_after(sess, sess.index[0] + SESSIONS_LEAD) if len(sess.index) else None


def _is_start(extra, p) -> bool:
    return pd.Timestamp(extra["day"]) == _start_session(extra, p)


# ============================================================================================ fixed mixes
@weights("inv_fixed_mix")
def inv_fixed_mix(panel, p, extra):
    """A fixed allocation `mix`, bought on the first session of the test and rebalanced on the first
    session of each year (rebalance: yearly), of each quarter (quarterly) or never."""
    day = pd.Timestamp(extra["day"])
    nth, _left = extra["pos"]
    reb = p.get("rebalance", "yearly")
    due = _is_start(extra, p) or (nth == 1 and ((reb == "yearly" and day.month == 1)
                                                 or (reb == "quarterly" and day.month in (1, 4, 7, 10))))
    return _row(panel, p["mix"]) if due else _none()


# ============================================================================================ dated books
def _entries(p) -> list[dict]:
    book = _book(p["file"])
    ent = book[p["key"]] if "key" in p else book
    if isinstance(ent, dict):
        ent = ent["portfolios"]
    return sorted(ent, key=lambda e: e["effective"])


@weights("inv_dated_portfolio")
def inv_dated_portfolio(panel, p, extra):
    """Target portfolios with an `effective` date each (13F filings: the day after the SEC filing
    date; screens: the start). The latest portfolio effective by today is bought on the first session
    on or after its effective date (or the start, if later) and then held until the next one."""
    day = pd.Timestamp(extra["day"])
    sess = extra["sessions"]
    start = _start_session(extra, p)
    active, due = None, False
    for e in _entries(p):
        s = _session_on_or_after(sess, e["effective"])
        if s is None:
            continue
        s = max(s, start) if start is not None else s
        if s <= day:
            active, due = e, (s == day)
    if active is None or not due:
        return _none()
    return _row(panel, active["weights"])


# ============================================================================================ stock anomalies
def _score(c: pd.DataFrame, hi: pd.DataFrame, metric: str, bench: str) -> pd.Series:
    if metric == "mom12_1":                     # Jegadeesh & Titman: 12-month return skipping the last month
        return c.iloc[-22] / c.iloc[-253] - 1 if len(c) >= 253 else pd.Series(dtype=float)
    if metric == "high52":                      # George & Hwang: nearness to the 52-week high
        return c.iloc[-1] / hi.iloc[-252:].max() if len(c) >= 252 else pd.Series(dtype=float)
    if metric == "rev1m":                       # Jegadeesh 1990: last month's losers
        return -(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else pd.Series(dtype=float)
    r = c.pct_change(fill_method=None).iloc[-252:]
    if len(r) < 200:
        return pd.Series(dtype=float)
    if metric == "lowvol":                      # Ang, Hodrick, Xing & Zhang; Blitz & van Vliet: low volatility
        return -r.std()
    if metric == "lowbeta":                     # Frazzini & Pedersen: betting against beta (long leg)
        m = r[bench]
        beta = r.apply(lambda s: s.cov(m)) / m.var()
        return -(0.6 * beta + 0.4)
    raise ValueError(metric)


@weights("inv_xs_rank")
def inv_xs_rank(panel, p, extra):
    """Rank the stock universe on `metric` at each month-end (and on the first session of the test)
    and hold the best `n` stocks equally weighted. A stock needs a full year of prices to be ranked.
    Only the universe's own trading days count (other tickers in the panel may add dates)."""
    _nth, left = extra["pos"]
    if not (_is_start(extra, p) or left == 0):
        return _none()
    bench = p.get("bench", "SPY")
    cols = [k for k in dict.fromkeys(list(p["universe"]) + [bench]) if k in panel.names]
    c, hi = panel.close[cols], panel.high[cols]
    keep = c.notna().mean(axis=1) > 0.5
    c, hi = c[keep], hi[keep]
    if not len(c):
        return _none()
    sc = _score(c, hi, p["metric"], bench)
    uni = set(p["universe"])
    sc = sc[[k for k in sc.index if k in uni]].replace([np.inf, -np.inf], np.nan).dropna()
    full = c.iloc[-253:].notna().all()
    sc = sc[[k for k in sc.index if bool(full.get(k, False))]]
    if sc.empty:
        return _none()
    pick = list(sc.sort_values(ascending=False).index[:int(p.get("n", 50))])
    return _row(panel, {k: 1.0 / len(pick) for k in pick})


# ============================================================================================ timing rules
@weights("inv_greedy_fearful")
def inv_greedy_fearful(panel, p, extra):
    """'Be fearful when others are greedy and greedy when others are fearful' (Buffett, 1986 letter;
    'Buy American. I Am.', NYT 2008): a base stock / T-bill mix, all in stocks while fear is high:
    VIX close above `vix` (mode vix), or the S&P 500 at least `dd` below its 52-week closing high
    (mode drawdown)."""
    c = panel.close
    stock, cash = p.get("stock", "SPY"), p.get("cash", "SHY")
    if stock not in c or not len(c):
        return _none()
    if p.get("mode", "vix") == "vix":
        v = c["^VIX"].dropna() if "^VIX" in c else pd.Series(dtype=float)
        if v.empty:
            return _none()
        fear = float(v.iloc[-1]) > float(p.get("vix", 30))
    else:
        s = c[stock].dropna()
        fear = len(s) > 20 and float(s.iloc[-1] / s.iloc[-252:].max() - 1) <= -float(p.get("dd", 0.2))
    base = float(p.get("base", 0.6))
    return _row(panel, {stock: 1.0} if fear else {stock: base, cash: 1 - base})


@weights("inv_buffett_indicator")
def inv_buffett_indicator(panel, p, extra):
    """The Buffett indicator (US stock market value / GDP), from a published reading (`ref_ratio` on
    `ref_date`) carried forward with the total-market ETF's price. Buffett (Fortune, 2001): around 70-80%
    buying stocks works very well; approaching 200% 'you are playing with fire'. Hold the total market
    (VTI) while the ratio is below `max_ratio`, T-bills (SHY) at or above it."""
    c = panel.close
    m = p.get("market", "VTI")
    if m not in c:
        return _none()
    s = c[m].dropna()
    base = s.loc[:pd.Timestamp(p["ref_date"])]
    if base.empty or s.empty:
        return _none()
    ratio = float(p["ref_ratio"]) * float(s.iloc[-1]) / float(base.iloc[-1])
    return _row(panel, {m: 1.0} if ratio < float(p.get("max_ratio", 2.0)) else {p.get("cash", "SHY"): 1.0})


@weights("inv_brk_book")
def inv_brk_book(panel, p, extra):
    """Berkshire's buyback threshold (2011-2018: repurchases at up to 110%, then 120% of book value,
    where Buffett said the shares were clearly undervalued): hold BRK-B while its price is at most
    `max_pb` x the last reported book value per class-B share, otherwise the S&P 500 index fund Buffett
    recommends (VOO). Book values become usable the session after their 10-Q / 10-K."""
    c = panel.close
    if "BRK-B" not in c or not len(c):
        return _none()
    day = pd.Timestamp(extra["day"])
    bv = None
    for e in sorted(p["book"], key=lambda e: e["effective"]):
        if pd.Timestamp(e["effective"]) <= day:
            bv = float(e["bvps_b"])
    px = c["BRK-B"].dropna()
    if bv is None or px.empty:
        return _none()
    return _row(panel, {"BRK-B": 1.0} if float(px.iloc[-1]) <= float(p.get("max_pb", 1.2)) * bv else
                {p.get("index", "VOO"): 1.0})
