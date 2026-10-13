"""Families from Reddit / social media / quant blogs (13 Oct 2026): the calendar rules hit the
published dates, and the pattern rules fire exactly where their sources say."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.bots.families import social
from quantdesk.bots.generic import Panel

FOMC = ["2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09", "2027-01-27"]


def _sessions(start, end, holidays=()):
    days = [d for d in pd.bdate_range(start, end) if d.strftime("%Y-%m-%d") not in holidays]
    return pd.DataFrame({"open": 0, "close": 0}, index=pd.DatetimeIndex(days))


SESS = _sessions("2026-10-01", "2027-01-15", {"2026-11-26", "2026-12-25", "2027-01-01"})


def test_fomc_even_weeks_match_the_paper_schedule():
    held = [d for d in SESS.index if "2026-10-12" < d.strftime("%Y-%m-%d") <= "2026-12-31"
            and social.fomc_even_week(d, FOMC)]
    got = [d.strftime("%m-%d") for d in held]
    want = ["10-13", "10-14", "10-15", "10-16", "10-19",            # days 19-23 after 16 Sep
            "10-27", "10-28", "10-29", "10-30", "11-02",            # days -1..3 around 28 Oct
            "11-10", "11-11", "11-12", "11-13", "11-16",            # days 9-13
            "11-24", "11-25", "11-27", "11-30",                     # days 19-23 (26 Nov is a holiday)
            "12-08", "12-09", "12-10", "12-11", "12-14",            # days -1..3 around 9 Dec
            "12-22", "12-23", "12-24", "12-28"]                     # days 9-13
    assert got == want


def _one(fn, day, p, tickers=("SPY",)):
    idx = pd.bdate_range(end=day, periods=30)
    panel = Panel({t: pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "volume")}, index=idx)
                   for t in tickers})
    w = fn(panel, p, {"day": day, "sessions": SESS, "pos": (5, 5)})
    return w.iloc[-1][w.iloc[-1] > 0].to_dict()


def test_calendar_rules_hold_on_the_right_days():
    p = {"asset": "SPY"}
    assert _one(social.halloween, "2026-10-29", p) == {}
    assert _one(social.halloween, "2026-10-30", p) == {"SPY": 1.0}       # next session is 2 Nov
    assert _one(social.santa_claus_rally, "2026-12-22", p) == {}
    assert _one(social.santa_claus_rally, "2026-12-23", p) == {"SPY": 1.0}  # last 5: 24, 28, 29, 30, 31 Dec
    assert _one(social.santa_claus_rally, "2027-01-04", p) == {"SPY": 1.0}  # 5 Jan is the 2nd January session
    assert _one(social.santa_claus_rally, "2027-01-05", p) == {}
    assert _one(social.payday_16th, "2026-11-13", p) == {"SPY": 1.0}       # next session is Monday 16 Nov
    assert _one(social.payday_16th, "2026-11-16", p) == {}


def test_sector_seasonality_ranks_the_month_eleven_before_the_holding_month():
    idx = pd.bdate_range("2025-01-01", "2026-10-30")
    frames = {}
    for t, best in (("XLK", "2025-11"), ("XLE", "2025-12"), ("XLU", None)):
        px = pd.Series(100.0, index=idx)
        if best:
            m = idx.to_period("M") == pd.Period(best)
            px[m] = 100 + np.arange(m.sum()) * 1.0       # rises through that month only
            px[idx.to_period("M") > pd.Period(best)] = px[m].iloc[-1]
        frames[t] = pd.DataFrame({c: px for c in ("open", "high", "low", "close", "volume")})
    panel = Panel(frames)
    end = social.sector_seasonality(panel, {"k": 1}, {"pos": (22, 0)}).iloc[-1]     # 30 Oct: holding Nov
    assert end.idxmax() == "XLE"                                                  # Nov - 11 = Dec 2025
    start = social.sector_seasonality(panel, {"k": 1}, {"pos": (9, 13)}).iloc[-1]   # mid-October: holding Oct
    assert start.idxmax() == "XLK"                                                # Oct - 11 = Nov 2025


def test_spy_band_has_hysteresis_and_deleverages():
    idx = pd.bdate_range("2025-01-01", periods=400)
    spy = pd.Series(100.0, index=idx)
    spy.iloc[250:300] = 106.0      # +6% over the 200-day average -> TQQQ
    spy.iloc[300:] = 101.0         # inside the band -> keep TQQQ
    qqq = pd.Series(100.0, index=idx)
    mk = lambda s: pd.DataFrame({c: s for c in ("open", "high", "low", "close", "volume")})
    w = social.spy_band_tqqq(Panel({"SPY": mk(spy), "QQQ": mk(qqq), "TQQQ": mk(qqq)}), {}, {}).iloc[-1]
    assert w["TQQQ"] == 1.0
    spy2 = spy.copy()
    spy2.iloc[350:] = 95.0         # 3% under -> QQQ
    w = social.spy_band_tqqq(Panel({"SPY": mk(spy2), "QQQ": mk(qqq), "TQQQ": mk(qqq)}), {}, {}).iloc[-1]
    assert w["QQQ"] == 1.0
    hot = qqq.copy()
    hot.iloc[-5:] = 140.0          # QQQ 30%+ above its own average -> QQQ instead of TQQQ
    w = social.spy_band_tqqq(Panel({"SPY": mk(spy), "QQQ": mk(hot), "TQQQ": mk(hot)}), {}, {}).iloc[-1]
    assert w["QQQ"] == 1.0


def _bars_ny(rows):
    """rows: list of (NY timestamp string, open, high, low, close) -> UTC-indexed 5-minute frame."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz="America/New_York") for r in rows]).tz_convert("UTC")
    return pd.DataFrame([r[1:] + (1.0,) for r in rows], columns=["open", "high", "low", "close", "volume"],
                        index=idx)


def test_ict_silver_bullet_needs_a_sweep_then_takes_the_first_gap_in_the_window():
    rows = []
    for t in pd.date_range("2026-10-13 18:00", "2026-10-14 16:00", freq="5min", inclusive="left"):
        rows.append([t.strftime("%Y-%m-%d %H:%M"), 100.0, 100.1, 99.9, 100.0])
    df = _bars_ny([tuple(r) for r in rows])
    ny = df.index.tz_convert("America/New_York").strftime("%H:%M")
    day14 = df.index.tz_convert("America/New_York").day == 14

    def setbar(hhmm, o, h, lo, c):
        df.loc[(ny == hhmm) & day14, ["open", "high", "low", "close"]] = [o, h, lo, c]
    setbar("03:00", 100.0, 100.1, 99.5, 99.8)           # London low 99.5
    setbar("09:40", 99.6, 100.1, 99.65, 100.0)          # a gap before the window does not count
    setbar("10:05", 99.9, 99.95, 99.4, 99.6)            # sweeps the London and Asian lows
    setbar("10:10", 99.6, 99.8, 99.3, 99.7)             # candle 1
    setbar("10:15", 99.7, 100.6, 99.7, 100.5)           # candle 2
    setbar("10:20", 100.5, 100.8, 100.0, 100.7)         # candle 3: low 100.0 > candle-1 high 99.8
    setbar("10:30", 100.7, 101.2, 100.9, 101.0)         # a later gap in the same window: ignored
    out = social.ict_silver_bullet(df, {}, {})
    hits = out[out["entry"]]
    assert len(hits) == 1
    hit = hits.index[0].tz_convert("America/New_York")
    assert hit.strftime("%H:%M") == "10:20"
    assert hits["entry_price"].iloc[0] == 100.0 and hits["sl_dist"].iloc[0] == pytest.approx(0.7)
    assert hits["tp_dist"].iloc[0] == pytest.approx(1.4)
    assert out["exit"][out["exit"]].index.tz_convert("America/New_York").strftime("%H:%M").tolist() == ["15:55"]


def test_ict_silver_bullet_without_a_sweep_does_not_trade():
    rows = [(t.strftime("%Y-%m-%d %H:%M"), 100.0, 100.1, 99.9, 100.0)
            for t in pd.date_range("2026-10-13 18:00", "2026-10-14 16:00", freq="5min", inclusive="left")]
    df = _bars_ny(rows)
    ny = df.index.tz_convert("America/New_York").strftime("%H:%M")
    day14 = df.index.tz_convert("America/New_York").day == 14
    df.loc[(ny == "10:15") & day14, ["high", "close"]] = [100.6, 100.5]
    df.loc[(ny == "10:20") & day14, ["low", "high"]] = [100.2, 100.8]
    assert not social.ict_silver_bullet(df, {}, {})["entry"].any()


def _days_5m(highs, lows):
    rows = []
    for d, (h, lo) in zip(pd.bdate_range("2026-10-05", periods=len(highs)), zip(highs, lows)):
        for t in pd.date_range(f"{d.date()} 09:30", f"{d.date()} 16:00", freq="5min", inclusive="left"):
            rows.append((t.strftime("%Y-%m-%d %H:%M"), (h + lo) / 2, h, lo, (h + lo) / 2))
    return _bars_ny(rows)


def test_lower_highs_and_reversal_day_setups():
    highs = [110, 109, 108, 107, 106, 108, 107]
    lows = [100, 101, 99, 98, 97, 99, 98]
    df = _days_5m(highs, lows)
    first = df.index.tz_convert("America/New_York").strftime("%H:%M") == "09:30"
    lh = social.lower_highs_breakout(df, {}, {})
    days = lh.index[lh["entry"]].tz_convert("America/New_York").strftime("%m-%d").tolist()
    assert days == ["10-12"]                                    # after 4 lower highs (6th-9th Oct)
    assert lh.loc[lh["entry"], "entry_price"].iloc[0] == 106
    rv = social.reversal_day_breakout(df, {}, {})
    rdays = rv.index[rv["entry"]].tz_convert("America/New_York").strftime("%m-%d").tolist()
    assert rdays == ["10-08", "10-09", "10-12"]                 # days after a lower-high AND lower-low day
    assert rv["entry"].sum() == (rv["entry"] & first).sum()


def test_minervini_holds_only_qualifiers():
    idx = pd.bdate_range("2025-01-01", periods=320)
    mk = lambda s: pd.DataFrame({"open": s, "high": s * 1.01, "low": s * 0.99, "close": s, "volume": 1.0})
    t = np.arange(320)
    frames = {f"S{i}": mk(pd.Series(100.0 * np.exp(0.0005 * i * t), index=idx)) for i in range(8)}
    frames["DOWN"] = mk(pd.Series(200.0 - 0.2 * t, index=idx))
    w = social.minervini_template(Panel(frames), {"max_names": 10}, {}).iloc[-1]
    held = sorted(w[w > 0].index)
    # trend conditions pass for the risers, relative strength keeps only the top 30% of 9 names
    assert held == ["S5", "S6", "S7"] and w["S7"] == pytest.approx(1 / 3)


def test_top1_market_cap_switches_when_dethroned():
    idx = pd.bdate_range("2026-10-01", "2026-11-03")
    a = pd.Series(100.0, index=idx)
    b = pd.Series(100.0, index=idx)
    b[idx > "2026-10-12"] = 130.0                                  # B's cap grows 30% after the reference date
    mk = lambda s: pd.DataFrame({c: s for c in ("open", "high", "low", "close", "volume")})
    p = {"ref_date": "2026-10-12", "ref_caps": {"A": 5.0e12, "B": 4.0e12}}
    w = social.top1_market_cap(Panel({"A": mk(a), "B": mk(b)}), p, {}).iloc[-1]
    assert w["B"] == 1.0
    w0 = social.top1_market_cap(Panel({"A": mk(a[:"2026-10-12"]), "B": mk(b[:"2026-10-12"])}), p, {}).iloc[-1]
    assert w0["A"] == 1.0


def test_bull_market_support_band_only_changes_on_weekly_closes():
    idx = pd.date_range("2025-01-06", periods=300, freq="1D", tz="UTC")       # starts on a Monday
    px = pd.Series(np.r_[np.linspace(100, 200, 200), np.linspace(200, 120, 100)], index=idx)
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0})
    out = social.bull_market_support_band(df, {}, {})
    changes = out["entry"].astype(int).diff().fillna(0) != 0
    assert set(out.index[changes].day_name()) <= {"Sunday"}                  # Sunday's candle closes the week
    assert out["entry"].iloc[190] and out["exit"].iloc[-1]


class _Pos:
    def __init__(self, q):
        self.qty = q


class _Ctx:
    """Just enough of BotContext for an allocation bot's on_close."""

    def __init__(self, cash, held, px):
        self.cash, self.held, self.px, self.state, self.orders, self._hist = cash, dict(held), px, {}, [], {}
        self.ts = pd.Timestamp("2026-12-31T20:50:00Z")

    @property
    def ledger(self):
        return type("L", (), {"positions": {k: _Pos(v) for k, v in self.held.items()}})()

    def qty(self, i):
        return self.held.get(i, 0.0)

    def equity(self):
        return self.cash + sum(q * self.px[i] for i, q in self.held.items())

    def cash_eur(self):
        return self.cash

    def place(self, iid, side, type_, qty=None, eur=None, price=None, tag=""):
        self.orders.append((iid, side, qty if qty is not None else eur / self.px[iid]))


SESS_1231 = {"open": pd.Timestamp("2026-12-31T14:30:00Z"), "close": pd.Timestamp("2026-12-31T21:00:00Z")}


def _ninesig():
    from quantdesk.bots.families.custom import NineSigBot
    bot = NineSigBot("nine_sig", {"timeframe": "5m", "band": 0.0,
                                  "params": {"universe": ["TQQQ", "AGG"], "risky": "TQQQ", "safe": "AGG"}})
    bot.sessions = SESS
    return bot


def test_nine_sig_sells_the_surplus_above_the_9pct_signal_line():
    bot = _ninesig()
    ctx = _Ctx(0.0, {"TQQQ": 8.0, "AGG": 4.0}, {"TQQQ": 100.0, "AGG": 100.0})   # TQQQ 800, AGG 400
    ctx.state.update({"started": True, "ninesig": {"base": 600.0, "skip_sells": 0}})
    bars = {i: {"close": p} for i, p in ctx.px.items()}
    bot.on_close(ctx, bars, SESS_1231, "2026-12-30")                     # not the quarter's last session
    assert not ctx.orders
    bot.on_close(ctx, bars, SESS_1231, "2026-12-31")
    sold = [q for i, s, q in ctx.orders if i == "TQQQ" and s == "sell"]
    assert sold and sold[0] == pytest.approx((800 - 654) / 100)    # target = 600 x 1.09
    assert ctx.state["ninesig"]["base"] == pytest.approx(654)


def test_nine_sig_buys_the_shortfall_from_bonds_only():
    bot = _ninesig()
    ctx = _Ctx(0.0, {"TQQQ": 3.0, "AGG": 1.0}, {"TQQQ": 100.0, "AGG": 100.0})   # TQQQ 300, AGG 100
    ctx.state.update({"started": True, "ninesig": {"base": 600.0, "skip_sells": 0}})
    bot.on_close(ctx, {i: {"close": p} for i, p in ctx.px.items()}, SESS_1231, "2026-12-31")
    assert ctx.state["ninesig"]["base"] == pytest.approx(400)       # shortfall 354, but AGG holds 100
    assert ("AGG", "sell", 1.0) in [(i, s, round(q, 6)) for i, s, q in ctx.orders]
