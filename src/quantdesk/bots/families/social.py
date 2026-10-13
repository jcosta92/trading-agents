"""Strategies from Reddit, X / Instagram / YouTube trading content and quant blogs (added 13 Oct 2026).

Each function follows the rules as published by its source (links in config/families.yaml). Where
a source leaves a detail open, the choice made here is written in the family's `rules` text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import TF
from ..generic import signal, weights
from ..lib import crossed_above, ema, highest, lowest, macd, merge_informative, sma
from .crypto_ft import _out
from .etf import _best, _last_only

NY = "America/New_York"


def _f(index) -> pd.Series:
    return pd.Series(False, index=index)


def _next_session(extra) -> pd.Timestamp | None:
    idx = extra["sessions"].index
    nxt = idx[idx > pd.Timestamp(extra["day"])]
    return nxt[0] if len(nxt) else None


# ============================================================================================ crypto
@signal("crypto_turn_of_month")
def crypto_turn_of_month(df, p, info):
    """Unger Academy, turn of the month on Bitcoin (1-hour bars): buy when the 02:00 bar of the 25th
    closes, sell when the 12:00 bar of the 3rd closes (bar times as close times, UTC)."""
    end = df.index + TF["1h"]
    entry = pd.Series((end.day == 25) & (end.hour == 2), index=df.index)
    exit_ = pd.Series((end.day == 3) & (end.hour == 12), index=df.index)
    return _out(df, entry, exit_)


@signal("bull_market_support_band")
def bull_market_support_band(df, p, info):
    """Bull Market Support Band (20-week SMA and 21-week EMA): long after a weekly close above both
    lines, cash after a weekly close below both, unchanged inside the band. Weeks run Monday 00:00 to
    Sunday 24:00 UTC; a week counts once it has closed."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    wk = df.resample("W-MON", label="left", closed="left").agg(agg).dropna()   # [Monday, next Monday)
    wk["sma20"] = sma(wk["close"], 20)
    wk["ema21"] = ema(wk["close"], 21)
    m = merge_informative(df[[]], wk[["close", "sma20", "ema21"]], TF["1d"], pd.Timedelta(days=7))
    wc, a, b = m["close_inf"], m["sma20_inf"], m["ema21_inf"]
    ready = a.notna() & b.notna() & wc.notna()
    return _out(df, ready & (wc > np.maximum(a, b)), ready & (wc < np.minimum(a, b)))


@signal("ict_silver_bullet")
def ict_silver_bullet(df, p, info):
    """ICT Silver Bullet as codified by JadeCap / PickMyTrade (long side): in the 10:00-11:00 and
    14:00-15:00 New York windows, after the day's price has swept the Asian-session low (20:00-24:00
    NY), the London-session low (02:00-05:00 NY) or the low of the 9:00 NY hour, the first bullish
    fair value gap (candle 3 low above candle 1 high) gets a limit buy at its top (candle 3 low), stop
    under candle 1, target 2R; flat at 16:00 NY."""
    ny = df.index.tz_convert(NY)
    mins = pd.Series(ny.hour * 60 + ny.minute, index=df.index)
    date = pd.Series(ny.normalize().tz_localize(None), index=df.index)
    lo, hi = df["low"], df["high"]
    asian_key = date.where(mins >= 20 * 60) + pd.Timedelta(days=1)
    asian_low = lo.groupby(asian_key).min()
    london_low = lo[(mins >= 120) & (mins < 300)].groupby(date).min()
    nine_low = lo[(mins >= 540) & (mins < 600)].groupby(date).min()

    def swept(after_min: int, level: pd.Series) -> pd.Series:
        run = lo.where(mins >= after_min).groupby(date).cummin()
        lvl = date.map(level)
        return (run < lvl) & (mins >= after_min)

    sweep = swept(0, asian_low) | swept(300, london_low) | swept(600, nine_low)
    fvg = lo > hi.shift(2)
    win = ((mins >= 600) & (mins < 660)) | ((mins >= 840) & (mins < 900))
    window_id = date.astype(str) + np.where(mins < 720, "am", "pm")
    cand = fvg & win & sweep
    first = cand & (cand.astype(int).groupby(window_id).cumsum() == 1)
    out = _out(df, first, mins == 955)                      # 15:55 bar closes at 16:00 NY
    out["entry_price"] = lo.where(first)
    out["sl_dist"] = (lo - lo.shift(2)).where(first)
    out["tp_dist"] = 2 * out["sl_dist"]
    return out


@signal("pos_5ema")
def pos_5ema(df, p, info):
    """Power of Stocks (Subasish Pani) 5 EMA, buy side: an alert candle closes completely below the
    5 EMA (its high under the EMA); buy when the next candle breaks the alert high; stop under the
    alert low; target 3R. A later candle that is again fully below the EMA becomes the new alert."""
    e5 = ema(df["close"], 5)
    alert = df["high"] < e5
    out = _out(df, alert, _f(df.index))
    rng = df["high"] - df["low"]
    out["entry_price"] = df["high"].where(alert)
    out["sl_dist"] = rng.where(alert)
    out["tp_dist"] = 3 * rng.where(alert)
    return out


@signal("trading_rush_macd200")
def trading_rush_macd200(df, p, info):
    """Trading Rush 'MACD + 200 EMA' (tested 100 times on YouTube): price above EMA 200 and MACD
    (12, 26, 9) crossing above its signal line below the zero line; stop under the pullback (lowest
    low of the last 10 candles), target 1.5 x risk."""
    c = df["close"]
    m, s, _ = macd(c, 12, 26, 9)
    entry = crossed_above(m, s) & (m < 0) & (c > ema(c, 200))
    risk = (c - lowest(df["low"], 10)).where(entry)
    out = _out(df, entry & (risk > 0), _f(df.index))
    out["sl_dist"] = risk
    out["tp_dist"] = 1.5 * risk
    return out


# ============================================================================================ US ETFs, daily
@signal("ibs_lower_band")
def ibs_lower_band(df, p, info):
    """Quantitativo's IBS lower-band mean reversion ('A Mean Reversion Strategy with 2.11 Sharpe',
    most upvoted on r/algotrading and r/Daytrading): buy a close under (10-day highest high - 2.5 x
    25-day average high-low range) with IBS < 0.3; sell a close above the previous day's high."""
    c, h, lo = df["close"], df["high"], df["low"]
    band = highest(h, 10) - 2.5 * (h - lo).rolling(25).mean()
    ibs = (c - lo) / (h - lo).replace(0, np.nan)
    return _out(df, (c < band) & (ibs < 0.3), c > h.shift(1))


@signal("turnaround_tuesday")
def turnaround_tuesday(df, p, info):
    """Turnaround Tuesday: buy at Monday's close if it closed below Friday's close; sell at
    Tuesday's close."""
    wd = pd.Series(pd.DatetimeIndex(df.index).weekday, index=df.index)
    c = df["close"]
    return _out(df, (wd == 0) & (c < c.shift(1)), wd != 0)


# ============================================================================================ US ETFs, 5-minute
def _ny_day(df):
    ny = df.index.tz_convert(NY)
    return (pd.Series(ny.normalize().tz_localize(None), index=df.index),
            pd.Series(ny.hour * 60 + ny.minute, index=df.index))


def _prior_days(df, day):
    """Regular-session high/low of completed earlier days, aligned to each 5-minute bar."""
    dh, dl = df["high"].groupby(day).max(), df["low"].groupby(day).min()
    return dh, dl


@signal("orb_5min")
def orb_5min(df, p, info):
    """Zarattini & Aziz (2023) 5-minute opening range breakout, long side: if the 09:30-09:35 candle
    closes up, buy at the next open; stop at that candle's low; target 10 x the risk; otherwise
    out at the close."""
    day, mins = _ny_day(df)
    first = mins == 570
    entry = first & (df["close"] > df["open"])
    out = _out(df, entry, _f(df.index))
    out["orb_low"] = df["low"].where(first).groupby(day).ffill()
    out["tp_dist"] = (10 * (df["close"] - df["low"])).where(entry)
    return out


@signal("reversal_day_breakout")
def reversal_day_breakout(df, p, info):
    """r/algotrading 'Simple Reversal Strategy' (russs123): after a day with a lower low AND a lower
    high than the day before, buy a break of that day's high the next day; sell at the close."""
    day, mins = _ny_day(df)
    dh, dl = _prior_days(df, day)
    setup = (dl.shift(1) < dl.shift(2)) & (dh.shift(1) < dh.shift(2))
    first = mins == 570
    entry = first & day.map(setup).fillna(False).astype(bool)
    out = _out(df, entry, _f(df.index))
    out["entry_price"] = day.map(dh.shift(1)).where(entry)
    return out


@signal("lower_highs_breakout")
def lower_highs_breakout(df, p, info):
    """r/algotrading 'Multiple Lower Highs' (russs123, code on GitHub): after at least 4 lower daily
    highs in a row, buy a break of the latest high the next day (at the open if it gaps above); sell
    at the close. Unfilled orders move to the new high the next day."""
    day, mins = _ny_day(df)
    dh, _ = _prior_days(df, day)
    lower = dh < dh.shift(1)
    run = lower & lower.shift(1, fill_value=False) & lower.shift(2, fill_value=False) & lower.shift(3, fill_value=False)
    setup = run.shift(1, fill_value=False)
    first = mins == 570
    entry = first & day.map(setup).fillna(False).astype(bool)
    out = _out(df, entry, _f(df.index))
    out["entry_price"] = day.map(dh.shift(1)).where(entry)
    return out


# ============================================================================================ calendar rules
FOMC_EVEN = [(0, 3), (9, 13), (19, 23), (29, 33)]


def fomc_even_week(x: pd.Timestamp, fomc_days: list) -> bool:
    """Cieslak, Morse & Vissing-Jorgensen (2019): day 0 = scheduled FOMC announcement; weekdays are
    counted (holidays included); even weeks are days -1..3, 9..13, 19..23 and 29..33."""
    xd = np.datetime64(pd.Timestamp(x).date())
    days = sorted(np.datetime64(pd.Timestamp(d).date()) for d in fomc_days)
    nxt = [d for d in days if d > xd]
    if nxt and np.busday_count(xd, nxt[0]) == 1:
        return True
    prev = [d for d in days if d <= xd]
    if not prev:
        return False
    k = int(np.busday_count(prev[-1], xd))
    return any(a <= k <= b for a, b in FOMC_EVEN)


@weights("fomc_even_weeks")
def fomc_even_weeks(panel, p, extra):
    """Hold the asset over the next session if it falls in an even FOMC-cycle week, else cash."""
    x = _next_session(extra)
    asset = p["asset"]
    on = x is not None and fomc_even_week(x, p["fomc_days"])
    return _last_only(panel, lambda i: {asset: 1.0} if on else {})


@weights("halloween")
def halloween(panel, p, extra):
    """Halloween indicator (Bouman & Jacobsen 2002): equities from the end of October to the
    beginning of May, cash otherwise."""
    x = _next_session(extra)
    on = x is not None and x.month in (11, 12, 1, 2, 3, 4)
    return _last_only(panel, lambda i: {p["asset"]: 1.0} if on else {})


@weights("santa_claus_rally")
def santa_claus_rally(panel, p, extra):
    """Santa Claus rally (Hirsch 1972): hold over the last 5 trading days of December and the first
    2 of January."""
    x = _next_session(extra)
    on = False
    if x is not None:
        idx = extra["sessions"].index
        if x.month == 12:
            dec = idx[(idx.year == x.year) & (idx.month == 12)]
            on = x in dec[-5:]
        elif x.month == 1:
            jan = idx[(idx.year == x.year) & (idx.month == 1)]
            on = x in jan[:2]
    return _last_only(panel, lambda i: {p["asset"]: 1.0} if on else {})


@weights("payday_16th")
def payday_16th(panel, p, extra):
    """Payday anomaly (Ma & Pratt 2018): hold the index over the 16th of the month."""
    x = _next_session(extra)
    on = x is not None and x.day == 16
    return _last_only(panel, lambda i: {p["asset"]: 1.0} if on else {})


# ============================================================================================ rotations
@weights("utilities_rotation")
def utilities_rotation(panel, p, extra):
    """Bilello & Gayed (2014) utilities beta rotation: if utilities (XLU) beat the market (VTI) over
    the prior 4 weeks, hold utilities for the next week, otherwise the market."""
    r = panel.close / panel.close.shift(20) - 1

    def f(i):
        row = r.iloc[i]
        if row.reindex(["XLU", "VTI"]).isna().any():
            return {}
        return {"XLU": 1.0} if row["XLU"] > row["VTI"] else {"VTI": 1.0}
    return _last_only(panel, f)


@weights("sector_seasonality")
def sector_seasonality(panel, p, extra):
    """Quantpedia 'Front-running seasonality in US stock sectors': at each month-end hold the 2
    sector SPDRs with the best return in the calendar month that comes 11 months before the holding
    month (e.g. at the end of December, last February's winners)."""
    c = panel.close
    me = c.groupby(c.index.to_period("M")).last()
    mret = me / me.shift(1) - 1

    # decided on the month's last session -> holding next month; otherwise (the bot's first day)
    # holding the current month: the ranking month is always the holding month minus 11
    pos = extra.get("pos") if isinstance(extra, dict) else None
    month_end = pos is not None and pos[1] == 0

    def f(i):
        cur = c.index[i].to_period("M")
        rank_month = cur - 10 if month_end else cur - 11
        if rank_month not in mret.index:
            return {}
        row = mret.loc[rank_month].dropna()
        top = _best(row, int(p.get("k", 2)))
        return {a: 1 / len(top) for a in top} if top else {}
    return _last_only(panel, f)


@weights("multi_asset_pullback")
def multi_asset_pullback(panel, p, extra):
    """Quantpedia (2026) multi-asset 3-day pullback: buy at the close each ETF that is above its
    200-day average after three down days in a row, equal weight among the active ones; hold one day."""
    c = panel.close
    r = c.pct_change()
    act = (c > c.rolling(200).mean()) & (r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0)

    def f(i):
        on = [a for a in c.columns if bool(act[a].iloc[i])]
        return {a: 1 / len(on) for a in on} if on else {}
    return _last_only(panel, f)


@weights("spy_band_tqqq")
def spy_band_tqqq(panel, p, extra):
    """r/LETFs 'SPY 200SMA +4%/-3% TQQQ/QQQ': hold TQQQ after SPY closes 4% above its 200-day
    average, QQQ after it closes 3% below; keep the holding in between; if QQQ is 30% above its own
    200-day average, hold QQQ instead of TQQQ."""
    c = panel.close
    ratio = (c["SPY"] / c["SPY"].rolling(200).mean()).to_numpy()
    q_ratio = (c["QQQ"] / c["QQQ"].rolling(200).mean()).to_numpy()
    lev = False
    for x in ratio:
        if np.isnan(x):
            continue
        if x >= float(p.get("up", 1.04)):
            lev = True
        elif x <= float(p.get("down", 0.97)):
            lev = False
    if len(ratio) == 0 or np.isnan(ratio[-1]):
        return _last_only(panel, lambda i: {})
    hot = not np.isnan(q_ratio[-1]) and q_ratio[-1] > float(p.get("deleverage", 1.30))
    pick = "TQQQ" if lev and not hot else "QQQ"
    return _last_only(panel, lambda i: {pick: 1.0})


@weights("static_mix")
def static_mix(panel, p, extra):
    """A fixed mix (lazy portfolio); the bot's schedule and band decide when it is rebalanced."""
    return _last_only(panel, lambda i: dict(p["mix"]))


@weights("minervini_template")
def minervini_template(panel, p, extra):
    """Mark Minervini's Trend Template on the 40 largest US stocks: close above the 150- and 200-day
    averages, 150 above 200, 200 rising for a month, 50 above 150 and 200, close above the 50, at
    least 30% above the 52-week low and within 25% of the 52-week high, relative strength in the top
    30% (IBD-style 12-month weighted return, ranked within the universe). Hold the qualifiers with the
    best relative strength (up to 10), equal weight."""
    c = panel.close
    s50, s150, s200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    lo52, hi52 = panel.low.rolling(252).min(), panel.high.rolling(252).max()
    rs = 0.4 * (c / c.shift(63) - 1) + 0.2 * (c / c.shift(126) - 1) + 0.2 * (c / c.shift(189) - 1) \
        + 0.2 * (c / c.shift(252) - 1)

    def f(i):
        row = rs.iloc[i].dropna()
        if len(row) < 5:
            return {}
        pct = row.rank(pct=True)
        ok = []
        for a in row.index:
            x = c[a].iloc[i]
            conds = [x > s150[a].iloc[i], x > s200[a].iloc[i], s150[a].iloc[i] > s200[a].iloc[i],
                     s200[a].iloc[i] > s200[a].iloc[max(0, i - 21)], s50[a].iloc[i] > s150[a].iloc[i],
                     s50[a].iloc[i] > s200[a].iloc[i], x > s50[a].iloc[i], x >= 1.30 * lo52[a].iloc[i],
                     x >= 0.75 * hi52[a].iloc[i], pct[a] >= 0.70]
            if all(bool(v) for v in conds):
                ok.append(a)
        top = sorted(ok, key=lambda a: -row[a])[:int(p.get("max_names", 10))]
        return {a: 1 / len(top) for a in top} if top else {}
    return _last_only(panel, f)


@weights("top1_market_cap")
def top1_market_cap(panel, p, extra):
    """r/Daytrading 'buy the #1 company by market cap, switch on the 1st of the month when it is
    dethroned'. Market caps are the 12 Oct 2026 values scaled by each stock's price change since."""
    c = panel.close
    ref = pd.Timestamp(p["ref_date"])
    caps = {k: float(v) for k, v in p["ref_caps"].items() if k in c}

    def f(i):
        mc = {}
        for a, cap in caps.items():
            base = c[a].loc[:ref].dropna()
            b = float(base.iloc[-1]) if len(base) else float(c[a].iloc[i])
            x = float(c[a].iloc[i])
            if b > 0 and not np.isnan(x):
                mc[a] = cap * x / b
        return {max(mc, key=mc.get): 1.0} if mc else {}
    return _last_only(panel, f)
