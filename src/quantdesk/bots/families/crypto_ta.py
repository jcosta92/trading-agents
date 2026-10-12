"""Classic technical systems and popular TradingView scripts / built-in strategies (long-only:
a short signal means exit to cash). Formulas follow the original books or the open Pine code."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib as tl

from ..data import TF
from ..generic import signal
from ..lib import (S, adx, atr, bb_pop, cci, connors_rsi, crossed_above, crossed_below, cross_above_level,
                   cross_below_level, ema, f, heikin_ashi, highest, ichimoku, lowest, merge_informative, mfi,
                   minus_di, plus_di, rsi, sar, sma, stoch, wma)
from .crypto_ft import _out


def _lvl(df, v):
    return pd.Series(float(v), index=df.index)


@signal("ichimoku_tk")
def ichimoku_tk(df, p, info):
    t, k, a, b = ichimoku(df, 9, 26, 52, 26)
    c = df["close"]
    cloud_top = pd.concat([a, b], axis=1).max(axis=1)
    return _out(df, crossed_above(t, k) & (c > cloud_top), crossed_below(t, k) | (c < k))


@signal("psar_reversal")
def psar_reversal(df, p, info):
    s = sar(df, 0.02, 0.2)
    above = df["close"] > s
    return _out(df, above & ~above.shift(1, fill_value=True), ~above & above.shift(1, fill_value=False))


@signal("dmi_cross")
def dmi_cross(df, p, info):
    pdi, mdi, a = plus_di(df, 14), minus_di(df, 14), adx(df, 14)
    return _out(df, crossed_above(pdi, mdi) & (a > 25), crossed_above(mdi, pdi))


@signal("donchian_55_20")
def donchian_55_20(df, p, info):
    o = _out(df, _lvl(df, 1) > 0, _lvl(df, 0) > 0)
    o["entry_price"] = highest(df["high"], 55)
    o["stop_level"] = lowest(df["low"], 20)
    o["sl_dist"] = 2 * atr(df, 20)
    return o


@signal("keltner_tv")
def keltner_tv(df, p, info):
    mid = ema(df["close"], 20)
    up = mid + 2 * atr(df, 10)
    o = _out(df, crossed_above(df["close"], up), df["close"] < mid)
    o["entry_price"] = df["high"] * 1.0001
    return o


@signal("bb_tv")
def bb_tv(df, p, info):
    lower, mid, upper = bb_pop(df["close"], 20, 2.0)
    return _out(df, crossed_above(df["close"], lower), crossed_below(df["close"], upper))


@signal("bollinger_method2")
def bollinger_method2(df, p, info):
    lower, mid, upper = bb_pop(df["close"], 20, 2.0)
    pb = (df["close"] - lower) / (upper - lower)
    m = mfi(df, 10)
    return _out(df, (pb > 0.8) & (m > 80), (pb < 0.2) & (m < 20))


@signal("stoch_slow")
def stoch_slow(df, p, info):
    k, d = stoch(df, 14, 3, 3)
    o = _out(df, crossed_above(k, d) & (k < 20), crossed_below(k, d) & (k > 80))
    o["sl_dist"] = 2 * atr(df, 14)
    return o


@signal("stochrsi_trend")
def stochrsi_trend(df, p, info):
    r = rsi(df["close"], 14)
    srsi = (r - lowest(r, 14)) / (highest(r, 14) - lowest(r, 14)) * 100
    k = sma(srsi, 3)
    d = sma(k, 3)
    e200 = ema(df["close"], 200)
    return _out(df, crossed_above(k, d) & (k < 20) & (d < 20) & (df["close"] > e200),
                (crossed_below(k, d) & (k > 80)) | (df["close"] < e200))


@signal("cci_breakout")
def cci_breakout(df, p, info):
    c = cci(df, 20)
    return _out(df, cross_above_level(c, 100), cross_below_level(c, 100))


@signal("aroon")
def aroon(df, p, info):
    dn, up = tl.AROON(f(df["high"]), f(df["low"]), 25)
    up, dn = S(up, df.index), S(dn, df.index)
    return _out(df, crossed_above(up, dn) & (up >= 70), crossed_above(dn, up))


def chandelier_dir(df, n=22, mult=3.0):
    c = df["close"].to_numpy()
    a = f(atr(df, n)) * mult
    hh = f(highest(df["close"], n))
    ll = f(lowest(df["close"], n))
    ls = hh - a
    ss = ll + a
    d = np.ones(len(c))
    for i in range(1, len(c)):
        if not np.isnan(ls[i - 1]) and c[i - 1] > ls[i - 1]:
            ls[i] = max(ls[i], ls[i - 1])
        if not np.isnan(ss[i - 1]) and c[i - 1] < ss[i - 1]:
            ss[i] = min(ss[i], ss[i - 1])
        if not np.isnan(ss[i - 1]) and c[i] > ss[i - 1]:
            d[i] = 1
        elif not np.isnan(ls[i - 1]) and c[i] < ls[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return pd.Series(d, index=df.index)


@signal("chandelier_exit")
def chandelier_exit(df, p, info):
    d = chandelier_dir(df, 22, 3.0)
    return _out(df, (d == 1) & (d.shift(1) == -1), (d == -1) & (d.shift(1) == 1))


def hma(s: pd.Series, n: int) -> pd.Series:
    return wma(2 * wma(s, n // 2) - wma(s, n), int(round(np.sqrt(n))))


@signal("hull_suite")
def hull_suite(df, p, info):
    h = hma(df["close"], 55)
    up = h > h.shift(2)
    return _out(df, up, h < h.shift(2))


@signal("elder_triple_screen")
def elder_triple_screen(df, p, info):
    c = df["close"]
    d1 = info.get("1d")
    if d1 is None or not len(d1):
        z = pd.Series(False, index=df.index)
        return _out(df, z, z)
    d1 = d1.copy()
    m, s, hist = tl.MACD(f(d1["close"]), 12, 26, 9)
    d1["hist"] = hist
    d1["e13"] = ema(d1["close"], 13)
    d1["tide_up"] = (d1["hist"] > d1["hist"].shift(1)).astype(float)
    d1["red"] = ((d1["e13"] < d1["e13"].shift(1)) & (d1["hist"] < d1["hist"].shift(1))).astype(float)
    mm = merge_informative(df[[]], d1[["tide_up", "red"]], TF["4h"], TF["1d"])
    force2 = ema((c - c.shift()) * df["volume"], 2)
    _, _, h4 = tl.MACD(f(c), 12, 26, 9)
    h4 = S(h4, df.index)
    e13 = ema(c, 13)
    red4 = (e13 < e13.shift()) & (h4 < h4.shift())
    o = _out(df, (mm["tide_up_inf"] == 1) & (force2 < 0) & ~red4, mm["red_inf"] == 1)
    o["entry_price"] = df["high"] * 1.0001
    o["stop_level"] = lowest(df["low"], 5)
    return o


@signal("heikin_ashi_trend")
def heikin_ashi_trend(df, p, info):
    ha = heikin_ashi(df)
    green = ha["close"] > ha["open"]
    no_lower = ha["open"] <= ha["low"] * (1 + 1e-12)
    return _out(df, green & green.shift(1, fill_value=False) & no_lower, ~green)


@signal("kama_cross")
def kama_cross(df, p, info):
    k = S(tl.KAMA(f(df["close"]), 10), df.index)
    c = df["close"]
    return _out(df, crossed_above(c, k) & (k > k.shift(1)), crossed_below(c, k))


@signal("trix")
def trix(df, p, info):
    t = S(tl.TRIX(f(df["close"]), 15), df.index)
    s = ema(t, 9)
    return _out(df, crossed_above(t, s) & (t < 0), crossed_below(t, s))


@signal("chaikin_money_flow")
def cmf_sig(df, p, info):
    h, lo, c, v = df["high"], df["low"], df["close"], df["volume"]
    mfm = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
    cmf = (mfm.fillna(0) * v).rolling(20).sum() / v.rolling(20).sum()
    return _out(df, cross_above_level(cmf, 0.05) & (c > ema(c, 50)), cmf < -0.05)


@signal("obv_breakout")
def obv_breakout(df, p, info):
    c = df["close"]
    ob = S(tl.OBV(f(c), f(df["volume"])), df.index)
    e50 = ema(c, 50)
    entry = (ob > highest(ob, 20).shift(1)) & (c > highest(c, 20).shift(1) * 0.98) & (c > e50)
    return _out(df, entry, (ob < ema(ob, 20)) | (c < e50))


@signal("vwap_reversion")
def vwap_reversion(df, p, info):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"]
    day = df.index.floor("D")
    cv = v.groupby(day).cumsum()
    vwap = (tp * v).groupby(day).cumsum() / cv.replace(0, np.nan)
    var = (v * (tp - vwap) ** 2).groupby(day).cumsum() / cv.replace(0, np.nan)
    sd = np.sqrt(var)
    lb = vwap - 2 * sd
    d1 = info.get("1d")
    if d1 is not None and len(d1):
        d1 = d1.copy()
        d1["up"] = (d1["close"] > ema(d1["close"], 50)).astype(float)
        trend = merge_informative(df[[]], d1[["up"]], TF["15m"], TF["1d"])["up_inf"] == 1
    else:
        trend = pd.Series(False, index=df.index)
    c = df["close"]
    end = df.index + TF["15m"]
    late = (end.hour == 23) & (end.minute >= 45)
    entry = (c.shift(1) < lb.shift(1)) & (c > lb) & trend & ~late
    o = _out(df, entry, pd.Series(late, index=df.index))
    o["tp_level"] = vwap
    o["stop_level"] = vwap - 3 * sd
    return o


@signal("pivot_bounce")
def pivot_bounce(df, p, info):
    d = df.resample("1D", label="left", closed="left").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()
    d["P"] = (d["high"] + d["low"] + d["close"]) / 3
    d["S1"] = 2 * d["P"] - d["high"]
    d["S2"] = d["P"] - (d["high"] - d["low"])
    m = merge_informative(df[[]], d[["P", "S1", "S2"]], TF["1h"], TF["1d"])
    entry = (df["low"] <= m["S1_inf"]) & (df["close"] > m["S1_inf"])
    o = _out(df, entry, pd.Series(False, index=df.index))
    o["tp_level"], o["stop_level"] = m["P_inf"], m["S2_inf"]
    return o


@signal("golden_cross")
def golden_cross(df, p, info):
    s50, s200 = sma(df["close"], 50), sma(df["close"], 200)
    return _out(df, crossed_above(s50, s200), crossed_below(s50, s200))


@signal("connors_rsi_crypto")
def connors_rsi_crypto(df, p, info):
    c = df["close"]
    cr = connors_rsi(c)
    o = _out(df, (c > sma(c, 200)) & (cr < 10), cr > 70)
    o["entry_price"] = c * 0.98
    return o


def _ut_bot(c: np.ndarray, a: np.ndarray, key: float):
    n = len(c)
    ts = np.zeros(n)
    for i in range(1, n):
        nloss = key * a[i] if not np.isnan(a[i]) else np.nan
        s = ts[i - 1]
        if np.isnan(nloss):
            ts[i] = c[i]
            continue
        if c[i] > s and c[i - 1] > s:
            ts[i] = max(s, c[i] - nloss)
        elif c[i] < s and c[i - 1] < s:
            ts[i] = min(s, c[i] + nloss)
        elif c[i] > s:
            ts[i] = c[i] - nloss
        else:
            ts[i] = c[i] + nloss
    return ts


@signal("ut_bot")
def ut_bot(df, p, info):
    c = df["close"]
    ts = pd.Series(_ut_bot(f(c), f(atr(df, int(p.get("atr_period", 10)))), float(p.get("key", 2.0))), index=df.index)
    buy = (c > ts) & crossed_above(c, ts)
    sell = (c < ts) & crossed_above(ts, c)
    return _out(df, buy, sell)


@signal("ssl_channel")
def ssl_channel(df, p, info):
    sh, sl = sma(df["high"], 10), sma(df["low"], 10)
    c = df["close"].to_numpy()
    hlv = np.zeros(len(c))
    shv, slv = f(sh), f(sl)
    for i in range(len(c)):
        prev = hlv[i - 1] if i else 0
        hlv[i] = 1 if c[i] > shv[i] else (-1 if c[i] < slv[i] else prev)
    h = pd.Series(hlv, index=df.index)
    return _out(df, (h == 1) & (h.shift(1) != 1), (h == -1) & (h.shift(1) != -1))


@signal("qqe")
def qqe(df, p, info):
    r = rsi(df["close"], 14)
    rm = ema(r, 5)
    tr = (rm - rm.shift()).abs()
    dar = ema(ema(tr, 27), 27) * 4.236
    rmv, darv = f(rm), f(dar)
    n = len(rmv)
    lb, sb, tr_ = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        if np.isnan(rmv[i]) or np.isnan(darv[i]):
            continue
        nl, ns = rmv[i] - darv[i], rmv[i] + darv[i]
        lb[i] = max(lb[i - 1], nl) if (rmv[i - 1] > lb[i - 1] and rmv[i] > lb[i - 1]) else nl
        sb[i] = min(sb[i - 1], ns) if (rmv[i - 1] < sb[i - 1] and rmv[i] < sb[i - 1]) else ns
        if rmv[i] > sb[i - 1] and rmv[i - 1] <= sb[i - 1]:
            tr_[i] = 1
        elif rmv[i] < lb[i - 1] and rmv[i - 1] >= lb[i - 1]:
            tr_[i] = -1
        else:
            tr_[i] = tr_[i - 1] if tr_[i - 1] != 0 else 1
    t = pd.Series(tr_, index=df.index)
    return _out(df, (t == 1) & (t.shift(1) == -1), (t == -1) & (t.shift(1) == 1))


@signal("wavetrend")
def wavetrend(df, p, info):
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ema(ap, 10)
    d = ema((ap - esa).abs(), 10)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ema(ci, 21)
    wt2 = sma(wt1, 4)
    o = _out(df, crossed_above(wt1, wt2) & (wt1 < -53), crossed_below(wt1, wt2) & (wt1 > 53))
    o["sl_dist"] = 3 * atr(df, 14)
    return o


@signal("range_filter")
def range_filter(df, p, info):
    src = df["close"]
    avrng = ema((src - src.shift()).abs(), 100)
    smrng = f(ema(avrng, 199) * 3.0)
    s = f(src)
    n = len(s)
    filt = np.copy(s)
    upw, dnw = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        r = smrng[i]
        if np.isnan(r):
            filt[i] = s[i]
            continue
        if s[i] > filt[i - 1]:
            filt[i] = filt[i - 1] if s[i] - r < filt[i - 1] else s[i] - r
        else:
            filt[i] = filt[i - 1] if s[i] + r > filt[i - 1] else s[i] + r
        upw[i] = upw[i - 1] + 1 if filt[i] > filt[i - 1] else (0 if filt[i] < filt[i - 1] else upw[i - 1])
        dnw[i] = dnw[i - 1] + 1 if filt[i] < filt[i - 1] else (0 if filt[i] > filt[i - 1] else dnw[i - 1])
    long_c = (s > filt) & (upw > 0)
    short_c = (s < filt) & (dnw > 0)
    ini = np.zeros(n)
    for i in range(1, n):
        ini[i] = 1 if long_c[i] else (-1 if short_c[i] else ini[i - 1])
    buy = long_c & (np.r_[0, ini[:-1]] == -1)
    sell = short_c & (np.r_[0, ini[:-1]] == 1)
    return _out(df, pd.Series(buy, index=df.index), pd.Series(sell, index=df.index))


@signal("halftrend")
def halftrend(df, p, info):
    amp = 2
    h, lo, c = f(df["high"]), f(df["low"]), f(df["close"])
    hp, lp = f(highest(df["high"], amp)), f(lowest(df["low"], amp))
    hma_, lma = f(sma(df["high"], amp)), f(sma(df["low"], amp))
    n = len(c)
    trend = np.zeros(n)
    nxt = 0
    maxlow = lo[0] if n else 0.0
    minhigh = h[0] if n else 0.0
    tr = 0
    for i in range(1, n):
        if np.isnan(hp[i]) or np.isnan(hma_[i]):
            trend[i] = tr
            continue
        if nxt == 1:
            maxlow = max(lp[i], maxlow)
            if hma_[i] < maxlow and c[i] < lo[i - 1]:
                tr, nxt, minhigh = 1, 0, hp[i]
        else:
            minhigh = min(hp[i], minhigh)
            if lma[i] > minhigh and c[i] > h[i - 1]:
                tr, nxt, maxlow = 0, 1, lp[i]
        trend[i] = tr
    t = pd.Series(trend, index=df.index)
    return _out(df, (t == 0) & (t.shift(1) == 1), (t == 1) & (t.shift(1) == 0))


@signal("coral_trend")
def coral_trend(df, p, info):
    sm, cd = 21, 0.4
    di = (sm - 1.0) / 2.0 + 1.0
    c1 = 2 / (di + 1.0)
    c2 = 1 - c1
    c3 = 3.0 * (cd * cd + cd * cd * cd)
    c4 = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5 = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd
    src = f(df["close"])
    n = len(src)
    i_ = np.zeros((7, n))
    i_[0] = src
    for t in range(n):
        for k in range(1, 7):
            prev = i_[k, t - 1] if t else 0.0
            i_[k, t] = c1 * i_[k - 1, t] + c2 * prev
    bfr = -cd ** 3 * i_[6] + c3 * i_[5] + c4 * i_[4] + c5 * i_[3]
    b = pd.Series(bfr, index=df.index)
    up = b > b.shift(1)
    return _out(df, up & ~up.shift(1, fill_value=True), ~up & up.shift(1, fill_value=False))


@signal("holy_grail")
def holy_grail(df, p, info):
    a = adx(df, 14)
    e20 = ema(df["close"], 20)
    touch = (df["low"] <= e20) & (a > 30) & (a > a.shift(1))
    o = _out(df, touch, pd.Series(False, index=df.index))
    o["entry_price"] = df["high"] * 1.0001
    o["sl_dist"] = df["high"] - lowest(df["low"], 5)
    o["tp_dist"] = (highest(df["high"], 20).shift(1) - df["high"]).clip(lower=0) + 1e-9
    return o


@signal("turtle_soup")
def turtle_soup(df, p, info):
    lo = f(df["low"])
    n = len(lo)
    setup = np.zeros(n, dtype=bool)
    prior = np.full(n, np.nan)
    for i in range(21, n):
        w = lo[i - 20:i]                     # the 20 bars before today
        j = int(np.argmin(w))
        prev_low = w[j]
        ago = 20 - j                         # bars before today
        if lo[i] < prev_low and ago >= 4 and lo[i] <= np.min(lo[i - 19:i + 1]):
            setup[i] = True
            prior[i] = prev_low
    o = _out(df, pd.Series(setup, index=df.index), pd.Series(False, index=df.index))
    o["entry_price"] = pd.Series(prior, index=df.index) * 1.0005
    o["sl_dist"] = o["entry_price"] - df["low"]
    return o


@signal("nr7_breakout")
def nr7_breakout(df, p, info):
    rng = df["high"] - df["low"]
    nr7 = rng <= rng.rolling(7).min()
    o = _out(df, nr7, pd.Series(False, index=df.index))
    o["entry_price"] = df["high"] * 1.0001
    o["sl_dist"] = rng
    return o


@signal("darvas_box")
def darvas_box(df, p, info):
    h, lo = f(df["high"]), f(df["low"])
    n = len(h)
    top_a, bot_a = np.full(n, np.nan), np.full(n, np.nan)
    top = bot = None
    top_i = None
    for i in range(23, n):
        if top is None:
            cand = h[i - 3]
            if cand >= np.max(h[i - 23:i - 2]) and np.max(h[i - 2:i + 1]) < cand:
                top, top_i, bot = cand, i - 3, None
        else:
            if h[i] > top and bot is None:
                top = None
                continue
            if bot is None:
                seg = lo[top_i + 1:i + 1]
                k = int(np.argmin(seg))
                if (len(seg) - 1 - k) >= 3:
                    bot = seg[k]
            if bot is not None and lo[i] < bot:
                top = bot = None
                continue
        if top is not None and bot is not None:
            top_a[i], bot_a[i] = top, bot
            if df["close"].iat[i] > top:     # broke out: look for the next box
                top = bot = None
    box = pd.Series(~np.isnan(top_a), index=df.index)
    o = _out(df, box, pd.Series(False, index=df.index))
    o["entry_price"] = pd.Series(top_a, index=df.index) * 1.0001
    o["stop_level"] = pd.Series(bot_a, index=df.index).ffill() * 0.999
    return o


@signal("wyckoff_spring")
def wyckoff_spring(df, p, info):
    a = atr(df, 14)
    h40 = highest(df["high"], 40).shift(1)
    l40 = lowest(df["low"], 40).shift(1)
    side = (h40 - l40) < 6 * a
    spring = side & (df["low"] < l40) & (df["low"] >= l40 - a) & (df["close"] > l40)
    o = _out(df, spring, pd.Series(False, index=df.index))
    o["entry_price"] = df["high"] * 1.0001
    o["sl_dist"] = df["high"] - (df["low"] - 0.2 * a)
    o["tp_dist"] = (h40 - df["high"]).clip(lower=0) + 1e-9
    return o


@signal("vortex")
def vortex(df, p, info):
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    vip = (h - lo.shift()).abs().rolling(14).sum() / tr.rolling(14).sum()
    vim = (lo - h.shift()).abs().rolling(14).sum() / tr.rolling(14).sum()
    return _out(df, crossed_above(vip, vim), crossed_above(vim, vip))


@signal("alphatrend")
def alphatrend(df, p, info):
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    a = f(sma(tr, 14))
    up, dn = f(lo) - a, f(h) + a
    m = f(mfi(df, 14))
    n = len(up)
    at = np.zeros(n)
    for i in range(1, n):
        if np.isnan(a[i]) or np.isnan(m[i]):
            at[i] = at[i - 1]
            continue
        if m[i] >= 50:
            at[i] = at[i - 1] if up[i] < at[i - 1] else up[i]
        else:
            at[i] = at[i - 1] if dn[i] > at[i - 1] else dn[i]
    s = pd.Series(at, index=df.index)
    return _out(df, crossed_above(s, s.shift(2)), crossed_below(s, s.shift(2)))


@signal("tv_momentum")
def tv_momentum(df, p, info):
    c = df["close"]
    m0 = c - c.shift(12)
    m1 = m0 - m0.shift(1)
    o = _out(df, (m0 > 0) & (m1 > 0), (m0 < 0) & (m1 < 0))
    o["entry_price"] = df["high"] * 1.0001
    return o


@signal("consecutive_updown")
def consecutive_updown(df, p, info):
    c = df["close"]
    up = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3))
    dn = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3))
    return _out(df, up, dn)
