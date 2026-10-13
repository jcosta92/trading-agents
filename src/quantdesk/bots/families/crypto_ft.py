"""Open-source crypto bot strategies (Freqtrade, Jesse), rules copied from the published files.

Each function returns a frame with boolean `entry` / `exit` columns (plus extra exit columns or
price levels where the strategy needs them). Risk settings (stoploss, ROI table, trailing stop)
live in config/families.yaml next to the source link.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib as tl

from ..generic import signal
from ..lib import (S, adx, atr, bb_qtpy, cci, cmo, cross_above_level, crossed_above, crossed_below, ema, f,
                   heikin_ashi, highest, ichimoku, linreg, lowest, macd, merge_informative, mfi, minus_di, mom,
                   plus_di, resample_closed, rsi, sar, sma, stoch, stochf, tema, typical)
from ..data import TF


def _out(df, entry, exit_, **extra) -> pd.DataFrame:
    o = pd.DataFrame({"entry": entry.fillna(False).astype(bool), "exit": exit_.fillna(False).astype(bool)},
                     index=df.index)
    for k, v in extra.items():
        o[k] = v
    return o


def _vol_ok(df):
    return df["volume"] > 0


# --------------------------------------------------------------------------------------- ssssi E0V1E
@signal("e0v1e")
def e0v1e(df, p, info):
    c = df["close"]
    sma15 = sma(c, 15)
    idx = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    cti = c.rolling(20).corr(idx)                       # pandas_ta cti: correlation of price with time
    r, rf, rs = rsi(c, 14), rsi(c, 4), rsi(c, 20)
    fk, _ = stochf(df, 5, 3)
    cc = cci(df, 20)
    falling = rs < rs.shift(1)
    b1 = falling & (rf < 40) & (r > 42) & (c < sma15 * 0.973) & (cti < 0.69)
    bn = falling & (rf < 34) & (r > 28) & (c < sma15 * 0.96) & (cti < 0.69)
    return _out(df, b1 | bn, pd.Series(False, index=df.index), exit_fastk=fk > 84, exit_cci=cc > 80)


# --------------------------------------------------------------------------------------- ElliotV5HO
def _ewo(c, a=50, b=200):
    return (ema(c, a) - ema(c, b)) / c * 100


@signal("elliotv5")
def elliotv5(df, p, info):
    c = df["close"]
    e17, e39 = ema(c, 17), ema(c, 39)
    ewo, r = _ewo(c), rsi(c, 14)
    dip = c < e17 * 0.978
    entry = ((dip & (ewo > 3.34) & (r < 60)) | (dip & (ewo < -17.457))) & _vol_ok(df)
    return _out(df, entry, (c > e39 * 1.011) & _vol_ok(df))


@signal("smaoffset")
def smaoffset(df, p, info):
    c = df["close"]
    return _out(df, (c < sma(c, 30) * 0.958) & _vol_ok(df), (c > ema(c, 30) * 1.012) & _vol_ok(df))


# --------------------------------------------------------------------------------------- ClucHAnix_5m
@signal("cluchanix")
def cluchanix(df, p, info):
    ha = heikin_ashi(df)
    hc, hh, hl = ha["close"], ha["high"], ha["low"]
    tp = (hh + hl + hc) / 3
    mid = tp.rolling(40).mean()
    sd = tp.rolling(40).std()
    lower = mid - 2 * sd
    bbdelta = (mid - lower).abs()
    closedelta = (hc - hc.shift()).abs()
    tail = (hc - hl).abs()
    ema3, ema50 = ema(hc, 3), ema(hc, 50)
    fisher = np.tanh(0.1 * (rsi(df["close"], 14) - 50))
    inf = info.get("1h")
    if inf is not None and len(inf):
        hai = heikin_ashi(inf)
        hai["rocr"] = hai["close"] / hai["close"].shift(168)
        m = merge_informative(df[[]], hai[["rocr"]], TF["5m"], TF["1h"])
        rocr = m["rocr_inf"]
    else:
        rocr = pd.Series(np.nan, index=df.index)
    a = (lower.shift() > 0) & (bbdelta > hc * 0.01889) & (closedelta > hc * 0.00916) & (tail < bbdelta * 0.72235) \
        & (hc < lower.shift()) & (hc <= hc.shift())
    b = (hc < ema50) & (hc < 0.0127 * lower)
    entry = (rocr > 0.79492) & (a | b)
    exit_ = (fisher > 0.39075) & (hh <= hh.shift(1)) & (hh.shift(1) <= hh.shift(2)) & (hc <= hc.shift(1)) \
        & (ema3 > hc) & (hc * 0.99754 > mid)
    return _out(df, entry, exit_)


# --------------------------------------------------------------------------------------- ClucMay72018
@signal("clucmay72018")
def clucmay(df, p, info):
    c = df["close"]
    lower, mid, _ = bb_qtpy(typical(df), 20, 2)
    e = ema(c, 50)
    entry = (c < e) & (c < 0.985 * lower) & (df["volume"] < df["volume"].rolling(30).mean().shift(1) * 20)
    return _out(df, entry, c > mid)


# --------------------------------------------------------------------------------------- Jesse IFR2
@signal("ifr2")
def ifr2(df, p, info):
    c = df["close"]
    _, _, sa, sb = ichimoku(df, 20, 30, 120, 60)
    trendmode = S(tl.HT_TRENDMODE(f(c)), df.index)
    entry = (rsi(c, 2) < 10) & (c > sa) & (c > sb) & (trendmode == 1)
    exit_ = (c > df["high"].shift(1)) & (c > df["high"].shift(2))
    return _out(df, entry, exit_)


@signal("rsi2_jesse")
def rsi2_jesse(df, p, info):
    c = df["close"]
    return _out(df, (c > sma(c, 200)) & (rsi(c, 2) <= 10), c > sma(c, 5))


# --------------------------------------------------------------------------------------- paulcpk "that work"
@signal("rsi_directional_trend")
def rsi_dir(df, p, info):
    r, e = rsi(df["close"], 4), ema(df["close"], 100)
    entry = crossed_above(r, pd.Series(15.0, index=df.index)) & (df["low"] > e) & _vol_ok(df)
    exit_ = crossed_above(r, pd.Series(85.0, index=df.index)) | (df["low"] < e)
    return _out(df, entry, exit_)


@signal("ema800_cross")
def ema800(df, p, info):
    c = df["close"]
    e = ema(c, 800)
    return _out(df, crossed_above(c, e) & _vol_ok(df), crossed_below(c, e * 0.99))


@signal("macd_cross_trend")
def macd_trend(df, p, info):
    m, s, _ = macd(df["close"])
    e = ema(df["close"], 100)
    entry = (m < 0) & crossed_above(m, s) & (df["low"] > e) & _vol_ok(df)
    exit_ = crossed_below(m, pd.Series(0.0, index=df.index)) | (df["low"] < e)
    return _out(df, entry, exit_)


# --------------------------------------------------------------------------------------- official repo
@signal("hlhb")
def hlhb(df, p, info):
    hl2 = (df["close"] + df["open"]) / 2
    r = rsi(hl2, 10)
    e5, e10 = ema(hl2, 5), ema(hl2, 10)
    a = adx(df, 14)
    fifty = pd.Series(50.0, index=df.index)
    entry = crossed_above(r, fifty) & crossed_above(e5, e10) & (a > 25) & _vol_ok(df)
    exit_ = crossed_below(r, fifty) & crossed_below(e5, e10) & (a > 25)
    return _out(df, entry, exit_)


@signal("cofibit")
def cofibit(df, p, info):
    eh, el = ema(df["high"], 5), ema(df["low"], 5)
    fk, fd = stochf(df, 5, 3)
    a = adx(df, 14)
    entry = (df["open"] < el) & crossed_above(fk, fd) & (fk < 25) & (fd < 25) & (a > 25)
    lvl = pd.Series(75.0, index=df.index)
    exit_ = (df["open"] >= eh) | crossed_above(fk, lvl) | crossed_above(fd, lvl)
    return _out(df, entry, exit_)


@signal("bandtastic")
def bandtastic(df, p, info):
    tp = typical(df)
    lo1, _, _ = bb_qtpy(tp, 20, 1)
    _, _, up2 = bb_qtpy(tp, 20, 2)
    entry = (df["close"] < lo1) & _vol_ok(df)
    exit_ = (mfi(df, 14) > 46) & (df["close"] > up2) & _vol_ok(df)
    return _out(df, entry, exit_)


def _td_setup(cond: pd.Series) -> pd.Series:
    out = np.zeros(len(cond))
    a = cond.to_numpy()
    for i in range(len(a)):
        out[i] = out[i - 1] + 1 if (a[i] and i > 0) else (1 if a[i] else 0)
    return pd.Series(out, index=cond.index)


@signal("tdsequential")
def tdseq(df, p, info):
    c, lo, hi = df["close"], df["low"], df["high"]
    sb = _td_setup(c < c.shift(4))
    ss = _td_setup(c > c.shift(4))
    # perfection: the low of bar 8 or 9 is below the lows of bars 6 and 7 of the same setup
    lo6_7 = pd.concat([lo.shift(2), lo.shift(1)], axis=1).min(axis=1)
    lo6_7_at9 = pd.concat([lo.shift(3), lo.shift(2)], axis=1).min(axis=1)
    perf_buy = ((sb == 8) & (lo <= lo6_7)) | ((sb >= 9) & ((lo <= lo6_7_at9) | (lo.shift(1) <= lo6_7.shift(1))))
    hi6_7 = pd.concat([hi.shift(2), hi.shift(1)], axis=1).max(axis=1)
    hi6_7_at9 = pd.concat([hi.shift(3), hi.shift(2)], axis=1).max(axis=1)
    perf_sell = ((ss == 8) & (hi >= hi6_7)) | ((ss >= 9) & ((hi >= hi6_7_at9) | (hi.shift(1) >= hi6_7.shift(1))))
    return _out(df, (sb > 8) & perf_buy, (ss > 8) | perf_sell)


@signal("awesome_macd")
def awesome_macd(df, p, info):
    m, _, _ = macd(df["close"])
    mp = (df["high"] + df["low"]) / 2
    ao = sma(mp, 5) - sma(mp, 34)
    return _out(df, (m > 0) & (ao > 0) & (ao.shift(1) < 0), (m < 0) & (ao < 0) & (ao.shift(1) > 0))


@signal("cmcwinner")
def cmcwinner(df, p, info):
    cc, mf, cm = cci(df, 14), mfi(df, 14), cmo(df["close"], 14)
    entry = (cc.shift(1) < -100) & (mf.shift(1) < 20) & (cm.shift(1) < -50)
    exit_ = (cc.shift(1) > 100) & (mf.shift(1) > 80) & (cm.shift(1) > 50)
    return _out(df, entry, exit_)


@signal("simple_bollinger_jesse")
def simple_bollinger(df, p, info):
    hl2 = (df["high"] + df["low"]) / 2
    mid = sma(hl2, 20)
    up = mid + 2 * hl2.rolling(20).std(ddof=0)
    _, _, sa, sb = ichimoku(df, 9, 26, 52, 26)
    c = df["close"]
    return _out(df, (c > up) & (c > sa) & (c > sb), c < mid)


@signal("tv_rsi_jesse")
def tv_rsi(df, p, info):
    r = rsi(df["close"], 5)
    s = lambda v: pd.Series(float(v), index=df.index)  # noqa: E731
    o = _out(df, crossed_above(r, s(35)), crossed_below(r, s(75)) | crossed_below(r, s(10)))
    o["sl_pct"], o["tp_pct"] = 0.05, 0.10
    return o


@signal("magen")
def magen(df, p, info):
    c = df["close"]
    fast, slow = linreg(c, 5), linreg(c, 20)
    a8 = adx(df, 8)
    a = atr(df, 32)
    o = _out(df, crossed_above(fast, slow) & (a8 > 13), (fast < slow) & (a8 < 15))
    o["sl_dist"], o["tp_dist"] = 2 * a, 5 * a
    return o


def _ott(c: pd.Series, length=2, percent=1.4):
    """Optimized Trend Tracker (Anıl Özekşi) with the VAR (VIDYA-type) moving average, as in
    Freqtrade's FOttStrategy."""
    alpha = 2 / (length + 1)
    up = (c - c.shift()).clip(lower=0).rolling(9).sum()
    dn = (c.shift() - c).clip(lower=0).rolling(9).sum()
    cmo_abs = ((up - dn) / (up + dn)).abs().fillna(0).to_numpy()
    cv = c.to_numpy()
    var = np.zeros(len(cv))
    for i in range(1, len(cv)):
        var[i] = alpha * cmo_abs[i] * cv[i] + (1 - alpha * cmo_abs[i]) * var[i - 1]
    fark = var * percent * 0.01
    ls = var - fark
    ss = var + fark
    d = np.ones(len(cv))
    for i in range(1, len(cv)):
        ls[i] = max(ls[i], ls[i - 1]) if var[i] > ls[i - 1] else ls[i]
        ss[i] = min(ss[i], ss[i - 1]) if var[i] < ss[i - 1] else ss[i]
        if d[i - 1] == -1 and var[i] > ss[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and var[i] < ls[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    mt = np.where(d == 1, ls, ss)
    ott = np.where(var > mt, mt * (200 + percent) / 200, mt * (200 - percent) / 200)
    return pd.Series(var, index=c.index), pd.Series(ott, index=c.index).shift(2)


@signal("fott")
def fott(df, p, info):
    var, ott = _ott(df["close"])
    return _out(df, crossed_above(var, ott), adx(df, 14) > 60)


@signal("trend_following_obv")
def trend_obv(df, p, info):
    c = df["close"]
    e = c.ewm(span=20, adjust=False).mean()
    ob = S(tl.OBV(f(c), f(df["volume"])), df.index)
    return _out(df, crossed_above(c, e) & (ob > ob.shift(1)), crossed_below(c, e) & (ob > ob.shift(1)))


@signal("emaskippump")
def emaskippump(df, p, info):
    c = df["close"]
    lower, _, upper = bb_qtpy(typical(df), 20, 2)
    e5, e12 = ema(c, 5), ema(c, 12)
    entry = (df["volume"] < df["volume"].rolling(30).mean().shift(1) * 20) & (c < e5) & (c < e12) \
        & (c == lowest(c, 12)) & (c <= lower)
    exit_ = (c > e5) & (c > e12) & (c >= highest(c, 12)) & (c >= upper)
    return _out(df, entry, exit_)


@signal("quickie")
def quickie(df, p, info):
    c = df["close"]
    t = tema(c, 9)
    _, mid, _ = bb_qtpy(c, 20, 2)
    a = adx(df, 14)
    s200 = sma(c, 200)
    entry = (a > 30) & (t < mid) & (t > t.shift(1)) & (s200 > c)
    exit_ = (a > 70) & (t > mid) & (t < t.shift(1))
    return _out(df, entry, exit_)


@signal("strategy001")
def strategy001(df, p, info):
    c = df["close"]
    e20, e50, e100 = ema(c, 20), ema(c, 50), ema(c, 100)
    ha = heikin_ashi(df)
    entry = crossed_above(e20, e50) & (ha["close"] > e20) & (ha["open"] < ha["close"])
    exit_ = crossed_above(e50, e100) & (ha["close"] < e20) & (ha["open"] > ha["close"])
    return _out(df, entry, exit_)


def _fisher_rsi(c):
    return np.tanh(0.1 * (rsi(c, 14) - 50))


@signal("strategy005")
def strategy005(df, p, info):
    c = df["close"]
    r = rsi(c, 14)
    fk, fd = stochf(df, 5, 3)
    fr_norm = 50 * (_fisher_rsi(c) + 1)
    m, _, _ = macd(c)
    entry = (df["volume"] > df["volume"].rolling(150).mean() * 4) & (c < sma(c, 40)) & (fd > fk) & (r > 26) \
        & (fd > 1) & (fr_norm < 5)
    exit_ = crossed_above(r, pd.Series(74.0, index=df.index)) & (m < 0) & (minus_di(df, 14) > 4)
    return _out(df, entry, exit_)


@signal("swinghightosky")
def swinghightosky(df, p, info):
    return _out(df, (cci(df, 72) < -175) & (rsi(df["close"], 36) < 90),
                (cci(df, 66) > -106) & (rsi(df["close"], 45) > 88))


@signal("binhv27")
def binhv27(df, p, info):
    c = df["close"]
    r = S(tl.RSI(f(c), 5), df.index)
    emarsi = S(tl.EMA(f(r.fillna(50)), 5), df.index)
    a = adx(df, 14)
    mdi, pdi = minus_di(df, 14), plus_di(df, 14)
    minusdiema, plusdiema = ema(mdi.fillna(0), 25), ema(pdi.fillna(0), 5)  # noqa: F841
    lowsma, highsma = ema(c, 60), ema(c, 120)
    fastsma, slowsma = sma(c, 120), sma(c, 240)
    bigup = (fastsma > slowsma) & ((fastsma - slowsma) > c / 300)
    bigdown = ~bigup
    trend = fastsma - slowsma
    prep = trend > trend.shift()
    prepconf = prep & (trend.shift() > trend.shift(2))
    contup = (slowsma > slowsma.shift()) & (slowsma.shift() > slowsma.shift(2))
    delta = fastsma - fastsma.shift()
    slowing = delta < delta.shift()
    base = (c < highsma) & (c < lowsma) & (mdi > minusdiema) & (r >= r.shift())
    e1 = ~prep & ~contup & (a > 25) & bigdown & (emarsi <= 20)
    e2 = ~prep & contup & (a > 30) & bigdown & (emarsi <= 20)
    e3 = ~contup & (a > 35) & bigup & (emarsi <= 20)
    e4 = contup & (a > 30) & bigup & (emarsi <= 25)
    entry = base & (e1 | e2 | e3 | e4)
    x1 = ~prepconf & ~contup & ((c > lowsma) | (c > highsma)) & bigdown
    x2 = ~prepconf & ~contup & (c > highsma) & ((emarsi >= 75) | (c > slowsma)) & bigdown
    x3 = ~prepconf & (c > highsma) & (a > 30) & (emarsi >= 80) & bigup
    x4 = prepconf & ~contup & slowing & (emarsi >= 75)
    x5 = prepconf & (mdi < pdi) & (c > lowsma)
    return _out(df, entry, x1 | x2 | x3 | x4 | x5)


@signal("strategy004")
def strategy004(df, p, info):
    c = df["close"]
    a14, a35 = adx(df, 14), adx(df, 35)
    cc = cci(df, 14)
    fk, fd = stochf(df, 5, 3)
    sk, sd = stochf(df, 50, 3)
    mean_vol = df["volume"].rolling(12).mean()
    entry = ((a14 > 50) | (a35 > 26)) & (cc < -100) & (fk.shift(1) < 20) & (fd.shift(1) < 20) \
        & (sk.shift(1) < 30) & (sd.shift(1) < 30) & (fk.shift(1) < fd.shift(1)) & (fk > fd) & (mean_vol > 0.75)
    exit_ = (a35 < 25) & ((fk > 70) | (fd > 70)) & (fk.shift(1) < fd.shift(1)) & (c > ema(c, 5))
    return _out(df, entry, exit_)


@signal("strategy003")
def strategy003(df, p, info):
    c = df["close"]
    r = rsi(c, 14)
    fr = _fisher_rsi(c)
    fk, fd = stochf(df, 5, 3)
    e5, e10, e50, e100 = ema(c, 5), ema(c, 10), ema(c, 50), ema(c, 100)
    entry = (r < 28) & (r > 0) & (c < sma(c, 40)) & (fr < -0.94) & (mfi(df, 14) < 16) \
        & ((e50 > e100) | crossed_above(e5, e10)) & (fd > fk) & (fd > 0)
    exit_ = (sar(df) > c) & (fr > 0.3)
    return _out(df, entry, exit_)


@signal("reinforced_average")
def reinforced_average(df, p, info):
    c = df["close"]
    e8, e21 = ema(c, 8), ema(c, 21)
    big = df.resample("48h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    big["sma50"] = sma(big["close"], 50)
    m = merge_informative(df[[]], big[["sma50"]], TF["4h"], pd.Timedelta(hours=48))
    return _out(df, crossed_above(e8, e21) & (c > m["sma50_inf"]), crossed_above(e21, e8))


@signal("adxsmas")
def adxsmas(df, p, info):
    c = df["close"]
    a = adx(df, 14)
    s3, s6 = sma(c, 3), sma(c, 6)
    return _out(df, (a > 25) & crossed_above(s3, s6), (a < 25) & crossed_above(s6, s3))


@signal("adxmomentum")
def adxmomentum(df, p, info):
    c = df["close"]
    a = adx(df, 14)
    pdi, mdi = plus_di(df, 25), minus_di(df, 25)
    m = mom(c, 14)
    return _out(df, (a > 25) & (m > 0) & (pdi > 25) & (pdi > mdi), (a > 25) & (m < 0) & (mdi > 25) & (pdi < mdi))


@signal("macd_crossed_cci")
def macd_crossed(df, p, info):
    m, s, _ = macd(df["close"])
    cc = cci(df, 14)
    return _out(df, crossed_above(m, s) & (cc <= -50), crossed_below(m, s) & (cc >= 100))


# ======================================================================================= added 13 Oct 2026
def _tl_ema(s: pd.Series, n: int) -> pd.Series:
    """TA-Lib EMA (seeded with an SMA), as `ta.EMA` in Freqtrade strategies."""
    return S(tl.EMA(f(s), n), s.index)


def _ssl_atr(df: pd.DataFrame, length: int, atr_s: pd.Series):
    """SSL channel around SMA(high/low) +- ATR (NostalgiaForInfinity / Obelisk `SSLChannels`)."""
    sma_hi = df["high"].rolling(length).mean() + atr_s
    sma_lo = df["low"].rolling(length).mean() - atr_s
    hlv = pd.Series(np.where(df["close"] > sma_hi, 1.0, np.where(df["close"] < sma_lo, -1.0, np.nan)),
                    index=df.index).ffill()
    down = pd.Series(np.where(hlv < 0, sma_hi, sma_lo), index=df.index)
    up = pd.Series(np.where(hlv < 0, sma_lo, sma_hi), index=df.index)
    return down, up


# --------------------------------------------------------------------------------------- Freqtrade sample
@signal("ft_sample_strategy")
def ft_sample_strategy(df, p, info):
    """freqtrade/templates/sample_strategy.py (the official template, long side)."""
    c = df["close"]
    r = rsi(c, 14)
    _, mid, _ = bb_qtpy(typical(df), 20, 2)
    t = tema(c, 9)
    v = _vol_ok(df)
    entry = cross_above_level(r, 30) & (t <= mid) & (t > t.shift(1)) & v
    exit_ = cross_above_level(r, 70) & (t > mid) & (t < t.shift(1)) & v
    return _out(df, entry, exit_)


# --------------------------------------------------------------------------------------- Strategy002
@signal("strategy002")
def strategy002(df, p, info):
    """freqtrade-strategies Strategy002: oversold hammer below the lower band; exit on SAR + Fisher RSI."""
    c = df["close"]
    slowk, _ = stoch(df, 5, 3, 3)                       # ta.STOCH defaults
    r = rsi(c, 14)
    fisher = np.tanh(0.1 * (r - 50))                    # (e^2x - 1) / (e^2x + 1)
    lower, _, _ = bb_qtpy(typical(df), 20, 2)
    hammer = S(tl.CDLHAMMER(f(df["open"]), f(df["high"]), f(df["low"]), f(c)), df.index)
    entry = (r < 30) & (slowk < 20) & (lower > c) & (hammer == 100)
    return _out(df, entry, (sar(df) > c) & (fisher > 0.3))


# --------------------------------------------------------------------------------------- UniversalMACD
@signal("universal_macd")
def universal_macd(df, p, info):
    """freqtrade-strategies UniversalMACD (mablue): EMA12 / EMA26 - 1 in a band. The published exit
    band has min > max, so it never fires: trades end on the ROI table or the stop, as in the source."""
    c = df["close"]
    u = _tl_ema(c, 12) / _tl_ema(c, 26) - 1
    return _out(df, u.between(-0.01416, -0.01176), u.between(-0.00707, -0.02323))


# --------------------------------------------------------------------------------------- MultiRSI
@signal("multi_rsi")
def multi_rsi(df, p, info):
    """freqtrade-strategies berlinguyinca/MultiRSI: 5m RSI against RSI(14) of 10m and 40m candles
    resampled from the 5m ones (a resampled candle is used once it has closed)."""
    c = df["close"]
    r = rsi(c, 14)
    rr = {}
    for mins in (10, 40):
        big = resample_closed(df, f"{mins}min")
        big["rsi"] = rsi(big["close"], 14)
        rr[mins] = merge_informative(df[[]], big[["rsi"]], TF["5m"], pd.Timedelta(minutes=mins))["rsi_inf"]
    entry = (sma(c, 5) >= sma(c, 200)) & (r < rr[40] - 20)
    return _out(df, entry, (r > rr[10]) & (r > rr[40]))


# --------------------------------------------------------------------------------------- VolatilitySystem
@signal("volatility_system")
def volatility_system(df, p, info):
    """freqtrade-strategies futures/VolatilitySystem (long side): a 3-hour close-to-close move larger
    than 2 x ATR(14) of 3-hour candles. Exit on the opposite move. `add` = a fresh signal, used to add
    the second half-stake (adjust_trade_position)."""
    big = resample_closed(df, "3h")
    big["atr"] = atr(big, 14) * 2.0
    big["chg"] = big["close"].diff()
    m = merge_informative(df[[]], big[["atr", "chg"]], TF["1h"], pd.Timedelta(hours=3))
    a, chg = m["atr_inf"], m["chg_inf"]
    entry = (chg > a.shift(1)).fillna(False)
    exit_ = (-chg > a.shift(1)).fillna(False)
    return _out(df, entry, exit_, add=entry & ~entry.shift(1, fill_value=False))


# --------------------------------------------------------------------------------------- NostalgiaForInfinityV1
@signal("nfi_v1")
def nfi_v1(df, p, info):
    """NostalgiaForInfinityV1 (iterativ), 5m with 1h informative candles: three dip-buy conditions,
    three exit conditions. `roi_exit` reproduces the ROI veto (no ROI exit while RSI > 50 or while the
    buy signal is on)."""
    c, v = df["close"], df["volume"]
    inf = info.get("1h")
    cols = ["ema_50", "ema_200", "rsi", "ssl_down", "ssl_up"]
    if inf is not None and len(inf):
        h = pd.DataFrame(index=inf.index)
        h["ema_50"], h["ema_200"] = _tl_ema(inf["close"], 50), _tl_ema(inf["close"], 200)
        h["rsi"] = rsi(inf["close"], 14)
        h["ssl_down"], h["ssl_up"] = _ssl_atr(inf, 20, atr(inf, 14))
        m = merge_informative(df[[]], h, TF["5m"], TF["1h"])
    else:
        m = pd.DataFrame({k + "_inf": np.nan for k in cols}, index=df.index)
    e50h, e200h, rsih = m["ema_50_inf"], m["ema_200_inf"], m["rsi_inf"]
    lower, mid, _ = bb_qtpy(c, 40, 2)
    bbdelta = (mid - lower).abs()
    closedelta = (c - c.shift()).abs()
    tail = (c - df["low"]).abs()
    bb_lower, _, bb_upper = bb_qtpy(typical(df), 20, 2)
    ema_slow = _tl_ema(c, 50)
    vol_mean_slow = v.rolling(30).mean()
    e50, e200 = _tl_ema(c, 50), _tl_ema(c, 200)
    sma5, sma9 = _tl_ema(c, 5), _tl_ema(c, 9)              # named sma_5 / sma_9 in the source, but EMAs
    r = rsi(c, 14)
    a = (c < sma9) & (c > e200h) & (e50 > e200) & (e50h > e200h) & (lower.shift() > 0) \
        & (bbdelta > c * 0.045) & (closedelta > c * 0.023) & (tail < bbdelta * 0.266) \
        & (c < lower.shift()) & (c <= c.shift()) & (v > 0)
    b = (c < sma9) & (c > e200) & (c > e200h) & (c < ema_slow) & (c < 0.992 * bb_lower) \
        & (v < vol_mean_slow.shift(1) * 34)
    cc = (c < sma5) & (m["ssl_up_inf"] > m["ssl_down_inf"]) & (e50 > e200) & (e50h > e200h) \
        & (r < rsih - 36.815) & (v > 0)
    entry = (a | b | cc).fillna(False)
    above = c > bb_upper
    exit_ = (above & above.shift(1, fill_value=False) & above.shift(2, fill_value=False) & (v > 0)) \
        | ((r > 78) & (v > 0)) | ((c < e200) & (c > e50) & (r > 50) & (v > 0))
    return _out(df, entry, exit_, roi_exit=(r <= 50) & ~entry)


# --------------------------------------------------------------------------------------- ichiV1
@signal("ichiv1")
def ichiv1(df, p, info):
    """IchisV1 (Freqtrade community): Heikin-Ashi open/high/low with the real close; close above the
    Ichimoku cloud (20/60/120, displacement 30), close above open on 5 EMA scales, and a rising
    EMA12/EMA96 'fan'. Exit when the close crosses below EMA24."""
    c = df["close"]
    ha = heikin_ashi(df)
    o = ha["open"]
    tc = {n: _tl_ema(c, n) for n in (3, 6, 12, 24, 48, 96)}
    to = {n: _tl_ema(o, n) for n in (3, 6, 12, 24, 48)}
    fan = tc[12] / tc[96]
    gain = fan / fan.shift(1)
    _, _, sa, sb = ichimoku(pd.DataFrame({"high": ha["high"], "low": ha["low"]}, index=df.index), 20, 60, 120, 29)
    entry = (c > sa) & (c > sb) & (c > o)
    for n in (3, 6, 12, 24, 48):
        entry &= tc[n] > to[n]
    entry &= (gain >= 1.002) & (fan > 1) & (fan.shift(1) < fan) & (fan.shift(2) < fan) & (fan.shift(3) < fan)
    return _out(df, entry, crossed_below(c, tc[24]))


# --------------------------------------------------------------------------------------- Obelisk Ichimoku Slow
@signal("obelisk_ichimoku_slow")
def obelisk_ichimoku_slow(df, p, info):
    """Obelisk_Ichimoku_Slow_v1_3 (brookmiles/freqtrade-stuff), 1h: a trend state switched on when
    Ichimoku (20/60/120, displacement 30), the SSL channel and EMA50 > EMA200 all agree, and off when
    the SSL channel turns down; buy pullbacks (Elder force index > 0, candle below the SSL top)."""
    c, o = df["close"], df["open"]
    tenkan, kijun, sa, sb = ichimoku(df, 20, 60, 120, 29)
    lead_a = (tenkan + kijun) / 2
    lead_b = (highest(df["high"], 120) + lowest(df["low"], 120)) / 2
    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)          # NaN-skipping, as DataFrame.max
    future_green = lead_a > lead_b
    # source: ((close.shift(-29) > senkou) ...).shift(30), i.e. the previous close vs the cloud 30 candles ago
    chikou_high = (c.shift(1) > sa.shift(30)) & (c.shift(1) > sb.shift(30))
    e50, e200 = _tl_ema(c, 50), _tl_ema(c, 200)
    ema_ok = (c > e50) & (e50 > e200)
    efi = _tl_ema((c - c.shift()) * df["volume"], 13)
    ssl_down, ssl_up = _ssl_atr(df, 10, atr(df, 14))
    ssl_ok = ssl_up > ssl_down
    ichimoku_ok = (tenkan > kijun) & (c > cloud_top) & future_green & chikou_high
    entry_ok = (efi > 0) & (o < ssl_up) & (c < ssl_up)
    trending = pd.Series(np.nan, index=df.index)
    trending[ichimoku_ok & ssl_ok & ema_ok] = 3.0
    trending[~ssl_ok] = 0.0
    trending = trending.ffill()
    return _out(df, (trending > 0) & entry_ok, trending == 0)
