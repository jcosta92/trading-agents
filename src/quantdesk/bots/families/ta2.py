"""Second batch of technical systems (added 24 Nov 2026, replayed from 12 Oct): TradingView built-in
strategies, John Ehlers' indicators and classic book systems (Williams, Raschke, Elder, Chande, Pring,
Dorsey, Guppy, Nison/Bulkowski candlesticks...). Long only: a short signal means exit to cash.
Every function is causal: a row only uses candles up to and including its own."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib as tl

from ..generic import signal
from ..lib import (S, adx, atr, cmo, cross_above_level, cross_below_level, crossed_above, crossed_below, ema, f,
                   highest, hl2, lowest, rma, sma, stoch, typical, wma)
from .crypto_ft import _out


def _ser(a, df):
    return pd.Series(a, index=df.index)


def _true(df):
    return pd.Series(True, index=df.index)


def _false(df):
    return pd.Series(False, index=df.index)


def _flip_on(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


def _turn_up(s: pd.Series) -> pd.Series:
    return (s > s.shift(1)) & (s.shift(1) <= s.shift(2))


def _turn_down(s: pd.Series) -> pd.Series:
    return (s < s.shift(1)) & (s.shift(1) >= s.shift(2))


def _pivots(df, left: int, right: int):
    """TradingView ta.pivothigh/pivotlow(left, right): the value of a pivot, reported `right` bars
    after it (when it is confirmed), NaN elsewhere."""
    h, l = df["high"], df["low"]
    ch, cl = h.shift(right), l.shift(right)
    lh = h.shift(right + 1).rolling(left).max()
    rh = h.rolling(right).max()
    ll = l.shift(right + 1).rolling(left).min()
    rl = l.rolling(right).min()
    ph = ch.where((ch > lh) & (ch > rh))
    pl = cl.where((cl < ll) & (cl < rl))
    return ph, pl


# ============================================================================== TradingView built-ins
@signal("ta2_inside_bar")
def inside_bar(df, p, info):
    ib = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    return _out(df, ib & (df["close"] > df["open"]), ib & (df["close"] < df["open"]))


@signal("ta2_outside_bar")
def outside_bar(df, p, info):
    ob = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    return _out(df, ob & (df["close"] > df["open"]), ob & (df["close"] < df["open"]))


@signal("ta2_pivot_reversal")
def pivot_reversal(df, p, info):
    ph, pl = _pivots(df, int(p.get("left", 4)), int(p.get("right", 2)))
    hprice, lprice = ph.ffill(), pl.ffill()
    # the long stop order stays while the high has not traded through the last pivot high since it formed
    seg = ph.notna().cumsum()
    hit = (df["high"] > hprice).groupby(seg).cummax()
    le = hprice.notna() & ~hit
    o = _out(df, le, _false(df))
    o["entry_price"] = hprice * 1.0001
    o["stop_level"] = lprice * 0.9999
    return o


@signal("ta2_pivot_extension")
def pivot_extension(df, p, info):
    ph, pl = _pivots(df, int(p.get("left", 4)), int(p.get("right", 2)))
    return _out(df, pl.notna(), ph.notna())


def _channel(df, n):
    o = _out(df, _true(df), _false(df))
    o["entry_price"] = highest(df["high"], n) * 1.0001
    o["stop_level"] = lowest(df["low"], n) * 0.9999
    return o


@signal("ta2_channel_breakout")
def channel_breakout(df, p, info):
    return _channel(df, int(p.get("length", 5)))


@signal("ta2_price_channel")
def price_channel(df, p, info):
    return _channel(df, int(p.get("length", 20)))


@signal("ta2_ma2line_cross")
def ma2line_cross(df, p, info):
    fa, sl = sma(df["close"], int(p.get("fast", 9))), sma(df["close"], int(p.get("slow", 18)))
    return _out(df, crossed_above(fa, sl), crossed_below(fa, sl))


@signal("ta2_ma_cross")
def ma_cross(df, p, info):
    m = sma(df["close"], int(p.get("length", 9)))
    above = df["close"] > m
    below = df["close"] < m
    return _out(df, above & ~above.shift(1, fill_value=False), below & ~below.shift(1, fill_value=False))


@signal("ta2_bar_updn")
def bar_updn(df, p, info):
    o, c = df["open"], df["close"]
    return _out(df, (c > o) & (o > c.shift(1)), (c < o) & (o < c.shift(1)))


@signal("ta2_volty_expan")
def volty_expan(df, p, info):
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    atrs = sma(tr, int(p.get("length", 5))) * float(p.get("num_atrs", 0.75))
    o = _out(df, atrs.notna(), _false(df))
    o["entry_price"] = df["close"] + atrs
    o["stop_level"] = df["close"] - atrs
    return o


@signal("ta2_macd_tv")
def macd_tv(df, p, info):
    c = df["close"]
    m = ema(c, 12) - ema(c, 26)
    delta = m - ema(m, 9)
    return _out(df, cross_above_level(delta, 0), cross_below_level(delta, 0))


# ============================================================================== John Ehlers
@signal("ta2_fisher")
def fisher(df, p, info):
    n = int(p.get("length", 10))
    x = hl2(df)
    hh, ll = f(highest(x, n)), f(lowest(x, n))
    xv = f(x)
    val = np.zeros(len(df))
    fish = np.zeros(len(df))
    for i in range(1, len(df)):
        rng = hh[i] - ll[i]
        raw = (xv[i] - ll[i]) / rng - 0.5 if rng > 0 and not np.isnan(rng) else 0.0
        v = 0.66 * raw + 0.67 * val[i - 1]
        v = min(max(v, -0.999), 0.999)
        val[i] = v
        fish[i] = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * fish[i - 1]
    fs = _ser(fish, df)
    fs.iloc[:n] = np.nan
    trig = fs.shift(1)
    return _out(df, crossed_above(fs, trig), crossed_below(fs, trig))


@signal("ta2_mama")
def mama(df, p, info):
    m, fa = tl.MAMA(f(df["close"]), fastlimit=0.5, slowlimit=0.05)
    m, fa = S(m, df.index), S(fa, df.index)
    return _out(df, crossed_above(m, fa), crossed_below(m, fa))


@signal("ta2_laguerre_rsi")
def laguerre_rsi(df, p, info):
    g = float(p.get("gamma", 0.5))
    c = f(df["close"])
    n = len(c)
    L0 = np.zeros(n); L1 = np.zeros(n); L2 = np.zeros(n); L3 = np.zeros(n)
    out = np.full(n, np.nan)
    for i in range(n):
        p0, p1, p2, p3 = (L0[i - 1], L1[i - 1], L2[i - 1], L3[i - 1]) if i else (c[0], c[0], c[0], c[0])
        L0[i] = (1 - g) * c[i] + g * p0
        L1[i] = -g * L0[i] + p0 + g * p1
        L2[i] = -g * L1[i] + p1 + g * p2
        L3[i] = -g * L2[i] + p2 + g * p3
        cu = max(L0[i] - L1[i], 0) + max(L1[i] - L2[i], 0) + max(L2[i] - L3[i], 0)
        cd = max(L1[i] - L0[i], 0) + max(L2[i] - L1[i], 0) + max(L3[i] - L2[i], 0)
        out[i] = cu / (cu + cd) if cu + cd > 0 else (out[i - 1] if i else 0.5)
    r = _ser(out, df)
    r.iloc[:4] = np.nan
    return _out(df, cross_above_level(r, 0.2), cross_below_level(r, 0.8))


@signal("ta2_itrend")
def itrend(df, p, info):
    a = float(p.get("alpha", 0.07))
    x = f(hl2(df))
    n = len(x)
    it = np.zeros(n)
    for i in range(n):
        if i < 7:
            it[i] = (x[i] + 2 * x[max(i - 1, 0)] + x[max(i - 2, 0)]) / 4
        else:
            it[i] = ((a - a * a / 4) * x[i] + 0.5 * a * a * x[i - 1] - (a - 0.75 * a * a) * x[i - 2]
                     + 2 * (1 - a) * it[i - 1] - (1 - a) ** 2 * it[i - 2])
    its = _ser(it, df)
    trig = 2 * its - its.shift(2)
    its.iloc[:10] = np.nan
    return _out(df, crossed_above(trig, its), crossed_below(trig, its))


@signal("ta2_cyber_cycle")
def cyber_cycle(df, p, info):
    a = float(p.get("alpha", 0.07))
    x = f(hl2(df))
    n = len(x)
    sm = np.zeros(n)
    cy = np.zeros(n)
    for i in range(n):
        if i >= 3:
            sm[i] = (x[i] + 2 * x[i - 1] + 2 * x[i - 2] + x[i - 3]) / 6
        if i < 7:
            cy[i] = (x[i] - 2 * x[max(i - 1, 0)] + x[max(i - 2, 0)]) / 4
        else:
            cy[i] = ((1 - 0.5 * a) ** 2 * (sm[i] - 2 * sm[i - 1] + sm[i - 2]) + 2 * (1 - a) * cy[i - 1]
                     - (1 - a) ** 2 * cy[i - 2])
    cs = _ser(cy, df)
    trig = cs.shift(1)
    cs.iloc[:10] = np.nan
    return _out(df, crossed_above(cs, trig), crossed_below(cs, trig))


@signal("ta2_zero_lag_ec")
def zero_lag_ec(df, p, info):
    """Ehlers & Way (2010) error-correcting EMA: the gain in [-5, 5] that best fits the close."""
    n = int(p.get("length", 20))
    gl = int(p.get("gain_limit", 50))
    thr = float(p.get("threshold", 1.0))
    alpha = 2 / (n + 1)
    c = f(df["close"])
    e = f(ema(df["close"], n))
    gains = np.arange(-gl, gl + 1) / 10.0
    ec = np.zeros(len(c))
    err = np.zeros(len(c))
    ec[0] = c[0]
    for i in range(1, len(c)):
        cand = alpha * (e[i] + gains * (c[i] - ec[i - 1])) + (1 - alpha) * ec[i - 1]
        j = int(np.argmin(np.abs(c[i] - cand)))
        ec[i] = cand[j]
        err[i] = abs(c[i] - cand[j])
    ecs, es = _ser(ec, df), _ser(e, df)
    ok = 100 * _ser(err, df) / df["close"] > thr
    ecs.iloc[:n] = np.nan
    return _out(df, crossed_above(ecs, es) & ok, crossed_below(ecs, es) & ok)


@signal("ta2_ht_sine")
def ht_sine(df, p, info):
    s, lead = tl.HT_SINE(f(df["close"]))
    s, lead = S(s, df.index), S(lead, df.index)
    return _out(df, crossed_above(lead, s), crossed_below(lead, s))


# ============================================================================== oscillators and averages
@signal("ta2_schaff")
def schaff(df, p, info):
    fast, slow, cyc, fac = int(p.get("fast", 23)), int(p.get("slow", 50)), int(p.get("cycle", 10)), 0.5
    m = f(ema(df["close"], fast) - ema(df["close"], slow))
    n = len(m)

    def stoch_smooth(x):
        out = np.full(n, np.nan)
        prev = np.nan
        for i in range(n):
            w = x[max(0, i - cyc + 1):i + 1]
            w = w[~np.isnan(w)]
            if len(w) == 0:
                continue
            lo, hi = w.min(), w.max()
            k = (x[i] - lo) / (hi - lo) * 100 if hi > lo else (prev if not np.isnan(prev) else 0.0)
            prev = k if np.isnan(prev) else prev + fac * (k - prev)
            out[i] = prev
        return out
    stc = _ser(stoch_smooth(stoch_smooth(m)), df)
    stc.iloc[:slow] = np.nan
    return _out(df, cross_above_level(stc, 25), cross_below_level(stc, 75))


@signal("ta2_coppock")
def coppock(df, p, info):
    c = df["close"]
    cc = wma(c.pct_change(14) * 100 + c.pct_change(11) * 100, 10)
    return _out(df, _turn_up(cc) & (cc < 0), _turn_down(cc) & (cc > 0))


@signal("ta2_kst")
def kst(df, p, info):
    c = df["close"]
    r = lambda n: c.pct_change(n) * 100          # noqa: E731
    k = sma(r(10), 10) + 2 * sma(r(15), 10) + 3 * sma(r(20), 10) + 4 * sma(r(30), 15)
    sig = sma(k, 9)
    return _out(df, crossed_above(k, sig), crossed_below(k, sig))


@signal("ta2_mass_index")
def mass_index(df, p, info):
    rng = df["high"] - df["low"]
    e1 = ema(rng, 9)
    mi = (e1 / ema(e1, 9)).rolling(25).sum()
    e9 = ema(df["close"], 9)
    v = f(mi)
    bulge = np.zeros(len(v), dtype=bool)
    armed = False
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        if v[i] > 27:
            armed = True
        elif armed and v[i] < 26.5:
            bulge[i] = True
            armed = False
    b = _ser(bulge, df)
    falling = e9 < e9.shift(1)
    return _out(df, b & falling, b & ~falling)


@signal("ta2_williams_r")
def williams_r(df, p, info):
    r = S(tl.WILLR(f(df["high"]), f(df["low"]), f(df["close"]), 14), df.index)
    return _out(df, cross_above_level(r, -80), cross_below_level(r, -20))


def _alligator(df):
    x = hl2(df)
    jaw = rma(x, 13).shift(8)
    teeth = rma(x, 8).shift(5)
    lips = rma(x, 5).shift(3)
    return jaw, teeth, lips


@signal("ta2_alligator")
def alligator(df, p, info):
    jaw, teeth, lips = _alligator(df)
    aligned = (lips > teeth) & (teeth > jaw)
    return _out(df, _flip_on(aligned), crossed_below(lips, teeth))


@signal("ta2_fractals")
def fractals(df, p, info):
    h, l = df["high"], df["low"]
    hc, lc = h.shift(2), l.shift(2)
    up = hc.where((hc > h.shift(4)) & (hc > h.shift(3)) & (hc > h.shift(1)) & (hc > h))
    dn = lc.where((lc < l.shift(4)) & (lc < l.shift(3)) & (lc < l.shift(1)) & (lc < l))
    _, teeth, _ = _alligator(df)
    upl, dnl = up.ffill(), dn.ffill()
    o = _out(df, upl.notna() & (upl > teeth) & (df["close"] < upl), _false(df))
    o["entry_price"] = upl * 1.0001
    o["stop_level"] = dnl * 0.9999
    return o


@signal("ta2_raschke_anti")
def raschke_anti(df, p, info):
    lo, hi = lowest(df["low"], 7), highest(df["high"], 7)
    k = sma((df["close"] - lo) / (hi - lo) * 100, 4)
    d = sma(k, 10)
    entry = (d > d.shift(1)) & _turn_up(k) & (k < d)
    return _out(df, entry, (d < d.shift(1)) & _turn_down(k))


@signal("ta2_cmo")
def chande_mo(df, p, info):
    c = cmo(df["close"], int(p.get("length", 20)))
    return _out(df, cross_above_level(c, -50), cross_below_level(c, 50))


@signal("ta2_vidya")
def vidya(df, p, info):
    n = int(p.get("length", 9))
    k = np.abs(f(cmo(df["close"], n))) / 100
    a = 2 / (n + 1)
    c = f(df["close"])
    v = np.full(len(c), np.nan)
    for i in range(len(c)):
        if np.isnan(k[i]):
            continue
        prev = v[i - 1] if i and not np.isnan(v[i - 1]) else c[i]
        v[i] = a * k[i] * c[i] + (1 - a * k[i]) * prev
    vs = _ser(v, df)
    return _out(df, crossed_above(df["close"], vs), crossed_below(df["close"], vs))


@signal("ta2_qstick")
def qstick(df, p, info):
    q = sma(df["close"] - df["open"], int(p.get("length", 8)))
    return _out(df, cross_above_level(q, 0), cross_below_level(q, 0))


@signal("ta2_t3")
def t3(df, p, info):
    t = S(tl.T3(f(df["close"]), timeperiod=5, vfactor=0.7), df.index)
    return _out(df, _turn_up(t), _turn_down(t))


@signal("ta2_mcginley")
def mcginley(df, p, info):
    n = int(p.get("length", 14))
    c = f(df["close"])
    md = np.full(len(c), np.nan)
    md[0] = c[0]
    for i in range(1, len(c)):
        md[i] = md[i - 1] + (c[i] - md[i - 1]) / (n * (c[i] / md[i - 1]) ** 4)
    m = _ser(md, df)
    m.iloc[:n] = np.nan
    return _out(df, crossed_above(df["close"], m), crossed_below(df["close"], m))


@signal("ta2_chop")
def chop(df, p, info):
    n = 14
    tr1 = atr(df, 1)
    ch = 100 * np.log10(tr1.rolling(n).sum() / (highest(df["high"], n) - lowest(df["low"], n))) / np.log10(n)
    m = sma(df["close"], 20)
    return _out(df, cross_below_level(ch, 38.2) & (df["close"] > m),
                cross_above_level(ch, 61.8) | (df["close"] < m))


def _swma(s):
    return (s + 2 * s.shift(1) + 2 * s.shift(2) + s.shift(3)) / 6


@signal("ta2_rvi")
def rvi(df, p, info):
    n = int(p.get("length", 10))
    r = _swma(df["close"] - df["open"]).rolling(n).sum() / _swma(df["high"] - df["low"]).rolling(n).sum()
    sig = _swma(r)
    return _out(df, crossed_above(r, sig) & (r < 0), crossed_below(r, sig) & (r > 0))


@signal("ta2_smi")
def smi(df, p, info):
    k, d, e = int(p.get("k", 10)), int(p.get("d", 3)), int(p.get("ema", 3))
    hh, ll = highest(df["high"], k), lowest(df["low"], k)
    rel = df["close"] - (hh + ll) / 2
    s = 200 * ema(ema(rel, d), d) / ema(ema(hh - ll, d), d)
    sig = ema(s, e)
    return _out(df, crossed_above(s, sig) & (s < -40), crossed_below(s, sig) & (s > 40))


@signal("ta2_demarker")
def demarker(df, p, info):
    n = 14
    dmax = (df["high"] - df["high"].shift(1)).clip(lower=0)
    dmin = (df["low"].shift(1) - df["low"]).clip(lower=0)
    dem = sma(dmax, n) / (sma(dmax, n) + sma(dmin, n))
    return _out(df, cross_above_level(dem, 0.3), cross_below_level(dem, 0.7))


@signal("ta2_klinger")
def klinger(df, p, info):
    t = typical(df)
    sv = df["volume"].where(t.diff() >= 0, -df["volume"])
    kvo = ema(sv, 34) - ema(sv, 55)
    sig = ema(kvo, 13)
    return _out(df, crossed_above(kvo, sig), crossed_below(kvo, sig))


@signal("ta2_keltner_1960")
def keltner_1960(df, p, info):
    mid = sma(typical(df), 10)
    band = sma(df["high"] - df["low"], 10)
    return _out(df, crossed_above(df["close"], mid + band), crossed_below(df["close"], mid - band))


@signal("ta2_donchian_4week")
def donchian_4week(df, p, info):
    n = int(p.get("length", 20))
    o = _out(df, _true(df), _false(df))
    o["entry_price"] = highest(df["high"], n) * 1.0001
    o["stop_level"] = lowest(df["low"], n) * 0.9999
    return o


@signal("ta2_gann_hilo")
def gann_hilo(df, p, info):
    n = int(p.get("length", 3))
    hi, lo = sma(df["high"], n).shift(1), sma(df["low"], n).shift(1)
    c = f(df["close"])
    h, l = f(hi), f(lo)
    d = np.zeros(len(c))
    for i in range(len(c)):
        prev = d[i - 1] if i else 0
        d[i] = 1 if c[i] > h[i] else (-1 if c[i] < l[i] else prev)
    ds = _ser(d, df)
    return _out(df, (ds == 1) & (ds.shift(1) != 1), (ds == -1) & (ds.shift(1) != -1))


@signal("ta2_gmma")
def gmma(df, p, info):
    c = df["close"]
    short = pd.concat([ema(c, n) for n in (3, 5, 8, 10, 12, 15)], axis=1)
    long = pd.concat([ema(c, n) for n in (30, 35, 40, 45, 50, 60)], axis=1)
    up = short.min(axis=1) > long.max(axis=1)
    down = short.max(axis=1) < long.min(axis=1)
    up[c.index[:60]] = False
    return _out(df, _flip_on(up), down)


@signal("ta2_stoch_pop")
def stoch_pop(df, p, info):
    k, _ = stoch(df, 8, 3, 3)
    a = adx(df, 14)
    return _out(df, cross_above_level(k, 80) & (a < 20), cross_below_level(k, 80))


# ============================================================================== daily bar setups (US ETFs)
@signal("ta2_raschke_8020")
def raschke_8020(df, p, info):
    """Yesterday opened in the top 20% of its range and closed in the bottom 20%; today trades below
    yesterday's low and is back above it at 15:55 -> buy, held one day."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    setup = ((o - l) / rng >= 0.8) & ((c - l) / rng <= 0.2)
    ys = setup.shift(1, fill_value=False)
    yl = l.shift(1)
    return _out(df, ys & (l < yl) & (c > yl), _false(df))


@signal("ta2_smash_day")
def smash_day(df, p, info):
    """Larry Williams Smash Day (naked close, type A): a close below the prior day's low; buy when a
    following day trades and closes above the smash day's high (at 15:55); bail out at the first
    profitable close; stop at the smash day's low."""
    h, l, c = df["high"], df["low"], df["close"]
    smash = c < l.shift(1)
    sh = h.where(smash).ffill()
    sl = l.where(smash).ffill()
    since = smash.cumsum()
    fresh = (since > 0) & (smash.groupby(since).cumcount() <= 3) & ~smash
    o = _out(df, fresh & (c > sh), _false(df))
    o["stop_level"] = sl
    return o


@signal("ta2_elder_force")
def elder_force(df, p, info):
    c = df["close"]
    fi2 = ema(c.diff() * df["volume"], 2)
    e22 = ema(c, 22)
    up = e22 > e22.shift(1)
    return _out(df, up & (fi2 < 0), fi2 > 0)


@signal("ta2_elder_ray")
def elder_ray(df, p, info):
    e = ema(df["close"], 13)
    bear = df["low"] - e
    bull = df["high"] - e
    up = e > e.shift(1)
    return _out(df, up & (bear < 0) & (bear > bear.shift(1)), (~up) | ((bull > 0) & (bull < bull.shift(1)) & (bull.shift(1) < bull.shift(2))))


def _cdl(df, fn):
    return S(fn(f(df["open"]), f(df["high"]), f(df["low"]), f(df["close"])), df.index)


@signal("ta2_engulfing")
def engulfing(df, p, info):
    v = _cdl(df, tl.CDLENGULFING)
    c = df["close"]
    return _out(df, (v > 0) & (c.shift(1) < sma(c, 20).shift(1)), _false(df))


@signal("ta2_hammer")
def hammer(df, p, info):
    v = _cdl(df, tl.CDLHAMMER)
    c = df["close"]
    return _out(df, (v > 0) & (c < sma(c, 20)), _false(df))


@signal("ta2_morning_star")
def morning_star(df, p, info):
    v = S(tl.CDLMORNINGSTAR(f(df["open"]), f(df["high"]), f(df["low"]), f(df["close"]), penetration=0.3), df.index)
    return _out(df, v > 0, _false(df))


@signal("ta2_three_soldiers")
def three_soldiers(df, p, info):
    v = _cdl(df, tl.CDL3WHITESOLDIERS)
    return _out(df, v > 0, _false(df))


# ============================================================================== intraday (US ETFs)
NY = "America/New_York"


@signal("ta2_oops")
def oops(df, p, info):
    """Larry Williams 'Oops!': the session opens below yesterday's low -> buy stop at yesterday's low
    for the rest of the day; flat at the close."""
    ny = df.index.tz_convert(NY)
    day = pd.Series(ny.normalize().tz_localize(None), index=df.index)
    mins = pd.Series(ny.hour * 60 + ny.minute, index=df.index)
    dl = df["low"].groupby(day).min()
    dopen = df["open"].groupby(day).first()
    prior_low = day.map(dl.shift(1))
    first = mins == 570
    entry = first & (day.map(dopen) < prior_low)
    o = _out(df, entry, _false(df))
    o["entry_price"] = prior_low.where(entry) * 1.0001
    return o
