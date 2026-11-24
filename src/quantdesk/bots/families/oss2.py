"""Open-source bot strategies, second batch (added 24 Nov 2026, replayed from the 12 Oct start).

Freqtrade (official freqtrade-strategies repo and two well-known community files), Gekko and
Zenbot built-in strategies, Hummingbot V2 directional controllers, a Jesse example, and the
"hello world" examples of backtesting.py, backtrader, vectorbt and QuantConnect Lean, all run long
only on the bots' crypto markets. Each function returns boolean `entry` / `exit` columns (plus
extra columns where the risk rules need them); ROI, stoploss and trailing settings are next to the
source link in config/families.d/20_oss2.yaml. Rules and default parameters are copied from the
published files; where a source uses a candle size the data feed does not have (Zenbot's 2-, 10-
and 30-minute periods, Hummingbot's 3-minute candles) the yaml says what is used instead.

Everything is causal: values at a candle use that candle and earlier ones only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib as tl

from ..data import TF
from ..generic import signal
from ..lib import (S, adx, atr, bb_pop, bb_qtpy, cci, crossed_above, crossed_below, ema, f,
                   macd, merge_informative, mfi, resample_closed, rsi, sma, tema, typical, wma)
from .. import ta as _ta


def _out(df, entry, exit_, **extra) -> pd.DataFrame:
    o = pd.DataFrame({"entry": pd.Series(entry, index=df.index).fillna(False).astype(bool),
                      "exit": pd.Series(exit_, index=df.index).fillna(False).astype(bool)}, index=df.index)
    for k, v in extra.items():
        o[k] = v
    return o


def _vol(df):
    return df["volume"] > 0


def _info_col(df, info, tf_name, base_tf, fn, name):
    """Column `name` of fn(informative frame), aligned to `df` once the informative candle closed."""
    inf = info.get(tf_name)
    if inf is None or not len(inf):
        return pd.Series(np.nan, index=df.index)
    x = pd.DataFrame({name: fn(inf)}, index=inf.index)
    m = merge_informative(df[[]], x, TF[base_tf], TF[tf_name])
    return m[f"{name}_inf"]


def _periods(df, rule: str, base_tf: str, fn, p, cols=("entry", "exit")) -> pd.DataFrame:
    """Run a strategy on longer periods built from the bot's candles (Zenbot's 10- or 30-minute
    periods): fn on the closed resampled candles, its signal visible from the close of each period."""
    per = resample_closed(df, rule)
    out = fn(per, p)
    m = merge_informative(df[[]], out[list(cols)].astype(float), TF[base_tf], pd.Timedelta(rule))
    return _out(df, m["entry_inf"] > 0.5, m["exit_inf"] > 0.5)


def _gekko_ema(s, n):
    """Gekko's EMA: starts at the first price, then k = 2 / (n + 1)."""
    return s.ewm(span=n, adjust=False).mean()


def _persist(up: pd.Series, down: pd.Series, persistence: int, reset_neutral: bool = False):
    """Gekko's trend bookkeeping: a trend starts when the value crosses the `up` (or `down`)
    threshold; after `persistence` candles in that zone it advises once (long / short). Outside both
    zones the trend is kept (StochRSI only resets the candle count)."""
    u, d = up.fillna(False).to_numpy(), down.fillna(False).to_numpy()
    n = len(u)
    ent, ex = np.zeros(n, bool), np.zeros(n, bool)
    direction, dur, adv = None, 0, False
    for i in range(n):
        if u[i]:
            if direction != "up":
                direction, dur, adv = "up", 0, False
            dur += 1
            if dur >= persistence and not adv:
                adv, ent[i] = True, True
        elif d[i]:
            if direction != "down":
                direction, dur, adv = "down", 0, False
            dur += 1
            if dur >= persistence and not adv:
                adv, ex[i] = True, True
        elif reset_neutral:
            dur = 0
    return pd.Series(ent, index=up.index), pd.Series(ex, index=up.index)


def _trend_pulses(up: pd.Series, down: pd.Series):
    """Zenbot's `acted_on_trend`: buy once when the trend turns up, sell once when it turns down."""
    u, d = up.fillna(False).to_numpy(), down.fillna(False).to_numpy()
    ent, ex = np.zeros(len(u), bool), np.zeros(len(u), bool)
    trend = None
    for i in range(len(u)):
        if u[i]:
            if trend != "up":
                ent[i] = True
            trend = "up"
        elif d[i]:
            if trend != "down":
                ex[i] = True
            trend = "down"
    return pd.Series(ent, index=up.index), pd.Series(ex, index=up.index)


def _supertrend_line(df, period, mult):
    """Supertrend line and direction (+1 / -1), the TradingView / pandas_ta logic."""
    a = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    up, lo = hl2 + mult * a, hl2 - mult * a
    fu, fl = up.copy(), lo.copy()
    d = np.full(n, np.nan)
    line = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(d[i - 1]):
            d[i] = 1
            line[i] = fl[i]
            continue
        fl[i] = lo[i] if (lo[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        if d[i - 1] == -1:
            d[i] = 1 if c[i] > fu[i] else -1
        else:
            d[i] = -1 if c[i] < fl[i] else 1
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return pd.Series(line, index=df.index), pd.Series(d, index=df.index)


# =============================================================================== Freqtrade (official repo)
@signal("oss2_bbandrsi")
def bbandrsi(df, p, info):
    c = df["close"]
    r = rsi(c, 14)
    lower, mid, upper = bb_qtpy(typical(df), 20, 2)
    return _out(df, (r < 30) & (c < lower), r > 70)


@signal("oss2_combined_binh_cluc")
def combined_binh_cluc(df, p, info):
    c = df["close"]
    mid40 = c.rolling(40).mean().fillna(0)
    low40 = (c.rolling(40).mean() - c.rolling(40).std() * 2).fillna(0)
    bbdelta = (mid40 - low40).abs()
    closedelta = (c - c.shift()).abs()
    tail = (c - df["low"]).abs()
    lower, mid, _ = bb_qtpy(typical(df), 20, 2)
    ema50 = ema(c, 50)
    vmean = df["volume"].rolling(30).mean()
    binh = (low40.shift() > 0) & (bbdelta > c * 0.008) & (closedelta > c * 0.0175) & (tail < bbdelta * 0.25) \
        & (c < low40.shift()) & (c <= c.shift())
    cluc = (c < ema50) & (c < 0.985 * lower) & (df["volume"] < vmean.shift(1) * 20)
    return _out(df, binh | cluc, c > mid)


@signal("oss2_reinforced_quickie")
def reinforced_quickie(df, p, info):
    c, o = df["close"], df["open"]
    e5, e12 = ema(c, 5), ema(c, 12)
    lower_t, mid_t, upper_t = bb_qtpy(typical(df), 20, 2)
    lower, mid, upper = bb_qtpy(c, 20, 2)            # the file overwrites the typical-price bands
    mn, mx = S(tl.MIN(f(c), 12), df.index), S(tl.MAX(f(c), 12), df.index)
    cc, mf, r = cci(df, 14), mfi(df, 14), rsi(c, 7)
    avg = (c + o + df["high"] + df["low"]) / 4
    rs = _info_col(df, info, "1h", "5m", lambda x: sma(x["close"], 25), "rsma")
    a = (c < e5) & (c < e12) & (c == mn) & (c <= lower)
    b = (avg.shift(5) > avg.shift(4)) & (avg.shift(4) > avg.shift(3)) & (avg.shift(3) > avg.shift(2)) \
        & (avg.shift(2) > avg.shift(1)) & (avg.shift(1) < avg) & (df["low"].shift(1) < mid) \
        & (cc.shift(1) < -100) & (r.shift(1) < 30) & (mf.shift(1) < 30)
    trend = (df["volume"] < df["volume"].rolling(30).mean().shift(1) * 20) & (rs < c) & (rs.shift(1) < rs)
    entry = (a | b) & trend
    green = pd.concat([(o < c).shift(k) for k in range(8)], axis=1).fillna(False).all(axis=1)
    exit_ = ((c > e5) & (c > e12) & (c >= mx) & (c >= upper) & (mf > 80)) | (green & (r > 70))
    return _out(df, entry, exit_)


@signal("oss2_average_strategy")
def average_strategy(df, p, info):
    c = df["close"]
    s, lg = ema(c, 8), ema(c, 21)
    return _out(df, crossed_above(s, lg) & _vol(df), crossed_above(lg, s) & _vol(df))


@signal("oss2_asdts_rockwell")
def asdts_rockwell(df, p, info):
    m, si, _ = macd(df["close"])
    return _out(df, (m > 0) & (m > si), m < si)


def _cmf(df, n):
    h, lo, c, v = df["high"], df["low"], df["close"], df["volume"]
    mfv = (((c - lo) - (h - c)) / (h - lo)).fillna(0.0) * v
    return mfv.rolling(n).sum() / v.rolling(n).sum()


@signal("oss2_technical_example")
def technical_example(df, p, info):
    x = _cmf(df, 21)
    return _out(df, x < 0, x > 0)


@signal("oss2_simple")
def simple(df, p, info):
    c = df["close"]
    m, si, _ = macd(c)
    r = rsi(c, 7)
    _, _, upper = bb_qtpy(c, 12, 2)
    return _out(df, (m > 0) & (m > si) & (upper > upper.shift(1)) & (r > 70), r > 80)


@signal("oss2_macd_cci")
def macd_cci(df, p, info):
    m, si, _ = macd(df["close"])
    cc = cci(df, 14)
    return _out(df, (m > si) & (cc <= -48) & _vol(df), (m < si) & (cc >= 687) & _vol(df))


@signal("oss2_backtest_validation")
def backtest_validation(df, p, info):
    c = df["close"]
    fa, sl = sma(c, 14), sma(c, 28)
    return _out(df, fa > sl, fa < sl)


def _st_dir(df, m, period):
    return _ta.supertrend(df, period, m)


@signal("oss2_supertrend_triple")
def supertrend_triple(df, p, info):
    b = [_st_dir(df, 4, 8), _st_dir(df, 7, 9), _st_dir(df, 1, 8)]
    s = [_st_dir(df, 1, 16), _st_dir(df, 3, 18), _st_dir(df, 6, 18)]
    entry = (b[0] == 1) & (b[1] == 1) & (b[2] == 1) & _vol(df)
    exit_ = (s[0] == -1) & (s[1] == -1) & (s[2] == -1) & _vol(df)
    return _out(df, entry, exit_)


@signal("oss2_fsupertrend")
def fsupertrend(df, p, info):
    b = [_st_dir(df, 4, 8), _st_dir(df, 7, 9), _st_dir(df, 1, 8)]
    entry = (b[0] == 1) & (b[1] == 1) & (b[2] == 1) & _vol(df)
    return _out(df, entry, _st_dir(df, 3, 18) == -1)


@signal("oss2_freinforced")
def freinforced(df, p, info):
    c = df["close"]
    rs = _info_col(df, info, "1h", "5m", lambda x: sma(x["close"], 50), "rsma")
    entry = (c > rs) & crossed_above(ema(c, 8), ema(c, 21))
    return _out(df, entry, adx(df, 14) < 30)


@signal("oss2_fadxsma")
def fadxsma(df, p, info):
    c = df["close"]
    a = adx(df, 14)
    return _out(df, (a > 30) & crossed_above(sma(c, 12), sma(c, 48)), a < 30)


@signal("oss2_multima")
def multima(df, p, info):
    c = df["close"]
    t = {}

    def T(n):
        if n not in t:
            t[n] = tema(c, n)
        return t[n]
    buy = (T(30) < T(15)) & (T(45) < T(30))                       # buy_ma_count 4, gap 15
    sells = [T(68 * k) > T(68 * (k - 1)) for k in range(2, 12)]     # sell_ma_count 12, gap 68 (any)
    sell = pd.concat(sells, axis=1).fillna(False).any(axis=1)
    return _out(df, buy & _vol(df), sell & _vol(df))


@signal("oss2_diamond")
def diamond(df, p, info):
    entry = crossed_above(df["high"].shift(7), df["volume"] * 0.942)
    exit_ = crossed_below(df["high"].shift(10), df["low"] * 1.184)
    return _out(df, entry, exit_)


@signal("oss2_heracles")
def heracles(df, p, info):
    h, lo, c = df["high"], df["low"], df["close"]
    center = typical(df).rolling(20).mean()                       # ta: keltner, original_version=True
    band = (h - lo).rolling(20).mean()
    kcw = ((center + band) - (center - band)) / center * 100
    dhi, dlo = h.rolling(10).max(), lo.rolling(10).min()          # ta: donchian_channel_pband(window=10)
    dcp = (c - dlo) / (dhi - dlo)
    d = dcp.shift(15) / kcw.shift(9)
    return _out(df, d.between(0.16, 0.75), pd.Series(False, index=df.index))


@signal("oss2_mabstra")
def mabstra(df, p, info):
    c = df["close"]
    mojo, fast, slow = sma(c, 7), sma(c, 14), sma(c, 28)
    bmax, bmin, smax, smin = 2.25446, 0.29497, 1.54593, 2.81436
    entry = (mojo / fast > bmin) & (mojo / fast < bmax) & (fast / slow > bmin) & (fast / slow < bmax)
    exit_ = (fast / mojo > smin) & (fast / mojo < smax) & (slow / fast > smin) & (slow / fast < smax)
    return _out(df, entry, exit_)


@signal("oss2_pattern_recognition")
def pattern_recognition(df, p, info):
    x = tl.CDLHIGHWAVE(f(df["open"]), f(df["high"]), f(df["low"]), f(df["close"]))
    return _out(df, S(x, df.index) == -100, pd.Series(False, index=df.index))


@signal("oss2_informative_sample")
def informative_sample(df, p, info):
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    inf = info.get("15m")
    if inf is not None and len(inf):
        x = pd.DataFrame({"c": inf["close"], "s": inf["close"].rolling(20).mean()}, index=inf.index)
        m = merge_informative(df[[]], x, TF["5m"], TF["15m"])
        c15, s15 = m["c_inf"], m["s_inf"]
    else:
        c15 = s15 = pd.Series(np.nan, index=df.index)
    return _out(df, (e20 > e50) & (c15 > s15), (e20 < e50) & (c15 < s15))


@signal("oss2_smooth_operator")
def smooth_operator(df, p, info):
    c, o, h, lo = df["close"], df["open"], df["high"], df["low"]
    cc, r, mf = cci(df, 20), rsi(c, 14), mfi(df, 14)
    sm = lambda s: S(tl.EMA(f(s), 11), df.index)                 # noqa: E731
    lower, mid, upper = bb_qtpy(c, 20, 2)
    combo = (sm(r) * 1.125 + sm(mf) * 1.125 + sm(cc)) / 3
    combo = S(tl.TEMA(f(combo), 21), df.index)
    avg = (c + o + h + lo) / 4
    a = (avg.shift(5) > avg.shift(4)) & (avg.shift(4) > avg.shift(3)) & (avg.shift(3) > avg.shift(2)) \
        & (avg.shift(2) > avg.shift(1)) & (avg.shift(1) < avg) & (lo.shift(1) < mid) & (cc.shift(1) < -100) \
        & (r.shift(1) < 30)
    b = (lo < mid) & (cc < -200) & (r < 30) & (mf < 30)
    d = (mf < 10) & (cc < -150) & (r < mf)
    entry = (a | b | d) & (c > c.shift())
    green8 = pd.concat([(o < c).shift(k) for k in range(8)], axis=1).fillna(False).all(axis=1)
    exit_ = ((combo > 100) & (combo.shift(1) > combo) & (combo.shift(2) < combo.shift(1))
             & (combo.shift(3) < combo.shift(2))) | green8 | ((cc > 200) & (r > 70))
    return _out(df, entry, exit_)


def _ewo(df, a, b, denom):
    c = df["close"]
    return (ema(c, a) - ema(c, b)) / df[denom] * 100


@signal("oss2_notanother_smaoffset")
def notanother_smaoffset(df, p, info):
    c = df["close"]
    mb, ms = ema(c, 14), ema(c, 24)
    ewo = _ewo(df, 50, 200, "low")
    r, rf, rsl = rsi(c, 14), rsi(c, 4), rsi(c, 20)
    hma50 = _hma(c, 50)
    e100 = ema(c, 100)
    base = (rf < 35) & (c < mb * 0.975) & _vol(df) & (c < ms * 0.991)
    entry = (base & (ewo > 2.327) & (r < 69)) | (base & (ewo < -20.988))
    ex = ((c > sma(c, 9)) & (c > ms * 0.997) & (r > 50) & _vol(df) & (rf > rsl)) | \
         ((c < hma50) & (c > ms * 0.991) & _vol(df) & (rf > rsl))
    block = (hma50 * 1.149 > e100) & (c < e100 * 0.951)           # confirm_trade_exit refuses the sell
    return _out(df, entry, ex & ~block)


@signal("oss2_smaoffset_protect_v1")
def smaoffset_protect_v1(df, p, info):
    c = df["close"]
    mb, ms = ema(c, 16), ema(c, 49)
    ewo = _ewo(df, 50, 200, "close")
    r = rsi(c, 14)
    entry = ((c < mb * 0.978) & (ewo > 5.638) & (r < 61) & _vol(df)) | ((c < mb * 0.978) & (ewo < -19.993) & _vol(df))
    return _out(df, entry, (c > ms * 1.006) & _vol(df))


def _hma(s, n):
    """qtpylib.hull_moving_average: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))."""
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))


# =============================================================================== Gekko built-in strategies
@signal("oss2_gekko_dema")
def gekko_dema(df, p, info):
    c = df["close"]
    e = _gekko_ema(c, 21)
    dema = 2 * e - _gekko_ema(e, 21)
    diff = sma(c, 21) - dema
    ent, ex = _persist(diff > 0.025, diff < -0.025, 1)
    return _out(df, ent, ex)


@signal("oss2_gekko_macd")
def gekko_macd(df, p, info):
    c = df["close"]
    m = _gekko_ema(c, 10) - _gekko_ema(c, 21)
    h = m - _gekko_ema(m, 9)
    ent, ex = _persist(h > 0.025, h < -0.025, 1)
    return _out(df, ent, ex)


def _gekko_ppo_hist(c, short=12, long=26, sig=9):
    lg = _gekko_ema(c, long)
    ppo = 100 * (_gekko_ema(c, short) - lg) / lg
    return ppo - _gekko_ema(ppo, sig)


@signal("oss2_gekko_ppo")
def gekko_ppo(df, p, info):
    h = _gekko_ppo_hist(df["close"])
    ent, ex = _persist(h > 0.025, h < -0.025, 2)
    return _out(df, ent, ex)


@signal("oss2_gekko_varppo")
def gekko_varppo(df, p, info):
    c = df["close"]
    h = _gekko_ppo_hist(c)
    mom = c.diff()
    tsi = 100 * _gekko_ema(_gekko_ema(mom, 25), 13) / _gekko_ema(_gekko_ema(mom.abs(), 25), 13)
    low, high = -25 + h * 120, 25 + h * -120
    ent, ex = _persist(tsi < low, tsi > high, 0)
    return _out(df, ent, ex)


@signal("oss2_gekko_cci")
def gekko_cci(df, p, info):
    x = cci(df, 90)
    ent, ex = _persist(x <= -100, x >= 100, 0)
    return _out(df, ent, ex)


@signal("oss2_gekko_stochrsi")
def gekko_stochrsi(df, p, info):
    r = rsi(df["close"], 3)
    lo, hi = r.rolling(3).min(), r.rolling(3).max()
    st = ((r - lo) / (hi - lo) * 100).where(hi > lo, 0.0)
    ent, ex = _persist(st < 20, st > 80, 3, reset_neutral=True)
    return _out(df, ent, ex)


# =============================================================================== Zenbot built-in strategies
def _overbought(c, n, level):
    return rsi(c, n) >= level


@signal("oss2_zenbot_bollinger")
def zenbot_bollinger(df, p, info):
    c = df["close"]
    lower, _, upper = bb_pop(c, 20, 2)
    sell = c > upper
    return _out(df, (c < lower) & ~sell, sell)


@signal("oss2_zenbot_macd")
def zenbot_macd(df, p, info):
    c = df["close"]
    m = ema(c, 12) - ema(c, 26)
    h = m - ema(m, 9)
    buy = (h > 0) & (h.shift(1) <= 0)
    sell = ((h < 0) & (h.shift(1) >= 0)) | _overbought(c, 25, 70)
    return _out(df, buy & ~sell, sell)


@signal("oss2_zenbot_dema")
def zenbot_dema(df, p, info):
    c = df["close"]
    h = ema(c, 10) - ema(c, 21)
    # the overbought branch sets the signal 'sold', which Zenbot does not act on
    return _out(df, (h > 0) & (h.shift(1) <= 0), (h < 0) & (h.shift(1) >= 0))


def _zen_trend(df, up, down, c, ob_n, ob_level):
    ent, ex = _trend_pulses(up, down)
    ob = _overbought(c, ob_n, ob_level)
    return _out(df, ent & ~ob, ex | ob)


@signal("oss2_zenbot_ta_trix")
def zenbot_ta_trix(df, p, info):
    c = df["close"]
    t = S(tl.TRIX(f(c), 30), df.index)
    return _zen_trend(df, t >= 0, t < 0, c, 25, 70)


@signal("oss2_zenbot_ta_ultosc")
def zenbot_ta_ultosc(df, p, info):
    c = df["close"]
    u = S(tl.ULTOSC(f(df["high"]), f(df["low"]), f(c), 7, 14, 28), df.index)
    return _zen_trend(df, u > 65, (u < 50) & (u > 0), c, 25, 90)


@signal("oss2_zenbot_ti_hma")
def zenbot_ti_hma(df, p, info):
    c = df["close"]
    h = _hma(c, 36)
    rate = h.pct_change() * 100
    return _zen_trend(df, rate > 0, rate < 0, c, 25, 70)


def _ta_ppo_period(per, p):
    c = per["close"]
    ppo = S(tl.PPO(f(c), 12, 26, 0), per.index)
    return _zen_trend(per, ppo >= 0, ppo < 0, c, 25, 70)


@signal("oss2_zenbot_ta_ppo")
def zenbot_ta_ppo(df, p, info):
    return _periods(df, "10min", "5m", _ta_ppo_period, p)


def _srsi_macd_period(per, p):
    c = per["close"]
    r = rsi(c, 14)
    lo, hi = r.rolling(14).min(), r.rolling(14).max()
    st = ((r - lo) / (hi - lo) * 100).where(hi > lo, 0.0)
    k = st.rolling(5).mean()
    d = k.rolling(3).mean()
    m = ema(c, 24) - ema(c, 200)
    h = m - ema(m, 9)
    buy = (h >= 0) & (k > d) & (k > k.shift(1)) & (k < 20)
    sell = (h < 0) & (k < d) & (k < k.shift(1)) & (k > 80)
    return _out(per, buy & ~sell, sell)


@signal("oss2_zenbot_srsi_macd")
def zenbot_srsi_macd(df, p, info):
    return _periods(df, "30min", "15m", _srsi_macd_period, p)


def _ehlers_period(per, p):
    src = ((per["high"] + per["low"]) / 2).to_numpy()
    n = len(src)
    n1 = np.zeros(n)
    fish = np.zeros(n)
    pos = np.zeros(n)
    for i in range(n):
        lo_i = max(0, i - 9)
        mx, mn = src[lo_i:i + 1].max(), src[lo_i:i + 1].min()
        prev_n1 = n1[i - 1] if i else 0.0
        v = 0.33 * 2 * ((src[i] - mn) / (mx - mn) - 0.5) + 0.67 * prev_n1 if mx > mn else 0.67 * prev_n1
        n1[i] = v
        n2 = 0.999 if v > 0.99 else (-0.999 if v < -0.99 else v)
        fish[i] = 0.5 * np.log((1 + n2) / (1 - n2)) + 0.5 * (fish[i - 1] if i else 0.0)
        prev_f = fish[i - 1] if i else 0.0
        pos[i] = 1 if fish[i] > prev_f else (-1 if fish[i] < prev_f else (pos[i - 1] if i else 0))
    pos_s = pd.Series(pos, index=per.index)
    pos_s.iloc[:10] = 0                                 # preroll
    return _out(per, pos_s == 1, pos_s == -1)


@signal("oss2_zenbot_ehlers_ft")
def zenbot_ehlers_ft(df, p, info):
    return _periods(df, "30min", "15m", _ehlers_period, p)


def _zen_srsi(c, n=14, k=5, d=3):
    r = rsi(c, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    st = ((r - lo) / (hi - lo) * 100).where(hi > lo, 0.0)
    kk = st.rolling(k).mean()
    return kk, kk.rolling(d).mean()


def _cci_srsi_period(per, p):
    c = per["close"]
    te = ema(c, 30)                                     # trend_ema over min_periods (30)
    up = te > te.shift(1)
    acc = (te / te.shift(1) - 1).abs() * 100
    k, _ = _zen_srsi(c, 14, 5, 3)
    cc = cci(per, 14)
    cci_down, k_down = cc < cc.shift(1), k < k.shift(1)
    side = (acc < 0.03)
    buy = (side | up) & (cc <= -90) & (k <= 18) & ~cci_down & ~k_down
    sell = (side | ~up) & (cc >= 140) & (k >= 85) & (cci_down | k_down)
    return _out(per, buy & ~sell, sell)


@signal("oss2_zenbot_cci_srsi")
def zenbot_cci_srsi(df, p, info):
    return _periods(df, "20min", "5m", _cci_srsi_period, p)


@signal("oss2_zenbot_sar")
def zenbot_sar(df, p, info):
    h, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    af0, afmax = 0.015, 0.3
    trend = np.zeros(n)                                  # +1 up / -1 down, 0 before min_periods
    tr = sar = ep = af = None
    for i in range(n):
        if i < 52:
            continue
        if tr is None:
            if h[i] > h[0]:
                tr, sar, ep = 1, min(lo[i - 2], lo[i - 1]), max(h[:i + 1].max(), h[i])
            else:
                tr, sar, ep = -1, max(h[i - 2], h[i - 1]), min(lo[:i + 1].min(), lo[i])
            af = af0
            trend[i] = tr
            continue
        if tr == 1:
            sar = min(lo[i - 2], lo[i - 1], sar + af * (ep - sar))
        else:
            sar = max(h[i - 2], h[i - 1], sar - af * (sar - ep))
        if tr == -1:
            if h[i] >= sar and c[i] > c[i - 1]:
                tr, ep, af = 1, lo[i], af0
                sar = min(lo[i - 1], lo[i], sar + af * (ep - sar))
            elif lo[i] < ep:
                ep = lo[i]
                af = af + af0 if af < afmax else af
        else:
            if lo[i] <= sar and c[i] < c[i - 1]:
                tr, ep, af = -1, h[i], af0
                sar = max(h[i - 1], h[i], sar - af * (sar - ep))
            elif h[i] > ep:
                ep = h[i]
                af = af + af0 if af < afmax else af
        trend[i] = tr
    t = pd.Series(trend, index=df.index)
    return _out(df, t == 1, t == -1)


# =============================================================================== Hummingbot V2 controllers
@signal("oss2_hb_bollinger_v1")
def hb_bollinger_v1(df, p, info):
    c = df["close"]
    lower, _, upper = bb_pop(c, 100, 2.0)
    bbp = (c - lower) / (upper - lower)
    return _out(df, bbp < 0.0, pd.Series(False, index=df.index), sl_pct=0.03, tp_pct=0.02)


@signal("oss2_hb_supertrend_v1")
def hb_supertrend_v1(df, p, info):
    c = df["close"]
    line, d = _supertrend_line(df, 20, 4.0)
    dist = (c - line).abs() / c
    return _out(df, (d == 1) & (dist < 0.01), pd.Series(False, index=df.index), sl_pct=0.03, tp_pct=0.02)


# =============================================================================== Jesse / backtesting libraries
@signal("oss2_jesse_donchian")
def jesse_donchian(df, p, info):
    c = df["close"]
    up = df["high"].rolling(20).max().shift(1)          # donchian(self.candles[:-1])
    dn = df["low"].rolling(20).min().shift(1)
    return _out(df, (c > up) & (c > sma(c, 200)), c < dn)


@signal("oss2_backtestingpy_smacross")
def backtestingpy_smacross(df, p, info):
    c = df["close"]
    a, b = sma(c, 10), sma(c, 20)
    return _out(df, crossed_above(a, b), crossed_above(b, a))


@signal("oss2_backtrader_sma15")
def backtrader_sma15(df, p, info):
    c = df["close"]
    s = sma(c, 15)
    return _out(df, c > s, c < s)


@signal("oss2_vectorbt_ma_10_50")
def vectorbt_ma(df, p, info):
    c = df["close"]
    a, b = sma(c, 10), sma(c, 50)
    return _out(df, crossed_above(a, b), crossed_below(a, b))


@signal("oss2_lean_macd_trend")
def lean_macd_trend(df, p, info):
    c = df["close"]
    m, si, _ = macd(c, 12, 26, 9)
    d = (m - si) / ema(c, 12)
    return _out(df, d > 0.0025, d < -0.0025)


@signal("oss2_lean_ma_cross")
def lean_ma_cross(df, p, info):
    c = df["close"]
    fa, sl = ema(c, 15), ema(c, 30)
    return _out(df, fa > sl * 1.00015, fa < sl)


@signal("oss2_lean_rsi_alpha")
def lean_rsi_alpha(df, p, info):
    r = rsi(df["close"], 14).to_numpy()
    n = len(r)
    ent, ex = np.zeros(n, bool), np.zeros(n, bool)
    state = "middle"
    for i in range(n):
        x = r[i]
        if np.isnan(x):
            continue
        if x > 70:
            new = "high"
        elif x < 30:
            new = "low"
        elif state == "low" and x > 35:
            new = "middle"
        elif state == "high" and x < 65:
            new = "middle"
        else:
            new = state
        if new != state:
            ent[i] = new == "low"
            ex[i] = new == "high"
        state = new
    return _out(df, ent, ex)


@signal("oss2_lean_ema_cross_alpha")
def lean_ema_cross_alpha(df, p, info):
    c = df["close"]
    fa, sl = ema(c, 12), ema(c, 26)
    return _out(df, crossed_above(fa, sl), crossed_below(fa, sl))


@signal("oss2_lean_macd_alpha")
def lean_macd_alpha(df, p, info):
    c = df["close"]
    _, si, _ = macd(c, 12, 26, 9)
    ns = si / c
    d = pd.Series(np.where(ns > 0.01, 1, np.where(ns < -0.01, -1, 0)), index=df.index).where(si.notna())
    chg = d != d.shift(1)
    return _out(df, chg & (d == 1), chg & (d <= 0))
