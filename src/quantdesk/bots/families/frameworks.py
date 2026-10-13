"""Strategies from open-source bot frameworks other than Freqtrade/Jesse: Gekko, Zenbot, Hummingbot V2,
NautilusTrader and OctoBot. Rules and default parameters were read from each project's source file
(links in config/families.yaml). Indicators that the frameworks compute in their own way (Gekko's and
NautilusTrader's EMA seeded with the first value, Tulip's Klinger oscillator, Zenbot's band) are
reproduced here rather than swapped for a TA-Lib look-alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib as tl

from ..generic import signal
from ..lib import S, f, macd, rsi, sma
from ..ta import true_range
from .crypto_ft import _out


def _ema_first(s: pd.Series, n: int) -> pd.Series:
    """EMA seeded with the first value (Gekko, Tulip, NautilusTrader)."""
    return s.ewm(span=n, adjust=False).mean()


def _zone_transitions(x: pd.Series, low: float, high: float) -> tuple[pd.Series, pd.Series]:
    """Gekko's oscillator pattern (persistence 1): advise 'long' on the first candle below `low` after
    the last advice was not 'long', 'short' on the first candle above `high` after it was not 'short'.
    A return to the neutral zone does not reset the direction."""
    a = x.to_numpy(dtype=float)
    ent, ex = np.zeros(len(a), bool), np.zeros(len(a), bool)
    state = ""
    for i, v in enumerate(a):
        if np.isnan(v):
            continue
        if v > high:
            if state != "high":
                ex[i], state = True, "high"
        elif v < low:
            if state != "low":
                ent[i], state = True, "low"
    return pd.Series(ent, index=x.index), pd.Series(ex, index=x.index)


# ============================================================================================ Gekko
@signal("rsi_bull_bear")
def rsi_bull_bear(df, p, info):
    """Tommie Hansen's RSI_BULL_BEAR (Gekko): SMA50 vs SMA1000 picks the regime; bull market RSI(10)
    buys below 60 and sells above 80, bear market RSI(15) buys below 20 and sells above 50."""
    c = df["close"]
    slow, fast = sma(c, 1000), sma(c, 50)
    bull_r, bear_r = rsi(c, 10), rsi(c, 15)
    ready = slow.notna() & fast.notna()
    bear = fast < slow
    sell = ready & ((bear & (bear_r > 50)) | (~bear & (bull_r > 80)))
    buy = ready & ~sell & ((bear & (bear_r < 20)) | (~bear & (bull_r < 60)))
    return _out(df, buy, sell)


@signal("gekko_tsi")
def gekko_tsi(df, p, info):
    """Gekko's TSI strategy: True Strength Index (25, 13) below -25 buys, above +25 sells."""
    m = df["close"].diff()
    num = _ema_first(_ema_first(m.dropna(), 25), 13)
    den = _ema_first(_ema_first(m.dropna().abs(), 25), 13)
    tsi = (100 * num / den).reindex(df.index)
    ent, ex = _zone_transitions(tsi, -25, 25)
    return _out(df, ent, ex)


@signal("gekko_uo")
def gekko_uo(df, p, info):
    """Gekko's UO strategy: Ultimate Oscillator (7/14/28, weights 4/2/1) below 30 buys, above 70 sells."""
    uo = S(tl.ULTOSC(f(df["high"]), f(df["low"]), f(df["close"]), 7, 14, 28), df.index)
    ent, ex = _zone_transitions(uo, 30, 70)
    return _out(df, ent, ex)


# ============================================================================================ Zenbot
def _zenbot_stddev(x: pd.Series, length: int) -> pd.Series:
    """Zenbot lib/stddev.js: mean over the current value and `length` previous ones, squared
    deviations summed over the previous ones only, divided by length + 1."""
    n = length + 1
    avg = x.rolling(n).mean()
    ss_all = x.rolling(n).var(ddof=0) * n
    return np.sqrt((ss_all - (x - avg) ** 2).clip(lower=0) / n)


@signal("zenbot_trend_ema")
def zenbot_trend_ema(df, p, info):
    """Zenbot's default trend_ema strategy: buy when the EMA(26) rate of change rises above its own
    13-period standard deviation band, sell when it falls below minus the band; RSI(14) <= 10 buys
    the dip and blocks trend sells until the rate turns up again."""
    c = df["close"]
    e = S(tl.EMA(f(c), 26), df.index)
    rate = (e - e.shift(1)) / e.shift(1) * 100
    band = _zenbot_stddev(rate, 13)
    r = rsi(c, 14)
    ra, ba, rr = rate.to_numpy(float), band.to_numpy(float), r.to_numpy(float)
    ent, ex = np.zeros(len(c), bool), np.zeros(len(c), bool)
    cancel_down = False
    for i in range(len(c)):
        if not np.isnan(rr[i]) and rr[i] <= 10 and not cancel_down:
            ent[i], cancel_down = True, True                    # oversold buy; skip the trend logic
            continue
        if np.isnan(ra[i]) or np.isnan(ba[i]):
            continue
        if ra[i] > ba[i]:
            ent[i], cancel_down = True, False
        elif not cancel_down and ra[i] < -ba[i]:
            ex[i] = True
    return _out(df, pd.Series(ent, index=df.index), pd.Series(ex, index=df.index))


# ============================================================================================ Hummingbot
@signal("hb_macd_bb")
def hb_macd_bb(df, p, info):
    """Hummingbot V2 directional controller macd_bb_v1 (long side): close below the Bollinger band
    (100, 2; %B < 0) while MACD(21, 42, 9) is below zero with a positive histogram. Triple barrier from
    the controller defaults: stop 3%, take profit 2%, time limit 45 minutes."""
    c = df["close"]
    mid = sma(c, 100)
    sd = c.rolling(100).std(ddof=0)
    bbp = (c - (mid - 2 * sd)) / (4 * sd)
    m, _, h = macd(c, 21, 42, 9)
    entry = (bbp < 0.0) & (h > 0) & (m < 0)
    out = _out(df, entry, pd.Series(False, index=df.index))
    out["sl_pct"], out["tp_pct"] = 0.03, 0.02
    return out


# ============================================================================================ NautilusTrader
@signal("nautilus_ema_cross_trailing")
def nautilus_ema_cross_trailing(df, p, info):
    """NautilusTrader example EMACrossTrailingStop (long side): when flat and EMA(10) >= EMA(20), buy;
    the only exit is a trailing stop 3 x ATR(20) below the highest price since entry."""
    c = df["close"]
    fast, slow = _ema_first(c, 10), _ema_first(c, 20)
    a = true_range(df).rolling(20).mean()                      # AverageTrueRange, simple average
    ready = pd.Series(np.arange(len(c)) >= 19, index=df.index) & a.notna()
    out = _out(df, ready & (fast >= slow), pd.Series(False, index=df.index))
    out["trail_dist"] = 3.0 * a
    return out


# ============================================================================================ OctoBot
def tulip_kvo(df: pd.DataFrame, short: int = 35, long: int = 55) -> pd.Series:
    """Tulip Indicators' Klinger volume oscillator (tulipy.kvo), first value at the second candle."""
    h, lo, c, v = (df[k].to_numpy(float) for k in ("high", "low", "close", "volume"))
    out = np.full(len(h), np.nan)
    if len(h) < 2:
        return pd.Series(out, index=df.index)
    sp, lp = 2.0 / (short + 1), 2.0 / (long + 1)
    cm, trend = 0.0, -1
    prev_hlc = h[0] + lo[0] + c[0]
    s_ema = l_ema = 0.0
    for i in range(1, len(h)):
        hlc = h[i] + lo[i] + c[i]
        dm = h[i] - lo[i]
        if hlc > prev_hlc and trend != 1:
            trend, cm = 1, h[i - 1] - lo[i - 1]
        elif hlc < prev_hlc and trend != 0:
            trend, cm = 0, h[i - 1] - lo[i - 1]
        cm += dm
        vf = v[i] * abs(dm / cm * 2 - 1) * 100 * (1.0 if trend else -1.0) if cm != 0 else 0.0
        if i == 1:
            s_ema = l_ema = vf
        else:
            s_ema += (vf - s_ema) * sp
            l_ema += (vf - l_ema) * lp
        out[i] = s_ema - l_ema
        prev_hlc = hlc
    return pd.Series(out, index=df.index)


def _min_just_reached(window: np.ndarray, acceptance: float, delay: int) -> bool:
    """OctoBot TrendAnalysis.min_has_just_been_reached."""
    if len(window) <= 1:
        return False
    mn = window.min()
    return bool(np.any(window[-(delay + 1):] == mn) and window[-1] / acceptance > mn)


def _klinger_reversal(diff: np.ndarray) -> np.ndarray:
    """KlingerOscillatorReversalConfirmationMomentumEvaluator at every candle: the values since the
    last zero crossing of KVO - EMA13(KVO) (at most 7) must show a minimum just reached. The crossing
    index is what TrendAnalysis.get_threshold_change_indexes returns last: the first value after the
    last run of values <= 0 when the series ends above 0, else the start of that final run."""
    out = np.zeros(len(diff), bool)
    last_sub, run_start = -1, -1
    for k in range(len(diff)):
        if diff[k] <= 0:
            if last_sub != k - 1:
                run_start = k
            last_sub = k
        if last_sub < 0 or k < 1:
            continue                                   # no value <= 0 yet: OctoBot raises, no signal
        zc = last_sub + 1 if diff[k] > 0 else run_start
        n = min(7, k + 1 - zc)
        out[k] = _min_just_reached(diff[k + 1 - n:k + 1], 0.9, 1)
    return out


RSI_TO_WEIGHT = [  # OctoBot RSIWeightMomentumEvaluator default config: (slow, [(fast, price, volume), ...])
    (30, [(20, 2, 2), (30, 1, 1)]),
    (35, [(20, 3, 3), (35, 1, 1)]),
    (45, [(20, 3, 3), (40, 2, 1)]),
    (55, [(45, 1, 1)]),
    (65, [(45, 1, 1), (55, 3, 2), (60, 2, 1)]),
    (70, [(55, 3, 2), (70, 2, 2)]),
]


def _dip_weight(slow: float, fast: float, cur: float):
    for s_thr, fasts in RSI_TO_WEIGHT:
        if slow < s_thr:
            for f_thr, pw, vw in fasts:
                if fast < f_thr and cur < f_thr:
                    return pw, vw
            break
    return None, None


@signal("octobot_dip_analyser")
def octobot_dip_analyser(df, p, info):
    """OctoBot Dip Analyser strategy evaluator: a Klinger oscillator reversal (KVO 35/55 vs its EMA13)
    confirmed by an RSI(14) low just reached; the RSI averages of the last 16 and 4 values pick the
    price weight (sell targets) and volume weight (order size). Orders: see DipAnalyserBot."""
    kvo = tulip_kvo(df, 35, 55)
    k = kvo.dropna()
    diff = pd.Series(np.nan, index=df.index)
    if len(k) >= 13:
        diff.loc[k.index] = (k - _ema_first(k, 13)).to_numpy()
    d = diff.dropna()
    klinger = pd.Series(False, index=df.index)
    if len(d) > 1:
        klinger.loc[d.index] = _klinger_reversal(d.to_numpy(float))
    r = rsi(df["close"], 14)
    rv = r.to_numpy(float)
    first = int(np.argmax(~np.isnan(rv))) if np.any(~np.isnan(rv)) else len(rv)
    entry = np.zeros(len(rv), bool)
    pw_a, vw_a = np.full(len(rv), np.nan), np.full(len(rv), np.nan)
    kl = klinger.to_numpy()
    for i in range(first, len(rv)):
        hist = rv[first:i + 1]
        if not kl[i] or len(hist) < 2:
            continue
        if not _min_just_reached(hist[-5:], 0.95, 2):
            continue
        pw, vw = _dip_weight(hist[-16:].mean(), hist[-4:].mean(), hist[-1])
        if pw is not None:
            entry[i], pw_a[i], vw_a[i] = True, pw, vw
    out = _out(df, pd.Series(entry, index=df.index), pd.Series(False, index=df.index))
    out["price_weight"], out["volume_weight"] = pw_a, vw_a
    return out
