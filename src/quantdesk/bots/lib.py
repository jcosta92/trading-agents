"""Indicator helpers for the strategy families.

Everything here is causal: the value at a candle only uses that candle and earlier ones, so a
signal computed on the full history equals the signal computed when that candle was the newest
(tests/test_families.py checks this for every family). TA-Lib is used where the original
strategies use it (Freqtrade), with qtpylib-equivalent helpers for the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib as tl

from . import ta


# ------------------------------------------------------------------------------------------ basics
def f(s) -> np.ndarray:
    return np.asarray(s, dtype=float)


def S(a, index) -> pd.Series:
    return pd.Series(a, index=index, dtype=float)


def crossed_above(a, b) -> pd.Series:
    a = a if isinstance(a, pd.Series) else pd.Series(a)
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_below(a, b) -> pd.Series:
    a = a if isinstance(a, pd.Series) else pd.Series(a)
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


def cross_above_level(a: pd.Series, level: float) -> pd.Series:
    return (a > level) & (a.shift(1) <= level)


def cross_below_level(a: pd.Series, level: float) -> pd.Series:
    return (a < level) & (a.shift(1) >= level)


def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return ta.ema(s, n)


def rma(s, n):
    return ta.rma(s, n)


def wma(s: pd.Series, n: int) -> pd.Series:
    return S(tl.WMA(f(s), n), s.index)


def typical(df):
    return (df["high"] + df["low"] + df["close"]) / 3


def hl2(df):
    return (df["high"] + df["low"]) / 2


def bb_qtpy(series: pd.Series, window: int = 20, stds: float = 2.0):
    """qtpylib.bollinger_bands: SMA +- k * sample std (ddof=1), min_periods=1."""
    mid = series.rolling(window, min_periods=1).mean()
    sd = series.rolling(window, min_periods=1).std(ddof=1)
    return mid - stds * sd, mid, mid + stds * sd


def bb_pop(series: pd.Series, window: int = 20, stds: float = 2.0):
    """TradingView / talib bands: SMA +- k * population std."""
    return ta.bollinger(series, window, stds)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    c = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4).to_numpy()
    o = np.empty(len(df))
    if len(df):
        o[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        o[i] = (o[i - 1] + c[i - 1]) / 2
    h = np.maximum.reduce([df["high"].to_numpy(), o, c])
    lo = np.minimum.reduce([df["low"].to_numpy(), o, c])
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": df["volume"].to_numpy()},
                        index=df.index)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """TA-Lib RSI (Wilder, seeded with an SMA, as Freqtrade and TradingView)."""
    return S(tl.RSI(f(s), n), s.index)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return S(tl.ATR(f(df["high"]), f(df["low"]), f(df["close"]), n), df.index)


def adx(df, n=14):
    return S(tl.ADX(f(df["high"]), f(df["low"]), f(df["close"]), n), df.index)


def plus_di(df, n=14):
    return S(tl.PLUS_DI(f(df["high"]), f(df["low"]), f(df["close"]), n), df.index)


def minus_di(df, n=14):
    return S(tl.MINUS_DI(f(df["high"]), f(df["low"]), f(df["close"]), n), df.index)


def cci(df, n=14):
    return S(tl.CCI(f(df["high"]), f(df["low"]), f(df["close"]), n), df.index)


def mfi(df, n=14):
    return S(tl.MFI(f(df["high"]), f(df["low"]), f(df["close"]), f(df["volume"]), n), df.index)


def cmo(s, n=14):
    return S(tl.CMO(f(s), n), s.index)


def macd(s, fast=12, slow=26, sig=9):
    m, si, h = tl.MACD(f(s), fast, slow, sig)
    return S(m, s.index), S(si, s.index), S(h, s.index)


def stochf(df, fastk=5, fastd=3):
    k, d = tl.STOCHF(f(df["high"]), f(df["low"]), f(df["close"]), fastk_period=fastk, fastd_period=fastd,
                     fastd_matype=0)
    return S(k, df.index), S(d, df.index)


def stoch(df, k=14, ks=3, d=3):
    sk, sd = tl.STOCH(f(df["high"]), f(df["low"]), f(df["close"]), fastk_period=k, slowk_period=ks,
                      slowk_matype=0, slowd_period=d, slowd_matype=0)
    return S(sk, df.index), S(sd, df.index)


def sar(df, acc=0.02, maximum=0.2):
    return S(tl.SAR(f(df["high"]), f(df["low"]), acceleration=acc, maximum=maximum), df.index)


def tema(s, n):
    return S(tl.TEMA(f(s), n), s.index)


def obv(df):
    return S(tl.OBV(f(df["close"]), f(df["volume"])), df.index)


def mom(s, n):
    return S(tl.MOM(f(s), n), s.index)


def roc(s, n):
    return s / s.shift(n) - 1


def linreg(s, n):
    return S(tl.LINEARREG(f(s), n), s.index)


def highest(s, n):
    return s.rolling(n).max()


def lowest(s, n):
    return s.rolling(n).min()


def stdev(s, n):
    return s.rolling(n).std(ddof=0)


def zscore(s, n):
    return (s - s.rolling(n).mean()) / s.rolling(n).std(ddof=0)


def bars_since(cond: pd.Series) -> pd.Series:
    """Candles since `cond` was last True (0 on the candle itself); NaN before the first True."""
    idx = np.arange(len(cond))
    last = pd.Series(np.where(cond.to_numpy(), idx, np.nan), index=cond.index).ffill()
    return pd.Series(idx, index=cond.index) - last


def ichimoku(df, conv=9, base=26, span_b=52, disp=26):
    """Returns tenkan, kijun, and the cloud (span A, span B) that applies to each candle (computed
    `disp` candles earlier, as plotted)."""
    tenkan = (highest(df["high"], conv) + lowest(df["low"], conv)) / 2
    kijun = (highest(df["high"], base) + lowest(df["low"], base)) / 2
    a = ((tenkan + kijun) / 2).shift(disp)
    b = ((highest(df["high"], span_b) + lowest(df["low"], span_b)) / 2).shift(disp)
    return tenkan, kijun, a, b


def streak(s: pd.Series) -> pd.Series:
    """Consecutive up (+n) / down (-n) closes, 0 when unchanged (ConnorsRSI)."""
    d = np.sign(s.diff().fillna(0).to_numpy())
    out = np.zeros(len(d))
    for i in range(1, len(d)):
        if d[i] == 0:
            out[i] = 0
        elif d[i] > 0:
            out[i] = out[i - 1] + 1 if out[i - 1] > 0 else 1
        else:
            out[i] = out[i - 1] - 1 if out[i - 1] < 0 else -1
    return pd.Series(out, index=s.index)


def percent_rank(s: pd.Series, n: int) -> pd.Series:
    """Share of the previous n values strictly below the current one (TradingView ta.percentrank)."""
    a = s.to_numpy()
    out = np.full(len(a), np.nan)
    for i in range(n, len(a)):
        w = a[i - n:i]
        out[i] = 100.0 * np.sum(w < a[i]) / n
    return pd.Series(out, index=s.index)


def connors_rsi(close, n_rsi=3, n_streak=2, n_rank=100):
    return (rsi(close, n_rsi) + rsi(streak(close), n_streak) + percent_rank(close.pct_change() * 100, n_rank)) / 3


def resample_closed(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Higher-timeframe candles from lower ones, stamped at their open time, keeping only complete ones."""
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return agg


def merge_informative(df: pd.DataFrame, inf: pd.DataFrame, tf: pd.Timedelta, inf_tf: pd.Timedelta,
                      cols: list[str] | None = None, suffix: str = "_inf") -> pd.DataFrame:
    """Attach higher-timeframe columns to `df` as Freqtrade's merge_informative_pair does: a higher
    candle is visible only once it has closed (inf open + inf_tf <= df open + tf)."""
    inf = inf[cols] if cols else inf
    shifted = inf.copy()
    shifted.index = shifted.index + inf_tf                 # stamp at close time
    left = pd.DataFrame(index=df.index + tf)                # each candle's close time
    merged = pd.merge_asof(left.reset_index(names="t"), shifted.add_suffix(suffix).reset_index(names="t"),
                           on="t", direction="backward")
    merged.index = df.index
    return df.join(merged.drop(columns="t"))
