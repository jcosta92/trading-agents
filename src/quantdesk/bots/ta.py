"""Technical indicators on candle DataFrames (columns open/high/low/close/volume), TradingView conventions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average (TradingView `ta.rma`)."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(df), n)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(lower, mid, upper) with the population standard deviation, as qtpylib/TradingView."""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Direction series: +1 uptrend, -1 downtrend (TradingView `ta.supertrend` logic)."""
    a = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fu, fl = upper.copy(), lower.copy()
    d = np.ones(n)
    for i in range(1, n):
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        if d[i - 1] == -1:                       # the line was the upper band: flip up on a close above it
            d[i] = 1 if close[i] > fu[i] else -1
        else:                                    # the line was the lower band: flip down on a close below it
            d[i] = -1 if close[i] < fl[i] else 1
    out = pd.Series(d, index=df.index)
    out.iloc[:period] = np.nan
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (TradingView `ta.rsi`)."""
    d = close.diff()
    up, down = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def linreg_last(s: pd.Series, n: int) -> pd.Series:
    """TradingView `ta.linreg(s, n, 0)`: the least-squares line over the last n values, evaluated at the last one."""
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()

    def f(w: np.ndarray) -> float:
        b = ((x - xm) * (w - w.mean())).sum() / den
        return w.mean() + b * (x[-1] - xm)
    return s.rolling(n).apply(f, raw=True)


def squeeze_momentum(df: pd.DataFrame, n_bb: int = 20, k_bb: float = 2.0, n_kc: int = 20,
                     k_kc: float = 1.5) -> tuple[pd.Series, pd.Series]:
    """LazyBear's Squeeze Momentum (TradingView): (squeeze_on, momentum value).
    Squeeze is on while the Bollinger bands are inside the Keltner channel."""
    c = df["close"]
    basis = c.rolling(n_bb).mean()
    dev = k_bb * c.rolling(n_bb).std(ddof=0)
    ma = c.rolling(n_kc).mean()
    rng = true_range(df).rolling(n_kc).mean()
    on = ((basis - dev) > (ma - k_kc * rng)) & ((basis + dev) < (ma + k_kc * rng))
    mid = ((df["high"].rolling(n_kc).max() + df["low"].rolling(n_kc).min()) / 2 + ma) / 2
    return on, linreg_last(c - mid, n_kc)
