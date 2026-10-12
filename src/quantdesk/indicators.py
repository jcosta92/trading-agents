"""Technical indicators (pure pandas). Conventions follow the standard definitions:

  SMA/EMA          simple / exponential moving average of closes
  RSI(n)           Wilder's RSI (Wilder smoothing = EMA with alpha 1/n)
  ATR(n)           Wilder's Average True Range
  Donchian(n)      highest high / lowest low of the PREVIOUS n bars (today excluded, so a close
                   above it is a genuine breakout)
  ROC(n)           rate of change over n bars
  MACD             EMA(12) - EMA(26), signal EMA(9)
Every function returns a Series aligned to the input; values are NaN until enough history exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_down > 0, 100.0)               # no down moves -> RSI 100
    out = out.where(~((avg_up == 0) & (avg_down == 0)), 50.0)
    return out.where(avg_up.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    return high.shift(1).rolling(n, min_periods=n).max()


def donchian_low(low: pd.Series, n: int) -> pd.Series:
    return low.shift(1).rolling(n, min_periods=n).min()


def roc(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def last(s: pd.Series) -> float:
    """Last value as float (NaN if empty)."""
    if s is None or len(s) == 0:
        return float("nan")
    v = s.iloc[-1]
    return float(v) if v is not None else float("nan")


def ohlc_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing open/high/low from close (proxy history often has closes only)."""
    df = df.copy()
    for col in ("open", "high", "low"):
        if col not in df:
            df[col] = df["close"]
        df[col] = df[col].fillna(df["close"])
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
    return df
