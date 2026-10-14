"""Candle data for the bots: crypto from Bitvavo (Kraken as fallback) via ccxt, US stocks and gold
futures from Yahoo.

All frames are indexed by the candle's OPEN time (UTC, tz-aware) with columns open/high/low/close/volume.
Only CLOSED candles are returned (open time + timeframe <= now).
"""
from __future__ import annotations

import logging
import time

import pandas as pd

log = logging.getLogger(__name__)

TF = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1),
      "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
CRYPTO_SOURCES = ("bitvavo", "kraken")


def _frame(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
    return df


def closed_only(df: pd.DataFrame, timeframe: str, now: pd.Timestamp) -> pd.DataFrame:
    return df[df.index + TF[timeframe] <= now]


def _ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One ticker's columns from a yfinance download. With group_by="ticker" yfinance returns
    (ticker, field) columns even for a single ticker; older versions return plain fields."""
    if isinstance(raw.columns, pd.MultiIndex):
        return raw[ticker]
    return raw


class CandleSource:
    """Network candle source. `crypto()` / `stocks()` return closed candles from `since` to `now`."""

    def __init__(self):
        self._ex = {}
        self.used: dict[str, str] = {}

    def _exchange(self, name):
        import ccxt
        if name not in self._ex:
            self._ex[name] = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000})
        return self._ex[name]

    def crypto(self, symbol: str, timeframe: str, since: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
        last_err = None
        for name in CRYPTO_SOURCES:
            try:
                df = self._paged(name, symbol, timeframe, since, now)
                if len(df):
                    self.used[symbol] = name
                    return closed_only(df, timeframe, now)
            except Exception as exc:          # network / exchange error: try the next source
                last_err = exc
                log.warning("%s %s %s failed: %s", name, symbol, timeframe, exc)
        raise RuntimeError(f"no candles for {symbol} {timeframe}: {last_err}")

    def _paged(self, name, symbol, timeframe, since, now) -> pd.DataFrame:
        ex = self._exchange(name)
        ms = int(since.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        step = int(TF[timeframe].total_seconds() * 1000)
        rows = []
        for _ in range(60):
            batch = ex.fetch_ohlcv(symbol, timeframe, since=ms, limit=1000)
            if not batch:
                break
            rows += batch
            last = batch[-1][0]
            if last + step >= end_ms or last <= ms:
                break
            ms = last + step
            time.sleep(ex.rateLimit / 1000)
        df = _frame(rows)
        return df[df.index >= since]

    def stocks(self, tickers: list[str], timeframe: str, since: pd.Timestamp, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Intraday candles from Yahoo (5m history goes back ~60 days), regular session only."""
        import yfinance as yf
        days = max(2, int((now - since).total_seconds() // 86400) + 2)
        period = f"{min(days, 59)}d"
        raw = yf.download(tickers, period=period, interval=timeframe, auto_adjust=False, prepost=False,
                          progress=False, threads=True, group_by="ticker", timeout=30)
        out = {}
        for t in tickers:
            try:
                sub = _ticker_frame(raw, t)
                df = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            except (KeyError, TypeError):
                continue
            idx = pd.DatetimeIndex(df.index)
            df.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
            df = df[(df.index >= since)]
            out[t] = closed_only(df.astype(float), timeframe, now)
        return out

    def futures(self, symbol: str, timeframe: str, since: pd.Timestamp, now: pd.Timestamp,
                delay: pd.Timedelta = pd.Timedelta(minutes=15)) -> pd.DataFrame:
        """Intraday futures candles from Yahoo (every Globex hour). CME data on Yahoo is delayed, so a
        candle is only returned once it closed at least `delay` ago (a fresher one may be incomplete)."""
        import yfinance as yf
        days = max(2, int((now - since).total_seconds() // 86400) + 2)
        raw = yf.download(symbol, period=f"{min(days, 59)}d", interval=timeframe, auto_adjust=False, prepost=True,
                          progress=False, threads=False, group_by="ticker", timeout=30)
        try:
            df = _ticker_frame(raw, symbol).rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(
                subset=["open", "high", "low", "close"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"no {timeframe} candles for {symbol}: {exc}") from exc
        if df.empty:
            raise RuntimeError(f"no {timeframe} candles for {symbol}")
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.used[symbol] = "yahoo"
        return closed_only(df[df.index >= since].astype(float), timeframe, now - delay)

    def stock_daily(self, tickers: list[str], days: int, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        import yfinance as yf
        raw = yf.download(tickers, period=f"{days + 30}d", interval="1d", auto_adjust=False, progress=False,
                          threads=True, group_by="ticker", timeout=30)
        out = {}
        for t in tickers:
            try:
                sub = _ticker_frame(raw, t)
                df = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            except (KeyError, TypeError):
                continue
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            out[t] = df.astype(float)
        return out

    def eurusd(self) -> float:
        from ..data import yahoo
        q = yahoo.live_quote("EURUSD=X")
        if not q or not (0.7 < q["mid"] < 1.6):
            raise RuntimeError("no EURUSD quote")
        return float(q["mid"])
