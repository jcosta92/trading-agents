"""Deterministic synthetic market data for the family tests (any coin, ticker or timeframe)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from quantdesk.bots.data import TF

START = {"5m": "2026-07-20", "15m": "2026-06-01", "1h": "2026-01-01", "4h": "2025-01-01", "1d": "2022-01-01"}
END = pd.Timestamp("2026-11-30", tz="UTC")


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _walk(seed: int, n: int, start_px: float, vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.normal(0, vol / 5, n // 150 + 1), 150)[:n]
    return start_px * np.exp(np.cumsum(regime + rng.normal(0, vol, n)))


def _ohlcv(idx, seed, px0, vol):
    c = _walk(seed, len(idx), px0, vol)
    o = np.r_[c[0], c[:-1]]
    rng = np.random.default_rng(seed + 1)
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, len(c))))
    v = np.exp(rng.normal(10, 0.6, len(c)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)


class FamilySource:
    def __init__(self):
        self.used = {}
        self._c = {}

    def crypto(self, symbol, timeframe, since, now):
        key = (symbol, timeframe)
        if key not in self._c:
            tf = TF[timeframe]
            idx = pd.date_range(pd.Timestamp(START[timeframe], tz="UTC"), END, freq=tf, inclusive="left")
            vol = 0.02 * (tf / pd.Timedelta(days=1)) ** 0.5
            self._c[key] = _ohlcv(idx, _seed(symbol), 100.0 + _seed(symbol) % 900, vol)
        df = self._c[key]
        return df[(df.index >= since) & (df.index + TF[timeframe] <= now)]

    def eurusd(self):
        return 1.16

    def futures(self, symbol, timeframe, since, now, delay=pd.Timedelta(0)):
        """Gold-futures-like candles: every Globex hour (no 17:00-18:00 New York break, no weekend)."""
        key = ("fut", symbol, timeframe)
        if key not in self._c:
            idx = pd.date_range(pd.Timestamp(START[timeframe], tz="UTC"), END, freq=TF[timeframe], inclusive="left")
            ny = idx.tz_convert("America/New_York")
            wk = ny.dayofweek
            open_ = (ny.hour != 17) & (wk != 5) & ~((wk == 4) & (ny.hour >= 17)) & ~((wk == 6) & (ny.hour < 18))
            self._c[key] = _ohlcv(idx[open_], _seed(symbol, timeframe), 4100.0, 0.0009)
        df = self._c[key]
        return df[(df.index >= since) & (df.index + TF[timeframe] <= now - delay)]

    def _sched(self):
        if "sched" not in self._c:
            import exchange_calendars as xcals
            self._c["sched"] = xcals.get_calendar("XNYS").schedule.loc["2016-01-01":"2026-12-31"]
        return self._c["sched"]

    def _stock(self, t):
        key = ("stock", t)
        if key not in self._c:
            sched = self._sched()
            s5 = sched.loc["2026-08-01":]
            idx = pd.DatetimeIndex([x for o, c in zip(s5["open"], s5["close"])
                                    for x in pd.date_range(o, c, freq="5min", inclusive="left")])
            base = 20.0 + _seed(t) % 300
            intr = _ohlcv(idx, _seed(t, "5m"), base, 0.0015)
            didx = pd.DatetimeIndex(sched.index[sched.index < pd.Timestamp("2026-08-01")])
            daily = _ohlcv(didx, _seed(t, "1d"), base * 0.6, 0.012)
            daily = daily * (intr["open"].iloc[0] / daily["close"].iloc[-1])
            day5 = intr.copy()
            day5.index = day5.index.tz_convert("America/New_York").tz_localize(None).normalize()
            agg = day5.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                             "volume": "sum"})
            self._c[key] = (intr, pd.concat([daily, agg]))
        return self._c[key]

    def stocks(self, tickers, timeframe, since, now):
        out = {}
        for t in tickers:
            intr, _ = self._stock(t)
            out[t] = intr[(intr.index >= since) & (intr.index + TF["5m"] <= now)]
        return out

    def stock_daily(self, tickers, days, now):
        day = pd.Timestamp(now).tz_convert("America/New_York").tz_localize(None).normalize()
        return {t: self._stock(t)[1][lambda d: d.index < day] for t in tickers}
