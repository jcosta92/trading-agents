"""Crypto market data via ccxt (quotes, funding history) with multiple fallbacks for funding."""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)
_EXCHANGES: dict = {}


def exchange(name: str):
    import ccxt
    if name not in _EXCHANGES:
        _EXCHANGES[name] = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000})
    return _EXCHANGES[name]


def live_quote(exchange_name: str, symbol: str) -> dict | None:
    try:
        t = exchange(exchange_name).fetch_ticker(symbol)
    except Exception as exc:
        log.warning("ccxt quote failed %s %s: %s", exchange_name, symbol, exc)
        return None
    bid, ask, last = t.get("bid"), t.get("ask"), t.get("last")
    mid = (bid + ask) / 2 if bid and ask and ask >= bid > 0 else last
    if not mid:
        return None
    ts = t.get("datetime") or datetime.now(timezone.utc).isoformat()
    return {"mid": float(mid), "bid": bid, "ask": ask, "last": last, "ts": ts,
            "source": f"{exchange_name}:{symbol}"}


# ---------------------------------------------------------------------------------- funding
def funding_history(source: str, symbol: str, since: str = "2018-01-01") -> pd.Series:
    """Funding rates per funding event (fraction of notional, + = longs pay shorts), UTC timestamps."""
    try:
        if source == "krakenfutures":
            rows = exchange("krakenfutures").fetch_funding_rate_history(symbol)
            s = pd.Series({pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): r["fundingRate"] for r in rows})
        elif source == "hyperliquid":
            s = _paged_ccxt_funding("hyperliquid", symbol, since)
        elif source == "binance_vision":
            s = _binance_vision_funding(symbol, since)
        else:
            raise ValueError(source)
    except Exception as exc:
        log.warning("funding history failed %s %s: %s", source, symbol, exc)
        return pd.Series(dtype="float64")
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    return s[~s.index.duplicated()]


def _paged_ccxt_funding(ex_name: str, symbol: str, since: str) -> pd.Series:
    ex = exchange(ex_name)
    ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    out = {}
    now_ms = int(time.time() * 1000)
    while ms < now_ms:
        rows = ex.fetch_funding_rate_history(symbol, since=ms, limit=500)
        if not rows:
            break
        for r in rows:
            out[pd.Timestamp(r["timestamp"], unit="ms", tz="UTC")] = r["fundingRate"]
        last = rows[-1]["timestamp"]
        if last <= ms:
            break
        ms = last + 1
    return pd.Series(out)


def _binance_vision_funding(symbol: str, since: str) -> pd.Series:
    """Monthly zip files from data.binance.vision (USD-M futures)."""
    start = pd.Timestamp(since).to_period("M")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    out = []
    for per in pd.period_range(start, end, freq="M"):
        url = (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/"
               f"{symbol}-fundingRate-{per.year}-{per.month:02d}.zip")
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as fh:
                    df = pd.read_csv(fh)
            tcol = "calc_time" if "calc_time" in df.columns else df.columns[0]
            rcol = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]
            out.append(pd.Series(df[rcol].values, index=pd.to_datetime(df[tcol], unit="ms", utc=True)))
        except Exception as exc:
            log.info("binance vision %s %s: %s", symbol, per, exc)
    return pd.concat(out) if out else pd.Series(dtype="float64")


def daily_funding(events: pd.Series) -> pd.Series:
    """Sum funding events per UTC calendar day."""
    if events.empty:
        return events
    idx = events.index.tz_convert("UTC") if events.index.tz is not None else events.index
    s = pd.Series(events.values, index=idx)
    d = s.groupby(s.index.floor("D")).sum()
    d.index = d.index.tz_localize(None)
    return d


def funding_between(events: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Sum of funding events with timestamp in (start, end]."""
    if events.empty:
        return 0.0
    s = events[(events.index > start) & (events.index <= end)]
    return float(s.sum())
