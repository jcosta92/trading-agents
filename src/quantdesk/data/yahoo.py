"""Yahoo Finance access via yfinance: daily history, live quotes, option chains, news, fundamentals.

Yahoo is free and unofficial: expect gaps, rate limits and occasional bad prints. Every caller
must tolerate None/empty results; failures are logged, never silently replaced with fake data.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)

FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def _yf():
    import yfinance as yf  # imported lazily so tests run without network libraries configured
    return yf


def download_history(symbols: list[str], start: str = "2000-01-01", end: str | None = None,
                     chunk: int = 40, retries: int = 3, pause: float = 1.0,
                     fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")) -> dict[str, pd.DataFrame]:
    """Adjusted (splits + dividends) daily bars. Returns {field: DataFrame[date x symbol]}."""
    yf = _yf()
    symbols = list(dict.fromkeys(s for s in symbols if s))
    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in fields}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        df = None
        for attempt in range(retries):
            try:
                df = yf.download(batch, start=start, end=end, auto_adjust=True, actions=False,
                                 progress=False, threads=True, group_by="column", timeout=30)
                break
            except Exception as exc:  # network / rate limit
                log.warning("yfinance download failed (%s), attempt %d: %s", batch[:3], attempt + 1, exc)
                time.sleep(pause * (attempt + 1) * 3)
        if df is None or df.empty:
            continue
        for yf_field, field in FIELDS.items():
            if field not in fields:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                if yf_field not in df.columns.get_level_values(0):
                    continue
                sub = df[yf_field]
            else:
                sub = df[[yf_field]].rename(columns={yf_field: batch[0]})
            parts[field].append(sub)
        time.sleep(pause)
    out = {}
    for field, lst in parts.items():
        if lst:
            frame = pd.concat(lst, axis=1)
            frame = frame.loc[:, ~frame.columns.duplicated()]
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            out[field] = frame.dropna(how="all").sort_index()
    return out


def live_quote(symbol: str) -> dict | None:
    """Best-effort current quote: {'mid','bid','ask','last','ts','source'}."""
    yf = _yf()
    try:
        t = yf.Ticker(symbol)
        last = None
        try:
            last = float(t.fast_info["last_price"])
        except Exception:
            pass
        bid = ask = None
        ts = None
        try:
            info = t.info or {}
            bid = _num(info.get("bid"))
            ask = _num(info.get("ask"))
            last = last or _num(info.get("regularMarketPrice"))
            rmt = info.get("regularMarketTime")
            if rmt:
                ts = datetime.fromtimestamp(int(rmt), tz=timezone.utc).isoformat()
        except Exception as exc:
            log.info("info() failed for %s: %s", symbol, exc)
        if ts is None:
            # fallback: latest 1-minute bar gives both a price and a timestamp
            try:
                h = t.history(period="1d", interval="1m")
                if h is not None and len(h):
                    last = float(h["Close"].iloc[-1])
                    tsv = pd.Timestamp(h.index[-1])
                    ts = (tsv.tz_convert("UTC") if tsv.tzinfo else tsv.tz_localize("UTC")).isoformat()
            except Exception as exc:
                log.info("1m history failed for %s: %s", symbol, exc)
        if bid and ask and ask >= bid > 0 and last and abs((bid + ask) / 2 / last - 1) < 0.02:
            mid = (bid + ask) / 2
        else:
            bid = ask = None
            mid = last
        if mid is None or not math.isfinite(mid) or mid <= 0:
            return None
        return {"mid": mid, "bid": bid, "ask": ask, "last": last, "ts": ts, "source": f"yahoo:{symbol}"}
    except Exception as exc:
        log.warning("live quote failed for %s: %s", symbol, exc)
        return None


def dividends(symbol: str | None) -> pd.Series:
    """Cash dividends per share indexed by ex-date (tz-naive)."""
    if not symbol:
        return pd.Series(dtype="float64")
    try:
        d = _yf().Ticker(symbol).dividends
    except Exception as exc:
        log.info("dividends failed for %s: %s", symbol, exc)
        return pd.Series(dtype="float64")
    if d is None or len(d) == 0:
        return pd.Series(dtype="float64")
    d = d.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d.astype(float)


def option_expiries(symbol: str) -> list[str]:
    try:
        return list(_yf().Ticker(symbol).options or [])
    except Exception as exc:
        log.warning("option expiries failed for %s: %s", symbol, exc)
        return []


def option_chain(symbol: str, expiry: str) -> pd.DataFrame:
    """Puts and calls for one expiry with columns strike, bid, ask, lastPrice, impliedVolatility, right."""
    try:
        ch = _yf().Ticker(symbol).option_chain(expiry)
    except Exception as exc:
        log.warning("option chain failed for %s %s: %s", symbol, expiry, exc)
        return pd.DataFrame()
    frames = []
    for right, df in (("P", ch.puts), ("C", ch.calls)):
        if df is None or df.empty:
            continue
        d = df[["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]].copy()
        d["right"] = right
        d["expiry"] = expiry
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def news(symbol: str) -> list[dict]:
    """Recent headlines: [{'id','title','summary','publisher','published','ticker'}]."""
    try:
        items = _yf().Ticker(symbol).news or []
    except Exception as exc:
        log.info("news failed for %s: %s", symbol, exc)
        return []
    out = []
    for it in items:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        pub = c.get("pubDate") or c.get("displayTime")
        if pub is None and it.get("providerPublishTime"):
            pub = datetime.fromtimestamp(int(it["providerPublishTime"]), tz=timezone.utc).isoformat()
        provider = c.get("provider") or {}
        out.append({
            "id": it.get("id") or c.get("id") or title,
            "title": title.strip(),
            "summary": (c.get("summary") or c.get("description") or "")[:400],
            "publisher": provider.get("displayName") if isinstance(provider, dict) else c.get("publisher"),
            "published": pub,
            "ticker": symbol,
        })
    return out


FUNDAMENTAL_FIELDS = ("sector", "enterpriseToEbitda", "returnOnEquity", "grossMargins", "debtToEquity",
                      "marketCap", "trailingPE")


def fundamentals(symbols: list[str], pause: float = 0.3, retries: int = 2) -> pd.DataFrame:
    """Current fundamentals snapshot (NOT point-in-time; for live use only)."""
    yf = _yf()
    rows = {}
    for s in symbols:
        for attempt in range(retries):
            try:
                info = yf.Ticker(s).info or {}
                rows[s] = {k: info.get(k) for k in FUNDAMENTAL_FIELDS}
                break
            except Exception as exc:
                log.info("fundamentals failed for %s (attempt %d): %s", s, attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        time.sleep(pause)
    return pd.DataFrame.from_dict(rows, orient="index")


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError):
        return None
