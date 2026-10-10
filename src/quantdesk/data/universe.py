"""S&P 500 membership: current constituents (with GICS sector) and point-in-time history."""
from __future__ import annotations

import io
import logging
import urllib.parse

import pandas as pd
import requests

log = logging.getLogger(__name__)

CURRENT_SOURCES = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv",
]
HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               + urllib.parse.quote("S&P 500 Historical Components & Changes (Updated).csv"))


def yahoo_symbol(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def current_constituents(timeout: int = 30) -> pd.DataFrame:
    """DataFrame[ticker, name, sector] of current members (Yahoo-style tickers)."""
    for url in CURRENT_SOURCES:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            out = pd.DataFrame({
                "ticker": df["Symbol"].map(yahoo_symbol),
                "name": df.get("Security", df["Symbol"]),
                "sector": df.get("GICS Sector"),
            })
            if len(out) > 400:
                return out.drop_duplicates("ticker").reset_index(drop=True)
        except Exception as exc:
            log.warning("constituents source failed %s: %s", url, exc)
    raise RuntimeError("no S&P 500 constituent source reachable")


def membership_history(timeout: int = 60) -> pd.DataFrame:
    """Point-in-time membership: index = change date, column 'tickers' = list of Yahoo symbols.

    Tickers of companies that left the index carry a '-YYYYMM' suffix in the source and are kept
    as-is (they are usually absent from Yahoo, which is the survivorship bias we report)."""
    r = requests.get(HISTORY_URL, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].map(lambda s: [_hist_symbol(t) for t in str(s).split(",") if t])
    return df.set_index("date").sort_index()


def _hist_symbol(t: str) -> str:
    t = t.strip()
    if "-" in t and t.rsplit("-", 1)[1].isdigit():      # delisted marker, e.g. 'TMC-200006'
        base, suffix = t.rsplit("-", 1)
        return yahoo_symbol(base) + "~" + suffix          # '~' never appears in Yahoo symbols
    return yahoo_symbol(t)


def members_on(history: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    sub = history.loc[:pd.Timestamp(date)]
    if sub.empty:
        return []
    return list(sub["tickers"].iloc[-1])


def all_symbols(history: pd.DataFrame, start: str | pd.Timestamp) -> list[str]:
    sub = history.loc[pd.Timestamp(start) - pd.Timedelta(days=400):]
    syms = set()
    for lst in sub["tickers"]:
        syms.update(lst)
    # include the row in force at `start`
    syms.update(members_on(history, pd.Timestamp(start)))
    return sorted(syms)
