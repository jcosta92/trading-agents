"""Builds HistoryStore objects from the network (Yahoo, ccxt, GitHub-hosted membership data).

Used by:
  * `fetch-history` (full history for backtests, saved to data/history/)
  * the paper runner (recent history only, rebuilt every run)
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Instrument
from . import crypto, universe, yahoo
from .store import HistoryStore, splice_with_proxy, to_eur

log = logging.getLogger(__name__)


def build_core_store(instruments: dict[str, Instrument], ids: list[str] | None = None, start: str = "2003-01-01",
                     use_proxies: bool = True, funding: bool = True, funding_since: str = "2018-01-01") -> HistoryStore:
    """Static instruments (ETFs, crypto, indices, FX) with proxy splicing, plus perp funding."""
    ids = ids or [i for i, inst in instruments.items() if inst.yf]
    ids = list(dict.fromkeys(ids + ["EURUSD"]))
    symbols = []
    for iid in ids:
        inst = instruments[iid]
        symbols += list(inst.yf)
        if use_proxies and inst.proxy:
            symbols.append(inst.proxy["yf"])
    raw = yahoo.download_history(symbols, start=start, fields=("open", "high", "low", "close", "volume"))
    empty = pd.DataFrame()
    close = raw.get("close", empty)
    store = HistoryStore()
    meta = {"resolved": {}, "inception": {}, "proxy": {}, "missing": []}
    eurusd = close["EURUSD=X"].dropna() if "EURUSD=X" in close else pd.Series(dtype=float)
    for iid in ids:
        inst = instruments[iid]
        sym = next((s for s in inst.yf if s in close and close[s].dropna().shape[0] > 20), None)
        if sym is None:
            meta["missing"].append(iid)
            log.warning("no Yahoo data for %s (%s)", iid, inst.yf)
            continue
        c = close[sym].dropna() * inst.price_scale
        series = {"close": c}
        for f in ("open", "high", "low"):
            fr = raw.get(f, empty)
            series[f] = (fr[sym].dropna() * inst.price_scale) if sym in fr else c
        v = raw.get("volume", empty)[sym].dropna() if sym in raw.get("volume", empty) else None
        meta["resolved"][iid] = sym
        meta["inception"][iid] = str(c.index[0].date())
        psym = inst.proxy["yf"] if (use_proxies and inst.proxy) else None
        if psym and psym in close:
            pser = {f: (raw.get(f, empty)[psym].dropna() if psym in raw.get(f, empty) else close[psym].dropna())
                    for f in ("open", "high", "low", "close")}
            if inst.proxy.get("currency", inst.currency) == "USD" and inst.currency == "EUR" and len(eurusd):
                pser = {f: to_eur(x, eurusd) for f, x in pser.items()}
            pc = pser["close"]
            if len(pc) and pc.index[0] < c.index[0]:
                scale = c.iloc[0] / pc.loc[:c.index[0]].iloc[-1]
                series["close"] = splice_with_proxy(c, pc)
                for f in ("open", "high", "low"):
                    cur = series[f]
                    series[f] = pd.concat([pser[f][pser[f].index < cur.index[0]] * scale, cur]).sort_index()
                meta["proxy"][iid] = {"symbol": psym, "until": meta["inception"][iid]}
        for f, x in series.items():
            store.set_series(f, iid, x)
        if v is not None:
            store.set_series("volume", iid, v)
    if funding:
        fmeta = add_funding(store, instruments, since=funding_since)
        meta["funding"] = fmeta
    store.meta.update(meta)
    return store


def add_funding(store: HistoryStore, instruments: dict[str, Instrument], since: str = "2018-01-01",
                prefer: str = "longest") -> dict:
    """Daily funding sums per perp. `prefer`='longest' (backtests) or 'first' (live: the modelled venue)."""
    meta = {}
    for iid, inst in instruments.items():
        if not inst.is_perp or not inst.funding:
            continue
        best, best_src, info = None, None, {}
        for src in inst.funding:
            ev = crypto.funding_history(src["source"], src["symbol"], since=since)
            info[src["source"]] = {"events": int(len(ev)),
                                   "first": str(ev.index[0]) if len(ev) else None,
                                   "last": str(ev.index[-1]) if len(ev) else None}
            if ev.empty:
                continue
            daily = crypto.daily_funding(ev)
            store.set_series(f"funding_{src['source']}", iid, daily)
            if best is None or (prefer == "longest" and len(daily) > len(best)):
                best, best_src = daily, src["source"]
            if prefer == "first":
                break
        if best is not None:
            store.set_series("funding", iid, best)
        # unit check: sources should agree on the size of daily funding where they overlap
        daily = {src["source"]: store.series(f"funding_{src['source']}", iid) for src in inst.funding}
        daily = {k: v for k, v in daily.items() if len(v)}
        if len(daily) > 1:
            common = None
            for v in daily.values():
                common = v.index if common is None else common.intersection(v.index)
            if common is not None and len(common) > 30:
                info["overlap_days"] = int(len(common))
                info["mean_daily_annualised"] = {k: float(v.reindex(common).mean() * 365) for k, v in daily.items()}
        meta[iid] = {"chosen": best_src, "sources": info}
    return meta


def build_sp500_store(start: str = "2006-06-01", end: str | None = None, current_only: bool = False,
                      tickers: list[str] | None = None) -> tuple[HistoryStore, dict]:
    """Close/open/volume for S&P 500 members. Backtest: everyone who was a member since `start`."""
    extras: dict = {}
    cur = universe.current_constituents()
    extras["sectors"] = dict(zip(cur["ticker"], cur["sector"]))
    if tickers is None:
        if current_only:
            tickers = list(cur["ticker"])
        else:
            hist = universe.membership_history()
            extras["membership"] = hist
            tickers = [t for t in universe.all_symbols(hist, start) if "~" not in t]
    raw = yahoo.download_history(tickers, start=start, end=end, chunk=60)
    store = HistoryStore()
    for field in ("close", "open", "high", "low", "volume"):
        if field in raw:
            store.set_frame(field, raw[field])
    got = set(raw.get("close", pd.DataFrame()).columns[raw.get("close", pd.DataFrame()).notna().any()]) \
        if "close" in raw else set()
    store.meta["sp500"] = {"requested": len(tickers), "with_data": len(got),
                           "missing_sample": sorted(set(tickers) - got)[:50]}
    return store, extras
