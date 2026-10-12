"""Synthetic market data for tests and offline development. NOT used for any reported result."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.store import HistoryStore


def _gbm(rng, n, mu, sigma, s0, regime_len=250):
    """GBM with slowly switching drift regimes, so trend rules have something to find."""
    drift = np.repeat(rng.normal(mu, abs(mu) * 2 + 0.05, size=n // regime_len + 1), regime_len)[:n]
    r = drift / 252 + sigma / np.sqrt(252) * rng.standard_normal(n)
    return s0 * np.exp(np.cumsum(r))


def make_store(start="2006-01-02", end="2026-10-09", n_stocks=60, seed=42) -> tuple[HistoryStore, dict]:
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, end)
    days = pd.date_range(start, end, freq="D")
    nb, nd = len(bdays), len(days)
    store = HistoryStore()

    def put(key, idx, close, vol=None, open_jitter=0.002):
        close = pd.Series(close, index=idx)
        opn = close.shift(1).fillna(close.iloc[0]) * (1 + open_jitter * rng.standard_normal(len(idx)))
        rng_hi = np.abs(close.pct_change().fillna(0)).values * 0.6 + 0.004
        hi = np.maximum(opn, close) * (1 + rng_hi * rng.uniform(0.2, 1.0, len(idx)))
        lo = np.minimum(opn, close) * (1 - rng_hi * rng.uniform(0.2, 1.0, len(idx)))
        store.set_series("close", key, close)
        store.set_series("open", key, opn)
        store.set_series("high", key, pd.Series(hi, index=idx))
        store.set_series("low", key, pd.Series(lo, index=idx))
        if vol is not None:
            store.set_series("volume", key, pd.Series(vol, index=idx))

    specs = {"SP500": (0.08, 0.17, 100), "WORLD_EQ": (0.07, 0.15, 60), "EM_EQ": (0.05, 0.20, 30),
             "EUR_GOVT": (0.02, 0.05, 120), "GOLD": (0.05, 0.15, 40), "COMMOD": (0.02, 0.18, 25),
             "MOM_WORLD": (0.08, 0.16, 50), "VAL_WORLD": (0.06, 0.15, 40), "QUAL_WORLD": (0.08, 0.14, 55)}
    for k, (mu, sig, s0) in specs.items():
        put(k, bdays, _gbm(rng, nb, mu, sig, s0))
    eurusd = 1.15 * np.exp(np.cumsum(0.06 / np.sqrt(365) * rng.standard_normal(nd)))
    store.set_series("close", "EURUSD", pd.Series(eurusd, index=days))
    btc_usd = _gbm(rng, nd, 0.4, 0.70, 400, regime_len=365)
    eth_usd = _gbm(rng, nd, 0.4, 0.85, 20, regime_len=365)
    put("BTC_PERP", days, btc_usd)
    put("ETH_PERP", days, eth_usd)
    put("BTC_SPOT", days, btc_usd / eurusd)
    put("ETH_SPOT", days, eth_usd / eurusd)
    spx = _gbm(rng, nb, 0.08, 0.17, 1200)
    put("XSP", bdays, spx / 10)
    vix = np.clip(18 + np.cumsum(rng.standard_normal(nb)) * 0.6 - np.linspace(0, 0, nb), 10, 80)
    # mean revert VIX
    v = np.empty(nb)
    v[0] = 18
    for i in range(1, nb):
        v[i] = max(9.5, v[i - 1] + 0.05 * (18 - v[i - 1]) + 1.2 * rng.standard_normal())
    store.set_series("close", "VIX", pd.Series(v, index=bdays))
    store.set_series("close", "TBILL", pd.Series(np.clip(2 + np.cumsum(rng.standard_normal(nb)) * 0.02, 0, 6), index=bdays))
    # funding: daily sums, mostly low with occasional high regimes
    for k in ("BTC_PERP", "ETH_PERP"):
        base = np.where((np.arange(nd) // 120) % 4 == 0, 0.0006, 0.00008)
        store.set_series("funding", k, pd.Series(base + 0.0002 * rng.standard_normal(nd), index=days))
    # stocks
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    sectors = {t: ("Financials" if i % 10 == 0 else "Information Technology" if i % 3 == 0 else "Industrials")
               for i, t in enumerate(tickers)}
    for i, t in enumerate(tickers):
        px = _gbm(rng, nb, rng.normal(0.08, 0.06), rng.uniform(0.2, 0.4), rng.uniform(20, 300))
        vol = rng.uniform(2e5, 2e6) * np.exp(0.3 * rng.standard_normal(nb))
        put(t, bdays, px, vol)
    membership = pd.DataFrame({"tickers": [tickers[:50] + ["DEAD~200901"], tickers[5:55], tickers[10:60]]},
                              index=pd.to_datetime(["2005-01-01", "2012-06-01", "2019-03-01"]))
    store.meta["synthetic"] = True
    extras = {"membership": membership, "sectors": sectors}
    return store, extras
