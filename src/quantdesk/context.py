"""Point-in-time context handed to agents. Agents may only see data through this object.

`asof` is the last COMPLETED daily bar the agent may use. In a backtest that is the current
simulated date (orders then fill at the next bar's open); in the paper run it is yesterday
(orders fill at live quotes today). `history()` and `panel()` slice at `asof`, which is the
look-ahead guard; tests perturb data after `asof` and check that decisions do not change.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .data.store import HistoryStore
from .execution import Quote, QuoteSource
from .ledger import FxRates
from .registry import Registry


class Context:
    def __init__(self, asof: pd.Timestamp, store: HistoryStore, registry: Registry, quotes: QuoteSource,
                 mode: str, now: pd.Timestamp | None = None, extras: dict[str, Any] | None = None,
                 options=None):
        self.asof = pd.Timestamp(asof).normalize()
        self._store = store
        self.registry = registry
        self.quotes = quotes
        self.mode = mode                      # 'backtest' | 'live'
        self.now = pd.Timestamp(now) if now is not None else self.asof
        self.extras = extras or {}
        self.options = options                # SyntheticOptionModel (backtest) or LiveOptionChains (live)
        self.log: list[str] = []

    # ------------------------------------------------------------------ data (point in time)
    def history(self, key: str, field: str = "close", lookback: int | None = None) -> pd.Series:
        s = self._store.series(field, key).loc[:self.asof]
        return s.iloc[-lookback:] if lookback else s

    def panel(self, keys: list[str], field: str = "close", lookback: int | None = None) -> pd.DataFrame:
        df = self._store.frame(field)
        if df.empty:
            return df
        cols = [k for k in keys if k in df.columns]
        out = df.loc[:self.asof, cols].dropna(how="all")   # drop dates on which none of these trade
        return out.iloc[-lookback:] if lookback else out

    def has(self, key: str, field: str = "close") -> bool:
        return not self.history(key, field).empty

    # ------------------------------------------------------------------ prices for sizing
    def inst(self, iid: str):
        return self.registry.get(iid)

    def quote(self, iid: str) -> Quote | None:
        inst = self.registry.get(iid)
        return self.quotes.quote(inst) if inst else None

    def mark(self, iid: str) -> float | None:
        q = self.quote(iid)
        return q.mid if q else None

    @property
    def fx(self) -> FxRates:
        return self.quotes.fx()

    def option_chain(self, dte_min: int, dte_max: int, right: str = "P") -> pd.DataFrame | None:
        """Puts/calls with columns expiry, dte, strike, bid, ask, mid, delta (as of `asof`/now)."""
        if self.options is None:
            return None
        if self.mode == "backtest":
            return self.options.chain(self.asof, dte_min, dte_max, right)
        return self.options.chain_between(dte_min, dte_max, right)

    def note(self, msg: str) -> None:
        self.log.append(msg)
