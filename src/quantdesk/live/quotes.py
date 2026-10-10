"""Live quotes for the paper run (Yahoo for listed instruments, ccxt for crypto, Yahoo chains for options)."""
from __future__ import annotations

import logging
import math

import pandas as pd

from ..config import Instrument
from ..data import crypto, yahoo
from ..execution import Quote
from ..ledger import FxRates

log = logging.getLogger(__name__)

# a live quote this far from yesterday's close is treated as a bad print (marks fall back to the last good one)
MAX_JUMP = {"etf": 0.20, "stock": 0.35, "index": 0.15, "crypto_spot": 0.40, "perp": 0.40, "fx": 0.05}


class LiveQuotes:
    def __init__(self, resolved_symbols: dict[str, str] | None = None, option_chains=None,
                 yahoo_fn=yahoo.live_quote, ccxt_fn=crypto.live_quote, fallback_eurusd: float | None = None,
                 reference: dict[str, float] | None = None):
        self.resolved = resolved_symbols or {}
        self.options = option_chains
        self._yahoo = yahoo_fn
        self._ccxt = ccxt_fn
        self._cache: dict[str, Quote | None] = {}
        self._fx: FxRates | None = None
        self.fallback_eurusd = fallback_eurusd
        self.fx_source = None
        self.reference = reference or {}          # last daily close per instrument, for sanity checks
        self.suspicious: list[str] = []

    def quote(self, inst: Instrument) -> Quote | None:
        if inst is None:
            return None
        if inst.id in self._cache:
            return self._cache[inst.id]
        q = None
        try:
            if inst.is_option:
                if self.options is not None:
                    d = self.options.quote(inst.expiry, inst.strike, inst.right)
                    if d:
                        q = Quote(d["mid"], d.get("bid"), d.get("ask"), pd.Timestamp.now(tz="UTC").isoformat(),
                                  f"yahoo-chain:{self.options.source}")
            elif inst.ccxt:
                d = self._ccxt(inst.ccxt["exchange"], inst.ccxt["symbol"])
                if d:
                    q = Quote(float(d["mid"]), d.get("bid"), d.get("ask"), d.get("ts"), d.get("source", ""))
            else:
                sym = self.resolved.get(inst.id) or (inst.yf[0] if inst.yf else inst.id)
                d = self._yahoo(sym)
                if d:
                    scale = inst.price_scale
                    q = Quote(float(d["mid"]) * scale, (d.get("bid") or 0) * scale or None,
                              (d.get("ask") or 0) * scale or None, d.get("ts"), d.get("source", ""))
        except Exception as exc:
            log.warning("quote failed for %s: %s", inst.id, exc)
            q = None
        if q is not None and (not math.isfinite(q.mid) or q.mid <= 0):
            q = None
        ref = self.reference.get(inst.id)
        if q is not None and ref and not inst.is_option:
            limit = MAX_JUMP.get(inst.kind, 0.25)
            if abs(q.mid / ref - 1) > limit:
                self.suspicious.append(f"{inst.id}: live {q.mid:.4g} vs last close {ref:.4g} (> {limit:.0%} move) - quote ignored")
                log.warning(self.suspicious[-1])
                q = None
        self._cache[inst.id] = q
        return q

    def fx(self) -> FxRates:
        if self._fx is None:
            d = self._yahoo("EURUSD=X")
            rate = float(d["mid"]) if d else None
            self.fx_source = "live"
            if rate is None or not (0.7 < rate < 1.6):
                if self.fallback_eurusd and 0.7 < self.fallback_eurusd < 1.6:
                    rate, self.fx_source = self.fallback_eurusd, "previous close"
                else:
                    raise RuntimeError("no EURUSD quote - refusing to value USD positions with a guessed rate")
            self._fx = FxRates(rate)
        return self._fx


class ExecutionQuotes:
    """Wraps LiveQuotes for order execution: rejects quotes older than `max_age_minutes`."""

    def __init__(self, live: LiveQuotes, now: pd.Timestamp, max_age_minutes: int = 120):
        self.live = live
        self.now = now
        self.max_age = pd.Timedelta(minutes=max_age_minutes)
        self.stale: list[str] = []

    def quote(self, inst: Instrument) -> Quote | None:
        q = self.live.quote(inst)
        if q is None:
            return None
        if not q.ts:
            self.stale.append(f"{inst.id}: quote has no timestamp - not traded")
            return None
        if q.ts:
            try:
                ts = pd.Timestamp(q.ts)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
                if self.now - ts > self.max_age:
                    self.stale.append(f"{inst.id} quote from {q.ts}")
                    return None
            except Exception:
                pass
        return q

    def fx(self) -> FxRates:
        return self.live.fx()
