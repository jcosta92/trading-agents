"""Transaction cost model shared by the backtester and the paper-trading runner.

Every simulated order pays:
  1. explicit fees  - broker commission + exchange/clearing/regulatory pass-through (venues.yaml)
  2. the spread     - buys fill at the ask, sells at the bid (or mid +/- a fallback half-spread)
  3. slippage       - an extra adverse move in basis points
  4. FX conversion  - when EUR cash is converted to buy a USD instrument (handled by the ledger)
A `stress` multiplier scales 1-3 (and the FX fee): 0 = gross, 1 = modelled, 2 = doubled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Instrument


@dataclass
class FeeBreakdown:
    components: dict[str, float] = field(default_factory=dict)
    currency: str = "EUR"

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))


class CostModel:
    def __init__(self, venues: dict[str, dict], stress: float = 1.0):
        self.venues = venues
        self.stress = float(stress)

    # ------------------------------------------------------------------ helpers
    def venue_id(self, inst: Instrument, override: str | None = None) -> str:
        if override and inst.kind in ("etf", "stock"):
            return override
        return inst.venue

    def venue(self, inst: Instrument, override: str | None = None) -> dict:
        return self.venues[self.venue_id(inst, override)]

    def lot_size(self, inst: Instrument, override: str | None = None) -> float:
        if self.venue(inst, override).get("fractional"):
            return 0.0
        return float(inst.lot_size or 0.0)

    def fx_fee_rate(self, inst: Instrument, override: str | None = None) -> float:
        return float(self.venue(inst, override).get("fx_conversion", 0.0)) * self.stress

    # ------------------------------------------------------------------ explicit fees
    def fees(self, inst: Instrument, qty: float, price: float, side: str,
             override: str | None = None) -> FeeBreakdown:
        """Explicit fees for one order of |qty| units at `price` (per unit, before multiplier)."""
        q = abs(float(qty))
        out = FeeBreakdown(currency=inst.currency)
        if q <= 0 or self.stress == 0:
            return out
        value = q * abs(price) * inst.multiplier
        for f in self.venue(inst, override).get("fees", []) or []:
            sides = f.get("sides")
            if sides and side not in sides:
                continue
            basis = f.get("basis", "value")
            if basis == "value":
                b = value
            elif basis == "units":
                b = q
            elif basis == "order":
                b = 1.0
            else:
                raise ValueError(f"unknown fee basis {basis}")
            if "rate_by_premium" in f:
                rate = next(r for bound, r in f["rate_by_premium"] if abs(price) < bound)
            else:
                rate = float(f.get("rate", 0.0))
            amt = float(f.get("fixed", 0.0)) + rate * b
            if "min" in f:
                amt = max(amt, float(f["min"]))
            if "max" in f:
                amt = min(amt, float(f["max"]))
            if "max_pct_value" in f:
                amt = min(amt, float(f["max_pct_value"]) * value)
            out.components[f["name"]] = amt * self.stress
        return out

    # ------------------------------------------------------------------ spread + slippage
    def _fallback_half_spread(self, inst: Instrument, mid: float) -> float:
        if inst.is_option:
            return max(inst.half_spread_abs or 0.03, 0.0)
        return mid * inst.half_spread_bps / 1e4

    def exec_price(self, inst: Instrument, side: str, mid: float, bid: float | None = None,
                   ask: float | None = None, override: str | None = None) -> float | None:
        """Pessimistic fill price, never better than the quoted side of the book.

        * bid and ask both quoted: buy at the ask, sell at the bid (+ slippage), however wide the spread
        * only the relevant side quoted: cross it
        * a quoted bid of 0 on a sell: no fill (None)
        * nothing quoted: mid +/- the instrument's fallback half-spread
        `stress` scales the distance from mid (0 = mid, 2 = twice the spread + slippage)."""
        has_bid, has_ask = _pos(bid), _pos(ask)
        if side == "sell" and bid is not None and _finite(bid) and bid <= 0:
            return None
        if has_bid and has_ask and ask >= bid:
            mid = (bid + ask) / 2.0
            hs = (ask - bid) / 2.0
        elif side == "buy" and has_ask and _pos(mid) and ask >= mid:
            hs = ask - mid
        elif side == "sell" and has_bid and _pos(mid) and bid <= mid:
            hs = mid - bid
        else:
            if not _pos(mid):
                return None
            if side == "buy" and has_bid:
                mid = max(mid, bid)          # never buy below a quoted bid
            if side == "sell" and has_ask:
                mid = min(mid, ask)          # never sell above a quoted ask
            hs = self._fallback_half_spread(inst, mid)
        v = self.venue(inst, override)
        hs += mid * float(v.get("extra_half_spread_bps", 0.0)) / 1e4
        slip = mid * float(v.get("slippage_bps", 0.0)) / 1e4
        sign = 1.0 if side == "buy" else -1.0
        px = mid + sign * self.stress * (hs + slip)
        if px <= 0:
            return None
        return px

    def estimate_cost(self, inst: Instrument, qty: float, mid: float, bid: float | None = None,
                      ask: float | None = None, override: str | None = None) -> float:
        """Estimated total cost (fees + spread + slippage) of an order, in instrument currency."""
        side = "buy" if qty > 0 else "sell"
        px = self.exec_price(inst, side, mid, bid, ask, override)
        if px is None:
            return math.inf
        impl = abs(px - mid) * abs(qty) * inst.multiplier
        return impl + self.fees(inst, qty, px, side, override).total


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _pos(x) -> bool:
    return _finite(x) and float(x) > 0


def valid_quote(bid: float | None, ask: float | None, inst: Instrument | None = None) -> bool:
    """Both sides quoted, positive and not crossed."""
    return _pos(bid) and _pos(ask) and ask >= bid
