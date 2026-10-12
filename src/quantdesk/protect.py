"""Protective orders attached to open positions: stop-loss, take-profit and time exits.

An agent attaches a spec to an entry order (OrderIntent.meta["protect"]). When the entry fills,
the levels are computed from the actual fill price and stored in ledger.meta["protect"][inst]:

    stop       stop-loss price (sell everything if the price trades at/below it)
    target     take-profit price (sell everything if the price trades at/above it)
    hwm        highest close since entry (for trailing stops)
    opened     entry date (for time exits)

Spec fields (all optional):
    stop_atr, atr       initial stop = entry - stop_atr * atr       (atr in price units, from the decision)
    stop_pct            initial stop = entry * (1 - stop_pct)
    trail_atr, atr_n    chandelier trailing stop: max(stop, highest close since entry - trail_atr * ATR(atr_n))
    target_pct          take-profit = entry * (1 + target_pct)
    target_sma          take-profit = SMA(n) of closes, re-set every day (a resting limit sell at the average);
                        only armed while the average is above the entry price - a take-profit below the
                        entry would just be a second stop
    max_days            time exit after n business days
    label               human-readable description shown on the dashboard

Paper run: levels are checked against live prices every hourly run (a resting stop/limit order is
approximated by polling, so a fast move can fill beyond the level - which is realistic for stop
orders). Backtests check each day's open/high/low bar: a gap through a level fills at the open (the
open is the day's first trade), otherwise at the level; if the stop and the target are both inside
one bar, the STOP is assumed first. Time exits fill at the open of the due day in backtests and at
the first open cycle of that day in the paper run. Only long positions are protected.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import indicators as ind
from .execution import OrderIntent, Quote


def _book(ledger) -> dict:
    return ledger.meta.setdefault("protect", {})


def _armed(target: float, entry: float) -> float | None:
    return target if target > entry else None


def register(ledger, fills) -> None:
    """Create protection for positions opened by fills that carry a protect spec."""
    book = _book(ledger)
    for f in fills:
        spec = getattr(f, "protect", None)
        if not spec or f.qty <= 0:
            continue
        pos = ledger.positions.get(f.inst)
        if pos is None or pos.qty <= 0 or f.inst in book:
            continue
        entry = float(f.price)
        stop = None
        if spec.get("stop_atr") and spec.get("atr"):
            stop = entry - float(spec["stop_atr"]) * float(spec["atr"])
        if spec.get("stop_pct"):
            s2 = entry * (1 - float(spec["stop_pct"]))
            stop = s2 if stop is None else max(stop, s2)
        target = entry * (1 + float(spec["target_pct"])) if spec.get("target_pct") else None
        if spec.get("target_sma") and spec.get("sma_value"):
            target = _armed(float(spec["sma_value"]), entry)
        # "updated" = last completed daily bar folded into hwm/trailing stop; empty so that the entry
        # day's own close is used as soon as that bar is complete
        book[f.inst] = {"entry": entry, "opened": str(f.ts)[:10], "stop": stop, "target": target, "hwm": entry,
                        "spec": {k: v for k, v in spec.items() if k != "sma_value"}, "updated": ""}


def cleanup(ledger) -> None:
    book = _book(ledger)
    for iid in list(book):
        p = ledger.positions.get(iid)
        if p is None or p.qty <= 0:
            book.pop(iid)


def update_daily(ledger, ctx) -> None:
    """Move trailing stops up and re-set indicator targets from completed daily bars (never loosens a stop)."""
    cleanup(ledger)
    day = ctx.asof.date().isoformat()
    for iid, pr in _book(ledger).items():
        if pr.get("updated", "") >= day:
            continue
        spec = pr["spec"]
        d = ctx.ohlc(iid, 150)
        if d.empty:
            continue
        since = d[d.index >= pd.Timestamp(max(pr["opened"], pr.get("updated") or pr["opened"]))]
        if len(since):
            pr["hwm"] = max(pr["hwm"], float(since["close"].max()))
        if spec.get("trail_atr"):
            a = ind.last(ind.atr(d["high"], d["low"], d["close"], int(spec.get("atr_n", 20))))
            if math.isfinite(a):
                new = pr["hwm"] - float(spec["trail_atr"]) * a
                pr["stop"] = new if pr["stop"] is None else max(pr["stop"], new)
        if spec.get("target_sma"):
            t = ind.last(ind.sma(d["close"], int(spec["target_sma"])))
            if math.isfinite(t):
                pr["target"] = _armed(t, pr["entry"])
        pr["updated"] = day


def time_exits(ledger, ctx) -> list[OrderIntent]:
    now = ctx.now.tz_convert(None) if getattr(ctx.now, "tzinfo", None) else ctx.now
    return due_time_exits(ledger, now)


def due_time_exits(ledger, today: pd.Timestamp) -> list[OrderIntent]:
    """Positions that have been held for `max_days` business days on `today`."""
    out = []
    day = pd.Timestamp(today).normalize().date()
    for iid, pr in _book(ledger).items():
        n = pr["spec"].get("max_days")
        if not n or ledger.qty(iid) <= 0:
            continue
        age = int(np.busday_count(pd.Timestamp(pr["opened"]).date(), day))
        if age >= int(n):
            out.append(OrderIntent(iid, -ledger.qty(iid), f"TIME EXIT: held {age} business days (max {n})"))
    return out


def check_bar(ledger, bars: dict[str, tuple[float, float, float]]) -> list[tuple[OrderIntent, float]]:
    """Backtest: (order, fill-trigger price) for every level crossed inside today's (open, high, low) bar."""
    out = []
    for iid, pr in _book(ledger).items():
        q = ledger.qty(iid)
        if q <= 0 or iid not in bars:
            continue
        o, h, l = bars[iid]
        if any(x is None or not math.isfinite(x) for x in (o, h, l)):
            continue
        stop, target = pr.get("stop"), pr.get("target")
        label = pr["spec"].get("label", "")
        stop_msg = "STOP-LOSS hit: traded {px:.4g} <= stop {lvl:.4g} ({label})"
        tp_msg = "TAKE-PROFIT hit: traded {px:.4g} >= target {lvl:.4g} ({label})"
        if stop is not None and o <= stop:                     # gapped through the stop
            hit = (stop_msg, o, stop)
        elif target is not None and o >= target:               # gapped through the target
            hit = (tp_msg, o, target)
        elif stop is not None and l <= stop:                   # intrabar; if both, assume the stop first
            hit = (stop_msg, stop, stop)
        elif target is not None and h >= target:
            hit = (tp_msg, target, target)
        else:
            continue
        msg, px, lvl = hit
        out.append((OrderIntent(iid, -q, msg.format(px=px, lvl=lvl, label=label)), px))
    return out


def check_price(ledger, prices: dict[str, float], skip: set | None = None) -> list[OrderIntent]:
    """Paper run: market sell orders for every level crossed by the current price."""
    out = []
    for iid, pr in _book(ledger).items():
        if skip and iid in skip:
            continue
        q = ledger.qty(iid)
        p = prices.get(iid)
        if q <= 0 or p is None or not math.isfinite(p):
            continue
        stop, target = pr.get("stop"), pr.get("target")
        label = pr["spec"].get("label", "")
        if stop is not None and p <= stop:
            out.append(OrderIntent(iid, -q, f"STOP-LOSS hit: price {p:.4g} <= stop {stop:.4g} ({label})"))
        elif target is not None and p >= target:
            out.append(OrderIntent(iid, -q, f"TAKE-PROFIT hit: price {p:.4g} >= target {target:.4g} ({label})"))
    return out


def apply_split(ledger, iid: str, ratio: float) -> None:
    """Scale the protective levels of a position after a ratio:1 stock split."""
    pr = _book(ledger).get(iid)
    if not pr or not ratio or ratio <= 0:
        return
    for k in ("entry", "stop", "target", "hwm"):
        if pr.get(k) is not None:
            pr[k] = float(pr[k]) / ratio
    if pr["spec"].get("atr"):
        pr["spec"]["atr"] = float(pr["spec"]["atr"]) / ratio


def resting_orders(ledger) -> list[dict]:
    """The protective orders currently 'resting' for the dashboard."""
    rows = []
    for iid, pr in _book(ledger).items():
        q = ledger.qty(iid)
        if q <= 0:
            continue
        rows.append({"inst": iid, "qty": q, "entry": pr["entry"], "stop": pr.get("stop"), "target": pr.get("target"),
                     "max_days": pr["spec"].get("max_days"), "opened": pr["opened"], "label": pr["spec"].get("label", "")})
    return rows


class TriggerQuotes:
    """Quote source that prices given instruments at a trigger level (backtest stop/target fills).
    The cost model then applies the half-spread and slippage around that level."""

    def __init__(self, base, prices: dict[str, float]):
        self.base = base
        self.prices = prices

    def quote(self, inst):
        if inst is not None and inst.id in self.prices:
            return Quote(float(self.prices[inst.id]), source="trigger")
        return self.base.quote(inst)

    def fx(self):
        return self.base.fx()
