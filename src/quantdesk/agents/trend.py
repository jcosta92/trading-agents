"""Indicator-driven trend agents.

A1 TrendETF          UCITS ETFs: BUY when close > SMA(50) > SMA(200), 12-month ROC > 0 and the close
                     is a new 20-day high (the breakout trigger stops instant re-entry after a stop);
                     SELL when SMA(50) < SMA(200); 3xATR(20) chandelier trailing stop.
A3 DonchianBreakout  BTC/ETH perps: BUY on a close above the 20-day high; SELL on a close below the
                     10-day low; fixed 2xATR(20) stop (Turtle-style "2N").
Sizing for both: risk a fixed fraction of equity between entry and stop, capped per position.
Neither uses a fixed take-profit: trend systems earn their return from the few long trends, and a
fixed target would cut exactly those. The trailing stop / channel exit locks in gains instead.
"""
from __future__ import annotations

import math

from .. import indicators as ind
from ..context import Context
from ..execution import OrderIntent
from ..ledger import Ledger
from ..portfolio import sort_sells_first
from .base import Agent


def _fresh(ctx: Context, d, max_age_days: int) -> bool:
    return len(d) > 0 and (ctx.asof - d.index[-1]).days <= max_age_days


class TrendETF(Agent):
    def static_instruments(self) -> list[str]:
        return list(self.p["instruments"])

    def lookback_days(self) -> int:
        return int(self.p.get("roc_window", 252)) + 60

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        if not self.once_per_day(ctx, ledger):
            return []
        p = self.p
        fast, slow, rw = int(p["sma_fast"]), int(p["sma_slow"]), int(p["roc_window"])
        atr_n, stop_atr = int(p["atr_window"]), float(p["stop_atr"])
        signals, orders = {}, []
        for iid in p["instruments"]:
            d = ctx.ohlc(iid, max(slow, rw) + 60)
            held = ledger.qty(iid) > 0
            if len(d) < max(slow, rw) + 1 or not _fresh(ctx, d, 7):
                signals[iid] = {"state": "NO DATA", "bars": len(d)}
                continue
            c = d["close"]
            close = float(c.iloc[-1])
            s_fast, s_slow = ind.last(ind.sma(c, fast)), ind.last(ind.sma(c, slow))
            r = ind.last(ind.roc(c, rw))
            a = ind.last(ind.atr(d["high"], d["low"], c, atr_n))
            bw = int(p["breakout_window"])
            prior_high = float(c.iloc[-bw - 1:-1].max())
            buy = close > s_fast > s_slow and r > 0 and close >= prior_high
            sell = s_fast < s_slow
            sig = {"close": round(close, 4), f"sma{fast}": round(s_fast, 4), f"sma{slow}": round(s_slow, 4),
                   f"roc{rw}": round(r, 4), f"atr{atr_n}": round(a, 4), f"high_close{bw}": round(prior_high, 4),
                   "buy_rule": f"close > SMA{fast} > SMA{slow}, ROC{rw} > 0 and a new {bw}-day closing high",
                   "sell_rule": f"SMA{fast} < SMA{slow}"}
            if held and sell:
                sig["state"] = "SELL"
                orders.append(OrderIntent(iid, -ledger.qty(iid),
                                          f"SELL signal: SMA{fast} {s_fast:.2f} < SMA{slow} {s_slow:.2f} (death cross)"))
            elif held:
                sig["state"] = "HOLD"
            elif buy and math.isfinite(a) and a > 0:
                qty = self.risk_qty(ctx, ledger, iid, stop_atr * a, float(p["risk_per_trade"]), float(p["max_position"]))
                inst = ctx.inst(iid)
                val = ctx.fx.to_eur(qty * (ctx.mark(iid) or close) * inst.multiplier, inst.currency)
                if qty > 0 and val >= float(p.get("min_trade_eur", 25)):
                    sig["state"] = "BUY"
                    orders.append(OrderIntent(
                        iid, qty, f"BUY signal: close {close:.2f} > SMA{fast} {s_fast:.2f} > SMA{slow} {s_slow:.2f}, "
                                  f"ROC{rw} {r:+.1%}, new {bw}-day high",
                        meta={"protect": {"stop_atr": stop_atr, "atr": a, "trail_atr": stop_atr, "atr_n": atr_n,
                                          "label": f"{stop_atr:g}xATR({atr_n}) chandelier trailing stop"}}))
                else:
                    sig["state"] = "BUY (too small)"
                    ctx.note(f"{iid}: buy signal but position would be {val:.0f} EUR")
            else:
                sig["state"] = "FLAT"
            signals[iid] = sig
        ledger.meta["signals"] = signals
        return sort_sells_first(orders, ledger)


class DonchianBreakout(Agent):
    def static_instruments(self) -> list[str]:
        return list(self.p["instruments"])

    def lookback_days(self) -> int:
        return max(int(self.p["entry_window"]), int(self.p["atr_window"])) + 60

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        if not self.once_per_day(ctx, ledger):
            return []
        p = self.p
        ew, xw, atr_n, stop_atr = int(p["entry_window"]), int(p["exit_window"]), int(p["atr_window"]), float(p["stop_atr"])
        signals, orders = {}, []
        fx = ctx.fx
        eq = ledger.equity(fx)
        # current gross notional, to respect max_leverage when adding
        gross = sum(fx.to_eur(abs(pos.qty) * (ctx.mark(i) or pos.last_mark or pos.avg_price) * pos.multiplier, pos.currency)
                    for i, pos in ledger.positions.items() if pos.kind == "perp")
        for iid in p["instruments"]:
            d = ctx.ohlc(iid, max(ew, xw, atr_n) + 80)
            held = ledger.qty(iid) > 0
            if len(d) < max(ew, xw, atr_n) + 2 or not _fresh(ctx, d, 3):
                signals[iid] = {"state": "NO DATA", "bars": len(d)}
                continue
            close = float(d["close"].iloc[-1])
            hi = ind.last(ind.donchian_high(d["high"], ew))
            lo = ind.last(ind.donchian_low(d["low"], xw))
            n = ind.last(ind.atr(d["high"], d["low"], d["close"], atr_n))
            sig = {"close": round(close, 2), f"high{ew}": round(hi, 2), f"low{xw}": round(lo, 2), f"atr{atr_n}": round(n, 2),
                   "buy_rule": f"close > highest high of previous {ew} days", "sell_rule": f"close < lowest low of previous {xw} days"}
            if held and close < lo:
                sig["state"] = "SELL"
                orders.append(OrderIntent(iid, -ledger.qty(iid), f"SELL signal: close {close:.2f} < {xw}-day low {lo:.2f}"))
            elif held:
                sig["state"] = "HOLD"
            elif close > hi and math.isfinite(n) and n > 0:
                qty = self.risk_qty(ctx, ledger, iid, stop_atr * n, float(p["risk_per_trade"]), float(p["max_position"]))
                inst = ctx.inst(iid)
                notional = fx.to_eur(qty * (ctx.mark(iid) or close) * inst.multiplier, inst.currency)
                room = float(p.get("max_leverage", 1.0)) * eq - gross
                if notional > room > 0:
                    qty = math.floor(qty * room / notional * 1e6) / 1e6
                    notional = room
                if qty > 0 and notional >= float(p.get("min_trade_eur", 20)):
                    gross += notional
                    sig["state"] = "BUY"
                    orders.append(OrderIntent(
                        iid, qty, f"BUY signal: close {close:.2f} > {ew}-day high {hi:.2f} (breakout)",
                        meta={"protect": {"stop_atr": stop_atr, "atr": n,
                                          "label": f"{stop_atr:g}N stop, N = ATR({atr_n}) = {n:.2f}"}}))
                else:
                    sig["state"] = "BUY (no room)"
            else:
                sig["state"] = "FLAT"
            signals[iid] = sig
        ledger.meta["signals"] = signals
        return sort_sells_first(orders, ledger)
