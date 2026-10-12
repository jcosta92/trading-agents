"""A7: RSI(2) mean reversion on the S&P 500 UCITS ETF (Connors-style), Trading 212 cost model.

BUY  when close > SMA(200) (only buy dips in an uptrend) and RSI(2) < 10.
SELL when close > SMA(5); in the paper run this is also a resting take-profit at the 5-day average,
     checked hourly. Protective stop 3xATR(10) below entry and a 10-business-day time exit.
"""
from __future__ import annotations

import math

from .. import indicators as ind
from ..context import Context
from ..execution import OrderIntent
from ..ledger import Ledger
from .base import Agent


class RSI2MeanReversion(Agent):
    def static_instruments(self) -> list[str]:
        return [self.p["instrument"]]

    def lookback_days(self) -> int:
        return int(self.p["trend_sma"]) + 40

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        if not self.once_per_day(ctx, ledger):
            return []
        p = self.p
        iid = p["instrument"]
        d = ctx.ohlc(iid, int(p["trend_sma"]) + 60)
        if len(d) < int(p["trend_sma"]) + 1 or (ctx.asof - d.index[-1]).days > 7:
            ledger.meta["signals"] = {iid: {"state": "NO DATA", "bars": len(d)}}
            return []
        c = d["close"]
        close = float(c.iloc[-1])
        trend = ind.last(ind.sma(c, int(p["trend_sma"])))
        r = ind.last(ind.rsi(c, int(p["rsi_window"])))
        exit_sma = ind.last(ind.sma(c, int(p["exit_sma"])))
        a = ind.last(ind.atr(d["high"], d["low"], c, int(p["atr_window"])))
        sig = {"close": round(close, 4), f"sma{p['trend_sma']}": round(trend, 4), f"rsi{p['rsi_window']}": round(r, 2),
               f"sma{p['exit_sma']}": round(exit_sma, 4), f"atr{p['atr_window']}": round(a, 4),
               "buy_rule": f"close > SMA{p['trend_sma']} and RSI({p['rsi_window']}) < {p['rsi_entry']}",
               "sell_rule": f"close > SMA{p['exit_sma']} (take-profit order at SMA{p['exit_sma']})"}
        held = ledger.qty(iid) > 0
        orders: list[OrderIntent] = []
        if held and close > exit_sma:
            sig["state"] = "SELL"
            orders.append(OrderIntent(iid, -ledger.qty(iid), f"SELL signal: close {close:.2f} > SMA{p['exit_sma']} {exit_sma:.2f}"))
        elif held:
            sig["state"] = "HOLD"
        elif close > trend and r < float(p["rsi_entry"]) and math.isfinite(a) and a > 0:
            inst = ctx.inst(iid)
            px = ctx.mark(iid) or close
            eq = ledger.equity(ctx.fx)
            qty = float(p["allocation"]) * eq / ctx.fx.to_eur(px * inst.multiplier, inst.currency)
            lot = self.cm.lot_size(inst, self.venue_override)
            qty = math.floor(qty / lot) * lot if lot > 0 else math.floor(qty * 1e6) / 1e6
            if qty > 0:
                sig["state"] = "BUY"
                orders.append(OrderIntent(
                    iid, qty, f"BUY signal: RSI({p['rsi_window']}) {r:.1f} < {p['rsi_entry']} with close {close:.2f} > "
                              f"SMA{p['trend_sma']} {trend:.2f}",
                    meta={"protect": {"stop_atr": float(p["stop_atr"]), "atr": a, "target_sma": int(p["exit_sma"]),
                                      "sma_value": exit_sma, "max_days": int(p["max_days"]),
                                      "label": f"stop {p['stop_atr']}xATR({p['atr_window']}), take-profit at "
                                               f"SMA{p['exit_sma']}, time exit {p['max_days']} days"}}))
        else:
            sig["state"] = "FLAT"
        ledger.meta["signals"] = {iid: sig}
        return orders
