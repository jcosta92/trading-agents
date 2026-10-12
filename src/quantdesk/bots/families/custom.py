"""Strategies that need their own order logic: infinity grid, Avellaneda-Stoikov style quoting,
Connors TPS scale-in, and the pre-FOMC drift."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data import TF
from ..generic import EtfDailyBot
from ..strategies import Bot, UsEtfBot


class InfinityGridBot(Bot):
    """Pionex / KuCoin infinity grid: keep the coin holding worth a constant EUR value V0; sell the
    excess each time the price rises one grid step, buy back to V0 each time it falls one step;
    no upper limit, paused below a floor (60% of the starting price)."""
    timeframe = "5m"

    def warmup(self) -> int:
        return 2

    def on_candle(self, ctx, iid: str) -> None:
        st = ctx.state
        close = float(ctx.candle(iid)["close"])
        if "v0" not in st:
            st["v0"] = float(self.p.get("v0_frac", 0.5)) * ctx.equity()
            st["floor"] = close * float(self.p.get("floor", 0.6))
            st["ref"] = close
            ctx.place(iid, "buy", "market", eur=st["v0"], tag="ig:initial buy")
            return
        if ctx.orders("ig:"):
            return
        self._quote(ctx, iid)

    def _quote(self, ctx, iid):
        st = ctx.state
        g = float(self.p.get("grid", 0.015))
        q = ctx.qty(iid)
        ref = st["ref"]
        up, dn = ref * (1 + g), ref * (1 - g)
        sell_q = (q * up - st["v0"]) / up
        if sell_q * up >= 5.0 and sell_q <= q:
            ctx.place(iid, "sell", "limit", qty=sell_q, price=up, tag="ig:sell one step up")
        if dn >= st["floor"]:
            buy_q = (st["v0"] - q * dn) / dn
            if buy_q * dn >= 5.0:
                ctx.place(iid, "buy", "limit", qty=buy_q, price=dn, tag="ig:buy one step down")

    def on_fill(self, ctx, order, fill) -> None:
        if order["tag"] == "ig:initial buy":
            ctx.state["ref"] = fill.price
            return
        ctx.cancel("ig:")
        ctx.state["ref"] = fill.price
        ctx.state["steps"] = int(ctx.state.get("steps", 0)) + 1


class AvellanedaBot(Bot):
    """Hummingbot's Avellaneda-Stoikov market making, approximated on 5-minute candles: every 15
    minutes quote one bid and one ask around a reservation price skewed by inventory (target 50% in
    the coin); the order-book liquidity term is replaced by a fixed 0.35% half-spread; long only."""
    timeframe = "5m"

    def warmup(self) -> int:
        return 205

    def on_candle(self, ctx, iid: str) -> None:
        st = ctx.state
        h = ctx.hist(iid)
        mid = float(h["close"].iloc[-1])
        if "started" not in st:
            st["started"] = True
            ctx.place(iid, "buy", "market", eur=0.5 * ctx.equity(), tag="as:initial inventory")
            return
        end = ctx.ts + TF["5m"]
        if end.minute % 15 != 0:
            return
        ctx.cancel("as:quote")
        if len(h) < 201:
            return
        sigma = float(h["close"].diff().iloc[-200:].std())
        eq = ctx.equity()
        total_base = eq / mid
        q = (ctx.qty(iid) - 0.5 * total_base) / total_base
        gamma = float(self.p.get("gamma", 1.0))
        r = mid - q * gamma * sigma
        spread = gamma * sigma + 2 * float(self.p.get("half_spread", 0.0035)) * mid
        bid, ask = r - spread / 2, r + spread / 2
        size_eur = float(self.p.get("order_frac", 0.10)) * eq
        if ctx.cash_eur() >= size_eur and bid > 0:
            ctx.place(iid, "buy", "limit", qty=size_eur / bid, price=min(bid, mid * 0.9995), tag="as:quote bid")
        sell_q = min(ctx.qty(iid), size_eur / ask)
        if sell_q * ask >= 5.0:
            ctx.place(iid, "sell", "limit", qty=sell_q, price=max(ask, mid * 1.0005), tag="as:quote ask")
        st["last_quote"] = {"bid": round(bid, 4), "ask": round(ask, 4), "q": round(q, 3)}


class TPSBot(EtfDailyBot):
    """Connors (2009) TPS: buy 10% of the planned position on the signal; add 20%, 30%, 40% on each
    later close below the last entry; sell everything when RSI(2) closes above 70."""

    def entry_eur(self, ctx, iid) -> float:
        plan = ctx.equity() * 0.99
        ctx.state["tps"] = {"plan": plan, "tranche": 1, "last": None}
        return plan * 0.10

    def after_close(self, ctx, iid, r, close) -> None:
        t = ctx.state.get("tps")
        if not t or ctx.qty(iid) <= 0:
            return
        if ctx.orders("sbx:exit", iid):
            return
        last = t.get("last") or ctx.state.get("pos", {}).get(iid, {}).get("entry_px")
        if t["tranche"] < 4 and last and close < last:
            frac = [0.10, 0.20, 0.30, 0.40][t["tranche"]]
            ctx.place(iid, "buy", "market", eur=min(t["plan"] * frac, ctx.cash_eur() * 0.995), tag="sbe:tps add")
            t["tranche"] += 1
            t["last"] = close


class PreFomcBot(UsEtfBot):
    """Lucca & Moench (2015) pre-FOMC announcement drift: buy SPY at 15:55 the day before a
    scheduled FOMC decision, sell at 13:55 on decision day (the announcement is at 14:00 ET)."""

    def on_bar(self, ctx, bars: dict) -> None:
        sess = self._session(ctx.ts)
        if sess is None:
            return
        iid = self.instruments()[0]
        day = sess["open"].tz_convert("America/New_York").normalize().tz_localize(None)
        dates = [pd.Timestamp(d) for d in self.p.get("decision_days", [])]
        idx = self.sessions.index
        nxt = idx[idx > day]
        tf = TF["5m"]
        ny_end = (ctx.ts + tf).tz_convert("America/New_York")
        if day in dates and ny_end.hour == 13 and ny_end.minute == 55 and ctx.qty(iid) > 0:
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="fomc:sell before the 14:00 decision")
        if len(nxt) and nxt[0] in dates and ctx.ts + 2 * tf == sess["close"] and ctx.qty(iid) <= 0 \
                and not ctx.orders("fomc:", iid):
            ctx.place(iid, "buy", "market", eur=ctx.equity() * 0.99, tag="fomc:buy the day before")
