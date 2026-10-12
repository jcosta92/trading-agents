"""The bots. Each one reacts to closed candles and fills; the engine does the order matching and accounting."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import ta


class Bot:
    timeframe = "1h"

    def __init__(self, bot_id: str, spec: dict):
        self.id = bot_id
        self.spec = spec
        self.name = spec.get("name", bot_id)
        self.p = spec.get("params", {}) or {}
        self.timeframe = spec.get("timeframe", self.timeframe)

    def instruments(self) -> list[str]:
        if "instrument" in self.p:
            return [self.p["instrument"]]
        return list(self.p.get("instruments", []))

    def warmup(self) -> int:
        """Candles of history needed before the first decision."""
        return 50

    def on_candle(self, ctx, iid: str) -> None: ...
    def on_bar(self, ctx, bars: dict) -> None: ...
    def on_fill(self, ctx, order: dict, fill) -> None: ...

    @staticmethod
    def slot_eur(ctx, n: int) -> float:
        """Equal share of equity per slot, capped by cash."""
        return min(ctx.equity() / n, ctx.cash_eur()) * 0.995


# ============================================================================================ grid
class GridBot(Bot):
    """Spot grid: resting buys below the price, each filled buy re-sold one level higher."""
    timeframe = "15m"

    def warmup(self) -> int:
        return 2

    def _levels(self, center: float) -> list[float]:
        n, r = int(self.p["levels"]), float(self.p["range_pct"])
        lo, hi = center * (1 - r), center * (1 + r)
        g = (hi / lo) ** (1.0 / n)
        return [lo * g ** k for k in range(n + 1)]

    def on_candle(self, ctx, iid: str) -> None:
        st = ctx.state
        close = float(ctx.candle(iid)["close"])
        if st.get("resetting"):
            if ctx.qty(iid) > 0 or ctx.orders("grid:reset"):
                return                                    # wait for the reset sale to fill
            st["resetting"] = False
        if not st.get("levels"):
            self._build(ctx, iid, close)
            return
        lv = st["levels"]
        outside = close > lv[-1] or close < lv[0]
        if not outside:
            st["out_since"] = None
            return
        st["out_since"] = st.get("out_since") or ctx.ts.isoformat()
        hours = (ctx.ts - pd.Timestamp(st["out_since"])).total_seconds() / 3600
        if hours >= float(self.p["reset_hours"]):
            ctx.note(f"price {close:,.0f} outside the grid {lv[0]:,.0f}-{lv[-1]:,.0f} for {hours:.0f}h: reset")
            ctx.cancel("grid:")
            if ctx.qty(iid) > 0:
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="grid:reset sell")
            st.update({"levels": None, "out_since": None, "resetting": True, "resets": st.get("resets", 0) + 1})

    def _build(self, ctx, iid: str, price: float) -> None:
        st = ctx.state
        lv = self._levels(price)
        per = float(self.p["invest"]) * ctx.equity() / (len(lv) - 1)
        st.update({"levels": lv, "per_level_eur": per, "center": price, "out_since": None})
        # price sits between level i and i+1: buys at 0..i, sells at i+2..n and level i+1 left empty, so
        # that every level holds at most one order (a filled buy re-sells one level up into the gap)
        i = max([k for k in range(len(lv)) if lv[k] <= price], default=-1)
        below = list(range(0, i + 1))
        above = list(range(i + 2, len(lv)))
        # coins for the sell orders above the price, bought now at market
        coins = sum(per / lv[k - 1] for k in above)
        if coins > 0:
            ctx.place(iid, "buy", "market", qty=coins, tag="grid:initial buy")
        for k in above:
            ctx.place(iid, "sell", "limit", qty=per / lv[k - 1], price=lv[k], tag=f"grid:sell:{k}")
        for k in below:
            ctx.place(iid, "buy", "limit", qty=per / lv[k], price=lv[k], tag=f"grid:buy:{k}")
        ctx.note(f"grid built around {price:,.2f}: {lv[0]:,.2f}-{lv[-1]:,.2f}, {len(lv) - 1} levels, "
                 f"EUR {per:,.0f} per level")

    def on_fill(self, ctx, order, fill) -> None:
        parts = order["tag"].split(":")
        if len(parts) != 3 or not ctx.state.get("levels"):
            return
        lv, k = ctx.state["levels"], int(parts[2])
        per = ctx.state["per_level_eur"]
        iid = order["inst"]
        if parts[1] == "buy" and k + 1 < len(lv) and not ctx.orders(f"grid:sell:{k + 1}"):
            ctx.place(iid, "sell", "limit", qty=fill.qty, price=lv[k + 1], tag=f"grid:sell:{k + 1}")
        elif parts[1] == "sell" and k - 1 >= 0 and not ctx.orders(f"grid:buy:{k - 1}"):
            ctx.place(iid, "buy", "limit", qty=per / lv[k - 1], price=lv[k - 1], tag=f"grid:buy:{k - 1}")


# ============================================================================================ DCA
class DCABot(Bot):
    """Base order + safety orders at growing distances and sizes; one take-profit above the average."""
    timeframe = "15m"

    def warmup(self) -> int:
        return 2

    def on_candle(self, ctx, iid: str) -> None:
        st = ctx.state
        if st.get("deal") is not None and ctx.qty(iid) <= 0 and not ctx.orders("dca:"):
            st["deal"] = None                             # the base order could not be filled: start over
        if st.get("deal") is None and not ctx.orders("dca:"):
            ctx.place(iid, "buy", "market", eur=float(self.p["base_eur"]), tag="dca:base")
            st["deal"] = {"opened": ctx.ts.isoformat(), "safety_filled": 0}

    def on_fill(self, ctx, order, fill) -> None:
        st, p, iid = ctx.state, self.p, order["inst"]
        tag = order["tag"]
        if tag == "dca:base":
            dev, step, eur = float(p["deviation"]), float(p["deviation"]), float(p["so_eur"])
            for i in range(int(p["max_so"])):
                price = fill.price * (1 - dev)
                ctx.place(iid, "buy", "limit", qty=eur / price, price=price, tag=f"dca:so:{i + 1}")
                step *= float(p["step_scale"])
                dev += step
                eur *= float(p["volume_scale"])
            self._take_profit(ctx, iid)
        elif tag.startswith("dca:so"):
            st["deal"]["safety_filled"] = int(st["deal"].get("safety_filled", 0)) + 1
            self._take_profit(ctx, iid)
        elif tag == "dca:tp":
            ctx.cancel("dca:so")
            deals = st.setdefault("completed_deals", 0)
            st["completed_deals"] = deals + 1
            st["deal"] = None

    def _take_profit(self, ctx, iid: str) -> None:
        ctx.cancel("dca:tp")
        avg = ctx.avg_price(iid)
        if avg:
            ctx.place(iid, "sell", "limit", qty=ctx.qty(iid), price=avg * (1 + float(self.p["take_profit"])),
                      tag="dca:tp")


# ============================================================================================ BinHV45
class BinHV45Bot(Bot):
    """Freqtrade BinHV45: buy a sharp close below the lower Bollinger band; +1.25% ROI, -5% stop."""
    timeframe = "15m"

    def warmup(self) -> int:
        return int(self.p.get("bb_window", 40)) + 5

    def signal(self, df: pd.DataFrame) -> bool:
        p = self.p
        if len(df) < int(p["bb_window"]) + 2:
            return False
        lower, mid, _ = ta.bollinger(df["close"], int(p["bb_window"]), float(p["bb_std"]))
        bbdelta = (mid - lower).abs()
        closedelta = (df["close"] - df["close"].shift()).abs()
        tail = (df["close"] - df["low"]).abs()
        c = df["close"].iloc[-1]
        prev_lower = lower.iloc[-2]
        return bool(prev_lower > 0
                    and bbdelta.iloc[-1] > c * float(p["bbdelta"]) / 1000
                    and closedelta.iloc[-1] > c * float(p["closedelta"]) / 1000
                    and tail.iloc[-1] < bbdelta.iloc[-1] * float(p["tail"]) / 1000
                    and c < prev_lower
                    and c <= df["close"].iloc[-2])

    def on_candle(self, ctx, iid: str) -> None:
        if ctx.qty(iid) > 0 or ctx.orders("bb:", iid):
            return
        slots = int(self.p.get("max_open_trades", 3))
        busy = {o["inst"] for o in ctx.orders("bb:")} | {i for i, p in ctx.ledger.positions.items() if p.qty > 0}
        if len(busy) >= slots:
            return
        if self.signal(ctx.hist(iid)):
            ctx.place(iid, "buy", "market", eur=self.slot_eur(ctx, slots), tag="bb:entry")
            ctx.note(f"{iid}: BinHV45 buy signal at {float(ctx.candle(iid)['close']):,.2f}")

    def on_fill(self, ctx, order, fill) -> None:
        iid = order["inst"]
        if order["tag"] == "bb:entry":
            ctx.place(iid, "sell", "limit", qty=fill.qty, price=fill.price * (1 + float(self.p["roi"])), tag="bb:tp")
            ctx.place(iid, "sell", "stop", qty=fill.qty, price=fill.price * (1 - float(self.p["stoploss"])), tag="bb:sl")
        elif order["tag"] in ("bb:tp", "bb:sl"):
            ctx.cancel("bb:", iid)


# ============================================================================================ Supertrend
class SupertrendBot(Bot):
    timeframe = "1h"

    def warmup(self) -> int:
        return 150

    def on_candle(self, ctx, iid: str) -> None:
        d = ta.supertrend(ctx.hist(iid), int(self.p["atr_period"]), float(self.p["multiplier"]))
        if len(d) < 3 or np.isnan(d.iloc[-2]):
            return
        if d.iloc[-1] == 1 and d.iloc[-2] == -1 and ctx.qty(iid) <= 0 and not ctx.orders("st:", iid):
            ctx.place(iid, "buy", "market", eur=self.slot_eur(ctx, len(self.instruments())), tag="st:buy (flip up)")
        elif d.iloc[-1] == -1 and d.iloc[-2] == 1 and ctx.qty(iid) > 0:
            ctx.cancel("st:", iid)
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="st:sell (flip down)")


# ============================================================================================ EMA + MACD
class EmaMacdBot(Bot):
    timeframe = "1h"

    def warmup(self) -> int:
        return int(self.p.get("trend", 200)) * 3

    def on_candle(self, ctx, iid: str) -> None:
        p = self.p
        df = ctx.hist(iid)
        if len(df) < int(p["trend"]) + 2:
            return
        c = df["close"]
        f, s, t = ta.ema(c, int(p["fast"])), ta.ema(c, int(p["slow"])), ta.ema(c, int(p["trend"]))
        line, sig = ta.macd(c)
        up = f.iloc[-1] > s.iloc[-1] and f.iloc[-2] <= s.iloc[-2]
        down = f.iloc[-1] < s.iloc[-1] and f.iloc[-2] >= s.iloc[-2]
        if up and line.iloc[-1] > sig.iloc[-1] and c.iloc[-1] > t.iloc[-1] and ctx.qty(iid) <= 0 \
                and not ctx.orders("em:", iid):
            a = float(ta.atr(df, int(p["atr_period"])).iloc[-1])
            ctx.state.setdefault("stop_dist", {})[iid] = float(p["stop_atr"]) * a
            ctx.place(iid, "buy", "market", eur=self.slot_eur(ctx, len(self.instruments())), tag="em:buy (cross up)")
        elif down and ctx.qty(iid) > 0:
            ctx.cancel("em:", iid)
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="em:sell (cross down)")

    def on_fill(self, ctx, order, fill) -> None:
        iid = order["inst"]
        if order["tag"].startswith("em:buy"):
            dist = ctx.state.get("stop_dist", {}).get(iid, fill.price * 0.03)
            ctx.place(iid, "sell", "stop", qty=fill.qty, price=fill.price - dist, tag="em:stop")
        elif order["tag"] in ("em:stop",) or order["tag"].startswith("em:sell"):
            ctx.cancel("em:", iid)


# ============================================================================================ ORB
class ORBBot(Bot):
    """5-minute opening range breakout on the day's most active large caps, long only, out by the close."""
    timeframe = "5m"

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.universe: list[str] = []
        self.daily: dict[str, pd.DataFrame] = {}
        self.sessions = None                      # DataFrame of NYSE sessions (open/close UTC)

    def instruments(self) -> list[str]:
        return list(self.universe)

    def warmup(self) -> int:
        return 0

    def _session(self, ts: pd.Timestamp):
        if self.sessions is None:
            return None
        day = ts.tz_convert("America/New_York").normalize().tz_localize(None)
        if day not in self.sessions.index:
            return None
        return self.sessions.loc[day]

    def on_bar(self, ctx, bars: dict) -> None:
        sess = self._session(ctx.ts)
        if sess is None:
            return
        open_, close_ = sess["open"], sess["close"]
        tf = pd.Timedelta(minutes=5)
        day = open_.strftime("%Y-%m-%d")
        if ctx.ts == open_:
            self._select(ctx, bars, day)
        # the candle before the last one: sell at the last candle's open (about 5 minutes before the close)
        if ctx.ts + 2 * tf >= close_ and not ctx.state.get("closed_day") == day:
            ctx.cancel("orb:")
            for iid in list(ctx.ledger.positions):
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="orb:exit at the close")
            ctx.state["closed_day"] = day

    def _select(self, ctx, bars: dict, day: str) -> None:
        p = self.p
        n_days = int(p["rel_volume_days"])
        cands = []
        for iid, bar in bars.items():
            if not (bar["close"] > bar["open"] and bar["close"] >= float(p["min_price"])):
                continue
            h = ctx.hist(iid)
            prior = h[h.index < ctx.ts]
            if prior.empty:
                continue
            ny = prior.index.tz_convert("America/New_York")
            first = prior[(ny.hour == 9) & (ny.minute == 30)]["volume"].iloc[-n_days:]
            if len(first) < max(5, n_days // 2) or first.mean() <= 0:
                continue
            rv = float(bar["volume"]) / float(first.mean())
            if rv > 1.0:
                cands.append((rv, iid))
        picks = [iid for _, iid in sorted(cands, reverse=True)[: int(p["n_names"])]]
        eq = ctx.equity()
        fx = ctx.engine.fx
        for iid in picks:
            hi = float(bars[iid]["high"])
            atr = self._atr(iid, ctx.ts)
            if not atr:
                continue
            eur = eq / int(p["n_names"]) * 0.99
            qty = fx.from_eur(eur, "USD") / hi
            ctx.state.setdefault("stop_dist", {})[iid] = float(p["stop_atr_frac"]) * atr
            ctx.place(iid, "buy", "stop", qty=qty, price=hi, tag=f"orb:entry {day}")
        ctx.state["last_selection"] = {"day": day, "picks": picks,
                                       "rel_volume": {iid: round(rv, 2) for rv, iid in sorted(cands, reverse=True)[:10]}}
        ctx.note(f"ORB {day}: watching {', '.join(picks) or 'nothing'}")

    def _atr(self, iid: str, ts: pd.Timestamp) -> float | None:
        d = self.daily.get(iid)
        if d is None or d.empty:
            return None
        day = ts.tz_convert("America/New_York").normalize().tz_localize(None)
        d = d[d.index < day]
        if len(d) < int(self.p["atr_days"]) + 1:
            return None
        return float(ta.atr(d, int(self.p["atr_days"])).iloc[-1])

    def on_fill(self, ctx, order, fill) -> None:
        iid = order["inst"]
        if order["tag"].startswith("orb:entry"):
            dist = ctx.state.get("stop_dist", {}).get(iid)
            if dist:
                ctx.place(iid, "sell", "stop", qty=fill.qty, price=fill.price - dist, tag="orb:stop")
        elif order["tag"] == "orb:stop":
            ctx.cancel("orb:", iid)


CLASSES = {c.__name__: c for c in (GridBot, DCABot, BinHV45Bot, SupertrendBot, EmaMacdBot, ORBBot)}


def build_bot(bot_id: str, spec: dict) -> Bot:
    return CLASSES[spec["class"]](bot_id, spec)
