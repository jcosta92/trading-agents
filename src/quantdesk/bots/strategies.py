"""The bots. Each one reacts to closed candles and fills; the engine does the order matching and accounting."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import ta
from .data import TF


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
        n_up = self.p.get("trail_up_levels")
        if n_up and close > lv[-1] * (lv[1] / lv[0]) ** int(n_up):
            ctx.note(f"price {close:,.2f} is {n_up} levels above the grid: trailing the grid up")
            ctx.cancel("grid:")
            if ctx.qty(iid) > 0:
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="grid:trail-up sell")
            st.update({"levels": None, "out_since": None, "resetting": True, "trails": st.get("trails", 0) + 1})
            return
        outside = close > lv[-1] or close < lv[0]
        if not outside:
            st["out_since"] = None
            return
        st["out_since"] = st.get("out_since") or ctx.ts.isoformat()
        hours = (ctx.ts - pd.Timestamp(st["out_since"])).total_seconds() / 3600
        if self.p.get("reset_hours") is not None and hours >= float(self.p["reset_hours"]):
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


# ============================================================================================ volatility breakout
class VolBreakoutBot(Bot):
    """Larry Williams volatility breakout as run by the Korean Upbit bot community (pyupbit tutorials):
    the trading day runs 00:00-24:00 UTC (09:00-09:00 KST, Upbit's daily candle). Buy when the price
    breaks today's open + k x yesterday's range; sell everything at the next day's open."""
    timeframe = "15m"

    def warmup(self) -> int:
        return 2 * 96 + 4

    def on_candle(self, ctx, iid: str) -> None:
        tf = TF[self.timeframe]
        end = ctx.ts + tf
        if not (end.hour == 0 and end.minute == 0):
            return                                        # act only on the last candle of the UTC day
        ctx.cancel("vb:", iid)                            # an unfilled breakout order expires with its day
        if ctx.qty(iid) > 0:
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="vb:sell at the day's end (09:00 KST)")
        h = ctx.hist(iid)
        day = h[h.index >= end - pd.Timedelta(days=1)]
        if len(day) < 0.9 * (pd.Timedelta(days=1) / tf):
            return                                        # incomplete day of data: no signal
        rng = float(day["high"].max() - day["low"].min())
        next_open = float(day["close"].iloc[-1])          # 24h market: the next day opens at this close
        target = next_open + float(self.p["k"]) * rng
        n = len(self.instruments())
        ctx.place(iid, "buy", "stop", eur=ctx.equity() / n * 0.99, price=target,
                  tag=f"vb:buy breakout {end:%Y-%m-%d}")
        ctx.state.setdefault("targets", {})[iid] = {"day": f"{end:%Y-%m-%d}", "open": round(next_open, 6),
                                                    "range": round(rng, 6), "target": round(target, 6)}


# ============================================================================================ time of day
class TimeOfDayBot(Bot):
    """Quantpedia's Bitcoin intraday seasonality: hold BTC only from 21:00 to 23:00 UTC every day."""
    timeframe = "1h"

    def warmup(self) -> int:
        return 2

    def on_candle(self, ctx, iid: str) -> None:
        end_hour = (ctx.ts + TF[self.timeframe]).hour     # the next candle opens at this hour
        if end_hour == int(self.p["entry_hour"]) and ctx.qty(iid) <= 0 and not ctx.orders("tod:"):
            ctx.place(iid, "buy", "market", eur=ctx.cash_eur() * 0.995, tag="tod:buy at 21:00 UTC")
        elif end_hour == int(self.p["exit_hour"]) and ctx.qty(iid) > 0:
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="tod:sell at 23:00 UTC")


# ============================================================================================ rebalancing
class RebalanceBot(Bot):
    """Pionex / Shrimpy-style rebalancing bot: equal weights, rebalanced when a coin drifts past a band."""
    timeframe = "15m"

    def warmup(self) -> int:
        return 2

    def on_bar(self, ctx, bars: dict) -> None:
        ins = self.instruments()
        if ctx.orders("rb:") or any(i not in bars for i in ins):
            return
        px = {i: float(bars[i]["close"]) for i in ins}
        eq = ctx.equity()
        tgt = (1 - float(self.p["cash_buffer"])) / len(ins)
        val = {i: ctx.qty(i) * px[i] for i in ins}
        drift = max(abs(val[i] / eq / tgt - 1) for i in ins)        # relative to the target weight
        first = not ctx.state.get("started")
        if not first and drift <= float(self.p["band"]):
            return
        proceeds = 0.0
        for i in ins:
            over = val[i] - tgt * eq
            if over > 5.0:
                ctx.place(i, "sell", "market", qty=over / px[i], tag="rb:sell overweight")
                proceeds += over
        budget = (ctx.cash_eur() + proceeds * (1 - 0.004)) * 0.998
        need = {i: tgt * eq - val[i] for i in ins if tgt * eq - val[i] > 5.0}
        scale = min(1.0, budget / sum(need.values())) if need else 0.0
        for i, eur in need.items():
            ctx.place(i, "buy", "market", eur=eur * scale, tag="rb:buy underweight" if not first else "rb:initial buy")
        ctx.state["started"] = True
        ctx.state["rebalances"] = int(ctx.state.get("rebalances", 0)) + (0 if first else 1)
        if not first:
            ctx.note(f"rebalance: a coin's weight was {drift:.1%} away from its {tgt:.2%} target")


# ============================================================================================ momentum rotation
class MomentumRotationBot(Bot):
    """Pionex 'Momentum Rebalancing': at each interval rank the coins by their return over that interval
    and move all funds into the best one."""
    timeframe = "1h"

    def warmup(self) -> int:
        return int(self.p["interval_hours"]) + 3

    def on_bar(self, ctx, bars: dict) -> None:
        hrs = int(self.p["interval_hours"])
        end = ctx.ts + TF[self.timeframe]
        on_schedule = (end.hour == 0) if hrs >= 24 else (end.hour % hrs == 0)
        if not on_schedule:
            return
        if ctx.orders("mr:"):
            return
        perf = {}
        for i in self.instruments():
            h = ctx.hist(i)
            if i in bars and len(h) > hrs:
                perf[i] = float(h["close"].iloc[-1] / h["close"].iloc[-1 - hrs] - 1)
        if len(perf) < len(self.instruments()) / 2:
            return
        best = max(perf, key=perf.get)
        held = [i for i, p in ctx.ledger.positions.items() if p.qty > 0]
        if held == [best]:
            return
        for i in held:
            if i != best:
                ctx.place(i, "sell", "market", qty=ctx.qty(i), tag="mr:rotate out")
        ctx.place(best, "buy", "market", eur=ctx.equity() * 0.99, tag=f"mr:rotate into ({perf[best]:+.1%} in {hrs}h)")
        ctx.state["holding"] = best
        ctx.state["switches"] = int(ctx.state.get("switches", 0)) + 1
        ctx.note(f"best coin over {hrs}h: {best} {perf[best]:+.2%}")


# ============================================================================================ IBS on US ETFs
class UsEtfBot(Bot):
    """Base for bots that trade US ETFs on 5-minute candles. The runner sets `sessions` (NYSE open/close
    times in UTC, indexed by the New York date) and, if `needs_daily`, `daily` (daily candles).
    `on_close` is called on the candle that ends 5 minutes before the close, so a market order placed
    there fills at the open of the session's last 5-minute candle (15:55 New York time). If that candle
    is missing from the data, the day is skipped rather than traded after the close."""
    timeframe = "5m"
    needs_daily = False

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.sessions = None
        self.daily: dict[str, pd.DataFrame] = {}

    def warmup(self) -> int:
        return 0

    def data_tickers(self) -> list[str]:
        """Tickers whose candles are needed (traded ones plus any signal ticker)."""
        sig = self.p.get("signal")
        return list(dict.fromkeys(self.instruments() + ([sig] if sig else [])))

    def _session(self, ts: pd.Timestamp):
        if self.sessions is None:
            return None
        day = ts.tz_convert("America/New_York").normalize().tz_localize(None)
        return self.sessions.loc[day] if day in self.sessions.index else None

    def on_bar(self, ctx, bars: dict) -> None:
        sess = self._session(ctx.ts)
        if sess is None:
            return
        tf = TF[self.timeframe]
        day = sess["open"].strftime("%Y-%m-%d")
        if ctx.ts + tf == sess["close"]:
            self.on_last_candle(ctx, bars, sess, day)
        if ctx.ts + 2 * tf != sess["close"] or ctx.state.get("done_day") == day:
            return
        ctx.state["done_day"] = day
        self.on_close(ctx, bars, sess, day)

    def on_close(self, ctx, bars: dict, sess, day: str) -> None: ...
    def on_last_candle(self, ctx, bars: dict, sess, day: str) -> None: ...

    def day_range(self, ctx, iid: str, sess) -> pd.DataFrame:
        h = ctx.hist(iid)
        return h[h.index >= sess["open"]]

    def trading_day_position(self, day: str) -> tuple[int, int]:
        """(n-th trading day of its month, counted from 1; trading days left in the month after it)."""
        d = pd.Timestamp(day)
        idx = self.sessions.index
        month = idx[(idx.year == d.year) & (idx.month == d.month)]
        return int((month <= d).sum()), int((month > d).sum())


class IBSBot(UsEtfBot):
    """Internal Bar Strength mean reversion on US index ETFs: buy into a close near the day's low,
    sell into a close near the day's high. Orders go in 5 minutes before the close."""

    def on_close(self, ctx, bars: dict, sess, day: str) -> None:
        n = len(self.instruments())
        ibs_all = {}
        for iid in self.instruments():
            d = self.day_range(ctx, iid, sess)
            if d.empty:
                continue
            hi, lo, c = float(d["high"].max()), float(d["low"].min()), float(d["close"].iloc[-1])
            if hi <= lo:
                continue
            ibs = (c - lo) / (hi - lo)
            ibs_all[iid] = round(ibs, 3)
            if ctx.qty(iid) > 0 and ibs >= float(self.p["exit"]):
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"ibs:sell (IBS {ibs:.2f})")
            elif ctx.qty(iid) <= 0 and ibs < float(self.p["entry"]) and not ctx.orders("ibs:", iid):
                ctx.place(iid, "buy", "market", eur=ctx.equity() / n * 0.99, tag=f"ibs:buy (IBS {ibs:.2f})")
        ctx.state["last_ibs"] = {"day": day, **ibs_all}


class OvernightBot(UsEtfBot):
    """Overnight anomaly: own the ETF from the close to the next open."""

    def on_close(self, ctx, bars: dict, sess, day: str) -> None:
        n = len(self.instruments())
        for iid in self.instruments():
            if ctx.qty(iid) <= 0 and iid in bars:
                ctx.place(iid, "buy", "market", eur=ctx.equity() / n * 0.99, tag="on:buy at the close")

    def on_last_candle(self, ctx, bars: dict, sess, day: str) -> None:
        for iid in self.instruments():
            if ctx.qty(iid) > 0 and not ctx.orders("on:sell", iid):
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="on:sell at the next open")


class TurnOfMonthBot(UsEtfBot):
    """Turn-of-the-month effect: hold the index ETF over the last trading days of the month and the
    first days of the next one."""

    def on_close(self, ctx, bars: dict, sess, day: str) -> None:
        nth, left = self.trading_day_position(day)
        ctx.state["day_position"] = {"day": day, "nth": nth, "left_in_month": left}
        for iid in self.instruments():
            if left == int(self.p["days_before_end"]) and ctx.qty(iid) <= 0 and iid in bars:
                ctx.place(iid, "buy", "market", eur=ctx.equity() / len(self.instruments()) * 0.99,
                          tag=f"tom:buy ({left} trading days left in the month)")
            elif nth == int(self.p["exit_day"]) and ctx.qty(iid) > 0:
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"tom:sell (trading day {nth} of the month)")


class LeverageRotationBot(UsEtfBot):
    """Gayed & Bilello 'Leverage for the Long Run': hold a leveraged S&P 500 ETF while the S&P 500 is
    above its 200-day average, cash otherwise. Decided on the day's price 5 minutes before the close."""
    needs_daily = True

    def on_close(self, ctx, bars: dict, sess, day: str) -> None:
        sig, iid = self.p["signal"], self.p["instrument"]
        d = self.daily.get(sig)
        if d is None or sig not in bars or iid not in bars:
            return
        prior = d[d.index < pd.Timestamp(day)]["close"]
        n = int(self.p["sma_days"])
        if len(prior) < n - 1:
            return
        price = float(bars[sig]["close"])
        sma = (prior.iloc[-(n - 1):].sum() + price) / n
        above = price > sma
        ctx.state["signal"] = {"day": day, "spy": round(price, 2), "sma200": round(sma, 2), "above": bool(above)}
        if above and ctx.qty(iid) <= 0:
            ctx.place(iid, "buy", "market", eur=ctx.equity() * 0.99, tag=f"lr:buy ({sig} {price:.2f} > SMA {sma:.2f})")
        elif not above and ctx.qty(iid) > 0:
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"lr:sell ({sig} {price:.2f} < SMA {sma:.2f})")


# ============================================================================================ classic crypto
class TurtleBot(Bot):
    """Turtle Traders System 1 on 4-hour candles: buy a 20-candle high breakout, exit on a 10-candle low,
    stop 2N (N = 20-candle ATR) below the entry."""
    timeframe = "4h"

    def warmup(self) -> int:
        return int(self.p["entry_n"]) + 30

    def on_candle(self, ctx, iid: str) -> None:
        p = self.p
        h = ctx.hist(iid)
        if len(h) < int(p["entry_n"]) + 2:
            return
        if ctx.qty(iid) <= 0:
            if ctx.orders("tt:exit", iid):
                ctx.cancel("tt:exit", iid)
            hi = float(h["high"].iloc[-int(p["entry_n"]):].max())
            n_atr = float(ta.atr(h, int(p["atr_n"])).iloc[-1])
            ctx.cancel("tt:entry", iid)
            ctx.place(iid, "buy", "stop", eur=self.slot_eur_total(ctx), price=hi * 1.0001,
                      tag=f"tt:entry (20-candle high {hi:,.4g})")
            ctx.state.setdefault("n", {})[iid] = n_atr
            return
        # in a position: the exit is the higher of the 10-candle low and the 2N stop
        lo = float(h["low"].iloc[-int(p["exit_n"]):].min())
        entry = ctx.state.get("entry", {}).get(iid, ctx.avg_price(iid))
        stop = max(lo, entry - float(p["stop_n"]) * ctx.state.get("n", {}).get(iid, 0.0))
        ctx.cancel("tt:exit", iid)
        ctx.place(iid, "sell", "stop", qty=ctx.qty(iid), price=stop * 0.9999, tag=f"tt:exit (stop {stop:,.4g})")

    def slot_eur_total(self, ctx) -> float:
        return ctx.equity() / len(self.instruments()) * 0.99

    def on_fill(self, ctx, order, fill) -> None:
        if order["tag"].startswith("tt:entry"):
            ctx.state.setdefault("entry", {})[order["inst"]] = fill.price


class SqueezeBot(Bot):
    """LazyBear Squeeze Momentum: buy when a squeeze fires with rising positive momentum,
    sell when momentum starts to fade."""
    timeframe = "1h"

    def warmup(self) -> int:
        return 80

    def on_candle(self, ctx, iid: str) -> None:
        h = ctx.hist(iid)
        if len(h) < 45:
            return
        on, val = ta.squeeze_momentum(h)
        v0, v1 = float(val.iloc[-1]), float(val.iloc[-2])
        if np.isnan(v0) or np.isnan(v1):
            return
        fired = bool(on.iloc[-2]) and not bool(on.iloc[-1])
        if ctx.qty(iid) <= 0 and not ctx.orders("sq:", iid) and fired and v0 > 0 and v0 > v1:
            ctx.place(iid, "buy", "market", eur=self.slot_eur(ctx, len(self.instruments())), tag="sq:buy (squeeze fired up)")
        elif ctx.qty(iid) > 0 and v0 < v1:
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="sq:sell (momentum fading)")


class RSIBot(Bot):
    """Classic RSI(14) bot: buy oversold (< 30), sell overbought (> 70)."""
    timeframe = "1h"

    def warmup(self) -> int:
        return 100

    def on_candle(self, ctx, iid: str) -> None:
        h = ctx.hist(iid)
        if len(h) < 30:
            return
        r = float(ta.rsi(h["close"], int(self.p["period"])).iloc[-1])
        if ctx.qty(iid) <= 0 and not ctx.orders("rsi:", iid) and r < float(self.p["buy_below"]):
            ctx.place(iid, "buy", "market", eur=self.slot_eur(ctx, len(self.instruments())), tag=f"rsi:buy (RSI {r:.1f})")
        elif ctx.qty(iid) > 0 and r > float(self.p["sell_above"]):
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"rsi:sell (RSI {r:.1f})")


CLASSES = {c.__name__: c for c in (GridBot, DCABot, BinHV45Bot, SupertrendBot, EmaMacdBot, ORBBot,
                                   VolBreakoutBot, TimeOfDayBot, RebalanceBot, MomentumRotationBot, IBSBot,
                                   OvernightBot, TurnOfMonthBot, LeverageRotationBot, TurtleBot, SqueezeBot, RSIBot)}

def all_classes() -> dict:
    """Bot classes by name, including the generic family bots (imported lazily: they import this module)."""
    from . import generic
    from .families import custom  # noqa: F401  (also registers every signal / weight function)
    out = dict(CLASSES)
    for mod in (generic, custom):
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, Bot) and obj is not Bot:
                out.setdefault(name, obj)
    return out


def build_bot(bot_id: str, spec: dict) -> Bot:
    cls = CLASSES.get(spec["class"]) or all_classes()[spec["class"]]
    return cls(bot_id, spec)
