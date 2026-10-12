"""Candle-replay trading engine for the bots.

Every cycle the engine fetches the candles that closed since the last cycle and walks through them
in time order. For each candle it
  1. fills the bot's resting orders that this candle traded through
       market  -> at the candle's open, plus half the spread and slippage, taker fee
       stop    -> at the stop level (or the open if it gapped through), plus spread/slippage, taker fee
       limit   -> at the limit price (or the open if it gapped through), maker fee
     Orders placed on a candle only become active from the next candle; when a candle reaches
     both a stop and a take-profit, the stop is assumed to have filled first. Within a candle all
     sell orders (of every market) are handled before buy orders, so a switch from one coin to
     another can be paid with the proceeds of the sale, as on a real exchange a moment later.
  2. lets the bot look at the candle (and all earlier ones) and place or cancel orders.
Because only closed candles are used and they are processed strictly in order, nothing depends
on when the cycle runs: a cycle delayed by an hour produces exactly the same trades.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import Instrument
from ..costs import CostModel
from ..ledger import FxRates, Ledger
from .data import TF

log = logging.getLogger(__name__)
MIN_NOTIONAL_EUR = 5.0
EPS = 1e-12


@dataclass
class Market:
    inst: Instrument
    half_spread: float          # fraction of price
    slippage: float             # fraction of price
    maker: float | None = None  # crypto fee rates; None = use the venue's fee schedule (stocks)
    taker: float | None = None


@dataclass
class Fill:
    ts: str
    bot: str
    inst: str
    side: str
    qty: float
    price: float
    fee_eur: float
    value_eur: float
    liquidity: str
    tag: str
    pnl_eur: float | None = None


class BotContext:
    """What a strategy sees while handling one candle."""

    def __init__(self, engine: "BotEngine", bot, ledger: Ledger, hist: dict[str, pd.DataFrame], ts: pd.Timestamp):
        self.engine, self.bot, self.ledger, self._hist, self.ts = engine, bot, ledger, hist, ts
        self.state = ledger.meta.setdefault("state", {})

    # data
    def hist(self, iid: str) -> pd.DataFrame:
        """Candles of `iid` up to and including the current one."""
        df = self._hist.get(iid)
        if df is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return df.loc[:self.ts]

    def candle(self, iid: str):
        df = self._hist.get(iid)
        if df is None or self.ts not in df.index:
            return None
        return df.loc[self.ts]

    # account
    def qty(self, iid: str) -> float:
        return self.ledger.qty(iid)

    def avg_price(self, iid: str) -> float | None:
        p = self.ledger.positions.get(iid)
        return p.avg_price if p else None

    def equity(self) -> float:
        return self.ledger.equity(self.engine.fx)

    def cash_eur(self) -> float:
        return self.ledger.cash_eur(self.engine.fx)

    # orders
    def orders(self, tag_prefix: str = "", iid: str | None = None) -> list[dict]:
        return [o for o in self.ledger.meta.setdefault("orders", [])
                if o["tag"].startswith(tag_prefix) and (iid is None or o["inst"] == iid)]

    def place(self, iid: str, side: str, type_: str, *, qty: float | None = None, eur: float | None = None,
              price: float | None = None, tag: str = "") -> dict:
        assert side in ("buy", "sell") and type_ in ("market", "limit", "stop")
        if type_ != "market" and not (price and price > 0):
            raise ValueError(f"{type_} order needs a positive price")
        meta = self.ledger.meta
        meta["order_seq"] = int(meta.get("order_seq", 0)) + 1
        o = {"id": meta["order_seq"], "inst": iid, "side": side, "type": type_, "price": price,
             "qty": qty, "eur": eur, "tag": tag, "placed": self.ts.isoformat()}
        meta.setdefault("orders", []).append(o)
        return o

    def cancel(self, tag_prefix: str = "", iid: str | None = None) -> int:
        keep, n = [], 0
        for o in self.ledger.meta.setdefault("orders", []):
            if o["tag"].startswith(tag_prefix) and (iid is None or o["inst"] == iid):
                n += 1
            else:
                keep.append(o)
        self.ledger.meta["orders"] = keep
        return n

    def note(self, msg: str) -> None:
        self.engine.notes.setdefault(self.bot.id, []).append(f"{self.ts:%Y-%m-%d %H:%M} {msg}")


class BotEngine:
    def __init__(self, markets: dict[str, Market], cost_model: CostModel, fx: FxRates):
        self.markets = markets
        self.cm = cost_model
        self.fx = fx
        self.fills: list[Fill] = []
        self.notes: dict[str, list[str]] = {}
        self.equity_rows: list[dict] = []

    # ------------------------------------------------------------------ main loop
    def run(self, bot, ledger: Ledger, candles: dict[str, pd.DataFrame], start: pd.Timestamp,
            end: pd.Timestamp) -> None:
        meta = ledger.meta
        if meta.get("finished"):
            return
        last = pd.Timestamp(meta["last_ts"]) if meta.get("last_ts") else None
        stamps = sorted(set().union(*[set(df.index) for df in candles.values()])) if candles else []
        tf = TF[bot.timeframe]
        for ts in stamps:
            if last is not None and ts <= last:
                continue
            if ts < start:
                continue
            bars = {iid: df.loc[ts] for iid, df in candles.items() if ts in df.index}
            if ts >= end:
                self._liquidate(bot, ledger, bars, ts)
                meta["finished"] = ts.isoformat()
                meta["last_ts"] = ts.isoformat()
                break
            ctx = BotContext(self, bot, ledger, candles, ts)
            for side in ("sell", "buy"):              # sales first, so their cash can pay for buys
                for iid, bar in bars.items():
                    self._fill_orders(bot, ledger, ctx, iid, bar, ts, side)
            ledger.update_marks({iid: float(b["close"]) for iid, b in bars.items()})
            for iid in bars:
                bot.on_candle(ctx, iid)
            bot.on_bar(ctx, bars)
            meta["last_ts"] = ts.isoformat()
            close_ts = ts + tf
            if close_ts.minute == 0 or tf >= pd.Timedelta(hours=1):
                self.equity_rows.append({"ts": close_ts.isoformat(), "bot": bot.id,
                                         "equity_eur": round(ledger.equity(self.fx), 4),
                                         "cash_eur": round(ledger.cash_eur(self.fx), 4),
                                         "n_positions": len(ledger.positions)})

    # ------------------------------------------------------------------ fills
    def _fill_orders(self, bot, ledger: Ledger, ctx: BotContext, iid: str, bar, ts: pd.Timestamp,
                     side: str | None = None) -> None:
        orders = [o for o in ledger.meta.get("orders", []) if o["inst"] == iid and pd.Timestamp(o["placed"]) < ts
                  and (side is None or o["side"] == side)]
        rank = {"market": 0, "stop": 1, "limit": 2}
        for o in sorted(orders, key=lambda o: (rank[o["type"]], o["id"])):
            if o not in ledger.meta.get("orders", []):        # cancelled by an earlier fill's callback
                continue
            px, mid, liq = self._trigger(o, bar)
            if px is None:
                continue
            ledger.meta["orders"] = [x for x in ledger.meta["orders"] if x["id"] != o["id"]]
            f = self._execute(bot, ledger, o, px, mid, liq, ts)
            if f is not None:
                bot.on_fill(ctx, o, f)

    def _trigger(self, o: dict, bar) -> tuple[float | None, float | None, str]:
        m = self.markets[o["inst"]]
        op, hi, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
        cost = m.half_spread + m.slippage
        sgn = 1.0 if o["side"] == "buy" else -1.0
        if o["type"] == "market":
            return op * (1 + sgn * cost), op, "taker"
        p = float(o["price"])
        if o["type"] == "limit":
            if o["side"] == "buy":
                if op <= p:
                    return op, op, "maker"
                return (p, p, "maker") if lo < p else (None, None, "")
            if op >= p:
                return op, op, "maker"
            return (p, p, "maker") if hi > p else (None, None, "")
        # stop orders: sell stop below the market, buy stop above it
        if o["side"] == "sell":
            if op <= p:
                return op * (1 - cost), op, "taker"
            return (p * (1 - cost), p, "taker") if lo <= p else (None, None, "")
        if op >= p:
            return op * (1 + cost), op, "taker"
        return (p * (1 + cost), p, "taker") if hi >= p else (None, None, "")

    def _fee(self, m: Market, qty: float, px: float, side: str, liq: str) -> float:
        """Fee in the instrument's currency."""
        if m.maker is not None:
            return abs(qty) * px * (m.maker if liq == "maker" else m.taker)
        return self.cm.fees(m.inst, qty, px, side).total

    def _execute(self, bot, ledger: Ledger, o: dict, px: float, mid: float, liq: str,
                 ts: pd.Timestamp) -> Fill | None:
        m = self.markets[o["inst"]]
        inst, fx = m.inst, self.fx
        if o["side"] == "sell":
            held = ledger.qty(inst.id)
            qty = min(float(o["qty"]) if o.get("qty") else held, held)
            if held - qty <= 1e-6 * held:                 # selling (almost) everything: no dust left behind
                qty = held
            if qty <= EPS:
                return None
            fee = self._fee(m, qty, px, "sell", liq)
            pos = ledger.positions[inst.id]
            # round-trip P&L of the sold units, net of their share of the buy fees and of this sell fee
            buy_fees = ledger.meta.setdefault("open_buy_fees", {})
            share = buy_fees.get(inst.id, 0.0) * qty / held
            buy_fees[inst.id] = buy_fees.get(inst.id, 0.0) - share
            pnl = (px - pos.avg_price) * qty - fee - share
            ledger.trade(ts.isoformat(), inst, -qty, px, {inst.currency: fee}, fx, mid, reason=o["tag"])
            return self._record(bot, inst, "sell", qty, px, fee, liq, o["tag"], ts, fx.to_eur(pnl, inst.currency))
        # buy: size by EUR amount or quantity, capped by the cash available
        want = float(o["qty"]) if o.get("qty") else float(o["eur"]) / fx.to_eur(px, inst.currency)
        fee_rate = (m.taker if liq == "taker" else m.maker) if m.maker is not None else 0.002
        cash_ccy = self._spendable(ledger, inst.currency)
        qty = min(want, cash_ccy / (px * (1 + fee_rate)) * 0.999)
        if fx.to_eur(qty * px, inst.currency) < MIN_NOTIONAL_EUR:
            return None
        fee = self._fee(m, qty, px, "buy", liq)
        if inst.currency != ledger.base:
            need = qty * px + fee - ledger.cash.get(inst.currency, 0.0)
            if need > 0:
                ledger.convert(ts.isoformat(), ledger.base, inst.currency, need, fx,
                               float(self.cm.venue(inst).get("fx_conversion", 0.0003)), reason="FX for stock purchase")
        ledger.trade(ts.isoformat(), inst, qty, px, {inst.currency: fee}, fx, mid, reason=o["tag"])
        buy_fees = ledger.meta.setdefault("open_buy_fees", {})
        buy_fees[inst.id] = buy_fees.get(inst.id, 0.0) + fee
        return self._record(bot, inst, "buy", qty, px, fee, liq, o["tag"], ts, None)

    def _spendable(self, ledger: Ledger, ccy: str) -> float:
        own = max(0.0, ledger.cash.get(ccy, 0.0))
        if ccy == ledger.base:
            return own
        base = max(0.0, ledger.cash.get(ledger.base, 0.0))
        return own + self.fx.from_eur(base, ccy) * (1 - 0.0003)

    def _record(self, bot, inst, side, qty, px, fee, liq, tag, ts, pnl) -> Fill:
        f = Fill(ts.isoformat(), bot.id, inst.id, side, qty, round(px, 8),
                 round(self.fx.to_eur(fee, inst.currency), 6), round(self.fx.to_eur(qty * px, inst.currency), 4),
                 liq, tag, round(pnl, 4) if pnl is not None else None)
        self.fills.append(f)
        return f

    def _liquidate(self, bot, ledger: Ledger, bars: dict, ts: pd.Timestamp) -> None:
        """End of the month: cancel every order and sell all holdings at this candle's open."""
        ledger.meta["orders"] = []
        for iid, pos in list(ledger.positions.items()):
            bar = bars.get(iid)
            if bar is None or pos.qty <= 0:
                continue
            o = {"id": -1, "inst": iid, "side": "sell", "type": "market", "qty": pos.qty, "tag": "end of test: sell all"}
            px, mid, liq = self._trigger(o, bar)
            self._execute(bot, ledger, o, px, mid, liq, ts)
        self.equity_rows.append({"ts": ts.isoformat(), "bot": bot.id, "equity_eur": round(ledger.equity(self.fx), 4),
                                 "cash_eur": round(ledger.cash_eur(self.fx), 4), "n_positions": len(ledger.positions)})


# ---------------------------------------------------------------------------------------- persistence
def load_ledger(path: Path, bot_id: str, capital: float, ts: str) -> Ledger:
    if path.exists():
        led = Ledger.from_dict(json.loads(path.read_text()))
        ok, msg = led.verify()
        if not ok:
            raise RuntimeError(f"{bot_id}: ledger failed journal verification: {msg}")
        return led
    return Ledger(bot_id, capital, ts=ts)


def save_ledger(path: Path, led: Ledger) -> None:
    ok, msg = led.verify()
    if not ok:
        raise RuntimeError(f"{led.agent_id}: refusing to save an inconsistent ledger: {msg}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(led.to_dict(), indent=0, default=str))


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
