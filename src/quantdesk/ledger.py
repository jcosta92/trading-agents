"""Per-agent, multi-currency ledger with an append-only journal.

Every change to cash or positions is first written as a journal entry and then applied by
`_apply`. `verify()` replays the journal into a fresh ledger and checks that it reproduces the
stored balances exactly, so a corrupted or hand-edited state file is detected.

Accounting conventions
  * cash is held per currency; equity is reported in EUR at mid FX rates
  * spot/ETF/stock/option trades move cash by -qty * price * multiplier (short options => credit)
  * perpetual futures move no notional cash; realised P&L is posted to cash when a position is
    reduced, and unrealised P&L counts towards equity
  * funding on perps is posted to cash (negative = paid)
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

EPS = 1e-9


@dataclass
class FxRates:
    eurusd: float  # USD per 1 EUR (mid)

    def to_eur(self, amount: float, ccy: str) -> float:
        if ccy == "EUR":
            return amount
        if ccy == "USD":
            return amount / self.eurusd
        raise ValueError(f"unsupported currency {ccy}")

    def from_eur(self, amount_eur: float, ccy: str) -> float:
        if ccy == "EUR":
            return amount_eur
        if ccy == "USD":
            return amount_eur * self.eurusd
        raise ValueError(f"unsupported currency {ccy}")


@dataclass
class Position:
    inst_id: str
    qty: float
    avg_price: float
    currency: str
    multiplier: float
    kind: str
    opened: str
    last_mark: float | None = None

    def market_value(self, mark: float) -> float:
        """Contribution to equity in the position currency."""
        if self.kind == "perp":
            return self.qty * (mark - self.avg_price) * self.multiplier
        return self.qty * mark * self.multiplier

    def notional(self, mark: float) -> float:
        return abs(self.qty * mark * self.multiplier)


class Ledger:
    def __init__(self, agent_id: str, initial_capital: float = 1000.0, base: str = "EUR",
                 ts: str = "1970-01-01T00:00:00"):
        self.agent_id = agent_id
        self.base = base
        self.initial_capital = float(initial_capital)
        self.cash: dict[str, float] = defaultdict(float)
        self.positions: dict[str, Position] = {}
        self.journal: list[dict] = []
        self.costs: dict[str, float] = defaultdict(float)   # EUR, by category
        self.realized: list[dict] = []                       # closed/reduced lots, for trade stats
        self.meta: dict = {}                                 # agent state (persisted)
        self.halted = False
        self.halt_reason: str | None = None
        if initial_capital:
            self.post({"type": "deposit", "ts": ts, "ccy": base, "amount": float(initial_capital)})

    # ------------------------------------------------------------------ journal
    def post(self, entry: dict) -> dict:
        entry = dict(entry)
        entry["seq"] = len(self.journal)
        self._apply(entry)
        self.journal.append(entry)
        return entry

    def _apply(self, e: dict) -> None:
        t = e["type"]
        if t == "deposit":
            self.cash[e["ccy"]] += e["amount"]
        elif t == "fx":
            self.cash[e["from_ccy"]] -= e["amount_from"]
            self.cash[e["to_ccy"]] += e["amount_to"]
            self.costs["fx"] += e.get("fee_eur", 0.0)
        elif t == "trade":
            self._apply_trade(e)
        elif t == "funding":
            self.cash[e["ccy"]] += e["amount"]
            self.costs["funding"] -= e.get("amount_eur", 0.0)
        elif t == "income":
            self.cash[e["ccy"]] += e["amount"]
            self.costs["dividends_received"] -= e.get("amount_eur", 0.0)
        elif t == "charge":
            self.cash[e["ccy"]] += e["amount"]          # amount is negative
            self.costs[e.get("category", "other")] -= e.get("amount_eur", 0.0)
        else:
            raise ValueError(f"unknown journal entry type {t}")

    def _apply_trade(self, e: dict) -> None:
        iid, qty, px, ccy = e["inst"], float(e["qty"]), float(e["price"]), e["ccy"]
        mult, kind = float(e.get("multiplier", 1.0)), e["kind"]
        pos = self.positions.get(iid)
        old = pos.qty if pos else 0.0
        realized = 0.0
        closed = 0.0
        if pos and old * qty < 0:                         # reducing or flipping
            closed = min(abs(qty), abs(old))
            realized = closed * (px - pos.avg_price) * (1.0 if old > 0 else -1.0) * mult
        if kind == "perp":
            self.cash[ccy] += realized
        else:
            self.cash[ccy] -= qty * px * mult
        for fccy, amt in (e.get("fees") or {}).items():
            self.cash[fccy] -= amt
        self.costs["commissions"] += e.get("fees_eur", 0.0)
        self.costs["spread_slippage"] += e.get("spread_cost_eur", 0.0)

        new = old + qty
        if abs(new) < EPS:
            self.positions.pop(iid, None)
        elif pos is None or old == 0:
            self.positions[iid] = Position(iid, new, px, ccy, mult, kind, e["ts"], px)
        elif old * qty > 0:                               # adding
            pos.avg_price = (old * pos.avg_price + qty * px) / new
            pos.qty = new
            pos.last_mark = px
        elif old * new > 0:                               # partial reduce, avg unchanged
            pos.qty = new
            pos.last_mark = px
        else:                                             # flip through zero
            self.positions[iid] = Position(iid, new, px, ccy, mult, kind, e["ts"], px)
        if closed > 0:
            self.realized.append({"ts": e["ts"], "inst": iid, "qty": closed, "pnl": realized,
                                  "ccy": ccy, "pnl_eur": realized / e.get("eur_rate", 1.0) if ccy == "USD" else realized})

    # ------------------------------------------------------------------ high-level operations
    def trade(self, ts: str, inst, qty: float, price: float, fees: dict[str, float], fx: FxRates,
              mid: float, reason: str = "", combo: str | None = None) -> dict:
        fees_eur = sum(fx.to_eur(a, c) for c, a in fees.items())
        spread_cost = abs(price - mid) * abs(qty) * inst.multiplier
        entry = {
            "type": "trade", "ts": ts, "inst": inst.id, "kind": inst.kind, "qty": float(qty),
            "price": float(price), "mid": float(mid), "ccy": inst.currency, "multiplier": inst.multiplier,
            "fees": {c: float(a) for c, a in fees.items() if a}, "fees_eur": fees_eur,
            "spread_cost_eur": fx.to_eur(spread_cost, inst.currency),
            "value_eur": fx.to_eur(abs(qty) * price * inst.multiplier, inst.currency),
            "eur_rate": fx.eurusd if inst.currency == "USD" else 1.0, "reason": reason,
        }
        if combo:
            entry["combo"] = combo
        return self.post(entry)

    def convert(self, ts: str, from_ccy: str, to_ccy: str, amount_to: float, fx: FxRates,
                fee_rate: float, reason: str = "auto") -> dict:
        """Buy `amount_to` of to_ccy using from_ccy at mid FX worsened by fee_rate."""
        if from_ccy == "EUR" and to_ccy == "USD":
            amount_from = amount_to / (fx.eurusd * (1.0 - fee_rate))
            fee_eur = amount_from - amount_to / fx.eurusd
        elif from_ccy == "USD" and to_ccy == "EUR":
            amount_from = amount_to * fx.eurusd / (1.0 - fee_rate)
            fee_eur = amount_from / fx.eurusd - amount_to
        else:
            raise ValueError("only EUR<->USD supported")
        return self.post({"type": "fx", "ts": ts, "from_ccy": from_ccy, "to_ccy": to_ccy,
                          "amount_from": amount_from, "amount_to": amount_to, "rate": fx.eurusd,
                          "fee_rate": fee_rate, "fee_eur": fee_eur, "reason": reason})

    def funding(self, ts: str, inst_id: str, ccy: str, amount: float, fx: FxRates, rate: float) -> dict:
        return self.post({"type": "funding", "ts": ts, "inst": inst_id, "ccy": ccy, "amount": float(amount),
                          "amount_eur": fx.to_eur(amount, ccy), "rate": rate})

    def income(self, ts: str, inst_id: str, ccy: str, amount: float, fx: FxRates, note: str = "") -> dict:
        """Cash income such as dividends (net of withholding). `amount` is positive."""
        return self.post({"type": "income", "ts": ts, "inst": inst_id, "ccy": ccy, "amount": abs(float(amount)),
                          "amount_eur": fx.to_eur(abs(amount), ccy), "note": note})

    def charge(self, ts: str, ccy: str, amount: float, fx: FxRates, category: str, note: str = "") -> dict:
        """Debit an external cost (e.g. LLM inference). `amount` is positive."""
        return self.post({"type": "charge", "ts": ts, "ccy": ccy, "amount": -abs(float(amount)),
                          "amount_eur": -fx.to_eur(abs(amount), ccy), "category": category, "note": note})

    # ------------------------------------------------------------------ valuation
    def qty(self, inst_id: str) -> float:
        p = self.positions.get(inst_id)
        return p.qty if p else 0.0

    def update_marks(self, marks: dict[str, float]) -> None:
        for iid, p in self.positions.items():
            m = marks.get(iid)
            if m is not None and math.isfinite(m) and m > 0:
                p.last_mark = float(m)

    def _mark(self, p: Position, marks: dict[str, float] | None) -> float:
        m = (marks or {}).get(p.inst_id)
        if m is None or not math.isfinite(m) or m <= 0:
            m = p.last_mark if p.last_mark is not None else p.avg_price
        return float(m)

    def equity(self, fx: FxRates, marks: dict[str, float] | None = None) -> float:
        eq = sum(fx.to_eur(a, c) for c, a in self.cash.items())
        for p in self.positions.values():
            eq += fx.to_eur(p.market_value(self._mark(p, marks)), p.currency)
        return float(eq)

    def gross_exposure(self, fx: FxRates, marks: dict[str, float] | None = None,
                       include_options: bool = False) -> float:
        tot = 0.0
        for p in self.positions.values():
            if p.kind == "option" and not include_options:
                continue
            tot += fx.to_eur(p.notional(self._mark(p, marks)), p.currency)
        return float(tot)

    def cash_eur(self, fx: FxRates) -> float:
        return float(sum(fx.to_eur(a, c) for c, a in self.cash.items()))

    # ------------------------------------------------------------------ integrity
    def verify(self, tol: float = 1e-6) -> tuple[bool, str]:
        fresh = Ledger(self.agent_id, 0.0, self.base)
        for e in self.journal:
            fresh._apply(dict(e))
            fresh.journal.append(e)
        for c in set(self.cash) | set(fresh.cash):
            if abs(self.cash.get(c, 0.0) - fresh.cash.get(c, 0.0)) > tol:
                return False, f"cash mismatch {c}: {self.cash.get(c)} vs replay {fresh.cash.get(c)}"
        if set(self.positions) != set(fresh.positions):
            return False, f"position set mismatch {sorted(self.positions)} vs {sorted(fresh.positions)}"
        for iid, p in self.positions.items():
            q = fresh.positions[iid]
            if abs(p.qty - q.qty) > tol or abs(p.avg_price - q.avg_price) > max(tol, 1e-9 * abs(p.avg_price)):
                return False, f"position mismatch {iid}"
        return True, "ok"

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id, "base": self.base, "initial_capital": self.initial_capital,
            "cash": dict(self.cash), "positions": {k: asdict(v) for k, v in self.positions.items()},
            "journal": self.journal, "costs": dict(self.costs), "realized": self.realized,
            "meta": self.meta, "halted": self.halted, "halt_reason": self.halt_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ledger":
        led = cls(d["agent_id"], 0.0, d.get("base", "EUR"))
        led.initial_capital = float(d.get("initial_capital", 1000.0))
        led.cash = defaultdict(float, d.get("cash", {}))
        led.positions = {k: Position(**v) for k, v in d.get("positions", {}).items()}
        led.journal = list(d.get("journal", []))
        led.costs = defaultdict(float, d.get("costs", {}))
        led.realized = list(d.get("realized", []))
        led.meta = copy.deepcopy(d.get("meta", {}))
        led.halted = bool(d.get("halted", False))
        led.halt_reason = d.get("halt_reason")
        return led
