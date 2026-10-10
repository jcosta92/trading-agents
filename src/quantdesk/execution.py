"""Order intents, quotes and the simulated executor (shared by backtest and paper run)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .config import Instrument
from .costs import CostModel
from .ledger import FxRates, Ledger

EPS = 1e-9


@dataclass
class OrderIntent:
    inst_id: str
    qty: float                    # signed units (contracts for options)
    reason: str = ""
    combo: str | None = None      # orders sharing a combo id fill all-or-nothing
    created: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OrderIntent":
        return cls(**d)


@dataclass
class Quote:
    mid: float
    bid: float | None = None
    ask: float | None = None
    ts: str | None = None
    source: str = ""


class QuoteSource(Protocol):
    def quote(self, inst: Instrument) -> Quote | None: ...
    def fx(self) -> FxRates: ...


@dataclass
class Fill:
    ts: str
    agent: str
    inst: str
    qty: float
    price: float
    mid: float
    fees: float
    fee_ccy: str
    value_eur: float
    reason: str
    combo: str | None = None


@dataclass
class Reject:
    ts: str
    agent: str
    inst: str
    qty: float
    reason: str
    why: str


def round_qty(qty: float, lot: float, mode: str = "floor") -> float:
    if lot and lot > 0:
        n = abs(qty) / lot
        n = math.floor(n + 1e-9) if mode == "floor" else round(n)
        return math.copysign(n * lot, qty) if n else 0.0
    return float(f"{qty:.8f}")


class Executor:
    """Fills orders against quotes with the pessimistic cost model and posts them to the ledger."""

    def __init__(self, cost_model: CostModel, registry, venue_override: str | None = None):
        self.cm = cost_model
        self.registry = registry
        self.venue_override = venue_override

    def execute(self, ledger: Ledger, orders: list[OrderIntent], quotes: QuoteSource, ts: str
                ) -> tuple[list[Fill], list[Reject]]:
        fills: list[Fill] = []
        rejects: list[Reject] = []
        fx = quotes.fx()
        # group combos, keep original order otherwise
        groups: list[list[OrderIntent]] = []
        seen: dict[str, list[OrderIntent]] = {}
        for o in orders:
            if o.combo:
                if o.combo not in seen:
                    seen[o.combo] = []
                    groups.append(seen[o.combo])
                seen[o.combo].append(o)
            else:
                groups.append([o])
        for group in groups:
            priced = []
            ok = True
            for o in group:
                inst = self.registry.get(o.inst_id)
                q = quotes.quote(inst) if inst else None
                if inst is None or q is None or not (q.mid and q.mid > 0):
                    rejects.append(Reject(ts, ledger.agent_id, o.inst_id, o.qty, o.reason, "no quote"))
                    ok = False
                    break
                side = "buy" if o.qty > 0 else "sell"
                px = self.cm.exec_price(inst, side, q.mid, q.bid, q.ask, self.venue_override)
                if px is None:
                    rejects.append(Reject(ts, ledger.agent_id, o.inst_id, o.qty, o.reason, "no executable price"))
                    ok = False
                    break
                mid = (q.bid + q.ask) / 2 if (q.bid and q.ask and q.ask >= q.bid > 0) else q.mid
                priced.append((o, inst, side, px, mid))
            if not ok:
                failed = rejects[-1].inst
                if len(group) > 1:
                    for o in group:
                        if o.inst_id != failed:
                            rejects.append(Reject(ts, ledger.agent_id, o.inst_id, o.qty, o.reason, "combo leg rejected"))
                continue
            if len(group) == 1:
                o, inst, side, px, mid = priced[0]
                f = self._fill_one(ledger, o, inst, side, px, mid, fx, ts, False)
                (fills if isinstance(f, Fill) else rejects).append(f)
                continue
            # combo: size every leg BEFORE filling any, scale all legs together, then fill all or none
            qtys = [self._round_leg(ledger, o, inst) for o, inst, side, px, mid in priced]
            need = sum(self._cash_need_eur(ledger, inst, q, px, side, fx)
                       for (o, inst, side, px, mid), q in zip(priced, qtys))
            avail = max(0.0, ledger.cash.get(ledger.base, 0.0))
            if need > avail + 1e-9:
                ratio = avail / need * 0.995
                qtys = [round_qty(o.qty * ratio, self.cm.lot_size(inst, self.venue_override), "floor")
                        for o, inst, side, px, mid in priced]
            if any(abs(q) < EPS for q in qtys):
                for o, inst, side, px, mid in priced:
                    rejects.append(Reject(ts, ledger.agent_id, o.inst_id, o.qty, o.reason,
                                          "combo cannot be sized (cash or lot size)"))
                continue
            for (o, inst, side, px, mid), q in zip(priced, qtys):
                fills.append(self._post(ledger, o, inst, side, px, mid, q, fx, ts))
        return fills, rejects

    def _round_leg(self, ledger: Ledger, o: OrderIntent, inst: Instrument) -> float:
        held = ledger.qty(inst.id)
        if held != 0 and abs(o.qty + held) < EPS:
            return o.qty
        return round_qty(o.qty, self.cm.lot_size(inst, self.venue_override), "floor")

    def _cash_need_eur(self, ledger: Ledger, inst: Instrument, qty: float, px: float, side: str, fx: FxRates) -> float:
        """Base-currency cash a leg consumes (0 for sells, perps and options)."""
        venue = self.cm.venue(inst, self.venue_override)
        if side != "buy" or venue.get("margin") or venue.get("allow_negative_cash") or inst.is_option or qty <= 0:
            return 0.0
        need = qty * px * inst.multiplier + self.cm.fees(inst, qty, px, side, self.venue_override).total
        if inst.currency == ledger.base:
            return need
        short = max(0.0, need - max(0.0, ledger.cash.get(inst.currency, 0.0)))
        return fx.to_eur(short, inst.currency) / (1.0 - self.cm.fx_fee_rate(inst, self.venue_override))

    def _fill_one(self, ledger: Ledger, o: OrderIntent, inst: Instrument, side: str, px: float, mid: float,
                  fx: FxRates, ts: str, in_combo: bool) -> Fill | Reject:
        lot = self.cm.lot_size(inst, self.venue_override)
        qty = self._round_leg(ledger, o, inst)
        if abs(qty) < EPS:
            return Reject(ts, ledger.agent_id, inst.id, o.qty, o.reason, "rounds to zero lots")
        venue = self.cm.venue(inst, self.venue_override)
        mult = inst.multiplier
        # cash check for cash-settled buys (perps are margined; options are risk-budgeted by the agent)
        if side == "buy" and not venue.get("margin") and not inst.is_option and not venue.get("allow_negative_cash"):
            fx_fee = self.cm.fx_fee_rate(inst, self.venue_override)
            fees = self.cm.fees(inst, qty, px, side, self.venue_override).total
            avail_ccy = self._available(ledger, inst.currency, fx, fx_fee, venue)
            if qty * px * mult + fees > avail_ccy + 1e-9:
                per_unit = px * mult
                qmax = max(0.0, (avail_ccy - fees) / per_unit) if per_unit > 0 else 0.0
                qty = round_qty(min(qty, qmax), lot, "floor")
                if qty <= EPS:
                    return Reject(ts, ledger.agent_id, inst.id, o.qty, o.reason, "insufficient cash")
                fees = self.cm.fees(inst, qty, px, side, self.venue_override).total
                if qty * per_unit + fees > avail_ccy + 1e-9:     # fees changed with the smaller size
                    step = lot if lot else qty * 0.01
                    qty = round_qty(qty - step, lot, "floor")
                    if qty <= EPS:
                        return Reject(ts, ledger.agent_id, inst.id, o.qty, o.reason, "insufficient cash")
        return self._post(ledger, o, inst, side, px, mid, qty, fx, ts)

    def _post(self, ledger: Ledger, o: OrderIntent, inst: Instrument, side: str, px: float, mid: float,
              qty: float, fx: FxRates, ts: str) -> Fill:
        venue = self.cm.venue(inst, self.venue_override)
        fb = self.cm.fees(inst, qty, px, side, self.venue_override)
        # auto FX: make sure the instrument currency balance covers the cash leg + fees
        if inst.currency != ledger.base and venue.get("auto_fx"):
            cash_leg = -qty * px * inst.multiplier if inst.kind != "perp" else 0.0
            after = ledger.cash.get(inst.currency, 0.0) + cash_leg - fb.total
            if after < -1e-9:
                ledger.convert(ts, ledger.base, inst.currency, -after, fx,
                               self.cm.fx_fee_rate(inst, self.venue_override), reason=f"auto FX for {inst.id}")
        ledger.trade(ts, inst, qty, px, {fb.currency: fb.total} if fb.total else {}, fx, mid,
                     reason=o.reason, combo=o.combo)
        return Fill(ts, ledger.agent_id, inst.id, qty, px, mid, fb.total, fb.currency,
                    fx.to_eur(abs(qty) * px * inst.multiplier, inst.currency), o.reason, o.combo)

    @staticmethod
    def _available(ledger: Ledger, ccy: str, fx: FxRates, fx_fee: float, venue: dict) -> float:
        """Cash available in `ccy`, counting base-currency cash convertible via auto FX."""
        own = ledger.cash.get(ccy, 0.0)
        if ccy == ledger.base or not venue.get("auto_fx"):
            return own
        base = max(0.0, ledger.cash.get(ledger.base, 0.0))
        return own + fx.from_eur(base, ccy) * (1.0 - fx_fee)
