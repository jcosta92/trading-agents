"""Turning target weights into orders, with cost-aware no-trade rules for small accounts."""
from __future__ import annotations

import math

from .context import Context
from .costs import CostModel
from .execution import OrderIntent
from .ledger import Ledger

EPS = 1e-9


def rebalance_to_weights(ctx: Context, ledger: Ledger, targets: dict[str, float], cm: CostModel, *,
                         venue_override: str | None = None, drift_band: float = 0.3,
                         max_cost_frac: float = 0.02, min_trade_eur: float = 25.0,
                         manage: set[str] | None = None, reason: str = "rebalance") -> list[OrderIntent]:
    """Orders that move the book toward `targets` (fractions of equity, signed for perps).

    Rules, in order:
      * a target of 0 on a held position -> close it (exits are never skipped)
      * a new position -> open it if it is worth at least `min_trade_eur`
      * an existing position -> resize only if it is more than `drift_band` (relative) away from
        target AND the estimated cost is at most `max_cost_frac` of the trade value
    Instruments held but absent from `targets` are closed only if they are in `manage`.
    Sells are returned before buys so that cash is freed first.
    """
    fx = ctx.fx
    held = set(ledger.positions)
    universe = set(targets) | (held & (manage or set(targets)))
    marks = {iid: ctx.mark(iid) for iid in universe}
    equity = ledger.equity(fx, {k: v for k, v in marks.items() if v})
    orders: list[OrderIntent] = []
    for iid in sorted(universe):
        inst = ctx.inst(iid)
        m = marks.get(iid)
        if inst is None or m is None or not math.isfinite(m) or m <= 0:
            ctx.note(f"{iid}: no price, skipped")
            continue
        w = float(targets.get(iid, 0.0))
        cur = ledger.qty(iid)
        unit_eur = fx.to_eur(m * inst.multiplier, inst.currency)
        tgt = w * equity / unit_eur
        lot = cm.lot_size(inst, venue_override)
        if lot > 0:
            tgt = round(tgt / lot) * lot
        if abs(tgt) < EPS and abs(cur) > EPS:
            orders.append(OrderIntent(iid, -cur, f"{reason}: exit"))
            continue
        if abs(cur) < EPS:
            if abs(tgt) > EPS and abs(tgt) * unit_eur >= min_trade_eur:
                orders.append(OrderIntent(iid, tgt, f"{reason}: entry"))
            continue
        if cur * tgt < 0:                                   # direction flip (perps only)
            orders.append(OrderIntent(iid, tgt - cur, f"{reason}: flip"))
            continue
        dq = tgt - cur
        if abs(dq) < EPS:
            continue
        rel = abs(dq) / max(abs(tgt), EPS)
        trade_eur = abs(dq) * unit_eur
        if rel < drift_band or trade_eur < min_trade_eur:
            continue
        q = ctx.quote(iid)
        est = cm.estimate_cost(inst, dq, m, q.bid if q else None, q.ask if q else None, venue_override)
        if fx.to_eur(est, inst.currency) > max_cost_frac * trade_eur:
            ctx.note(f"{iid}: resize skipped, cost {fx.to_eur(est, inst.currency):.2f} EUR > {max_cost_frac:.0%} of {trade_eur:.0f}")
            continue
        orders.append(OrderIntent(iid, dq, f"{reason}: resize"))
    return sort_sells_first(orders, ledger)


def sort_sells_first(orders: list[OrderIntent], ledger: Ledger) -> list[OrderIntent]:
    def reduces(o: OrderIntent) -> bool:
        cur = ledger.qty(o.inst_id)
        return cur * o.qty < 0
    return [o for o in orders if reduces(o)] + [o for o in orders if not reduces(o)]


def close_all(ledger: Ledger, reason: str) -> list[OrderIntent]:
    """Orders that flatten the book (option positions are closed as one combo)."""
    orders = []
    opts = [iid for iid, p in ledger.positions.items() if p.kind == "option"]
    for iid, p in ledger.positions.items():
        if p.kind == "option":
            continue
        orders.append(OrderIntent(iid, -p.qty, reason))
    if opts:
        for iid in opts:
            orders.append(OrderIntent(iid, -ledger.positions[iid].qty, reason, combo=f"flatten-{reason}"))
    return orders
