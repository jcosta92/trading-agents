"""Per-agent kill switches and exposure limits.

Paper run (enforced):
  * loss from the starting capital beyond `max_drawdown` -> flatten everything and halt the agent
  * loss on the day beyond `daily_loss`                 -> no new entries until the next day
Backtests record when the kill switch WOULD have fired (on peak-to-trough drawdown) but keep
trading, so that the strategy's long-run behaviour is still measured; see metrics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    action: str          # ok | block_entries | halt
    reason: str = ""


def evaluate(equity: float, initial: float, day_start_equity: float | None, risk: dict) -> RiskDecision:
    mdd = risk.get("max_drawdown")
    if mdd and equity <= initial * (1.0 - mdd):
        return RiskDecision("halt", f"loss from start {equity / initial - 1:.1%} breached -{mdd:.0%}")
    dl = risk.get("daily_loss")
    if dl and day_start_equity and equity <= day_start_equity * (1.0 - dl):
        return RiskDecision("block_entries", f"daily loss {equity / day_start_equity - 1:.1%} breached -{dl:.0%}")
    return RiskDecision("ok")


def is_entry(order_qty: float, held_qty: float) -> bool:
    """True if the order increases absolute exposure."""
    return abs(held_qty + order_qty) > abs(held_qty) + 1e-12


def filter_entries(orders, ledger):
    return [o for o in orders if not is_entry(o.qty, ledger.qty(o.inst_id))]


def margin_state(ledger, fx, marks: dict, venues: dict, registry) -> dict | None:
    """Collateral view of a margined perp account (Kraken EEA).

    Everything except spot/ETF/stock/option holdings is treated as collateral at the perp venue:
    base and USD cash plus unrealised perp P&L. Spot coins bought for a carry trade sit at another
    venue (Bitvavo) and do NOT back the perp."""
    perps = [p for p in ledger.positions.values() if p.kind == "perp"]
    if not perps:
        return None
    notional = sum(fx.to_eur(p.notional(ledger._mark(p, marks)), p.currency) for p in perps)
    other = sum(fx.to_eur(p.market_value(ledger._mark(p, marks)), p.currency)
                for p in ledger.positions.values() if p.kind != "perp")
    collateral = ledger.equity(fx, marks) - other
    inst = registry.get(perps[0].inst_id)
    v = venues.get(inst.venue, {}) if inst else {}
    return {"notional_eur": notional, "collateral_eur": collateral,
            "leverage": notional / collateral if collateral > 0 else float("inf"),
            "maintenance": float(v.get("maintenance_margin", 0.01)),
            "liquidation_fee": float(v.get("liquidation_fee", 0.01))}


def needs_liquidation(state: dict | None) -> bool:
    return bool(state) and state["collateral_eur"] <= state["maintenance"] * state["notional_eur"]


def liquidate(ledger, executor, quotes, fx, state: dict, ts: str) -> list:
    """Force-close every perp at the quote (with normal costs) and charge the venue's liquidation fee."""
    from .execution import OrderIntent
    orders = [OrderIntent(iid, -p.qty, "LIQUIDATION: collateral below maintenance margin")
              for iid, p in ledger.positions.items() if p.kind == "perp"]
    fills, rejects = executor.execute(ledger, orders, quotes, ts)
    fee_eur = state["liquidation_fee"] * state["notional_eur"]
    ledger.charge(ts, "EUR", fee_eur, fx, "liquidation", note="venue liquidation fee")
    return fills


def gross_ratio(ledger, fx, marks) -> float:
    eq = ledger.equity(fx, marks)
    return ledger.gross_exposure(fx, marks) / eq if eq > 0 else float("inf")
