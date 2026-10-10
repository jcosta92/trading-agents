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
