"""Benchmarks and static-weight research portfolios."""
from __future__ import annotations

import math

import pandas as pd

from ..context import Context
from ..execution import OrderIntent
from ..ledger import Ledger
from ..portfolio import rebalance_to_weights
from .base import Agent


class BuyAndHold(Agent):
    """Buy the instrument once with (almost) all the cash, then never trade again."""

    def static_instruments(self) -> list[str]:
        return [self.p["instrument"]]

    def lookback_days(self) -> int:
        return 10

    def decision_due(self, asof, ledger) -> bool:
        return not ledger.meta.get("invested")

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        iid = self.p["instrument"]
        if ledger.meta.get("invested") or ledger.qty(iid) > 0:
            ledger.meta["invested"] = True
            return []
        w = 1.0 - float(self.p.get("cash_buffer", 0.01))
        return rebalance_to_weights(ctx, ledger, {iid: w}, self.cm, venue_override=self.venue_override,
                                    drift_band=0.0, max_cost_frac=math.inf, min_trade_eur=1.0,
                                    reason="buy and hold")


class StaticWeights(Agent):
    """Fixed target weights, rebalanced on a calendar with a drift band (research proxies)."""

    def static_instruments(self) -> list[str]:
        return list(self.p["weights"])

    def lookback_days(self) -> int:
        return 10

    def _due(self, asof: pd.Timestamp, ledger: Ledger) -> bool:
        key = self.month_key(asof)
        if ledger.meta.get("last_rebalance") == key:
            return False
        if not ledger.positions:
            return True
        if self.p.get("rebalance", "quarterly") == "quarterly":
            return asof.month in (1, 4, 7, 10)
        return True

    def decision_due(self, asof, ledger) -> bool:
        return self._due(pd.Timestamp(asof), ledger)

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        if not self._due(ctx.asof, ledger):
            return []
        ledger.meta["last_rebalance"] = self.month_key(ctx.asof)
        return rebalance_to_weights(ctx, ledger, dict(self.p["weights"]), self.cm,
                                    venue_override=self.venue_override,
                                    drift_band=float(self.p.get("drift_band", 0.2)),
                                    max_cost_frac=float(self.p.get("max_cost_frac", 0.02)),
                                    min_trade_eur=float(self.p.get("min_trade_eur", 25)), reason="static weights")
