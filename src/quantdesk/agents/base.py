"""Agent base class. Agents are pure decision functions: (point-in-time context, ledger) -> orders.

They never talk to a broker or fetch data themselves; the backtester and the paper runner do that,
which is what lets the exact same agent code run in both.
"""
from __future__ import annotations

import pandas as pd

from ..config import AgentSpec
from ..context import Context
from ..costs import CostModel
from ..execution import OrderIntent
from ..ledger import Ledger
from ..registry import Registry

PERIODS_PER_YEAR = {"XETR": 252, "XNYS": 252, "CRYPTO": 365}


class Agent:
    fill_timing = "next_open"          # backtest: 'next_open' or 'close' (decide and fill on the same close)
    backtestable = True

    def __init__(self, spec: AgentSpec, registry: Registry, cost_model: CostModel):
        self.spec = spec
        self.id = spec.id
        self.p = spec.params
        self.registry = registry
        self.cm = cost_model
        self.ppy = PERIODS_PER_YEAR[spec.calendar]

    @property
    def venue_override(self) -> str | None:
        return self.spec.venue_override

    # ------------------------------------------------------------------ data needs
    def static_instruments(self) -> list[str]:
        """Instrument ids whose daily history the agent needs."""
        return []

    def lookback_days(self) -> int:
        return 400

    def calendar_reference(self) -> list[str]:
        """Series whose dates define the agent's trading days in a backtest."""
        if self.spec.calendar == "XNYS" and not self.static_instruments():
            return ["XSP"]
        return self.static_instruments()

    def decision_due(self, asof: pd.Timestamp, ledger: Ledger) -> bool:
        """Cheap check used by the paper runner to decide whether heavy data must be fetched."""
        return True

    def extra_needs(self) -> set[str]:
        """Optional heavy inputs: 'sp500_panel', 'fundamentals', 'news', 'funding', 'options'."""
        return set()

    # ------------------------------------------------------------------ decisions
    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def month_key(ts: pd.Timestamp) -> str:
        return pd.Timestamp(ts).strftime("%Y-%m")
