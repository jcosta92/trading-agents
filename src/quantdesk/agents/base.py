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
        """Daily decision on completed bars (once per `asof`)."""
        raise NotImplementedError

    def intraday(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        """Checks that run on every paper-run cycle (hourly) and once per day in backtests.
        Generic stop-loss/take-profit levels are handled by quantdesk.protect; agents override this
        only for exits that are not simple price levels (e.g. option spread value)."""
        return []

    # ------------------------------------------------------------------ sizing
    def risk_qty(self, ctx: Context, ledger: Ledger, iid: str, stop_dist: float, risk_frac: float,
                 max_frac: float) -> float:
        """Units to buy so that hitting the stop loses `risk_frac` of equity, capped at `max_frac` of
        equity in value and rounded down to whole lots where the venue requires it."""
        import math as _m
        inst = ctx.inst(iid)
        px = ctx.mark(iid)
        if inst is None or not px or not stop_dist or stop_dist <= 0:
            return 0.0
        fx = ctx.fx
        eq = ledger.equity(fx)
        unit_eur = fx.to_eur(px * inst.multiplier, inst.currency)
        risk_unit_eur = fx.to_eur(stop_dist * inst.multiplier, inst.currency)
        qty = min(risk_frac * eq / risk_unit_eur, max_frac * eq / unit_eur)
        lot = self.cm.lot_size(inst, self.venue_override)
        if lot > 0:
            qty = _m.floor(qty / lot + 1e-9) * lot
        else:
            qty = _m.floor(qty * 1e6) / 1e6
        return max(0.0, qty)

    def day_key(self, ctx: Context) -> str:
        return ctx.asof.date().isoformat()

    def once_per_day(self, ctx: Context, ledger: Ledger) -> bool:
        """True the first time it is called for a given `asof` (and records it)."""
        k = self.day_key(ctx)
        if ledger.meta.get("last_decision") == k:
            return False
        ledger.meta["last_decision"] = k
        return True

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def month_key(ts: pd.Timestamp) -> str:
        return pd.Timestamp(ts).strftime("%Y-%m")
