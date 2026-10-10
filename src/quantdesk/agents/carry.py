"""A4: conditional funding carry - long spot, short the same quantity of the perpetual."""
from __future__ import annotations

import math

import pandas as pd

from ..context import Context
from ..execution import OrderIntent
from ..ledger import Ledger
from ..risk import margin_state
from .base import Agent


class FundingCarry(Agent):
    """Enter when the 7-day average funding (annualised) is above `entry_annualised`; exit when it
    drops below `exit_annualised` or the last day's funding is negative. One pair at a time.

    Funding history lives in the store under field 'funding' (daily sum of funding rates, keyed by the
    perp instrument id). Funding is credited/debited to the ledger by the engine, not here."""

    def static_instruments(self) -> list[str]:
        out = []
        for legs in self.p["pairs"].values():
            out += [legs["spot"], legs["perp"]]
        return out

    def lookback_days(self) -> int:
        return 40

    def extra_needs(self) -> set[str]:
        return {"funding"}

    def funding_stats(self, ctx: Context, perp: str) -> tuple[float | None, float | None]:
        """(7-day average, average over the last `exit_negative_hours`), both annualised.
        The funding store holds daily sums, so the recent window is rounded up to whole days."""
        f = ctx.history(perp, field="funding")
        n = int(self.p.get("avg_window_days", 7))
        if len(f) < n:
            return None, None
        # require the series to be current (last value within 2 days of asof)
        if (ctx.asof - f.index[-1]).days > 2:
            return None, None
        recent_days = max(1, math.ceil(float(self.p.get("exit_negative_hours", 24)) / 24.0))
        avg = float(f.iloc[-n:].mean() * 365.0)
        last = float(f.iloc[-recent_days:].mean() * 365.0)
        return avg, last

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        day = ctx.asof.date().isoformat()
        if ledger.meta.get("last_decision") == day:
            return []
        ledger.meta["last_decision"] = day
        stats = {name: self.funding_stats(ctx, legs["perp"]) for name, legs in self.p["pairs"].items()}
        ledger.meta["funding"] = {k: {"avg7_ann": v[0], "last_ann": v[1]} for k, v in stats.items()}
        # reconcile state with the book (an entry or exit may have failed to fill)
        active = None
        for name, legs in self.p["pairs"].items():
            if abs(ledger.qty(legs["spot"])) > 0 or abs(ledger.qty(legs["perp"])) > 0:
                active = name
        ledger.meta["active"] = active
        orders: list[OrderIntent] = []

        if active:
            legs = self.p["pairs"][active]
            avg, last = stats.get(active, (None, None))
            exit_now = (avg is None or avg < float(self.p["exit_annualised"])
                        or (last is not None and last < 0))
            if not exit_now:
                return self._rebalance_collateral(ctx, ledger, active, legs, day)
            combo = f"carry-exit-{day}"
            for iid in (legs["spot"], legs["perp"]):
                q = ledger.qty(iid)
                if abs(q) > 0:
                    orders.append(OrderIntent(iid, -q, f"carry exit ({active}, avg {avg})", combo=combo))
            ledger.meta["active"] = None
            return orders

        # flat: pick the best pair above the entry threshold
        cands = [(v[0], k) for k, v in stats.items() if v[0] is not None and v[0] > float(self.p["entry_annualised"])
                 and (v[1] is None or v[1] > 0)]
        if not cands:
            return []
        avg, name = max(cands)
        legs = self.p["pairs"][name]
        spot_px = ctx.mark(legs["spot"])
        perp_px = ctx.mark(legs["perp"])
        if not spot_px or not perp_px:
            return []
        equity = ledger.equity(ctx.fx, {legs["spot"]: spot_px, legs["perp"]: perp_px})
        spot_inst = ctx.inst(legs["spot"])
        spot_eur = ctx.fx.to_eur(spot_px, spot_inst.currency)
        budget = float(self.p.get("spot_fraction", 0.5)) * equity * 0.99
        if budget < float(self.p.get("min_trade_eur", 50)):
            return []
        qty = math.floor(budget / spot_eur * 1e6) / 1e6
        combo = f"carry-entry-{day}"
        ledger.meta["active"] = name
        ledger.meta["entered"] = day
        return [OrderIntent(legs["spot"], qty, f"carry entry ({name}, 7d avg {avg:.1%})", combo=combo),
                OrderIntent(legs["perp"], -qty, f"carry entry hedge ({name})", combo=combo)]

    def _rebalance_collateral(self, ctx: Context, ledger: Ledger, name: str, legs: dict, day: str) -> list[OrderIntent]:
        """Keep the perp venue solvent. The spot coins sit at another venue and cannot be used as perp
        collateral, so a rally eats the perp account's collateral. When perp notional exceeds
        `max_perp_leverage` x collateral, sell part of the spot (moving the cash to the perp venue) and
        buy back the same quantity of perp, so the account is back at 1x and still delta-neutral:
            f = (N - K) / (N + S)   (N = perp notional, K = collateral, S = spot value)"""
        lev_cap = float(self.p.get("max_perp_leverage", 2.0))
        marks = {legs["spot"]: ctx.mark(legs["spot"]), legs["perp"]: ctx.mark(legs["perp"])}
        if not all(marks.values()):
            return []
        ms = margin_state(ledger, ctx.fx, marks, self.cm.venues, self.registry)
        if ms is None:
            return []
        ledger.meta["perp_leverage"] = round(ms["leverage"], 3) if math.isfinite(ms["leverage"]) else None
        if ms["leverage"] <= lev_cap:
            return []
        spot_qty = ledger.qty(legs["spot"])
        spot_val = ctx.fx.to_eur(spot_qty * marks[legs["spot"]], ctx.inst(legs["spot"]).currency)
        N, K = ms["notional_eur"], ms["collateral_eur"]
        f = min(1.0, max(0.0, (N - K) / (N + spot_val))) if (N + spot_val) > 0 else 1.0
        q = math.floor(spot_qty * f * 1e6) / 1e6
        if q <= 0:
            return []
        ctx.note(f"perp leverage {ms['leverage']:.2f}x > {lev_cap}x: moving collateral, reducing both legs by {f:.0%}")
        combo = f"carry-collateral-{day}"
        return [OrderIntent(legs["spot"], -q, f"collateral top-up ({name})", combo=combo),
                OrderIntent(legs["perp"], q, f"collateral top-up hedge ({name})", combo=combo)]
