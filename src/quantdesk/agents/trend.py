"""A1: multi-asset time-series momentum on UCITS ETFs, and A3: crypto trend on perpetuals."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..context import Context
from ..data.store import annualised_vol
from ..execution import OrderIntent
from ..ledger import Ledger
from ..portfolio import rebalance_to_weights
from .base import Agent


class TrendETF(Agent):
    """Long an asset when its trend signal is positive, else cash; inverse-vol sized; monthly.

    Weight_i = min(max_weight, (target_vol / N) / vol_i) for assets with a positive signal, where N is
    the number of assets in the universe (so fewer positive signals = more cash). Gross capped at 1.
    Between monthly rebalances the agent only retries entries/exits that failed to fill.
    """

    def static_instruments(self) -> list[str]:
        return list(self.p["instruments"])

    def lookback_days(self) -> int:
        return int(self.p.get("lookback_days", 252)) + int(self.p.get("vol_window", 60)) + 40

    def decision_due(self, asof, ledger) -> bool:
        return True   # cheap: daily retry of missed entries/exits; heavy work only monthly

    def signals(self, ctx: Context) -> tuple[dict[str, float], dict[str, dict]]:
        lb = int(self.p.get("lookback_days", 252))
        vw = int(self.p.get("vol_window", 60))
        universe = list(self.p["instruments"])
        n = len(universe)
        tv = float(self.p.get("target_vol", 0.10))
        weights, diag = {}, {}
        for iid in universe:
            px = ctx.history(iid)
            if len(px) < lb + 1:
                diag[iid] = {"signal": None, "why": f"only {len(px)} bars"}
                weights[iid] = 0.0
                continue
            if self.p.get("signal", "tsmom") == "sma":
                sig = float(px.iloc[-1] / px.iloc[-lb:].mean() - 1.0)
            else:
                sig = float(px.iloc[-1] / px.iloc[-lb - 1] - 1.0)
            vol = annualised_vol(px, vw, self.ppy)
            w = 0.0
            if sig > 0 and math.isfinite(vol) and vol > 0:
                w = min(float(self.p.get("max_weight", 0.4)), (tv / n) / vol)
            weights[iid] = w
            diag[iid] = {"signal": round(sig, 4), "vol": round(vol, 4) if math.isfinite(vol) else None,
                         "weight": round(w, 4)}
        gross = sum(weights.values())
        cap = float(self.p.get("max_gross", 1.0))
        if gross > cap:
            weights = {k: v * cap / gross for k, v in weights.items()}
        return weights, diag

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        key = self.month_key(ctx.asof)
        if ledger.meta.get("last_rebalance") != key:
            weights, diag = self.signals(ctx)
            ledger.meta["last_rebalance"] = key
            ledger.meta["targets"] = weights
            ledger.meta["signals"] = diag
            return rebalance_to_weights(ctx, ledger, weights, self.cm, venue_override=self.venue_override,
                                        drift_band=float(self.p.get("drift_band", 0.3)),
                                        max_cost_frac=float(self.p.get("max_cost_frac", 0.02)),
                                        min_trade_eur=float(self.p.get("min_trade_eur", 25)),
                                        reason="monthly trend")
        # mid-month: only fix positions whose direction disagrees with the stored targets
        targets = ledger.meta.get("targets") or {}
        missing = {k: v for k, v in targets.items() if (v > 0) != (ledger.qty(k) > 0)}
        if not missing:
            return []
        return rebalance_to_weights(ctx, ledger, targets, self.cm, venue_override=self.venue_override,
                                    drift_band=math.inf, max_cost_frac=float(self.p.get("max_cost_frac", 0.02)),
                                    min_trade_eur=float(self.p.get("min_trade_eur", 25)), reason="retry")


class CryptoTrend(Agent):
    """BTC/ETH perps, long or flat: long when close > SMA(n) and n-day return > 0; vol-scaled; daily."""

    def static_instruments(self) -> list[str]:
        return list(self.p["instruments"])

    def lookback_days(self) -> int:
        return max(int(self.p.get("sma_window", 100)), int(self.p.get("vol_window", 60))) + 40

    def signals(self, ctx: Context) -> tuple[dict[str, float], dict[str, dict]]:
        sma_n = int(self.p.get("sma_window", 100))
        mom_n = int(self.p.get("momentum_window", 30))
        vw = int(self.p.get("vol_window", 60))
        tv = float(self.p.get("target_vol", 0.40))
        weights, diag = {}, {}
        for iid, alloc in self.p["instruments"].items():
            px = ctx.history(iid)
            if len(px) < max(sma_n, mom_n, vw) + 1:
                weights[iid] = 0.0
                diag[iid] = {"signal": None, "why": f"only {len(px)} bars"}
                continue
            above = px.iloc[-1] > px.iloc[-sma_n:].mean()
            mom_ok = True if mom_n <= 0 else px.iloc[-1] > px.iloc[-mom_n - 1]
            vol = annualised_vol(px, vw, self.ppy)
            w = 0.0
            if above and mom_ok and math.isfinite(vol) and vol > 0:
                w = float(alloc) * min(1.0, tv / vol)
            weights[iid] = w
            diag[iid] = {"above_sma": bool(above), "momentum_ok": bool(mom_ok),
                         "vol": round(vol, 4) if math.isfinite(vol) else None, "weight": round(w, 4)}
        cap = float(self.p.get("max_leverage", 1.0))
        gross = sum(abs(v) for v in weights.values())
        if gross > cap:
            weights = {k: v * cap / gross for k, v in weights.items()}
        return weights, diag

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        day = ctx.asof.date().isoformat()
        if ledger.meta.get("last_decision") == day:
            return []
        weights, diag = self.signals(ctx)
        ledger.meta["last_decision"] = day
        ledger.meta["signals"] = diag
        ledger.meta["targets"] = weights
        return rebalance_to_weights(ctx, ledger, weights, self.cm, venue_override=self.venue_override,
                                    drift_band=float(self.p.get("drift_band", 0.25)),
                                    max_cost_frac=float(self.p.get("max_cost_frac", 0.01)),
                                    min_trade_eur=float(self.p.get("min_trade_eur", 20)), reason="crypto trend")
