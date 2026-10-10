"""Daily event-driven backtester using the same agents, executor, cost model and ledger as the paper run.

Day loop (per agent calendar):
  1. fill yesterday's orders at today's open (agents with fill_timing='next_open')
  2. credit/debit perpetual funding for the day
  3. cash-settle any expired options
  4. mark to market at the close, record equity and exposure
  5. risk checks: daily-loss block is enforced; the drawdown kill switch is RECORDED, not enforced
  6. agent decides with data up to today's close; orders fill now (fill_timing='close') or tomorrow
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import risk as riskmod
from ..agents.base import Agent
from ..context import Context
from ..costs import CostModel
from ..data.options import ModelParams, SyntheticOptionModel
from ..data.store import HistoryStore
from ..execution import Executor, OrderIntent, Quote
from ..ledger import FxRates, Ledger
from ..registry import Registry

log = logging.getLogger(__name__)


class BacktestQuotes:
    """Quotes for one simulated moment: the open or the close of `date`."""

    def __init__(self, date: pd.Timestamp, exact: pd.DataFrame, marks: pd.DataFrame, fx_series: pd.Series,
                 registry: Registry, options: SyntheticOptionModel | None, exact_only: bool):
        self.date = date
        self._exact = exact
        self._marks = marks
        self._fx = fx_series
        self.registry = registry
        self.options = options
        self.exact_only = exact_only
        self._fx_cache: FxRates | None = None

    def quote(self, inst) -> Quote | None:
        if inst is None:
            return None
        if inst.is_option:
            if self.options is None:
                return None
            q = self.options.quote(self.date, inst.expiry, inst.strike, inst.right)
            if q is None:
                return None
            return Quote(q["mid"], q["bid"], q["ask"], source="model")
        frame = self._exact if self.exact_only else self._marks
        if inst.id not in frame.columns:
            return None
        try:
            v = frame.at[self.date, inst.id]
        except KeyError:
            return None
        if v is None or not math.isfinite(v) or v <= 0:
            return None
        return Quote(float(v), source="history")

    def fx(self) -> FxRates:
        if self._fx_cache is None:
            sub = self._fx.loc[:self.date]
            self._fx_cache = FxRates(float(sub.iloc[-1]) if len(sub) else 1.10)
        return self._fx_cache


@dataclass
class BacktestResult:
    agent_id: str
    variant: str
    stress: float
    equity: pd.Series
    exposure: pd.Series
    fills: pd.DataFrame
    rejects: pd.DataFrame
    costs: dict
    ledger: Ledger
    kill_events: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def trading_dates(store: HistoryStore, keys: list[str], calendar: str, start, end) -> pd.DatetimeIndex:
    df = store.frame("close")
    cols = [k for k in keys if k in df.columns]
    if not cols:
        return pd.DatetimeIndex([])
    sub = df[cols]
    idx = sub.dropna(how="all").index
    if calendar != "CRYPTO":
        idx = idx[idx.dayofweek < 5]
    idx = idx[(idx >= pd.Timestamp(start))]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx


def run_backtest(agent: Agent, store: HistoryStore, registry: Registry, cm: CostModel, *,
                 start: str, end: str | None = None, capital: float = 1000.0, extras: dict | None = None,
                 reference: list[str] | None = None, fallback_funding_annual: float = 0.10,
                 enforce_daily_loss: bool = True) -> BacktestResult:
    extras = dict(extras or {})
    ref = reference or agent.calendar_reference()
    dates = trading_dates(store, ref, agent.spec.calendar, start, end)
    if len(dates) == 0:
        raise ValueError(f"{agent.id}: no trading dates for {ref} from {start}")
    close = store.frame("close")
    opens = store.frame("open") if "open" in store.frames else close
    marks_ff = close.ffill(limit=7)
    fx_series = store.close("EURUSD") if store.has("EURUSD") else pd.Series([1.1], index=[dates[0]])
    fx_series = fx_series.ffill()
    funding = store.frame("funding")
    options = None
    if "options" in agent.extra_needs():
        options = SyntheticOptionModel(store.close("XSP"), store.close("VIX"), store.close("TBILL"),
                                       ModelParams.from_agent(agent.p))
    ledger = Ledger(agent.id, capital, ts=dates[0].isoformat())
    ex = Executor(cm, registry, agent.venue_override)
    pending: list[OrderIntent] = []
    eq_rec, expo_rec, fills_all, rejects_all, dd_events, liquidations = {}, {}, [], [], [], []
    prev_eq = capital
    peak = capital
    in_dd_event = False
    fallback_days = 0
    max_dd = agent.spec.risk.get("max_drawdown")
    max_gross = agent.spec.risk.get("max_gross")
    fill_at = {"next_open": "open", "next_close": "close", "close": "close"}[agent.fill_timing]

    for d in dates:
        ts = d.isoformat()
        q_open = BacktestQuotes(d, opens, marks_ff, fx_series, registry, options, exact_only=True)
        q_close = BacktestQuotes(d, close, marks_ff, fx_series, registry, options, exact_only=False)
        q_exec_close = BacktestQuotes(d, close, marks_ff, fx_series, registry, options, exact_only=True)
        fx = q_close.fx()
        # 1. orders from yesterday's decision: at today's open (or close for 'next_close' agents)
        if pending and fill_at == "open":
            pending = _execute_pending(ex, ledger, pending, q_open, ts, fills_all, rejects_all)
        if pending and fill_at == "close":
            pending = _execute_pending(ex, ledger, pending, q_exec_close, ts, fills_all, rejects_all)
        # 2. perp funding
        for iid, pos in list(ledger.positions.items()):
            if pos.kind != "perp":
                continue
            rate = None
            if iid in funding.columns and d in funding.index:
                v = funding.at[d, iid]
                if v is not None and math.isfinite(v):
                    rate = float(v)
            if rate is None:
                rate = fallback_funding_annual / 365.0
                fallback_days += 1
            q = q_close.quote(registry.get(iid))
            if q is None:
                continue
            amount = -pos.qty * q.mid * pos.multiplier * rate
            if amount:
                ledger.funding(ts, iid, pos.currency, amount, fx, rate)
        # 3. expired options
        _settle_expired(ledger, registry, d, store, fx, ts)
        # 4. mark to market (+ margin call / liquidation check for perps)
        marks = _marks(ledger, registry, q_close)
        ms = riskmod.margin_state(ledger, fx, marks, cm.venues, registry)
        if riskmod.needs_liquidation(ms):
            fills_all += riskmod.liquidate(ledger, ex, q_close, fx, ms, ts)
            liquidations.append({"date": ts, **{k: round(v, 2) for k, v in ms.items() if isinstance(v, float)}})
            marks = _marks(ledger, registry, q_close)
        ledger.update_marks(marks)
        eq = ledger.equity(fx, marks)
        eq_rec[d] = eq
        expo_rec[d] = ledger.gross_exposure(fx, marks) / eq if eq > 0 else np.nan
        # 5. risk: drawdown events are recorded (peak-to-trough) for information only
        peak = max(peak, eq)
        if max_dd and eq <= peak * (1 - max_dd) and not in_dd_event:
            dd_events.append({"date": ts, "equity": round(eq, 2), "drawdown": round(eq / peak - 1, 4)})
            in_dd_event = True
        if in_dd_event and eq >= peak * (1 - max_dd / 2 if max_dd else 1):
            in_dd_event = False
        rd = riskmod.evaluate(eq, capital, prev_eq, {"daily_loss": agent.spec.risk.get("daily_loss")})
        prev_eq = eq
        if eq <= 0:
            log.warning("%s: equity wiped out on %s", agent.id, ts)
            break
        # 6. decide (data up to today's close)
        ctx = Context(d, store, registry, q_close, "backtest", now=d, extras=extras, options=options)
        orders = agent.decide(ctx, ledger)
        if enforce_daily_loss and rd.action == "block_entries":
            orders = riskmod.filter_entries(orders, ledger)
        if max_gross and orders and expo_rec[d] >= max_gross:
            orders = riskmod.filter_entries(orders, ledger)
        if not orders:
            continue
        if agent.fill_timing == "close":
            f, r = ex.execute(ledger, orders, q_close, ts)
            fills_all += f
            rejects_all += r
        else:
            pending = pending + orders

    equity = pd.Series(eq_rec, name=agent.id).sort_index()
    fills = pd.DataFrame([f.__dict__ for f in fills_all])
    rejects = pd.DataFrame([r.__dict__ for r in rejects_all])
    ok, msg = ledger.verify()
    diag = {"ledger_verified": ok, "verify_msg": msg, "fallback_funding_days": fallback_days,
            "start": str(dates[0].date()), "end": str(dates[-1].date()), "n_days": len(dates)}
    if "coverage_log" in ledger.meta:
        diag["coverage_log"] = ledger.meta["coverage_log"]
    diag["liquidations"] = liquidations
    return BacktestResult(agent.id, agent.spec.variant, cm.stress, equity, pd.Series(expo_rec).sort_index(),
                          fills, rejects, dict(ledger.costs), ledger, dd_events, diag)


def _marks(ledger: Ledger, registry: Registry, quotes: BacktestQuotes) -> dict[str, float]:
    out = {}
    for iid in ledger.positions:
        q = quotes.quote(registry.get(iid))
        if q is not None:
            out[iid] = q.mid
    return out


def _execute_pending(ex: Executor, ledger: Ledger, pending: list[OrderIntent], quotes, ts: str,
                     fills_all: list, rejects_all: list) -> list[OrderIntent]:
    """Fill pending orders; orders (or whole combos) with no quote today wait one more day."""
    f, r = ex.execute(ledger, pending, quotes, ts)
    fills_all += f
    rejects_all += r
    noquote_insts = {rj.inst for rj in r if rj.why == "no quote"}
    noquote_combos = {o.combo for o in pending if o.combo and o.inst_id in noquote_insts}
    retry = []
    for o in pending:
        hit = (o.combo in noquote_combos) if o.combo else (o.inst_id in noquote_insts)
        if hit and not o.meta.get("retried"):
            o.meta["retried"] = True
            retry.append(o)
    return retry


def _settle_expired(ledger: Ledger, registry: Registry, d: pd.Timestamp, store: HistoryStore,
                    fx: FxRates, ts: str) -> None:
    for iid, pos in list(ledger.positions.items()):
        if pos.kind != "option":
            continue
        inst = registry.get(iid)
        if inst is None or pd.Timestamp(inst.expiry) > d:
            continue
        und = store.close(inst.underlying).loc[:pd.Timestamp(inst.expiry)]
        if und.empty:
            continue
        S = float(und.iloc[-1])
        intrinsic = max(0.0, (inst.strike - S) if inst.right == "P" else (S - inst.strike))
        ledger.trade(ts, inst, -pos.qty, intrinsic, {}, fx, intrinsic, reason="expiry cash settlement")
