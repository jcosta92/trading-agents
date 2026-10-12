"""Paper-trading runner: one call = one cycle for every agent. Runs unattended every hour.

Per agent and run:
  1. load its ledger from state/ (or open a new EUR 1,000 account)
  2. accrue perpetual funding since the last run; credit dividends; cash-settle expired options
  3. mark to market with live quotes; margin liquidation; daily-loss and loss-from-start kill switches
  4. if its market is open (and has been for 30 min):
       a. protective orders - trailing stops / indicator targets are moved with the last completed
          daily bar, and every stop-loss, take-profit and time exit is checked against the live price
       b. the strategy's indicator rules on completed daily bars (each agent acts once per day)
     and all resulting orders fill against live quotes with the same cost model as the backtest;
     entries carry their protective levels, which are set from the actual fill price
  5. snapshot equity, append fills/rejects, save the ledger (journal replay is verified on save)
"""
from __future__ import annotations

import json
import logging
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import protect
from .. import risk as riskmod
from ..agents import build_agent
from ..backtest.engine import dedupe_exits, merge_pending
from ..config import Settings
from ..context import Context
from ..costs import CostModel
from ..data import crypto, loader, yahoo
from ..data.store import HistoryStore
from ..execution import Executor, OrderIntent
from ..ledger import Ledger
from ..portfolio import close_all
from ..registry import Registry
from .quotes import ExecutionQuotes, LiveQuotes

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------- data
class NetworkData:
    """Real data sources. Tests substitute an offline implementation with the same methods."""

    def core_store(self, instruments, ids, start) -> HistoryStore:
        store = loader.build_core_store(instruments, ids=ids, start=start, use_proxies=False, funding=False)
        loader.add_funding(store, instruments, since=str((pd.Timestamp.now() - pd.Timedelta(days=60)).date()),
                           prefer="first")
        return store

    def sp500_store(self, start) -> tuple[HistoryStore, dict]:
        return loader.build_sp500_store(start=start, current_only=True)

    def history(self, tickers, start) -> HistoryStore:
        raw = yahoo.download_history(tickers, start=start)
        st = HistoryStore()
        for f, df in raw.items():
            st.set_frame(f, df)
        return st

    def fundamentals(self, tickers) -> pd.DataFrame:
        return yahoo.fundamentals(tickers)

    def news(self, ticker) -> list[dict]:
        return yahoo.news(ticker)

    def funding_events(self, inst) -> pd.Series:
        for src in inst.funding:
            ev = crypto.funding_history(src["source"], src["symbol"],
                                        since=str((pd.Timestamp.now() - pd.Timedelta(days=10)).date()))
            if len(ev):
                return ev
        return pd.Series(dtype="float64")

    def live_quotes(self, resolved, option_chains, fallback_eurusd=None, reference=None) -> LiveQuotes:
        return LiveQuotes(resolved, option_chains, fallback_eurusd=fallback_eurusd, reference=reference)

    def dividends(self, inst, symbol=None) -> pd.Series:
        return yahoo.dividends(symbol or (inst.yf[0] if inst and inst.yf else None))

    def option_chains(self, spot, rate):
        from ..data.options import LiveOptionChains
        return LiveOptionChains(spot, rate)


def market_open(calendar: str, now: pd.Timestamp, settle_minutes: int = 30) -> bool:
    """True when the exchange is open now and was already open `settle_minutes` ago (so the opening
    auction is done and 15-minute-delayed quotes are from the current session)."""
    if calendar == "CRYPTO":
        return True
    import exchange_calendars as xcals
    cal = xcals.get_calendar(calendar)
    m = now.tz_convert("UTC").floor("min")
    try:
        return bool(cal.is_open_on_minute(m) and cal.is_open_on_minute(m - pd.Timedelta(minutes=settle_minutes)))
    except Exception:
        return False


# ------------------------------------------------------------------------------------------- runner
class PaperRunner:
    def __init__(self, settings: Settings, instruments: dict, venues: dict, state_dir: str | Path = "state",
                 now: pd.Timestamp | None = None, data=None, agents: list[str] | None = None,
                 fallback_funding_annual: float = 0.10, market_open_fn=None):
        self.settings = settings
        self.instruments = instruments
        self.state_dir = Path(state_dir)
        self.now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        if self.now.tzinfo is None:
            self.now = self.now.tz_localize("UTC")
        self.ts = self.now.isoformat()
        self.asof = self.now.tz_convert("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
        self.registry = Registry(instruments)
        self.cm = CostModel(venues, 1.0)
        self.data = data or NetworkData()
        self.only = agents
        self.fallback_funding_annual = fallback_funding_annual
        settle = int(settings.paper.get("market_settle_minutes", 30))
        self.market_open_fn = market_open_fn or (lambda cal, now: market_open(cal, now, settle))
        self.store: HistoryStore | None = None
        self.quotes: LiveQuotes | None = None
        self.sp500_sectors: dict[str, str] = {}
        self._sp500_loaded = False
        self.llm_records: list[dict] = []
        self.summary: dict = {"ts": self.ts, "asof": str(self.asof.date()), "agents": {}, "errors": []}
        self._funding_cache: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------ services used by agents
    @property
    def fx(self):
        return self.quotes.fx()

    def context(self, asof: pd.Timestamp, extras: dict, options=None) -> Context:
        return Context(asof, self.store, self.registry, self.quotes, "live", now=self.now, extras=extras,
                       options=options)

    def previous_close(self, iid: str) -> float | None:
        """Last daily close on or before `asof` (yesterday) - the reference for bad-print checks."""
        if self.store is None:
            return None
        s = self.store.close(iid).loc[:self.asof]
        return float(s.iloc[-1]) if len(s) else None

    def ensure_sp500_panel(self) -> dict[str, str]:
        if not self._sp500_loaded:
            start = str((self.asof - pd.Timedelta(days=430)).date())
            st, extras = self.data.sp500_store(start)
            self.store.merge(st)
            self.sp500_sectors = extras.get("sectors", {})
            self._sp500_loaded = True
        return self.sp500_sectors

    def ensure_history(self, tickers: list[str], days: int = 90) -> None:
        missing = [t for t in tickers if not self.store.has(t)]
        if missing:
            start = str((self.asof - pd.Timedelta(days=int(days * 1.6) + 10)).date())
            self.store.merge(self.data.history(missing, start))
        for t in tickers:
            self.registry.stock(t, self.sp500_sectors.get(t))

    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        return self.data.fundamentals(tickers)

    def fetch_news(self, ticker: str) -> list[dict]:
        return self.data.news(ticker)

    def log_llm(self, record: dict) -> None:
        self.llm_records.append(record)

    # ------------------------------------------------------------------ state
    def _paper_meta(self) -> dict:
        p = self.state_dir / "paper.json"
        if p.exists():
            return json.loads(p.read_text())
        length = int(self.settings.paper.get("length_days", 31))
        meta = {"start": self.ts, "end": (self.now + pd.Timedelta(days=length)).isoformat(), "status": "running"}
        return meta

    def _load_ledger(self, aid: str) -> Ledger:
        p = self.state_dir / "ledgers" / f"{aid}.json"
        if p.exists():
            led = Ledger.from_dict(json.loads(p.read_text()))
            ok, msg = led.verify()
            if not ok:
                raise RuntimeError(f"ledger for {aid} failed journal verification: {msg}")
            return led
        return Ledger(aid, self.settings.initial_capital, ts=self.ts)

    def _register_positions(self, led: Ledger) -> None:
        for iid, p in led.positions.items():
            if self.registry.get(iid) is None and p.kind == "stock":
                self.registry.stock(iid)

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "ledgers").mkdir(exist_ok=True)
        meta = self._paper_meta()
        trading_allowed = meta["status"] == "running" and self.now <= pd.Timestamp(meta["end"])
        if meta["status"] == "running" and not trading_allowed:
            meta["status"] = "finished"
            meta["finished_at"] = self.ts
        self.summary["paper"] = meta

        specs = {aid: s for aid, s in self.settings.agents.items() if s.paper and (not self.only or aid in self.only)}
        agents = {aid: build_agent(s, self.registry, self.cm) for aid, s in specs.items()}
        ledgers = {aid: self._load_ledger(aid) for aid in agents}
        for led in ledgers.values():
            self._register_positions(led)

        # data: recent history for all static instruments + reference series
        ids = sorted({i for a in agents.values() for i in a.static_instruments()} | {"EURUSD", "XSP", "VIX", "TBILL"})
        lookback = max([a.lookback_days() for a in agents.values()] + [60])
        start = str((self.asof - pd.Timedelta(days=int(lookback * 1.6) + 15)).date())
        self.store = self.data.core_store(self.instruments, ids, start)
        resolved = self.store.meta.get("resolved", {})
        option_chains = None
        if any("options" in a.extra_needs() for a in agents.values()):
            spot = self.store.close("XSP")
            rate = self.store.close("TBILL")
            option_chains = self.data.option_chains(float(spot.iloc[-1]) if len(spot) else float("nan"),
                                                    float(rate.iloc[-1]) / 100 if len(rate) else 0.0)
        fx_hist = self.store.close("EURUSD").loc[:self.asof]
        self.quotes = self.data.live_quotes(resolved, option_chains,
                                            float(fx_hist.iloc[-1]) if len(fx_hist) else None, self.previous_close)
        if option_chains is not None:
            xq = self.quotes.quote(self.registry.get("XSP"))
            if xq:
                option_chains.S = xq.mid
        self.summary["fx_eurusd"] = self.fx.eurusd
        self.summary["fx_source"] = getattr(self.quotes, "fx_source", None)

        for aid, agent in agents.items():
            led = ledgers[aid]
            try:
                self.summary["agents"][aid] = self._run_agent(agent, led, trading_allowed, option_chains)
            except Exception as exc:
                log.exception("agent %s failed", aid)
                self.summary["agents"][aid] = {"error": repr(exc), "trace": traceback.format_exc()[-2000:]}
                self.summary["errors"].append(f"{aid}: {exc!r}")
            ok, msg = led.verify()
            if not ok:
                self.summary["errors"].append(f"{aid}: ledger verification failed after run: {msg}")
                continue
            (self.state_dir / "ledgers" / f"{aid}.json").write_text(json.dumps(led.to_dict(), indent=1, default=str))

        (self.state_dir / "paper.json").write_text(json.dumps(meta, indent=1))
        self._append_jsonl(self.state_dir / "runs.jsonl", _slim_summary(self.summary))
        if self.llm_records:
            month = self.now.strftime("%Y-%m")
            for r in self.llm_records:
                self._append_jsonl(self.state_dir / "llm" / f"{month}.jsonl", r)
        return self.summary

    # ------------------------------------------------------------------ one agent
    def _run_agent(self, agent, led: Ledger, trading_allowed: bool, option_chains) -> dict:
        spec = agent.spec
        out: dict = {"name": spec.name, "calendar": spec.calendar, "notes": [], "fills": [], "rejects": []}
        is_open = self.market_open_fn(spec.calendar, self.now)
        out["market_open"] = is_open
        fx = self.fx

        self._accrue_funding(led, fx, out)
        self._credit_dividends(led, fx, out)
        self._settle_expired_options(led, fx)

        marks = self._marks(led)
        led.update_marks(marks)
        equity = led.equity(fx, marks)
        day = str(self.now.date())
        ds = led.meta.get("day_start")
        if not ds or ds.get("date") != day:
            prev = led.meta.get("last_equity")        # equity at the end of the previous run (previous day)
            led.meta["day_start"] = {"date": day, "equity": prev["equity"] if prev else equity}
        rd = riskmod.evaluate(equity, led.initial_capital, led.meta["day_start"]["equity"], spec.risk)
        out["risk"] = asdict(rd)

        orders: list[OrderIntent] = []
        ex = Executor(self.cm, self.registry, agent.venue_override)
        # margined perps: liquidate if collateral is below maintenance (crypto venues are always open)
        ms = riskmod.margin_state(led, fx, marks, self.cm.venues, self.registry)
        if ms:
            out["perp_leverage"] = round(ms["leverage"], 3) if math.isfinite(ms["leverage"]) else None
        if riskmod.needs_liquidation(ms):
            liq = riskmod.liquidate(led, ex, ExecutionQuotes(self.quotes, self.now), fx, ms, self.ts)
            out["fills"] += [f.__dict__ for f in liq]
            out["notes"].append(f"LIQUIDATED: collateral {ms['collateral_eur']:.2f} EUR vs notional {ms['notional_eur']:.2f} EUR")
            protect.cleanup(led)
            marks = self._marks(led)

        if rd.action == "halt" and not led.halted:
            led.halted, led.halt_reason = True, rd.reason
            out["notes"].append(f"KILL SWITCH: {rd.reason}; flattening")
        if led.halted and led.positions:
            # re-issued on every run until the book is flat (never ages out, survives rejected fills)
            orders = close_all(led, "kill switch: " + (led.halt_reason or ""))
            pending = []
        else:
            pending = [OrderIntent.from_dict(o) for o in led.meta.get("pending", [])]
            max_h = float(self.settings.paper.get("pending_order_max_age_hours", 24))
            pending = [o for o in pending
                       if (self.now - _utc(o.created)).total_seconds() <= max_h * 3600]

        decided = False
        if is_open and trading_allowed and not led.halted:
            extras: dict = {"sectors": self.sp500_sectors}
            if hasattr(agent, "prepare_live") and agent.decision_due(self.asof, led):
                extras.update(agent.prepare_live(self, led, self.asof) or {})
                for k in ("llm_status", "llm_calls", "headlines"):
                    if k in extras:
                        out[k] = extras[k]
            self._ensure_protect_history(led)
            ctx = self.context(self.asof, extras, option_chains)
            # (a) protective orders, every run: move trailing stops / indicator targets with the last
            #     completed daily bar, then compare stop-loss / take-profit levels with the live price
            protect.update_daily(led, ctx)
            exits = (protect.check_price(led, self._marks(led)) + protect.time_exits(led, ctx)
                     + agent.intraday(ctx, led))
            # (b) the strategy's indicator rules on completed daily bars (agents act once per day)
            new = agent.decide(ctx, led)
            decided = True
            orders = dedupe_exits(exits + new, led)
            out["notes"] += ctx.log
            if rd.action == "block_entries":
                orders = riskmod.filter_entries(orders, led)
                out["notes"].append(rd.reason + " - entries blocked")
            max_gross = spec.risk.get("max_gross")
            if max_gross and orders and riskmod.gross_ratio(led, fx, marks) >= max_gross:
                orders = riskmod.filter_entries(orders, led)
                out["notes"].append(f"gross exposure at/above {max_gross}x - entries blocked")
        orders = merge_pending(pending, orders)
        max_n = int(self.settings.paper.get("max_orders_per_agent_per_run", 25))
        if len(orders) > max_n:
            out["notes"].append(f"{len(orders)} orders exceeds cap {max_n}; truncated")
            orders = orders[:max_n]
        if orders and is_open:
            exq = ExecutionQuotes(self.quotes, self.now)
            fills, rejects = ex.execute(led, orders, exq, self.ts)
            protect.register(led, fills)
            protect.cleanup(led)
            out["fills"] += [f.__dict__ for f in fills]
            out["rejects"] = [r.__dict__ for r in rejects]
            if exq.stale:
                out["notes"] += exq.stale
            retry = [] if led.halted else self._retry_orders(orders, rejects)
            led.meta["pending"] = [o.to_dict() for o in retry]
            if retry:
                out["notes"].append(f"{len(retry)} orders without a usable quote - retried next run")
        elif orders and not led.halted:
            for o in orders:
                o.created = o.created or self.ts
            led.meta["pending"] = [o.to_dict() for o in orders]
            out["notes"].append(f"market closed: {len(orders)} orders pending")
        elif orders:
            out["notes"].append("market closed: kill-switch flatten orders will be re-issued next run")
        else:
            led.meta["pending"] = []
        out["decided"] = decided
        out["signals"] = led.meta.get("signals")
        out["resting"] = protect.resting_orders(led)

        marks = self._marks(led)
        led.update_marks(marks)
        out["equity_eur"] = round(led.equity(fx, marks), 4)
        led.meta["last_equity"] = {"date": day, "equity": out["equity_eur"]}
        out["cash_eur"] = round(led.cash_eur(fx), 4)
        out["gross_exposure"] = round(led.gross_exposure(fx, marks) / out["equity_eur"], 4) if out["equity_eur"] > 0 else None
        out["positions"] = {iid: {"qty": p.qty, "avg": p.avg_price, "mark": p.last_mark, "ccy": p.currency}
                            for iid, p in led.positions.items()}
        out["halted"] = led.halted
        out["costs_eur"] = {k: round(v, 4) for k, v in led.costs.items()}
        self._append_csv("equity.csv", [{"ts": self.ts, "agent": agent.id, "equity_eur": out["equity_eur"],
                                         "cash_eur": out["cash_eur"], "gross_exposure": out["gross_exposure"],
                                         "n_positions": len(led.positions), "halted": led.halted,
                                         "market_open": is_open}])
        self._append_csv("fills.csv", out["fills"])
        self._append_csv("rejects.csv", out["rejects"])
        return out

    RETRYABLE = ("no quote", "no executable price")

    def _retry_orders(self, orders: list[OrderIntent], rejects) -> list[OrderIntent]:
        """Orders rejected only because no usable quote was available (stale, missing, one-sided book)
        are retried on the next run, like a day order that is still working. Whole combos are retried
        together. Other rejects (cash, lot size) are final."""
        bad = {(r.inst, r.qty) for r in rejects if r.why in self.RETRYABLE}
        bad_insts = {i for i, _ in bad}
        combos = {o.combo for o in orders if o.combo and o.inst_id in bad_insts}
        out = []
        for o in orders:
            hit = (o.combo in combos) if o.combo else ((o.inst_id, o.qty) in bad)
            if hit:
                o.created = o.created or self.ts
                out.append(o)
        return out

    def _ensure_protect_history(self, led: Ledger) -> None:
        """Daily bars for protected single stocks (not in the core store) so trailing stops can move."""
        day = str(self.asof.date())
        need = [iid for iid, pr in led.meta.get("protect", {}).items()
                if (pr.get("updated") or "") < day and not self.store.has(iid)]
        if need:
            self.ensure_history(need, days=150)

    def _marks(self, led: Ledger) -> dict[str, float]:
        marks = {}
        for iid in led.positions:
            q = self.quotes.quote(self.registry.get(iid))
            if q is not None:
                marks[iid] = q.mid
        return marks

    def _accrue_funding(self, led: Ledger, fx, out: dict) -> None:
        last = led.meta.get("funding_ts", {})
        for iid, pos in list(led.positions.items()):
            if pos.kind != "perp":
                continue
            inst = self.registry.get(iid)
            since = pd.Timestamp(last.get(iid) or pos.opened)
            since = since.tz_localize("UTC") if since.tzinfo is None else since
            if iid not in self._funding_cache:
                self._funding_cache[iid] = self.data.funding_events(inst)
            ev = self._funding_cache[iid]
            q = self.quotes.quote(inst)
            if q is None:
                continue
            if len(ev):
                rate = crypto.funding_between(ev, since, self.now)
                src = "venue"
            else:
                hours = max(0.0, (self.now - since).total_seconds() / 3600)
                rate = self.fallback_funding_annual * hours / 8760
                src = "fallback"
                out["notes"].append(f"{iid}: no funding data, charged fallback {self.fallback_funding_annual:.0%}/yr")
            amount = -pos.qty * q.mid * pos.multiplier * rate
            if amount:
                led.funding(self.ts, iid, pos.currency, amount, fx, rate)
            last[iid] = self.ts
            out.setdefault("funding", {})[iid] = {"rate": rate, "amount": amount, "source": src}
        for iid in list(last):
            if iid not in led.positions:
                last.pop(iid)
        led.meta["funding_ts"] = last

    def _credit_dividends(self, led: Ledger, fx, out: dict) -> None:
        """Credit cash dividends (ex-date since the last check) on distributing ETFs and US stocks.
        US stocks: 15% withholding (Portugal-US treaty rate with a W-8BEN). Irish UCITS: none."""
        last = led.meta.get("div_checked", {})
        today = self.now.tz_convert("UTC").tz_localize(None).normalize()
        for iid, pos in list(led.positions.items()):
            if pos.kind not in ("etf", "stock") or pos.qty <= 0:
                continue
            if last.get(iid) == str(today.date()):
                continue                               # hourly runs: check each holding once a day
            since = pd.Timestamp(last.get(iid) or pos.opened[:10])
            divs = self.data.dividends(self.registry.get(iid), self.store.meta.get("resolved", {}).get(iid))
            if divs is None or divs.empty:
                last[iid] = str(today.date())
                continue
            d = divs[(divs.index > since) & (divs.index <= today)]
            wht = 0.15 if pos.kind == "stock" and pos.currency == "USD" else 0.0
            for ex_date, amt in d.items():
                net = pos.qty * float(amt) * (1 - wht)
                if net > 0:
                    led.income(self.ts, iid, pos.currency, net, fx, note=f"dividend ex {ex_date.date()} x {pos.qty:g}, wht {wht:.0%}")
                    out["notes"].append(f"{iid}: dividend {net:.2f} {pos.currency}")
            last[iid] = str(today.date())
        led.meta["div_checked"] = {k: v for k, v in last.items() if k in led.positions}

    def _settle_expired_options(self, led: Ledger, fx) -> None:
        today = self.now.tz_convert("UTC").tz_localize(None).normalize()
        for iid, pos in list(led.positions.items()):
            if pos.kind != "option":
                continue
            inst = self.registry.get(iid)
            if inst is None or pd.Timestamp(inst.expiry) >= today:
                continue
            und = self.store.close(inst.underlying).loc[:pd.Timestamp(inst.expiry)]
            if und.empty:
                continue
            S = float(und.iloc[-1])
            intrinsic = max(0.0, (inst.strike - S) if inst.right == "P" else (S - inst.strike))
            led.trade(self.ts, inst, -pos.qty, intrinsic, {}, fx, intrinsic, reason="expiry cash settlement")

    # ------------------------------------------------------------------ files
    def _append_csv(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        p = self.state_dir / name
        df = pd.DataFrame(rows)
        df.to_csv(p, mode="a", header=not p.exists(), index=False)

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _slim_summary(s: dict) -> dict:
    out = {k: v for k, v in s.items() if k != "agents"}
    out["agents"] = {}
    for aid, a in s["agents"].items():
        out["agents"][aid] = {k: a.get(k) for k in ("equity_eur", "market_open", "decided", "halted", "risk", "notes",
                                                    "llm_status", "error") if k in a}
        out["agents"][aid]["n_fills"] = len(a.get("fills", []))
        out["agents"][aid]["n_rejects"] = len(a.get("rejects", []))
    return out
