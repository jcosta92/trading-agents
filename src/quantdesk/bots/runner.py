"""One cycle of the trading bots: fetch new candles, replay them through each bot, save state."""
from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import pandas as pd

from ..config import CONFIG_DIR, Instrument, load_instruments, load_venues, load_yaml, stock_instrument
from ..costs import CostModel
from ..ledger import FxRates
from .data import TF, CandleSource
from .engine import BotEngine, Market, append_csv, load_ledger, save_ledger
from .strategies import ORBBot, build_bot

log = logging.getLogger(__name__)
DEFAULT_UNIVERSE = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "MU", "PLTR", "NFLX",
                    "ORCL", "JPM", "LLY", "COST", "WMT", "XOM", "INTC", "CRM"]


def load_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "bots.yaml")


def run_cycle(state_dir: str | Path, now: pd.Timestamp | None = None, source=None, cfg: dict | None = None,
              only: list[str] | None = None, start: str | None = None, end: str | None = None) -> dict:
    """Process every candle that closed since the last cycle. `start`/`end` override the config
    (used for the preview replay of a past month)."""
    cfg = cfg or load_config()
    state = Path(state_dir)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    t0 = pd.Timestamp(start or cfg["start"])
    t1 = pd.Timestamp(end or cfg["end"])
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1
    source = source or CandleSource()
    instruments = load_instruments()
    cm = CostModel(load_venues(), 1.0)
    summary = {"ts": now.isoformat(), "start": t0.isoformat(), "end": t1.isoformat(), "bots": {}, "errors": []}

    fx_path = state / "fx.json"
    try:
        rate = source.eurusd()
        fx_path.parent.mkdir(parents=True, exist_ok=True)
        fx_path.write_text(json.dumps({"eurusd": rate, "ts": now.isoformat()}))
    except Exception as exc:
        rate = json.loads(fx_path.read_text())["eurusd"] if fx_path.exists() else 1.16
        summary["errors"].append(f"EURUSD quote failed ({exc}); using {rate:.4f}")
    fx = FxRates(rate)
    summary["eurusd"] = rate

    fees = cfg.get("crypto_fees", {})
    markets: dict[str, Market] = {}
    for iid, m in (cfg.get("markets") or {}).items():
        inst = instruments.get(iid) or Instrument(
            id=iid, kind="crypto_spot", currency="EUR", venue="bitvavo_spot", calendar="CRYPTO", lot_size=0.0,
            half_spread_bps=float(m.get("half_spread_bps", 5)), description=f"{m['symbol']} (Bitvavo fee schedule)")
        markets[iid] = Market(inst, float(m.get("half_spread_bps", 3)) / 1e4,
                              float(m.get("slippage_bps", 3)) / 1e4, float(fees["maker"]), float(fees["taker"]))

    bots = {bid: build_bot(bid, spec) for bid, spec in cfg["bots"].items() if not only or bid in only}
    engine = BotEngine(markets, cm, fx)
    cache: dict[tuple, pd.DataFrame] = {}
    for bid, bot in bots.items():
        path = state / "ledgers" / f"{bid}.json"
        try:
            led = load_ledger(path, bid, float(cfg.get("initial_capital", 1000.0)), t0.isoformat())
            if led.meta.get("finished"):
                summary["bots"][bid] = _status(bot, led, fx, engine)
                continue
            last = pd.Timestamp(led.meta["last_ts"]) if led.meta.get("last_ts") else None
            if isinstance(bot, ORBBot):
                candles = _orb_inputs(bot, led, source, cfg, state, markets, now, t0, last)
            else:
                symbols = {k: m["symbol"] for k, m in (cfg.get("markets") or {}).items()}
                candles = _crypto_inputs(bot, source, now, t0, last, cache, symbols)
            if candles:
                engine.run(bot, led, candles, t0, t1)
            save_ledger(path, led)
            summary["bots"][bid] = _status(bot, led, fx, engine)
        except Exception as exc:
            log.exception("bot %s failed", bid)
            summary["errors"].append(f"{bid}: {exc!r}")
            summary["bots"][bid] = {"name": bot.name, "error": repr(exc), "trace": traceback.format_exc()[-1500:]}
    _record_prices(state, cache, t0)
    append_csv(state / "fills.csv", [f.__dict__ for f in engine.fills])
    append_csv(state / "equity.csv", engine.equity_rows)
    summary["data_sources"] = getattr(source, "used", {})
    summary["n_fills"] = len(engine.fills)
    (state / "status.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(state / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": summary["ts"], "n_fills": len(engine.fills), "errors": summary["errors"],
                             "equity": {b: s.get("equity_eur") for b, s in summary["bots"].items()}}) + "\n")
    return summary


def _crypto_inputs(bot, source, now, t0, last, cache, symbols) -> dict[str, pd.DataFrame]:
    tf = TF[bot.timeframe]
    first = last if last is not None else t0
    since = first - tf * (bot.warmup() + 5)
    out = {}
    for iid in bot.instruments():
        key = (iid, bot.timeframe)
        df = cache.get(key)
        if df is None or df.index.min() > since:
            df = source.crypto(symbols[iid], bot.timeframe, since, now)
            cache[key] = df
        out[iid] = df[df.index >= since]
    return out


def _orb_inputs(bot: ORBBot, led, source, cfg, state, markets, now, t0, last) -> dict[str, pd.DataFrame]:
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS")
    sched = cal.schedule.loc[(t0 - pd.Timedelta(days=40)).tz_localize(None).normalize():
                             (now + pd.Timedelta(days=1)).tz_localize(None).normalize()]
    bot.sessions = sched[["open", "close"]]
    # nothing to do unless a session has candles we have not processed yet
    started = sched[sched["open"] <= now]
    if started.empty:
        return {}
    latest_open = started["open"].iloc[-1]
    if last is not None and last >= started["close"].iloc[-1] - pd.Timedelta(minutes=5):
        return {}
    uni_p = Path(state).parent / "llm" / "universe.json"
    universe = led.meta.get("universe")
    if not universe:
        universe = json.loads(uni_p.read_text())["tickers"] if uni_p.exists() else DEFAULT_UNIVERSE
        led.meta["universe"] = universe                  # fixed for the whole test
    bot.universe = list(universe)
    stock = cfg.get("stock_costs", {})
    for t in bot.universe:
        if t not in markets:
            markets[t] = Market(stock_instrument(t), float(stock.get("half_spread_bps", 3)) / 1e4,
                                float(stock.get("slippage_bps", 2)) / 1e4)
    first = last if last is not None else t0
    since = min(first, latest_open) - pd.Timedelta(days=int(bot.p["rel_volume_days"]) * 1.6 + 3)
    bot.daily = source.stock_daily(bot.universe, int(bot.p["atr_days"]) * 2 + 10, now)
    return source.stocks(bot.universe, bot.timeframe, since, now)


def _status(bot, led, fx: FxRates, engine: BotEngine) -> dict:
    eq = led.equity(fx)
    st = led.meta.get("state", {})
    return {
        "name": bot.name, "timeframe": bot.timeframe, "equity_eur": round(eq, 4),
        "return": round(eq / led.initial_capital - 1, 6), "cash_eur": round(led.cash_eur(fx), 4),
        "positions": {iid: {"qty": p.qty, "avg": p.avg_price, "mark": p.last_mark} for iid, p in led.positions.items()},
        "open_orders": [{k: o[k] for k in ("inst", "side", "type", "price", "qty", "eur", "tag")}
                        for o in led.meta.get("orders", [])],
        "last_candle": led.meta.get("last_ts"), "finished": led.meta.get("finished"),
        "costs_eur": {k: round(v, 4) for k, v in led.costs.items()},
        "notes": engine.notes.get(bot.id, [])[-20:],
        "state": {k: v for k, v in st.items() if k in ("completed_deals", "resets", "deal", "center", "last_selection")},
    }


def _record_prices(state: Path, cache: dict, t0: pd.Timestamp) -> None:
    """Hourly closes of BTC/ETH/SOL, for the buy-and-hold reference lines on the dashboard."""
    cols = {}
    for iid in ("BTC_SPOT", "ETH_SPOT", "SOL_SPOT"):
        df = cache.get((iid, "1h"))
        if df is None and cache.get((iid, "15m")) is not None:
            df = cache[(iid, "15m")].resample("1h").agg({"close": "last"}).dropna()
        if df is not None and len(df):
            s = df["close"].copy()
            s.index = s.index + pd.Timedelta(hours=1)          # stamp at the candle's close
            cols[iid] = s
    if not cols:
        return
    p = pd.DataFrame(cols).dropna()
    p = p[p.index >= t0]
    path = state / "prices.csv"
    if path.exists():
        old = pd.read_csv(path)
        if len(old):
            last = pd.Timestamp(old["ts"].iloc[-1])
            p = p[p.index > last]
    rows = [{"ts": t.isoformat(), **{k: round(float(v), 6) for k, v in r.items()}} for t, r in p.iterrows()]
    append_csv(path, rows)
