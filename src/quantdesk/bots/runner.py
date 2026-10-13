"""One cycle of the trading bots: fetch new candles, replay them through each bot, save state."""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path

import pandas as pd

from ..config import CONFIG_DIR, Instrument, load_instruments, load_venues, load_yaml, stock_instrument
from ..costs import CostModel
from ..ledger import FxRates
from .data import TF, CandleSource
from .engine import BotEngine, Market, append_csv, load_ledger, save_ledger
from .strategies import ORBBot, UsEtfBot, build_bot

log = logging.getLogger(__name__)
DEFAULT_UNIVERSE = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "AMD", "MU", "PLTR", "NFLX",
                    "ORCL", "JPM", "LLY", "COST", "WMT", "XOM", "INTC", "CRM"]
# candles of history kept per timeframe for the signal families (long enough for EMA800 on 1h,
# 3 years of daily data for the HMM, the 360-day Donchian, etc.)
HIST = {"5m": 2500, "15m": 2500, "1h": 3000, "4h": 1500, "1d": 1500}
DAILY_DAYS = 2700                     # US daily history (all-time highs need years)


def load_config(path: str | Path | None = None, families: bool = True) -> dict:
    cfg = load_yaml(path or CONFIG_DIR / "bots.yaml")
    for bid, spec in cfg["bots"].items():
        spec.setdefault("family", bid)
    cfg["families"] = {bid: {"name": s.get("name", bid), "group": s.get("group", "popular"), "source": s.get("source", ""),
                             "rules": s.get("rules", ""), "class": s["class"], "timeframe": s.get("timeframe"),
                             "bots": [bid]} for bid, s in cfg["bots"].items()}
    cfg["group_names"] = {"popular": "Most used bot strategies", "alternative": "Less common strategies",
                          "classic": "Classic trader systems and market anomalies"}
    if families and (CONFIG_DIR / "families.yaml").exists():
        from .families_config import expand, load_families
        fc = load_families()
        bots, fams = expand(fc)
        cfg["bots"].update(bots)
        cfg["families"].update(fams)
        cfg["group_names"].update(fc.get("groups", {}))
    return cfg


class DataHub:
    """Candles for one cycle, fetched once and shared by all bots. Crypto frames are cached on disk
    between cycles of the same loop job (only new candles are downloaded)."""

    def __init__(self, source, now: pd.Timestamp, symbols: dict[str, str], cache_dir: str | None):
        self.source, self.now, self.symbols = source, now, symbols
        self.mem: dict = {}
        self.cache = Path(cache_dir) if cache_dir else None
        self.us_intraday: dict[str, pd.DataFrame] = {}
        self.us_since: pd.Timestamp | None = None
        self.us_daily: dict[str, pd.DataFrame] | None = None
        self.used = getattr(source, "used", {})

    def crypto(self, iid: str, tf: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
        key = (iid, tf)
        want = self.now - TF[tf] * HIST.get(tf, 1500)
        if since is not None:
            want = min(want, since)
        df = self.mem.get(key)
        if df is None or (len(df) and df.index.min() > want + TF[tf]):
            df = self._load(iid, tf, want)
            self.mem[key] = df
        return df if since is None else df[df.index >= since]

    def _load(self, iid, tf, want) -> pd.DataFrame:
        sym = self.symbols[iid]
        path = self.cache / f"{sym.replace('/', '_')}_{tf}.pkl" if self.cache else None
        old = None
        if path is not None and path.exists():
            try:
                old = pd.read_pickle(path)
            except Exception:
                old = None
        if old is not None and len(old) and old.index.min() <= want + TF[tf]:
            new = self.source.crypto(sym, tf, old.index.max() - 2 * TF[tf], self.now)
            df = pd.concat([old, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = self.source.crypto(sym, tf, want, self.now)
        df = df[df.index + TF[tf] <= self.now]
        if path is not None and len(df):
            path.parent.mkdir(parents=True, exist_ok=True)
            df[df.index >= self.now - TF[tf] * (HIST.get(tf, 1500) + 50)].to_pickle(path)
        return df[df.index >= want]

    def intraday(self, tickers: list[str], since: pd.Timestamp, tf: str = "5m") -> dict[str, pd.DataFrame]:
        missing = [t for t in tickers if t not in self.us_intraday]
        if missing or self.us_since is None or since < self.us_since:
            need = sorted(set(tickers) | set(self.us_intraday))
            s = min(since, self.us_since) if self.us_since is not None else since
            got = self.source.stocks(need, tf, s, self.now)
            self.us_intraday.update(got)
            for t in need:
                self.us_intraday.setdefault(t, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
            self.us_since = s
        return {t: self.us_intraday[t][self.us_intraday[t].index >= since] for t in tickers
                if t in self.us_intraday and len(self.us_intraday[t])}

    def daily(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        if self.us_daily is None:
            self.us_daily = {}
        missing = [t for t in tickers if t not in self.us_daily]
        if missing:
            got = None
            path = self.cache / f"us_daily_{self.now:%Y%m%d}_{abs(hash(tuple(sorted(missing)))) % 10**8}.pkl" \
                if self.cache else None
            if path is not None and path.exists():
                try:
                    got = pd.read_pickle(path)
                except Exception:
                    got = None
            if got is None:
                got = self.source.stock_daily(missing, DAILY_DAYS, self.now)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.to_pickle(got, path)
            for t in missing:
                self.us_daily[t] = got.get(t, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        return {t: self.us_daily[t] for t in tickers}


def run_cycle(state_dir: str | Path, now: pd.Timestamp | None = None, source=None, cfg: dict | None = None,
              only: list[str] | None = None, start: str | None = None, end: str | None = None,
              cache_dir: str | None = None) -> dict:
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
    tic = time.time()

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
    symbols = {k: m["symbol"] for k, m in (cfg.get("markets") or {}).items()}
    hub = DataHub(source, now, symbols, cache_dir if cache_dir is not None else os.environ.get("QD_CACHE"))

    capital = float(cfg.get("initial_capital", 1000.0))
    reset = restart_on_capital_change(state, capital, now)
    if reset:
        summary["capital_reset"] = reset
    bots = {bid: build_bot(bid, spec) for bid, spec in cfg["bots"].items() if not only or bid in only}
    engine = BotEngine(markets, cm, fx)
    timing = {}
    n_fills = n_eq = 0
    for bid, bot in bots.items():
        path = state / "ledgers" / f"{bid}.json"
        t_bot = time.time()
        try:
            led = load_ledger(path, bid, capital, t0.isoformat())
            if led.meta.get("finished"):
                summary["bots"][bid] = _status(bot, led, fx, engine)
                continue
            last = pd.Timestamp(led.meta["last_ts"]) if led.meta.get("last_ts") else None
            candles = _inputs(bot, led, hub, cfg, state, markets, now, t0, last, fx)
            if candles:
                engine.run(bot, led, candles, t0, t1)
            save_ledger(path, led)
            # write this bot's fills and equity right after its ledger, so a cycle cut short (time
            # limit) never leaves a saved ledger without its fills
            append_csv(state / "fills.csv", [f.__dict__ for f in engine.fills[n_fills:]])
            append_csv(state / "equity.csv", engine.equity_rows[n_eq:])
            n_fills, n_eq = len(engine.fills), len(engine.equity_rows)
            summary["bots"][bid] = _status(bot, led, fx, engine)
        except Exception as exc:
            log.exception("bot %s failed", bid)
            summary["errors"].append(f"{bid}: {exc!r}")
            summary["bots"][bid] = {"name": bot.name, "error": repr(exc), "trace": traceback.format_exc()[-1500:]}
            n_fills, n_eq = len(engine.fills), len(engine.equity_rows)   # a failed bot's partial replay is not saved
        timing[bid] = round(time.time() - t_bot, 3)
    _record_prices(state, hub, t0)
    try:
        _record_all_prices(state, hub, t0, list(markets_cfg_keys(cfg)))
    except Exception as exc:                       # reference data only; never fail the cycle for it
        log.warning("price snapshot failed: %s", exc)
    summary["data_sources"] = getattr(source, "used", {})
    summary["n_fills"] = len(engine.fills)
    summary["seconds"] = round(time.time() - tic, 1)
    summary["slowest"] = dict(sorted(timing.items(), key=lambda kv: -kv[1])[:10])
    (state / "status.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(state / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": summary["ts"], "n_fills": len(engine.fills), "errors": summary["errors"][:20],
                             "n_errors": len(summary["errors"]), "seconds": summary["seconds"]}) + "\n")
    return summary



def restart_on_capital_change(state: Path, capital: float, now: pd.Timestamp) -> dict | None:
    """When the configured starting capital changes, every ledger opened with another amount is
    discarded (with its fills and equity rows) so that bot replays from the test start with the new
    capital. Ledgers of bots no longer in the config are cleared the same way. Runs once per change;
    state/bots/capital.json records the amount in force."""
    marker = state / "capital.json"
    if marker.exists():
        try:
            if float(json.loads(marker.read_text()).get("initial_capital")) == capital:
                return None
        except (ValueError, TypeError):
            pass
    stale = []
    for path in sorted((state / "ledgers").glob("*.json")) if (state / "ledgers").exists() else []:
        try:
            old = float(json.loads(path.read_text()).get("initial_capital", 1000.0))
        except (ValueError, TypeError, OSError):
            old = None
        if old != capital:
            stale.append(path.stem)
            path.unlink()
    if stale:
        for name in ("fills.csv", "equity.csv"):
            f = state / name
            if f.exists() and f.stat().st_size:
                df = pd.read_csv(f)
                if "bot" in df.columns:
                    df[~df["bot"].isin(stale)].to_csv(f, index=False)
        log.warning("starting capital is now %.2f: %d bots restart from the test start", capital, len(stale))
    state.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"initial_capital": capital, "since": now.isoformat(), "restarted_bots": len(stale)}))
    return {"initial_capital": capital, "restarted_bots": len(stale)} if stale else None

def _inputs(bot, led, hub: DataHub, cfg, state, markets, now, t0, last, fx=None) -> dict[str, pd.DataFrame]:
    from .generic import IntradayEtfBot, SignalBot, WeightBot
    first = last if last is not None else t0
    if isinstance(bot, ORBBot):
        return _orb_inputs(bot, led, hub, cfg, state, markets, now, t0, last)
    if isinstance(bot, UsEtfBot):
        candles = _etf_inputs(bot, hub, cfg, markets, now, t0, last)
        if candles and isinstance(bot, IntradayEtfBot):
            bot.prepare_intraday(candles)
        return candles
    if isinstance(bot, SignalBot):
        iid = bot.instruments()[0]
        df = hub.crypto(iid, bot.timeframe, _hist_since(first, bot.timeframe))
        info = {tf: hub.crypto(iid, tf, _hist_since(first, tf)) for tf in bot.informative()}
        info["_from"] = first - TF[bot.timeframe] * 5
        bot.prepare(df, info)
        return {iid: df} if len(df) else {}
    if isinstance(bot, WeightBot):
        candles = {}
        for iid in bot.instruments():
            try:                                         # a coin without data drops out of the universe
                candles[iid] = hub.crypto(iid, bot.timeframe, _hist_since(first, bot.timeframe))
            except Exception as exc:
                log.warning("%s: no %s candles for %s (%s)", bot.id, bot.timeframe, iid, exc)
        candles = {k: v for k, v in candles.items() if len(v)}
        bot.from_ts = first - TF.get(bot.timeframe, pd.Timedelta(days=1)) * 5
        bot.prepare(candles)
        return candles
    tf = TF[bot.timeframe]
    since = first - tf * (bot.warmup() + 5)
    return {iid: hub.crypto(iid, bot.timeframe, since) for iid in bot.instruments()}


def _hist_since(first: pd.Timestamp, tf: str) -> pd.Timestamp:
    """Start of the history window: HIST candles before the first candle still to be processed."""
    return first - TF[tf] * HIST.get(tf, 1500)


def _sessions(t0: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    import exchange_calendars as xcals
    sched = xcals.get_calendar("XNYS").schedule.loc[(t0 - pd.Timedelta(days=40)).tz_localize(None).normalize():
                                                    (now + pd.Timedelta(days=45)).tz_localize(None).normalize()]
    return sched[["open", "close"]]          # future sessions too: calendar bots count trading days ahead


def _us_market(markets, cfg, t):
    if t not in markets:
        stock = cfg.get("stock_costs", {})
        markets[t] = Market(stock_instrument(t), float(stock.get("half_spread_bps", 3)) / 1e4,
                            float(stock.get("slippage_bps", 2)) / 1e4)


def _etf_inputs(bot: UsEtfBot, hub: DataHub, cfg, markets, now, t0, last) -> dict[str, pd.DataFrame]:
    """5-minute candles of the bot's US tickers from the open of the earliest session not yet processed."""
    bot.sessions = _sessions(t0, now)
    started = bot.sessions[bot.sessions["open"] <= now]
    if started.empty or (last is not None and last >= started["close"].iloc[-1] - pd.Timedelta(minutes=5)):
        return {}
    tickers = bot.data_tickers()
    for t in tickers:
        if not t.startswith("^"):
            _us_market(markets, cfg, t)
    first = last if last is not None else t0
    since = min(first, started["open"].iloc[-1]) - pd.Timedelta(days=1 + int(bot.spec.get("intraday_days", 0)))
    if bot.needs_daily or bot.spec.get("intraday_days"):
        bot.daily = hub.daily(tickers)
    return hub.intraday(tickers, since, bot.timeframe)


def _orb_inputs(bot: ORBBot, led, hub: DataHub, cfg, state, markets, now, t0, last) -> dict[str, pd.DataFrame]:
    bot.sessions = _sessions(t0, now)
    started = bot.sessions[bot.sessions["open"] <= now]
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
    for t in bot.universe:
        _us_market(markets, cfg, t)
    first = last if last is not None else t0
    since = min(first, latest_open) - pd.Timedelta(days=int(bot.p["rel_volume_days"]) * 1.6 + 3)
    bot.daily = hub.source.stock_daily(bot.universe, int(bot.p["atr_days"]) * 2 + 10, now)
    return hub.intraday(bot.universe, since, bot.timeframe)


def _status(bot, led, fx: FxRates, engine: BotEngine) -> dict:
    eq = led.equity(fx)
    st = led.meta.get("state", {})
    return {
        "name": bot.name, "timeframe": bot.timeframe, "equity_eur": round(eq, 4),
        "return": round(eq / led.initial_capital - 1, 6), "cash_eur": round(led.cash_eur(fx), 4),
        "positions": {iid: {"qty": p.qty, "avg": p.avg_price, "mark": p.last_mark} for iid, p in led.positions.items()},
        "open_orders": [{k: o[k] for k in ("inst", "side", "type", "price", "qty", "eur", "tag")}
                        for o in led.meta.get("orders", [])][:20],
        "last_candle": led.meta.get("last_ts"), "finished": led.meta.get("finished"),
        "costs_eur": {k: round(v, 4) for k, v in led.costs.items()},
        "notes": engine.notes.get(bot.id, [])[-10:],
        "state": {k: v for k, v in st.items() if k in ("completed_deals", "resets", "deal", "center", "last_selection",
                                                       "targets", "rebalances", "switches", "holding", "last_ibs",
                                                       "signal", "day_position", "last_target", "steps", "trails",
                                                       "last_quote")},
    }


def markets_cfg_keys(cfg) -> list[str]:
    return list((cfg.get("markets") or {}).keys())


def _record_all_prices(state: Path, hub: DataHub, t0: pd.Timestamp, coins: list[str]) -> None:
    """Hourly closes of every coin (for per-market buy-and-hold comparisons in the reviews) and
    daily closes of every US ticker the bots use."""
    cols = {}
    for iid in coins:
        for tf in ("1h", "15m", "5m"):
            df = hub.mem.get((iid, tf))
            if df is not None and len(df):
                s = df["close"].resample("1h").last().dropna() if tf != "1h" else df["close"]
                s = s.copy()
                s.index = s.index + pd.Timedelta(hours=1)
                cols[iid] = s
                break
    if cols:
        p = pd.DataFrame(cols).reindex(columns=coins)
        p = p[p.index >= t0]
        path = state / "prices_all.csv"
        if path.exists():
            old = pd.read_csv(path, usecols=["ts"])
            if len(old):
                p = p[p.index > pd.Timestamp(old["ts"].iloc[-1])]
        p = p.dropna(how="all")
        append_csv(path, [{"ts": t.isoformat(), **{k: (round(float(v), 8) if pd.notna(v) else "") for k, v in r.items()}}
                          for t, r in p.iterrows()])
    if hub.us_daily:
        start = (t0 - pd.Timedelta(days=7)).tz_localize(None)
        wide = pd.DataFrame({t: d["close"] for t, d in hub.us_daily.items() if d is not None and len(d)})
        if len(wide):
            path = state / "us_closes.csv"
            if path.exists():
                old = pd.read_csv(path, index_col=0, parse_dates=True)
                wide = old.combine_first(wide)
            wide[wide.index >= start].sort_index().to_csv(path)


def _record_prices(state: Path, hub: DataHub, t0: pd.Timestamp) -> None:
    """Hourly closes of BTC/ETH/SOL, for the buy-and-hold reference lines on the dashboard."""
    cols = {}
    for iid in ("BTC_SPOT", "ETH_SPOT", "SOL_SPOT"):
        df = hub.mem.get((iid, "1h"))
        if df is None and hub.mem.get((iid, "15m")) is not None:
            df = hub.mem[(iid, "15m")].resample("1h").agg({"close": "last"}).dropna()
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
