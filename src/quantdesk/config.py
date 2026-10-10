"""Configuration loading: instruments, venues and pre-registered agent specs."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Instrument:
    id: str
    kind: str                      # etf | stock | crypto_spot | perp | option | index | fx
    currency: str
    venue: str
    calendar: str                  # XETR | XNYS | CRYPTO
    lot_size: float = 0.0          # 1 = whole units, 0 = fractional
    half_spread_bps: float = 5.0
    yf: tuple[str, ...] = ()
    proxy: dict | None = None
    ccxt: dict | None = None
    funding: tuple[dict, ...] = ()
    multiplier: float = 1.0
    price_scale: float = 1.0
    option_chain_yf: tuple[str, ...] = ()
    description: str = ""
    # options
    underlying: str | None = None
    expiry: str | None = None      # ISO date
    strike: float | None = None
    right: str | None = None       # 'P' or 'C'
    half_spread_abs: float | None = None   # options: absolute half spread fallback
    sector: str | None = None

    @property
    def is_option(self) -> bool:
        return self.kind == "option"

    @property
    def is_perp(self) -> bool:
        return self.kind == "perp"


def _inst_from_dict(iid: str, d: dict) -> Instrument:
    d = dict(d)
    for key in ("yf", "option_chain_yf"):
        if key in d and d[key] is not None:
            d[key] = tuple(d[key])
    if "funding" in d and d["funding"] is not None:
        d["funding"] = tuple(d["funding"])
    return Instrument(id=iid, **d)


def load_instruments(path: str | Path | None = None) -> dict[str, Instrument]:
    raw = load_yaml(path or CONFIG_DIR / "instruments.yaml")
    return {iid: _inst_from_dict(iid, spec) for iid, spec in raw.items()}


def load_venues(path: str | Path | None = None) -> dict[str, dict]:
    raw = load_yaml(path or CONFIG_DIR / "venues.yaml")
    for v in raw.values():
        for f in v.get("fees", []) or []:
            if "rate_by_premium" in f:
                f["rate_by_premium"] = [
                    [math.inf if (isinstance(b, str) and b.lower() in (".inf", "inf")) else float(b), float(r)]
                    for b, r in f["rate_by_premium"]
                ]
    return raw


def stock_instrument(ticker: str, sector: str | None = None) -> Instrument:
    """Dynamic US stock traded at IBKR (fractional shares allowed)."""
    return Instrument(
        id=ticker, kind="stock", currency="USD", venue="ibkr_us_stock", calendar="XNYS",
        lot_size=0.0, half_spread_bps=3.0, yf=(ticker,), description=f"US stock {ticker}", sector=sector,
    )


def option_id(underlying: str, expiry: str, right: str, strike: float) -> str:
    return f"{underlying}:{expiry}:{right}:{strike:g}"


def option_instrument(underlying: Instrument, expiry: str, right: str, strike: float,
                      half_spread_abs: float = 0.03) -> Instrument:
    return Instrument(
        id=option_id(underlying.id, expiry, right, strike), kind="option", currency=underlying.currency,
        venue="ibkr_us_option", calendar=underlying.calendar, lot_size=1.0, half_spread_bps=0.0,
        multiplier=100.0, description=f"{underlying.id} {expiry} {strike:g}{right}",
        underlying=underlying.id, expiry=expiry, strike=float(strike), right=right,
        half_spread_abs=half_spread_abs,
    )


def parse_option_id(oid: str) -> tuple[str, str, str, float]:
    und, expiry, right, strike = oid.split(":")
    return und, expiry, right, float(strike)


@dataclass
class AgentSpec:
    id: str
    name: str
    cls: str
    calendar: str
    params: dict
    risk: dict = field(default_factory=dict)
    venue_override: str | None = None
    backtest_start: str | None = None
    backtest: bool = True
    paper: bool = True
    thesis: str = ""
    backtest_note: str = ""
    variants: dict = field(default_factory=dict)
    variant: str = "main"

    def with_variant(self, name: str) -> "AgentSpec":
        if name == "main":
            return self
        params = copy.deepcopy(self.params)
        params.update(self.variants[name])
        return replace(self, params=params, variant=name)


@dataclass
class Settings:
    base_currency: str
    initial_capital: float
    backtest: dict
    paper: dict
    agents: dict[str, AgentSpec]


def load_settings(path: str | Path | None = None) -> Settings:
    raw = load_yaml(path or CONFIG_DIR / "agents.yaml")
    agents = {}
    for aid, a in raw["agents"].items():
        agents[aid] = AgentSpec(
            id=aid, name=a.get("name", aid), cls=a["class"], calendar=a["calendar"],
            params=a.get("params", {}) or {}, risk=a.get("risk", {}) or {},
            venue_override=a.get("venue_override"), backtest_start=a.get("backtest_start"),
            backtest=a.get("backtest", True), paper=a.get("paper", True),
            thesis=(a.get("thesis") or "").strip(), backtest_note=(a.get("backtest_note") or "").strip(),
            variants=a.get("variants", {}) or {},
        )
    return Settings(
        base_currency=raw.get("base_currency", "EUR"), initial_capital=float(raw.get("initial_capital", 1000.0)),
        backtest=raw.get("backtest", {}), paper=raw.get("paper", {}), agents=agents,
    )
