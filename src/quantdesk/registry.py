"""Instrument registry: static instruments from config plus dynamically created stocks/options."""
from __future__ import annotations

from .config import Instrument, option_instrument, parse_option_id, stock_instrument


class Registry:
    def __init__(self, instruments: dict[str, Instrument]):
        self._inst = dict(instruments)

    def get(self, iid: str) -> Instrument | None:
        inst = self._inst.get(iid)
        if inst is not None:
            return inst
        if iid.count(":") == 3:                       # option id: UND:EXPIRY:RIGHT:STRIKE
            und, expiry, right, strike = parse_option_id(iid)
            if und in self._inst:
                inst = option_instrument(self._inst[und], expiry, right, strike)
                self._inst[iid] = inst
                return inst
        return None

    def add(self, inst: Instrument) -> Instrument:
        self._inst.setdefault(inst.id, inst)
        return self._inst[inst.id]

    def stock(self, ticker: str, sector: str | None = None) -> Instrument:
        return self.add(stock_instrument(ticker, sector))

    def option(self, underlying: str, expiry: str, right: str, strike: float, half_spread_abs: float = 0.03) -> Instrument:
        inst = option_instrument(self._inst[underlying], expiry, right, strike, half_spread_abs)
        return self.add(inst)

    def __contains__(self, iid: str) -> bool:
        return self.get(iid) is not None

    def all(self) -> dict[str, Instrument]:
        return dict(self._inst)
