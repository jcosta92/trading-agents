"""Option pricing: Black-Scholes, a synthetic XSP chain for backtests, and live chains from Yahoo.

The synthetic chain is a MODEL: implied vol = atm_ratio * VIX plus a linear put skew, with a
spread model on top. It reproduces the mechanics and the cost drag of a put-spread strategy; it
cannot reproduce real historical option prices, so backtest results are indicative only.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

log = logging.getLogger(__name__)


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (K - S) if right == "P" else (S - K))
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    if right == "P":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        if right == "P":
            return -1.0 if K > S else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return (norm.cdf(d1) - 1.0) * math.exp(-q * T) if right == "P" else norm.cdf(d1) * math.exp(-q * T)


@dataclass
class ModelParams:
    atm_iv_ratio: float = 0.90
    skew_per_pct: float = 1.0
    dividend_yield: float = 0.013
    half_spread_min: float = 0.025
    half_spread_frac: float = 0.05

    @classmethod
    def from_agent(cls, p: dict) -> "ModelParams":
        return cls(p.get("model_atm_iv_ratio", 0.9), p.get("model_skew_per_pct", 1.0),
                   p.get("model_dividend_yield", 0.013), p.get("model_half_spread_min", 0.025),
                   p.get("model_half_spread_frac", 0.05))


class SyntheticOptionModel:
    """Prices XSP options from the XSP level, VIX and the T-bill rate on a given date."""

    def __init__(self, xsp: pd.Series, vix: pd.Series, tbill: pd.Series, params: ModelParams):
        self.xsp = xsp.dropna()
        self.vix = vix.dropna()
        self.tbill = tbill.dropna()
        self.p = params

    def _state(self, date: pd.Timestamp) -> tuple[float, float, float] | None:
        S = _asof(self.xsp, date)
        v = _asof(self.vix, date)
        r = _asof(self.tbill, date)
        if S is None or v is None:
            return None
        return S, v / 100.0, (r or 0.0) / 100.0

    def iv(self, S: float, K: float, vix: float, right: str) -> float:
        atm = self.p.atm_iv_ratio * vix
        m = math.log(K / S)
        if right == "P":
            return max(0.03, atm + self.p.skew_per_pct * max(0.0, -m))
        return max(0.5 * atm, atm - 0.3 * self.p.skew_per_pct * max(0.0, m))

    def half_spread(self, mid: float) -> float:
        return max(self.p.half_spread_min, self.p.half_spread_frac * mid)

    def quote(self, date: pd.Timestamp, expiry: str, strike: float, right: str) -> dict | None:
        st = self._state(date)
        if st is None:
            return None
        S, vix, r = st
        T = max(0.0, (pd.Timestamp(expiry) - pd.Timestamp(date).normalize()).days / 365.0)
        if T <= 0:
            intrinsic = max(0.0, (strike - S) if right == "P" else (S - strike))
            return {"mid": intrinsic, "bid": intrinsic, "ask": intrinsic, "iv": 0.0, "delta": 0.0, "S": S,
                    "expired": True}
        sigma = self.iv(S, strike, vix, right)
        mid = bs_price(S, strike, T, r, self.p.dividend_yield, sigma, right)
        hs = self.half_spread(mid)
        bid = max(0.0, mid - hs)
        return {"mid": mid, "bid": bid, "ask": mid + hs, "iv": sigma, "S": S,
                "delta": bs_delta(S, strike, T, r, self.p.dividend_yield, sigma, right), "expired": False}

    def chain(self, date: pd.Timestamp, dte_min: int, dte_max: int, right: str = "P",
              moneyness: tuple[float, float] = (0.80, 1.02)) -> pd.DataFrame:
        st = self._state(date)
        if st is None:
            return pd.DataFrame()
        S, vix, r = st
        q = self.p.dividend_yield
        d0 = pd.Timestamp(date).normalize()
        K = np.arange(math.floor(S * moneyness[0]), math.ceil(S * moneyness[1]) + 1, 1.0)
        m = np.log(K / S)
        atm = self.p.atm_iv_ratio * vix
        if right == "P":
            sig = np.maximum(0.03, atm + self.p.skew_per_pct * np.maximum(0.0, -m))
        else:
            sig = np.maximum(0.5 * atm, atm - 0.3 * self.p.skew_per_pct * np.maximum(0.0, m))
        frames = []
        for exp in fridays_between(d0 + pd.Timedelta(days=dte_min), d0 + pd.Timedelta(days=dte_max)):
            T = (exp - d0).days / 365.0
            sq = sig * math.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / sq
            d2 = d1 - sq
            if right == "P":
                mid = K * math.exp(-r * T) * ndtr(-d2) - S * math.exp(-q * T) * ndtr(-d1)
                delta = (ndtr(d1) - 1.0) * math.exp(-q * T)
            else:
                mid = S * math.exp(-q * T) * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2)
                delta = ndtr(d1) * math.exp(-q * T)
            hs = np.maximum(self.p.half_spread_min, self.p.half_spread_frac * mid)
            df = pd.DataFrame({"expiry": exp.date().isoformat(), "dte": (exp - d0).days, "strike": K, "right": right,
                               "mid": mid, "bid": np.maximum(0.0, mid - hs), "ask": mid + hs, "iv": sig,
                               "delta": delta, "S": S})
            frames.append(df[df["mid"] >= 0.01])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fridays_between(a: pd.Timestamp, b: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.date_range(a, b, freq="D")
    return [d for d in days if d.weekday() == 4]


def _asof(s: pd.Series, date: pd.Timestamp) -> float | None:
    sub = s.loc[:pd.Timestamp(date)]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


# ------------------------------------------------------------------------------------- live
class LiveOptionChains:
    """XSP chains from Yahoo (^XSP, else ^SPX scaled by 1/10), with deltas from quoted IVs."""

    def __init__(self, spot: float, rate: float, dividend_yield: float = 0.013, symbols=("^XSP", "^SPX")):
        from . import yahoo
        self.yahoo = yahoo
        self.S = spot
        self.r = rate
        self.q = dividend_yield
        self.symbols = symbols
        self._expiries: dict[str, list[str]] = {}
        self._chains: dict[tuple[str, str], pd.DataFrame] = {}
        self.source: str | None = None

    def _pick_symbol(self) -> tuple[str, float] | None:
        for sym in self.symbols:
            exps = self._expiries.get(sym)
            if exps is None:
                exps = self.yahoo.option_expiries(sym)
                self._expiries[sym] = exps
            if exps:
                return sym, (1.0 if sym == "^XSP" else 0.1)
        return None

    def expiries(self) -> list[str]:
        pick = self._pick_symbol()
        return self._expiries[pick[0]] if pick else []

    def chain(self, expiry: str) -> pd.DataFrame:
        pick = self._pick_symbol()
        if not pick:
            return pd.DataFrame()
        sym, scale = pick
        key = (sym, expiry)
        if key not in self._chains:
            df = self.yahoo.option_chain(sym, expiry)
            if not df.empty:
                df = df.copy()
                for c in ("strike", "bid", "ask", "lastPrice"):
                    df[c] = df[c].astype(float) * scale
                T = max(1e-6, (pd.Timestamp(expiry) - pd.Timestamp.now().normalize()).days / 365.0)
                df["dte"] = (pd.Timestamp(expiry) - pd.Timestamp.now().normalize()).days
                df["iv"] = df["impliedVolatility"].astype(float)
                ok = df["iv"].between(0.03, 3.0)
                df["delta"] = [bs_delta(self.S, k, T, self.r, self.q, iv, rt) if good else np.nan
                               for k, iv, rt, good in zip(df["strike"], df["iv"], df["right"], ok)]
                df["mid"] = np.where((df["bid"] > 0) & (df["ask"] >= df["bid"]), (df["bid"] + df["ask"]) / 2,
                                     df["lastPrice"])
                if scale != 1.0:
                    # SPX chain used as a stand-in for XSP: keep strikes XSP actually lists (whole points)
                    # and never assume XSP is tighter than 0.03 either side of mid
                    df = df[np.isclose(df["strike"], df["strike"].round())].copy()
                    quoted = df["bid"] > 0
                    df.loc[quoted, "bid"] = np.minimum(df.loc[quoted, "bid"], df.loc[quoted, "mid"] - 0.03).clip(lower=0)
                    df["ask"] = np.where(df["ask"] > 0, np.maximum(df["ask"], df["mid"] + 0.03), df["ask"])
                df["S"] = self.S
            self._chains[key] = df
            self.source = sym
        return self._chains[key]

    def chain_between(self, dte_min: int, dte_max: int, right: str = "P") -> pd.DataFrame:
        today = pd.Timestamp.now().normalize()
        out = []
        for e in self.expiries():
            dte = (pd.Timestamp(e) - today).days
            if dte_min <= dte <= dte_max:
                c = self.chain(e)
                if not c.empty:
                    out.append(c[c["right"] == right])
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def quote(self, expiry: str, strike: float, right: str) -> dict | None:
        c = self.chain(expiry)
        if c.empty:
            return None
        row = c[(c["right"] == right) & (np.isclose(c["strike"], strike))]
        if row.empty:
            return None
        r = row.iloc[0]
        bid = float(r["bid"]) if r["bid"] > 0 else None
        ask = float(r["ask"]) if r["ask"] > 0 else None
        mid = float(r["mid"]) if r["mid"] and r["mid"] > 0 else None
        if mid is None:
            return None
        return {"mid": mid, "bid": bid, "ask": ask, "iv": float(r["iv"]), "delta": float(r["delta"])}
