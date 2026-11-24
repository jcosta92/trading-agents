"""Tactical asset allocation, risk-based portfolios and published calendar / macro anomalies on US
ETFs, plus five crypto cross-section papers (added 24 Nov 2026, replayed from the 12 Oct start).

US ETF rules (EtfWeightBot) are decided 5 minutes before the close on the day's price so far and
only fill the last row. Months are 21 trading days, as elsewhere in this package. Crypto rules
(WeightBot, daily candles) are causal over the whole history; costly ones are computed only on
their rebalance rows (and from the first candle still to replay) and carried forward.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ..generic import weights
from ..lib import sma
from .etf import _best, _last_only, _mom13612, _sma12m
from .etf_taa import min_variance

M1, M3, M6, M12 = 21, 63, 126, 252


def _r(c, n):
    return c / c.shift(n) - 1


def _sma10m(c):
    """Price / average of the last 10 month-end prices (21-day steps) - 1 (Faber's 10-month SMA)."""
    return c / (sum(c.shift(21 * k) for k in range(10)) / 10) - 1


def _ok(row, names) -> bool:
    return not row.reindex(names).isna().any()


def _cov(c: pd.DataFrame, names, n: int, i: int) -> np.ndarray | None:
    r = c[names].pct_change().iloc[max(1, i - n + 1):i + 1]
    if len(r) < n * 0.9 or r.isna().any().any():
        return None
    return r.cov().to_numpy() * 252


def _next_session(extra) -> pd.Timestamp | None:
    idx = extra["sessions"].index
    d = pd.Timestamp(extra["day"])
    nxt = idx[idx > d]
    return nxt[0] if len(nxt) else None


# ============================================================================== Keller & co.
@weights("taa2_gtaa_agg")
def gtaa_agg(panel, p, extra):
    """Faber (2010) GTAA Aggressive: the top k of 13 asset-class ETFs by the average of their 1, 3,
    6 and 12-month returns; a pick below its 10-month SMA goes to T-bills (BIL)."""
    c = panel.close
    uni = ["IWD", "MTUM", "IWN", "DWAS", "EFA", "EEM", "IEF", "BWX", "LQD", "TLT", "DBC", "GLD", "VNQ"]
    k = int(p.get("top", 3))
    score = (_r(c, M1) + _r(c, M3) + _r(c, M6) + _r(c, M12)) / 4
    trend = _sma10m(c)

    def f(i):
        row = score.iloc[i]
        if not _ok(row, uni) or trend.iloc[i][uni].isna().any():
            return {}
        out = {}
        for a in _best(row[uni], k):
            t = a if trend.iloc[i][a] > 0 else "BIL"
            out[t] = out.get(t, 0) + 1 / k
        return out
    return _last_only(panel, f)


@weights("taa2_gpm")
def gpm(panel, p, extra):
    """Keller & Keuning (2016) Generalized Protective Momentum: score z = r * (1 - c), r the
    average 1/3/6/12-month return, c the 12-month correlation with the equal-weight universe; top 3
    risky by z; bond fraction (12 - n) / 6 (n = risky with z > 0, capped at 100%) in the better of
    IEF / BIL."""
    c = panel.close
    risky = ["SPY", "QQQ", "IWM", "VGK", "EWJ", "EEM", "VNQ", "DBC", "GLD", "HYG", "LQD", "TLT"]
    cash = ["IEF", "BIL"]
    r = (_r(c, M1) + _r(c, M3) + _r(c, M6) + _r(c, M12)) / 4

    def f(i):
        row = r.iloc[i]
        if not _ok(row, risky + cash) or i < M12:
            return {}
        rets = c[risky].pct_change().iloc[i - M12 + 1:i + 1]
        ew = rets.mean(axis=1)
        corr = rets.apply(lambda s: s.corr(ew))
        z = row[risky] * (1 - corr)
        n = int((z > 0).sum())
        bf = min(1.0, max(0.0, (12 - n) / 6))
        out = {}
        if bf < 1:
            for a in _best(z, 3):
                out[a] = (1 - bf) / 3
        if bf > 0:
            b = _best(row[cash])[0]
            out[b] = out.get(b, 0) + bf
        return out
    return _last_only(panel, f)


@weights("taa2_eaa")
def eaa(panel, p, extra):
    """Keller & Butler (2014) Elastic Asset Allocation with Kipnis's defaults (wR=1, wC=0.5, wV=0,
    wS=2): z = (r^wR (1-c)^wC / v^wV)^(wS+eps) for assets with r > 0, r the average 1/3/6/12-month
    return, c the 12-month correlation with the equal-weight universe; top 1+ceil(sqrt(N)) by z,
    weighted by z; crash protection: the share of assets with r <= 0 goes to the cash asset (SHY)."""
    c = panel.close
    uni = list(p.get("assets", ["VTI", "EFA", "VWO", "SHY", "BND", "DBC", "VNQ"]))
    cash = p.get("cash", "SHY")
    wR, wC, wV, wS = (float(p.get(k, d)) for k, d in (("wR", 1), ("wC", 0.5), ("wV", 0), ("wS", 2)))
    r = (_r(c, M1) + _r(c, M3) + _r(c, M6) + _r(c, M12)) / 4
    top = 1 + math.ceil(math.sqrt(len(uni)))

    def f(i):
        row = r.iloc[i]
        if not _ok(row, uni) or i < M12:
            return {}
        rets = c[uni].pct_change().iloc[i - M12 + 1:i + 1]
        ew = rets.mean(axis=1)
        corr = rets.apply(lambda s: s.corr(ew))
        vol = rets.std() * math.sqrt(252)
        rr = row[uni]
        z = pd.Series(0.0, index=uni)
        pos = rr > 0
        z[pos] = ((rr[pos] ** wR) * ((1 - corr[pos]).clip(lower=0) ** wC) / (vol[pos] ** wV)) ** (wS + 1e-6)
        cp = float((rr <= 0).sum()) / len(uni)
        sel = [a for a in _best(z[z > 0], top)]
        out = {}
        tot = float(z[sel].sum()) if sel else 0.0
        for a in sel:
            out[a] = (1 - cp) * float(z[a]) / tot
        if cp > 0 or not sel:
            out[cash] = out.get(cash, 0) + (cp if sel else 1.0)
        return out
    return _last_only(panel, f)


@weights("taa2_raa")
def raa(panel, p, extra):
    """Alpha Architect Robust Asset Allocation, balanced (Gray & Vogel 2015), on plain index ETFs:
    20% US stocks, 20% international, 10% REITs, 10% commodities, 40% 10-year Treasuries. Each
    sleeve: half held if the price is above its 12-month average, half if its 12-month return beats
    T-bills (BIL); the rest in BIL."""
    c = panel.close
    sleeves = dict(p.get("sleeves", {"VTI": 0.2, "EFA": 0.2, "VNQ": 0.1, "DBC": 0.1, "IEF": 0.4}))
    ma = c / c.rolling(M12).mean() - 1
    mom = _r(c, M12)

    def f(i):
        if not _ok(mom.iloc[i], list(sleeves) + ["BIL"]) or ma.iloc[i][list(sleeves)].isna().any():
            return {}
        out = {}
        for a, w in sleeves.items():
            k = 0.5 * (ma.iloc[i][a] > 0) + 0.5 * (mom.iloc[i][a] > mom.iloc[i]["BIL"])
            out[a] = out.get(a, 0) + w * k
            out["BIL"] = out.get("BIL", 0) + w * (1 - k)
        return out
    return _last_only(panel, f)


@weights("taa2_paired_switching")
def paired_switching(panel, p, extra):
    """Maewal & Bock (2011) Paired Switching: each quarter hold whichever of SPY and TLT returned
    more over the previous quarter (63 trading days)."""
    c = panel.close
    m = _r(c, M3)

    def f(i):
        row = m.iloc[i]
        if not _ok(row, ["SPY", "TLT"]):
            return {}
        return {_best(row[["SPY", "TLT"]])[0]: 1.0}
    return _last_only(panel, f)


@weights("taa2_quint_switching")
def quint_switching(panel, p, extra):
    """Quint Switching, filtered (Logical Invest / CXO test): month-end, hold the one of QQQ, EEM,
    IWM, EFA and TLT with the best 3-month return; if that return is negative, hold SHY."""
    c = panel.close
    uni = ["QQQ", "EEM", "IWM", "EFA", "TLT"]
    m = _r(c, M3)

    def f(i):
        row = m.iloc[i]
        if not _ok(row, uni):
            return {}
        b = _best(row[uni])[0]
        return {b if row[b] > 0 else "SHY": 1.0}
    return _last_only(panel, f)


@weights("taa2_vaa_g12")
def vaa_g12(panel, p, extra):
    """Keller & Keuning (2017) VAA-G12: 13612W momentum; b = number of the 12 risky assets with
    negative momentum; cash share min(1, b/4) in the best of SHY / IEF / LQD; the rest equally in the
    top 2 risky assets."""
    m = _mom13612(panel.close)
    risky = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "EEM", "VNQ", "DBC", "GLD", "TLT", "LQD", "HYG"]
    cash = ["SHY", "IEF", "LQD"]

    def f(i):
        row = m.iloc[i]
        if not _ok(row, risky + cash):
            return {}
        b = int((row[risky] < 0).sum())
        cf = min(1.0, b / 4)
        out = {}
        if cf < 1:
            for a in _best(row[risky], 2):
                out[a] = (1 - cf) / 2
        if cf > 0:
            a = _best(row[cash])[0]
            out[a] = out.get(a, 0) + cf
        return out
    return _last_only(panel, f)


@weights("taa2_baa_g12")
def baa_g12(panel, p, extra):
    """Keller (2022) Bold Asset Allocation, balanced G12: canaries SPY/VWO/VEA/BND (13612W); all
    positive -> top 6 of 12 offensive ETFs by SMA(12) momentum, else the top 3 of the defensive set
    (TIP, DBC, BIL, IEF, TLT, LQD, BND), a pick worse than BIL replaced by BIL."""
    c = panel.close
    m = _mom13612(c)
    s12 = _sma12m(c)
    canary = ["SPY", "VWO", "VEA", "BND"]
    off = ["SPY", "QQQ", "IWM", "VGK", "EWJ", "VWO", "VNQ", "DBC", "GLD", "TLT", "HYG", "LQD"]
    deff = ["TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND"]

    def f(i):
        row = m.iloc[i]
        srow = s12.iloc[i]
        if not _ok(row, canary) or not _ok(srow, off + deff):
            return {}
        out = {}
        if (row[canary] < 0).any():
            for a in _best(srow[deff], 3):
                t = a if srow[a] >= srow["BIL"] else "BIL"
                out[t] = out.get(t, 0) + 1 / 3
            return out
        for a in _best(srow[off], 6):
            out[a] = out.get(a, 0) + 1 / 6
        return out
    return _last_only(panel, f)


@weights("taa2_haa_simple")
def haa_simple(panel, p, extra):
    """Keller & Keuning (2023) HAA-Simple: 13612U momentum; canary TIP <= 0 or SPY <= 0 -> the
    better of BIL / IEF, otherwise SPY."""
    c = panel.close
    m = (_r(c, M1) + _r(c, M3) + _r(c, M6) + _r(c, M12)) / 4

    def f(i):
        row = m.iloc[i]
        if not _ok(row, ["TIP", "SPY", "BIL", "IEF"]):
            return {}
        if row["TIP"] <= 0 or row["SPY"] <= 0:
            return {_best(row[["BIL", "IEF"]])[0]: 1.0}
        return {"SPY": 1.0}
    return _last_only(panel, f)


@weights("taa2_uis")
def uis(panel, p, extra):
    """Logical Invest Universal Investment Strategy (Kipnis replication): month-end, the SPY/TLT mix
    (0-100% in 10% steps) with the best modified Sharpe ratio over the last 72 trading days:
    annualised return / annualised volatility^2.5."""
    c = panel.close
    n, f_ = int(p.get("lookback", 72)), float(p.get("vol_factor", 2.5))

    def f(i):
        if i < n or c[["SPY", "TLT"]].iloc[i - n:i + 1].isna().any().any():
            return {}
        r = c[["SPY", "TLT"]].pct_change().iloc[i - n + 1:i + 1]
        best, bw = -np.inf, 0.5
        for w in np.linspace(0, 1, 11):
            pr = w * r["SPY"] + (1 - w) * r["TLT"]
            ann = (1 + pr).prod() ** (252 / len(pr)) - 1
            vol = pr.std() * math.sqrt(252)
            s = ann / vol ** f_ if vol > 0 else -np.inf
            if s > best:
                best, bw = s, w
        return {"SPY": bw, "TLT": 1 - bw}
    return _last_only(panel, f)


# ============================================================================== cross-sections of ETFs
@weights("taa2_xs_top")
def xs_top(panel, p, extra):
    """Cross-sectional rotation: hold the top (or bottom) k of a universe by its n-day return,
    equal weight. Used by country momentum, country reversal, asset-class, commodity, currency
    and style momentum."""
    c = panel.close
    uni = [a for a in p["universe"] if a in c]
    n, k = int(p.get("days", M12)), int(p.get("top", 3))
    skip = int(p.get("skip", 0))
    score = c[uni].shift(skip) / c[uni].shift(n) - 1

    def f(i):
        row = score.iloc[i]
        if row.isna().sum() > len(uni) // 4:
            return {}
        row = row.dropna()
        pick = list(row.sort_values(ascending=bool(p.get("lowest", False))).index[:k])
        return {a: 1 / k for a in pick}
    return _last_only(panel, f)


@weights("taa2_country_low_beta")
def country_low_beta(panel, p, extra):
    """Beta factor in country equity indexes (Quantpedia / QuantConnect library), long leg: each
    month hold the quintile of country ETFs with the lowest 1-year beta to the world index (ACWI)."""
    c = panel.close
    uni = [a for a in p["universe"] if a in c and a != "ACWI"]
    k = int(p.get("top", 4))

    def f(i):
        if i < M12 or "ACWI" not in c:
            return {}
        r = c[uni + ["ACWI"]].pct_change().iloc[i - M12 + 1:i + 1]
        mkt = r["ACWI"]
        if mkt.isna().any():
            return {}
        var = mkt.var()
        beta = r[uni].apply(lambda s: s.cov(mkt) / var).dropna()
        pick = list(beta.sort_values().index[:k])
        return {a: 1 / k for a in pick}
    return _last_only(panel, f)


@weights("taa2_tsmom")
def tsmom(panel, p, extra):
    """Time-series momentum, long only (Moskowitz, Ooi & Pedersen 2012): hold each asset whose
    12-month return is positive, sized 40% / its ex-ante volatility / N (EWMA of squared daily
    returns, centre of mass 60 days); total capped at 100%. With `lookbacks` [21, 63, 252] the
    signal is the average of the three signs (Hurst, Ooi & Pedersen 2017)."""
    c = panel.close
    uni = [a for a in p["universe"] if a in c]
    lbs = list(p.get("lookbacks", [M12]))
    r = c[uni].pct_change()
    vol = np.sqrt((r ** 2).ewm(com=60, min_periods=60).mean() * 261)

    def f(i):
        sig = sum(np.sign(_r(c[uni], n).iloc[i]) for n in lbs) / len(lbs)
        v = vol.iloc[i]
        if sig.isna().any() or v.isna().any():
            return {}
        w = (sig.clip(lower=0) * float(p.get("target", 0.40)) / v / len(uni)).fillna(0)
        tot = float(w.sum())
        if tot > 1:
            w = w / tot
        return w[w > 0].to_dict()
    return _last_only(panel, f)


def _erc(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution weights (long only, sum 1), Spinu's convex formulation."""
    n = len(cov)
    x = np.ones(n) / np.sqrt(np.diag(cov))
    x /= x.sum()
    for _ in range(200):                                  # cyclical coordinate descent
        for j in range(n):
            a = cov[j, j]
            b = cov[j] @ x - a * x[j]
            x[j] = (-b + math.sqrt(b * b + 4 * a / n)) / (2 * a)
    return x / x.sum()


def _max_div(cov: np.ndarray) -> np.ndarray:
    """Choueifaty & Coignard most-diversified portfolio: maximise w.sigma / sqrt(w' S w), long only."""
    n = len(cov)
    sd = np.sqrt(np.diag(cov))
    covn = cov / np.outer(sd, sd)                       # correlation: solve min w'Cw, then w_i / sd_i
    w = min_variance(covn)
    w = w / sd
    return w / w.sum()


def _min_corr(c: pd.DataFrame, names, n: int, i: int) -> np.ndarray | None:
    """Varadi, Kapler, Bee & Rittenhouse (2012) Minimum Correlation Algorithm (MCA, the 'mincorr'
    version with inverse-volatility scaling)."""
    r = c[names].pct_change().iloc[max(1, i - n + 1):i + 1]
    if len(r) < n * 0.9 or r.isna().any().any():
        return None
    cor = r.corr().to_numpy()
    up = cor[np.triu_indices(len(names), 1)]
    m, s = up.mean(), up.std()
    nd = 1 - norm.cdf(cor, m, s if s > 0 else 1e-9)
    np.fill_diagonal(nd, 0)
    avg = nd.mean(axis=1)
    rk = pd.Series(avg).rank().to_numpy()
    rk = rk / rk.sum()
    wt = (nd * rk[None, :]).sum(axis=1)                  # each pair's distance weighted by the rank share
    wt = wt / wt.sum()
    wt = wt / r.std().to_numpy()
    return wt / wt.sum()


@weights("taa2_risk_based")
def risk_based(panel, p, extra):
    """Risk-based portfolios on asset-class ETFs from the last n daily returns: `method` inverse_vol
    (Asness, Frazzini & Pedersen 2012, unlevered), erc (Maillard, Roncalli & Teiletche 2010),
    max_div (Choueifaty & Coignard 2008), min_var (Clarke, de Silva & Thorley 2006) or min_corr
    (Varadi et al. 2012). With trend: true only assets above their 10-month SMA are held, the rest
    of the capital in BIL (Clare, Seaton, Smith & Thomas 2016)."""
    c = panel.close
    uni = [a for a in p["assets"] if a in c]
    n = int(p.get("days", M12))
    method = p.get("method", "inverse_vol")
    trend = _sma10m(c) if p.get("trend") else None

    def f(i):
        if i < n:
            return {}
        if method == "min_corr":
            w = _min_corr(c, uni, n, i)
        else:
            cov = _cov(c, uni, n, i)
            if cov is None:
                return {}
            if method == "inverse_vol":
                w = 1 / np.sqrt(np.diag(cov))
                w = w / w.sum()
            elif method == "erc":
                w = _erc(cov)
            elif method == "max_div":
                w = _max_div(cov)
            else:
                w = min_variance(cov)
        if w is None:
            return {}
        out = dict(zip(uni, map(float, w)))
        if trend is not None:
            tr = trend.iloc[i]
            if tr[uni].isna().any():
                return {}
            off = sum(v for a, v in out.items() if tr[a] <= 0)
            out = {a: v for a, v in out.items() if tr[a] > 0}
            if off > 0:
                out["BIL"] = out.get("BIL", 0) + off
        return out
    return _last_only(panel, f)


@weights("taa2_fixed")
def fixed(panel, p, extra):
    """Fixed-weight benchmark portfolios of allocation research (60/40; 1/N of DeMiguel, Garlappi &
    Uppal 2009), rebalanced on the family's schedule."""
    w = dict(p["weights"]) if "weights" in p else {a: 1 / len(p["universe"]) for a in p["universe"]}
    return _last_only(panel, lambda i: w)


# ============================================================================== timing rules
@weights("taa2_crude_oil")
def crude_oil(panel, p, extra):
    """Crude oil predicts equity returns (Driesprong, Jacobsen & Maat 2008; Quantpedia / QuantConnect
    library): each month-end regress the S&P 500's monthly excess return on the previous month's
    crude-oil return over all months so far; if the forecast for next month (from this month's oil
    return) beats T-bills hold SPY, else BIL."""
    c = panel.close
    if not {"SPY", "CL=F", "^IRX"} <= set(c.columns):
        return _last_only(panel, lambda i: {})
    m = c[["SPY", "CL=F", "^IRX"]].dropna()
    me = m.groupby(m.index.to_period("M")).tail(1)
    spy = me["SPY"].pct_change()
    oil = me["CL=F"].pct_change()
    rf = me["^IRX"] / 100 / 12

    def f(i):
        x = oil.shift(1).iloc[1:]                         # last month's oil return -> this month's SPY
        y = (spy - rf.shift(1)).iloc[1:]
        d = pd.concat([x, y], axis=1).dropna()
        if len(d) < 24 or pd.isna(oil.iloc[-1]):
            return {}
        b, a = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
        fc = a + b * float(oil.iloc[-1])
        return {"SPY": 1.0} if fc > 0 else {"BIL": 1.0}
    return _last_only(panel, f)


@weights("taa2_yield_curve")
def yield_curve(panel, p, extra):
    """Yield-curve timing (Resnick & Shoesmith 2002; Quantpedia 'Avoid equity bear markets' part 3):
    hold SPY while the 10-year minus 3-month Treasury spread is positive, T-bills (BIL) when the
    curve is inverted."""
    c = panel.close

    def f(i):
        t, b = c["^TNX"].iloc[i], c["^IRX"].iloc[i]
        if pd.isna(t) or pd.isna(b):
            t, b = c["^TNX"].ffill().iloc[i], c["^IRX"].ffill().iloc[i]
        if pd.isna(t) or pd.isna(b):
            return {}
        return {"SPY": 1.0} if t - b > 0 else {"BIL": 1.0}
    return _last_only(panel, f)


@weights("taa2_ivy_timing")
def ivy_timing(panel, p, extra):
    """Faber (2007) / The Ivy Portfolio: equal weight in each asset class, each held only while its
    month-end price is above its 10-month SMA, otherwise that share in T-bills (BIL)."""
    c = panel.close
    uni = list(p["assets"])
    tr = _sma10m(c)

    def f(i):
        row = tr.iloc[i]
        if not _ok(row, uni):
            return {}
        out = {}
        for a in uni:
            t = a if row[a] > 0 else "BIL"
            out[t] = out.get(t, 0) + 1 / len(uni)
        return out
    return _last_only(panel, f)


@weights("taa2_calendar")
def calendar(panel, p, extra):
    """Calendar rules decided on the session before: `rule` is
      weekend      - SPY except over the weekend: sold on the week's last session, bought back on the
                     next (French 1980)
      gold_autumn  - GLD in September and November only (Baur 2013)
      january_small - IWM in January, SPY the rest of the year (QuantConnect 'January effect')
      january_barometer - SPY in January; Feb-Dec SPY if January's return was positive, else BIL
                     (Cooper, McConnell & Ovtchinnikov 2006)"""
    rule = p["rule"]
    c = panel.close
    nxt = _next_session(extra)
    d = pd.Timestamp(extra["day"])

    def f(i):
        if nxt is None:
            return {}
        if rule == "weekend":
            return {} if tuple(nxt.isocalendar())[:2] != tuple(d.isocalendar())[:2] else {"SPY": 1.0}
        if rule == "gold_autumn":
            return {"GLD": 1.0} if nxt.month in (9, 11) else {}
        if rule == "january_small":
            return {"IWM": 1.0} if nxt.month == 1 else {"SPY": 1.0}
        if rule == "january_barometer":
            if nxt.month == 1:
                return {"SPY": 1.0}
            s = c["SPY"].dropna()
            yr = nxt.year
            dec = s[s.index < pd.Timestamp(yr, 1, 1)]
            jan = s[(s.index >= pd.Timestamp(yr, 1, 1)) & (s.index < pd.Timestamp(yr, 2, 1))]
            if not len(dec) or not len(jan):
                return {}
            return {"SPY": 1.0} if jan.iloc[-1] > dec.iloc[-1] else {"BIL": 1.0}
        return {}
    return _last_only(panel, f)


def _macd(s, fast=12, slow=26, sig=9):
    line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return line, line.ewm(span=sig, adjust=False).mean()


@weights("taa2_sy_harding")
def sy_harding(panel, p, extra):
    """Sy Harding's Seasonal Timing Strategy: the best six months with MACD(12, 26, 9) entry/exit:
    buy SPY on the first day on or after 16 Oct when the MACD line is above its signal line; sell on
    the first day on or after 20 Apr when it is below; T-bills (BIL) in between."""
    s = panel.close["SPY"].dropna()
    line, sig = _macd(s)
    held = False
    for t, lv, sv in zip(s.index, line.to_numpy(), sig.to_numpy()):
        winter = (t.month, t.day) >= (10, 16) or (t.month, t.day) < (4, 20)
        if winter and not held and lv > sv:
            held = True
        elif not winter and held and lv < sv:
            held = False
    return _last_only(panel, lambda i: {"SPY": 1.0} if held else {"BIL": 1.0})


@weights("taa2_ma_band")
def ma_band(panel, p, extra):
    """Siegel (Stocks for the Long Run): buy the index when it closes 1% above its 200-day average,
    sell when it closes 1% below; cash (BIL) otherwise."""
    a = p.get("asset", "DIA")
    s = panel.close[a].dropna()
    m = sma(s, 200)
    band = float(p.get("band", 0.01))
    held = False
    for px, mv in zip(s.to_numpy(), m.to_numpy()):
        if np.isnan(mv):
            continue
        if not held and px > mv * (1 + band):
            held = True
        elif held and px < mv * (1 - band):
            held = False
    return _last_only(panel, lambda i: {a: 1.0} if held else {"BIL": 1.0})


@weights("taa2_lumber_gold")
def lumber_gold(panel, p, extra):
    """Bilello & Gayed (2015) 'Lumber: Worth Its Weight in Gold': at each week's close, if lumber
    futures outperformed gold futures over the prior 13 weeks hold the S&P 500 (SPY), otherwise
    Treasuries (IEF)."""
    c = panel.close

    def f(i):
        lu, go = c["LBR=F"].ffill(), c["GC=F"].ffill()
        if i < 65 or pd.isna(lu.iloc[i - 65]) or pd.isna(go.iloc[i - 65]):
            return {}
        return {"SPY": 1.0} if lu.iloc[i] / lu.iloc[i - 65] > go.iloc[i] / go.iloc[i - 65] else {"IEF": 1.0}
    return _last_only(panel, f)


@weights("taa2_vrp")
def vrp(panel, p, extra):
    """Volatility risk premium switch (Volatility Made Simple 'VRP', Kipnis replication): VRP = VIX
    minus SPY's 10-day realised volatility (annualised, in points); if its 5-day average is positive
    hold short-VIX (SVXY), otherwise long-VIX (VXX)."""
    c = panel.close
    spy, vix = c["SPY"].ffill(), c["^VIX"].ffill()
    hv = np.log(spy).diff().rolling(10).std() * math.sqrt(252) * 100
    v = (vix - hv).rolling(5).mean()

    def f(i):
        x = v.iloc[i]
        if pd.isna(x):
            return {}
        return {"SVXY": 1.0} if x > 0 else {"VXX": 1.0}
    return _last_only(panel, f)


# ============================================================================== crypto cross-sections
def _sched_rows(idx: pd.DatetimeIndex, weekly: bool) -> np.ndarray:
    """Rows a WeightBot rebalances on: weekly = candles that close on Sunday 00:00 UTC."""
    if not weekly:
        return np.ones(len(idx), bool)
    return ((idx + pd.Timedelta(days=1)).weekday == 6)


def _from(idx, extra) -> int:
    frm = extra.get("_from")
    if frm is None:
        return 0
    return int(np.searchsorted(idx.values, pd.Timestamp(frm).to_datetime64())) - 8


def _quintile(score: pd.Series, k: int, low: bool) -> dict:
    s = score.dropna()
    if len(s) < k:
        return {}
    return {a: 1 / k for a in s.sort_values(ascending=low).index[:k]}


@weights("taa2_crypto_xs")
def crypto_xs(panel, p, extra):
    """Weekly crypto cross-section, long one quintile (k coins) by a characteristic over the last n
    days: `score` low_vol (lowest std of daily returns; Burggraf & Rudolf 2021), illiquidity
    (highest Amihud |r| / EUR volume; Zhang & Li 2023), downside_beta (highest beta to the
    equal-weight market on days the market is below its mean; Dobrynskaya 2024), or min_var
    (long-only minimum-variance weights on all coins; Brauneis & Mestel 2019)."""
    C = panel.close
    idx = panel.index
    names = panel.names
    n, k = int(p.get("days", 30)), int(p.get("top", 4))
    what = p["score"]
    r = C.pct_change()
    w = pd.DataFrame(np.nan, index=idx, columns=names)
    rows = _sched_rows(idx, True)
    start = max(n, _from(idx, extra))
    eur_vol = panel.volume * C
    mkt = r.mean(axis=1)
    for i in np.flatnonzero(rows):
        if i < start:
            continue
        win = r.iloc[i - n + 1:i + 1]
        ok = [a for a in names if win[a].notna().sum() >= n * 0.9]
        if len(ok) < k:
            continue
        if what == "low_vol":
            out = _quintile(win[ok].std(), k, True)
        elif what == "illiquidity":
            ev = eur_vol[ok].iloc[i - n + 1:i + 1].replace(0, np.nan)
            out = _quintile((win[ok].abs() / ev).mean(), k, False)
        elif what == "downside_beta":
            m = mkt.iloc[i - n + 1:i + 1]
            dn = m < m.mean()
            mv = m[dn].var()
            out = _quintile(win[ok][dn].apply(lambda s: s.cov(m[dn]) / mv), k, False) if mv > 0 else {}
        else:
            cv = win[ok].dropna().cov().to_numpy() * 365
            out = dict(zip(ok, min_variance(cv))) if len(win[ok].dropna()) > n * 0.8 else {}
        w.iloc[i] = 0.0
        for a, v in out.items():
            w.iat[i, names.index(a)] = v
    return w.ffill().fillna(0.0)


@weights("taa2_crypto_weekday")
def crypto_weekday(panel, p, extra):
    """Long, Zaremba, Demir, Szczygielski & Vasenin (2020) seasonality in the cross-section of
    cryptocurrency returns: each day hold the quintile of coins with the highest average return on
    the same weekday over the previous 20 weeks, for one day."""
    C = panel.close
    r = C.pct_change()
    weeks = int(p.get("weeks", 20))
    k = int(p.get("top", 4))
    # weights on candle t are held over candle t+1: its weekday's past returns are r[t+1-7j]
    avg = sum(r.shift(6 + 7 * j) for j in range(weeks)) / weeks
    rank = avg.rank(axis=1, ascending=False, method="first")
    enough = avg.notna().sum(axis=1) >= k
    return ((rank <= k) & avg.notna()).astype(float).div(k).where(enough, 0.0)


@weights("taa2_crypto_equal")
def crypto_equal(panel, p, extra):
    """Platanakis, Sutcliffe & Urquhart (2018): the naive 1/N portfolio of the coins with a price."""
    c = panel.close
    n = c.notna().sum(axis=1).replace(0, np.nan)
    return c.notna().astype(float).div(n, axis=0).fillna(0.0)
