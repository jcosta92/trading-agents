"""More US ETF rules (added 13 Oct 2026): the backtesting.py multi-timeframe example system, and
published tactical / risk-based allocation models with public code: HAA, AAA, KDA, FAA, Composite
Dual Momentum, Accelerating Dual Momentum and Hierarchical Risk Parity.

Months are 21 trading days, as elsewhere in this package. Allocation functions only fill the last
row (the bot trades on it), computed from prices up to today's price at 15:50 New York time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from ..generic import signal, weights
from ..lib import sma
from .crypto_ft import _out
from .etf import _best, _last_only, _mom13612

M1, M3, M6, M12 = 21, 63, 126, 252


def _ret(c: pd.DataFrame | pd.Series, n: int):
    return c / c.shift(n) - 1


def min_variance(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights summing to 1 (quadprog / portfolio.optim with equal
    expected returns, as in the AAA and KDA code)."""
    n = len(cov)
    if n == 1:
        return np.ones(1)
    cov = np.asarray(cov, float)
    cov = cov / np.mean(np.diag(cov))                 # scale to ~1 so the solver's tolerances bite
    x0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x if res.success else 1 / np.diag(cov), 0, None)
    return w / w.sum()


# ============================================================================== backtesting.py example
def _bt_rsi(s: pd.Series, n: int) -> pd.Series:
    """The example's 'approximate' RSI: pandas ewm(com=n) of gains and losses."""
    gain = s.diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(com=n).mean() / loss.abs().ewm(com=n).mean()
    return 100 - 100 / (1 + rs)


@signal("bt_mtf_system")
def bt_mtf_system(df, p, info):
    """backtesting.py 'Multiple Time Frames' example System: daily RSI(30) and weekly RSI(30) both
    above 70 (weekly above daily), SMA10 > SMA20 > SMA50 > SMA100 and the close above SMA10: buy with
    a stop 8% below; sell when the close is 2% below SMA10. The weekly RSI uses completed W-FRI weeks
    (resample_apply: label='right', forward-filled onto the daily index)."""
    c = df["close"]
    d_rsi = _bt_rsi(c, 30)
    wk = c.resample("W-FRI", label="right").agg("last").dropna()
    w = _bt_rsi(wk, 30)
    w_rsi = w.reindex(c.index.union(w.index), method="ffill").reindex(c.index)
    m10, m20, m50, m100 = sma(c, 10), sma(c, 20), sma(c, 50), sma(c, 100)
    entry = (d_rsi > 70) & (w_rsi > 70) & (w_rsi > d_rsi) & (m10 > m20) & (m20 > m50) & (m50 > m100) & (c > m10)
    out = _out(df, entry, c < 0.98 * m10)
    out["sl_pct"] = 0.08
    return out


# ============================================================================== Keller & Keuning HAA
@weights("haa")
def haa(panel, p, extra):
    """Hybrid Asset Allocation, balanced (Keller & Keuning 2023): 13612U momentum (average of the 1,
    3, 6, 12-month returns). TIP momentum <= 0: all in the better of BIL / IEF. Otherwise the top 4 of
    SPY, IWM, VEA, VWO, VNQ, DBC, IEF, TLT at 25% each; a pick with momentum <= 0 is replaced by the
    better of BIL / IEF."""
    c = panel.close
    m = (_ret(c, M1) + _ret(c, M3) + _ret(c, M6) + _ret(c, M12)) / 4
    risky = ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
    cash = ["BIL", "IEF"]

    def f(i):
        row = m.iloc[i]
        if row.reindex(risky + cash + ["TIP"]).isna().any():
            return {}
        best_cash = _best(row[cash])[0]
        if row["TIP"] <= 0:
            return {best_cash: 1.0}
        out = {}
        for a in _best(row[risky], 4):
            tgt = a if row[a] > 0 else best_cash
            out[tgt] = out.get(tgt, 0) + 0.25
        return out
    return _last_only(panel, f)


# ============================================================================== Adaptive Asset Allocation
@weights("aaa")
def aaa(panel, p, extra):
    """Adaptive Asset Allocation (Butler, Philbrick, Gordillo & Varadi 2012), as in Systematic
    Investor's bt.aaa.test: the top 5 of 10 asset-class ETFs by 6-month (132-day) return, weighted
    by long-only minimum variance on the covariance of the last 22 daily log returns."""
    c = panel.close
    uni = [a for a in ("SPY", "EFA", "EWJ", "EEM", "IYR", "RWX", "IEF", "TLT", "DBC", "GLD") if a in c]
    mom = c[uni] / c[uni].shift(132) - 1
    lr = np.log(c[uni]).diff()

    def f(i):
        row = mom.iloc[i]
        if row.isna().any() or i < 22:
            return {}
        top = _best(row, 5)
        h = lr[top].iloc[i - 21:i + 1]
        sd = h.std().to_numpy()
        cov = np.outer(sd, sd) * h.corr().to_numpy()
        return dict(zip(top, min_variance(cov)))
    return _last_only(panel, f)


# ============================================================================== Kipnis KDA
@weights("kda")
def kda(panel, p, extra):
    """Kipnis Defensive Adaptive asset allocation (Kipnis 2019): AAA's universe; assets in the top 5
    by 13612W momentum AND with positive momentum, minimum-variance weights (1-month volatility,
    correlation = (12 C1 + 4 C3 + 2 C6 + C12) / 19); canaries VWO and BND scale the risky share to
    100 / 50 / 0%; the rest in IEF if its momentum is positive, otherwise cash."""
    c = panel.close
    risky = ["SPY", "VGK", "EWJ", "EEM", "VNQ", "RWX", "IEF", "TLT", "DBC", "GLD"]
    canary = ["VWO", "BND"]
    m = _mom13612(c)
    r = c.pct_change()

    def f(i):
        row = m.iloc[i]
        if row.reindex(risky + canary).isna().any():
            return {}
        rk = row[risky].rank()
        sel = [a for a in risky if rk[a] >= 6 and row[a] > 0]
        pct = float((row[canary] > 0).mean())
        out = {}
        if sel:
            if len(sel) == 1:
                w = np.ones(1)
            else:
                vols = r[sel].iloc[i - M1 + 1:i + 1].std().to_numpy()
                cors = sum(k * r[sel].iloc[i - n + 1:i + 1].corr().to_numpy()
                           for k, n in ((12, M1), (4, M3), (2, M6), (1, M12))) / 19
                w = min_variance(np.outer(vols, vols) * cors)
            for a, x in zip(sel, w):
                out[a] = x * pct
        if pct < 1 and row["IEF"] > 0:
            out["IEF"] = out.get("IEF", 0) + (1 - pct)
        return out
    return _last_only(panel, f)


# ============================================================================== Keller FAA
@weights("faa")
def faa(panel, p, extra):
    """Flexible Asset Allocation (Keller & van Putten 2012): over 4 months rank return (high best),
    volatility and average correlation (low best); score = rank R + 0.5 rank V + 0.5 rank C; hold the
    best 3 at 1/3 each; a pick with a negative 4-month return goes to SHY (cash)."""
    c = panel.close
    uni = ["VTI", "VEA", "VWO", "SHY", "BND", "GSG", "VNQ"]
    n = 4 * M1

    def f(i):
        if i < n or c[uni].iloc[i - n:i + 1].isna().any().any():
            return {}
        win = c[uni].iloc[i - n:i + 1]
        rets = win.pct_change().iloc[1:]
        r = win.iloc[-1] / win.iloc[0] - 1
        v = rets.std()
        cr = rets.corr().sum() - 1
        score = r.rank(ascending=False, method="first") + 0.5 * v.rank(method="first") \
            + 0.5 * cr.rank(method="first")
        out = {}
        for a in score.sort_values(kind="stable").index[:3]:
            tgt = a if r[a] >= 0 else "SHY"
            out[tgt] = out.get(tgt, 0) + 1 / 3
        return out
    return _last_only(panel, f)


# ============================================================================== Antonacci composite
@weights("composite_dual_momentum")
def composite_dual_momentum(panel, p, extra):
    """Composite Dual Momentum (Antonacci 2012): four 25% modules (SPY/EFA, HYG/LQD, VNQ/REM,
    GLD/TLT); each holds its asset with the higher 12-month return if that return beats T-bills
    (BIL), otherwise BIL."""
    c = panel.close
    r12 = _ret(c, M12)
    mods = [("SPY", "EFA"), ("HYG", "LQD"), ("VNQ", "REM"), ("GLD", "TLT")]

    def f(i):
        row = r12.iloc[i]
        if row.reindex([a for m in mods for a in m] + ["BIL"]).isna().any():
            return {}
        out = {}
        for a, b in mods:
            best = a if row[a] >= row[b] else b
            tgt = best if row[best] > row["BIL"] else "BIL"
            out[tgt] = out.get(tgt, 0) + 0.25
        return out
    return _last_only(panel, f)


# ============================================================================== Accelerating Dual Momentum
@weights("accelerating_dual_momentum")
def accelerating_dual_momentum(panel, p, extra):
    """Accelerating Dual Momentum (EngineeredPortfolio 2018): score = 1 + 3 + 6-month returns of SPY
    and SCZ; hold the higher one if its score is positive, otherwise long Treasuries (TLT)."""
    c = panel.close
    s = _ret(c, M1) + _ret(c, M3) + _ret(c, M6)

    def f(i):
        row = s.iloc[i]
        if row.reindex(["SPY", "SCZ", "TLT"]).isna().any():
            return {}
        best = "SPY" if row["SPY"] >= row["SCZ"] else "SCZ"
        return {best: 1.0} if row[best] > 0 else {"TLT": 1.0}
    return _last_only(panel, f)


# ============================================================================== Hierarchical Risk Parity
def hrp_weights(rets: pd.DataFrame) -> pd.Series:
    """PyPortfolioOpt HRPOpt (single linkage): cluster on sqrt((1 - corr) / 2), order the assets by
    the dendrogram, split weight top-down by inverse cluster variance."""
    cov, corr = rets.cov(), rets.corr()
    dist = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    link = sch.linkage(squareform(dist, checks=False), "single")
    order = list(corr.index[sch.to_tree(link, rd=False).pre_order()])
    w = pd.Series(1.0, index=order)
    items = [order]

    def cvar(names):
        cs = cov.loc[names, names].to_numpy()
        iv = 1 / np.diag(cs)
        iv /= iv.sum()
        return float(iv @ cs @ iv)
    while items:
        items = [i[j:k] for i in items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for t in range(0, len(items), 2):
            a, b = items[t], items[t + 1]
            va, vb = cvar(a), cvar(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    return w


@weights("hrp")
def hrp(panel, p, extra):
    """Hierarchical Risk Parity (Lopez de Prado 2016) on AAA's 10 asset-class ETFs: 260 daily
    returns, re-estimated monthly. A risk-based benchmark: always invested, no timing."""
    c = panel.close
    uni = [a for a in ("SPY", "EFA", "EWJ", "EEM", "IYR", "RWX", "IEF", "TLT", "DBC", "GLD") if a in c]

    def f(i):
        if i < 260:
            return {}
        rets = c[uni].iloc[i - 260:i + 1].pct_change().iloc[1:]
        if rets.isna().any().any():
            return {}
        return hrp_weights(rets).to_dict()
    return _last_only(panel, f)
