"""Portfolio strategies on several coins: target weights per candle (causal), traded by WeightBot
on the family's schedule. Long only; whatever is not allocated stays in cash (EUR)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..generic import weights
from ..lib import ema, sma


def _empty(panel):
    return pd.DataFrame(0.0, index=panel.index, columns=panel.names)


def _top_k(score: pd.DataFrame, k: int, ascending: bool = False, require_positive: pd.DataFrame | None = None):
    """Equal weight 1/k on the k best scores of each row (fewer if not enough data)."""
    r = score.rank(axis=1, ascending=ascending, method="first")
    sel = (r <= k) & score.notna()
    if require_positive is not None:
        sel &= require_positive > 0
    return sel.astype(float) / float(k)


@weights("zarattini_crypto_trends")
def zarattini_crypto_trends(panel, p, extra):
    """Zarattini, Pagani & Barbon (2025) 'Catching Crypto Trends': Donchian ensemble over nine
    lookbacks with a midpoint trailing stop, each coin sized to 25% annual volatility; no leverage."""
    lbs = p.get("lookbacks", [5, 10, 20, 30, 60, 90, 150, 250, 360])
    C = panel.close
    n_coins = len(panel.names)
    w = _empty(panel)
    for coin in panel.names:
        c = C[coin].to_numpy()
        vol = pd.Series(c).pct_change().rolling(90).std().to_numpy() * math.sqrt(365)
        scale = np.where(np.isnan(vol) | (vol <= 0), 0.0, np.minimum(1.0, 0.25 / np.where(vol > 0, vol, 1)))
        total = np.zeros(len(c))
        for L in lbs:
            hi = pd.Series(c).rolling(L).max().to_numpy()
            lo = pd.Series(c).rolling(L).min().to_numpy()
            prev_hi = np.r_[np.nan, hi[:-1]]
            inpos, stop = False, -np.inf
            state = np.zeros(len(c))
            for i in range(len(c)):
                if np.isnan(c[i]):
                    continue
                mid = (hi[i] + lo[i]) / 2 if not np.isnan(hi[i]) else np.nan
                if inpos:
                    if not np.isnan(mid):
                        stop = max(stop, mid)
                    if c[i] < stop:
                        inpos = False
                elif not np.isnan(prev_hi[i]) and c[i] > prev_hi[i]:
                    inpos, stop = True, mid
                state[i] = 1.0 if inpos else 0.0
            total += state
        w[coin] = total / len(lbs) / n_coins * scale
    s = w.sum(axis=1)
    return w.div(s.where(s > 1, 1.0), axis=0)


@weights("monash_vol_sma")
def monash_vol_sma(panel, p, extra):
    """Le & Ruthbah (Monash, 2024): hold when the price is above its n-day average, sized to 30%
    annual volatility. BTC with n=65 and ETH with n=20, half the capital each."""
    w = _empty(panel)
    cfg = p.get("sleeves", {"BTC_SPOT": 65, "ETH_SPOT": 20})
    for coin, n in cfg.items():
        if coin not in panel.names:
            continue
        c = panel.close[coin]
        vol = c.pct_change().rolling(int(n)).std() * math.sqrt(365)
        w[coin] = (c > sma(c, int(n))).astype(float) * (0.30 / vol).clip(upper=1.0).fillna(0) / len(cfg)
    return w


@weights("rohrbach_trend")
def rohrbach_trend(panel, p, extra):
    """Rohrbach, Suremann & Osterrieder (2017): normalised EMA crossover signals over three
    horizons (Baz et al. 2015 response function), long-only, equal risk budget per coin."""
    w = _empty(panel)
    for coin in panel.names:
        c = panel.close[coin]
        sig = []
        for fa, sl in ((8, 24), (16, 48), (32, 96)):
            x = (ema(c, fa) - ema(c, sl)) / c.rolling(63).std()
            y = x / x.rolling(252).std()
            sig.append(y * np.exp(-y ** 2 / 4) / 0.89)
        s = sum(sig) / 3
        w[coin] = s.clip(lower=0, upper=1) / len(panel.names)
    return w


def _ret(panel, n):
    return panel.close / panel.close.shift(n) - 1


@weights("cs_momentum")
def cs_momentum(panel, p, extra):
    """Liu, Tsyvinski & Wu (2022): weekly, hold the top 30% of coins by their 3-week return
    (equal weight); a coin with a negative return stays in cash."""
    r = _ret(panel, int(p.get("days", 21)))
    k = max(1, int(round(float(p.get("top_frac", 0.3)) * len(panel.names))))
    return _top_k(r, k, require_positive=r if p.get("absolute", True) else None)


@weights("vol_managed_momentum")
def vol_managed_momentum(panel, p, extra):
    """Grobys et al. (2025): the 3-week momentum basket scaled by min(1, 40% / its own realised
    volatility over the last 4 weeks)."""
    r = _ret(panel, 21)
    k = max(1, int(round(0.3 * len(panel.names))))
    base = _top_k(r, k)
    daily = panel.close.pct_change()
    idx = panel.index
    weekly = idx.weekday == 6
    held = base.where(pd.Series(weekly, index=idx), np.nan).ffill().shift(1).fillna(0.0)
    port = (held * daily).sum(axis=1)
    vol = port.rolling(28).std() * math.sqrt(365)
    scale = (0.40 / vol).clip(upper=1.0).fillna(0.0)
    return base.mul(scale, axis=0)


@weights("top_k_return")
def top_k_return(panel, p, extra):
    """Dobrynskaya (2022): equal weight in the top k coins by their n-day return."""
    return _top_k(_ret(panel, int(p.get("days", 14))), int(p.get("k", 5)))


@weights("near_week_high")
def near_week_high(panel, p, extra):
    """Ficura (2023): large coins closest to their 1-week high outperform."""
    score = panel.close / panel.high.rolling(7).max()
    return _top_k(score, int(p.get("k", 5)))


@weights("ctrend_proxy")
def ctrend_proxy(panel, p, extra):
    """Fieberg et al. (2025) CTREND, simplified: average cross-sectional rank of close / SMA(n)
    for n = 3 ... 200 (the paper's price-trend signals); hold the top 4."""
    ranks = []
    for n in (3, 5, 10, 20, 50, 100, 200):
        ranks.append((panel.close / panel.close.rolling(n).mean()).rank(axis=1, pct=True))
    score = sum(ranks) / len(ranks)
    return _top_k(score, int(p.get("k", 4)))


@weights("max_momentum")
def max_momentum(panel, p, extra):
    """Li, Urquhart, Wang & Zhang (2021): coins with the largest recent daily jumps keep winning."""
    r = panel.close.pct_change()
    mx = r.rolling(30).apply(lambda a: np.sort(a)[-5:].mean() if np.isfinite(a).all() else np.nan, raw=True)
    return _top_k(mx, int(p.get("k", 3)))


@weights("lottery_avoidance")
def lottery_avoidance(panel, p, extra):
    """Grobys & Junttila (2021): among large coins, the lowest-MAX portfolio beats the highest."""
    r = panel.close.pct_change()
    mx = r.rolling(7).max()
    return _top_k(mx, int(p.get("k", 6)), ascending=True)


@weights("weekly_contrarian")
def weekly_contrarian(panel, p, extra):
    """Kosc, Sakowski & Slepaczuk (2019): buy last week's losers (equal weight)."""
    return _top_k(_ret(panel, 7), int(p.get("k", 5)), ascending=True)


@weights("low_volume_reversal")
def low_volume_reversal(panel, p, extra):
    """Bianchi, Babiak & Dickerson (2022): yesterday's losers with unusually low volume rebound;
    hold one day."""
    r1 = panel.close.pct_change()
    lv = np.log(panel.volume.replace(0, np.nan))
    shock = (lv - lv.rolling(30).mean()) / lv.rolling(30).std()
    lo_r = r1.rank(axis=1, pct=True) <= 1 / 3
    lo_v = shock.rank(axis=1, pct=True) <= 1 / 3
    sel = (lo_r & lo_v).astype(float)
    cnt = sel.sum(axis=1).replace(0, np.nan)
    return sel.div(cnt, axis=0).fillna(0.0)


def _rolling_beta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    cov = y.rolling(n).cov(x)
    return cov / x.rolling(n).var()


@weights("btc_eth_distance")
def btc_eth_distance(panel, p, extra):
    """Fil & Kristoufek (2020) distance pairs, long-only: 50/50 BTC/ETH by default, 100% in the
    cheap leg when the spread z-score passes +/-2, back to 50/50 when it crosses 0."""
    a, b = "BTC_SPOT", "ETH_SPOT"
    w = _empty(panel)
    if a not in panel.names or b not in panel.names:
        return w
    n = int(p.get("window", 720))
    x, y = np.log(panel.close[a]), np.log(panel.close[b])
    beta = _rolling_beta(y, x, n)
    s = y - beta * x
    z = ((s - s.rolling(n).mean()) / s.rolling(n).std()).to_numpy()
    state = 0
    wa, wb = np.zeros(len(z)), np.zeros(len(z))
    for i in range(len(z)):
        if np.isnan(z[i]):
            continue
        if state == 0 and z[i] > 2:
            state = 1                         # ETH rich: hold BTC
        elif state == 0 and z[i] < -2:
            state = -1                        # ETH cheap: hold ETH
        elif state == 1 and z[i] < 0:
            state = 0
        elif state == -1 and z[i] > 0:
            state = 0
        wa[i], wb[i] = (1.0, 0.0) if state == 1 else ((0.0, 1.0) if state == -1 else (0.5, 0.5))
    w[a], w[b] = wa, wb
    return w


@weights("kalman_pairs")
def kalman_pairs(panel, p, extra):
    """Quantopian research (D. Edwards) Kalman-filter pairs on log ETH vs log BTC, long-only
    switch: |z| > 1.5 -> 100% in the cheap leg; back to 50/50 at the zero cross or after 20 days."""
    a, b = "BTC_SPOT", "ETH_SPOT"
    w = _empty(panel)
    if a not in panel.names or b not in panel.names:
        return w
    x, y = np.log(panel.close[a]).to_numpy(), np.log(panel.close[b]).to_numpy()
    n = len(x)
    delta = 1e-3
    burn = 300
    ok = ~(np.isnan(x) | np.isnan(y))
    first = int(np.argmax(ok)) if ok.any() else n
    if n - first < burn + 50:
        return w
    xb, yb = x[first:first + burn], y[first:first + burn]
    A = np.vstack([xb, np.ones(burn)]).T
    coef, *_ = np.linalg.lstsq(A, yb, rcond=None)
    ve = float(np.var(yb - A @ coef))
    Vw = delta / (1 - delta) * np.eye(2)
    theta = coef.copy()
    P = np.eye(2)
    spread = np.full(n, np.nan)
    for i in range(first, n):
        if not ok[i]:
            continue
        F = np.array([x[i], 1.0])
        R = P + Vw
        yhat = F @ theta
        e = y[i] - yhat
        Q = F @ R @ F + ve
        K = R @ F / Q
        theta = theta + K * e
        P = R - np.outer(K, F) @ R
        spread[i] = e
    sp = pd.Series(spread)
    z = ((sp - sp.rolling(300).mean()) / sp.rolling(300).std()).to_numpy()
    state, since = 0, 0
    wa, wb = np.full(n, 0.5), np.full(n, 0.5)
    max_bars = int(p.get("max_bars", 120))
    for i in range(n):
        if np.isnan(z[i]):
            continue
        if state == 0:
            if z[i] > 1.5:
                state, since = 1, i
            elif z[i] < -1.5:
                state, since = -1, i
        elif (state == 1 and z[i] < 0) or (state == -1 and z[i] > 0) or i - since >= max_bars:
            state = 0
        wa[i], wb[i] = (1.0, 0.0) if state == 1 else ((0.0, 1.0) if state == -1 else (0.5, 0.5))
    w[a], w[b] = wa, wb
    return w


@weights("ols_pairs_switch")
def ols_pairs_switch(panel, p, extra):
    """vectorbt PairsTrading example: rolling OLS spread of log P0 on log P1 (100 bars), z = +/-1.96;
    long-only form holds the leg that the source buys, switching only at the bands."""
    a, b = p.get("p0"), p.get("p1")
    w = _empty(panel)
    if a not in panel.names or b not in panel.names:
        return w
    n = int(p.get("window", 100))
    y, x = np.log(panel.close[a]), np.log(panel.close[b])
    beta = _rolling_beta(y, x, n)
    alpha = y.rolling(n).mean() - beta * x.rolling(n).mean()
    s = y - (beta * x + alpha)
    z = ((s - s.rolling(n).mean()) / s.rolling(n).std()).to_numpy()
    state = 0
    wa, wb = np.zeros(len(z)), np.zeros(len(z))
    for i in range(len(z)):
        if not np.isnan(z[i]):
            if z[i] > 1.96:
                state = -1                   # P0 rich: long P1
            elif z[i] < -1.96:
                state = 1                    # P0 cheap: long P0
        wa[i], wb[i] = (1.0, 0.0) if state == 1 else ((0.0, 1.0) if state == -1 else (0.0, 0.0))
    w[a], w[b] = wa, wb
    return w


@weights("half_kelly")
def half_kelly(panel, p, extra):
    """Quantopian KellyLeverage, long-only half Kelly: weight = clip(0.5 * mean / variance of daily
    log returns over 120 days, 0, 1)."""
    w = _empty(panel)
    for coin in panel.names:
        lr = np.log(panel.close[coin]).diff()
        n = int(p.get("window", 120))
        fk = lr.rolling(n).mean() / lr.rolling(n).var()
        w[coin] = (0.5 * fk).clip(lower=0, upper=1).fillna(0) / len(panel.names)
    return w
