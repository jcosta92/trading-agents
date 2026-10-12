"""Single-market crypto rules from academic and practitioner studies, plus rule-based strategies
from QuantConnect's library and the ai-hedge-fund repository."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import TF
from ..generic import signal
from ..lib import highest, linreg, lowest, merge_informative, sma, ema, ichimoku, rsi, stdev
from .crypto_ft import _out


def _false(df):
    return pd.Series(False, index=df.index)


@signal("price_above_sma")
def price_above_sma(df, p, info):
    """Detzel et al. (2021): hold while the daily close is above its n-day average."""
    c = df["close"]
    above = c > sma(c, int(p.get("n", 50)))
    return _out(df, above, ~above)


@signal("log_vma")
def log_vma(df, p, info):
    """Grobys, Ahmed & Sapkota (2020): VMA(1, n) on log prices."""
    lp = np.log(df["close"])
    above = lp > sma(lp, int(p.get("n", 20)))
    return _out(df, above, ~above)


@signal("qp_max")
def qp_max(df, p, info):
    """Padysak & Vojtko (Quantpedia 2022): hold for the next day when the price is the x-day maximum."""
    c = df["close"]
    x = int(p.get("x", 20))
    hit = c >= highest(c, x)
    return _out(df, hit, ~hit)


@signal("qp_minmax")
def qp_minmax(df, p, info):
    c = df["close"]
    x = int(p.get("x", 10))
    hit = (c >= highest(c, x)) | (c <= lowest(c, x))
    return _out(df, hit, ~hit)


@signal("trading_range_breakout")
def trb(df, p, info):
    """Brock, Lakonishok & LeBaron (1992) TRB as tested on Bitcoin by Gerritsen et al. (2020):
    buy when the close exceeds the prior n-day high by a 1% band; fixed 10-day hold."""
    c = df["close"]
    n = int(p.get("n", 50))
    return _out(df, c > highest(c, n).shift(1) * (1 + float(p.get("band", 0.01))), _false(df))


@signal("alexander_filter")
def alexander_filter(df, p, info):
    """Hudson & Urquhart (2021) filter rule: buy x% above the recent low; exit x% below the high since entry."""
    c = df["close"]
    x = float(p.get("x", 0.05))
    return _out(df, c >= (1 + x) * lowest(c, int(p.get("j", 20))), _false(df))


@signal("weekly_tsmom")
def weekly_tsmom(df, p, info):
    """Liu & Tsyvinski (2021): a strong week predicts the next. Weekly 7-day return above the 80th
    percentile of the past 104 weekly returns -> hold two weeks (daily candles, decided on Sundays)."""
    c = df["close"]
    r7 = c / c.shift(7) - 1
    sun = df.index.weekday == 6
    rs = r7[sun]
    thr = rs.rolling(104, min_periods=52).quantile(0.8).shift(1)
    hit = (rs > thr).reindex(df.index, fill_value=False)
    return _out(df, hit, _false(df))


@signal("daily_tsmom")
def daily_tsmom(df, p, info):
    """Liu & Tsyvinski (2021) daily momentum: at 12:00 UTC, a 24-hour return above the 80th
    percentile of the trailing daily samples -> hold 5 days (1-hour candles)."""
    c = df["close"]
    r24 = c / c.shift(24) - 1
    noon = (df.index + TF["1h"]).hour == 12
    rs = r24[noon]
    thr = rs.rolling(int(p.get("days", 120)), min_periods=60).quantile(0.8).shift(1)
    hit = (rs > thr).reindex(df.index, fill_value=False)
    return _out(df, hit, _false(df))


@signal("overreaction_day")
def overreaction_day(df, p, info):
    """Caporale & Plastun (2019/2020): once the day's return exceeds mean + k*sigma of daily returns
    (trailing year), buy in its direction and close at the end of the UTC day."""
    d1 = info.get("1d")
    if d1 is None or not len(d1):
        return _out(df, _false(df), _false(df))
    d = d1.copy()
    r = d["close"] / d["open"] - 1
    d["thr"] = r.rolling(365, min_periods=180).mean() + float(p.get("k", 1.5)) * r.rolling(365, min_periods=180).std()
    m = merge_informative(df[[]], d[["thr"]], TF["1h"], TF["1d"])        # yesterday's statistics
    day_open = df["open"].groupby(df.index.floor("D")).transform("first")
    rt = df["close"] / day_open - 1
    end = df.index + TF["1h"]
    last = end.hour == 0
    hit = (rt > m["thr_inf"]) & ~pd.Series(last, index=df.index)
    first_today = hit & ~hit.groupby(df.index.floor("D")).transform(lambda s: s.shift(1, fill_value=False).cummax())
    return _out(df, first_today, pd.Series(last, index=df.index))


@signal("monday_effect")
def monday_effect(df, p, info):
    """Caporale & Plastun (2019): Bitcoin returns are higher on Mondays (UTC)."""
    end = df.index + TF["1h"]
    entry = (end.weekday == 0) & (end.hour == 0)
    exit_ = (end.weekday == 1) & (end.hour == 0)
    return _out(df, pd.Series(entry, index=df.index), pd.Series(exit_, index=df.index))


@signal("weekend_window_trend")
def weekend_window_trend(df, p, info):
    """Concretum (Pagani, 2026): BTC intraday trend profits concentrate from Sunday 19:00 New York
    time for 24 hours. Inside the window hold while the close is above the midpoint of the last
    24 hourly closes, sized to 20% annual volatility."""
    c = df["close"]
    ny = (df.index + TF["1h"]).tz_convert("America/New_York")
    sunday = ny.weekday == 6
    after_sun = sunday & (ny.hour >= 19)
    mon = (ny.weekday == 0) & (ny.hour < 19)
    inside = pd.Series(after_sun | mon, index=df.index)
    mid = (highest(c, 24) + lowest(c, 24)) / 2
    vol = c.pct_change().rolling(24 * 30).std() * np.sqrt(24 * 365)
    o = _out(df, inside & (c > mid), ~inside | (c < mid))
    o["weight"] = (0.20 / vol).clip(upper=1.0)
    return o


@signal("smart_trade_dip")
def smart_trade_dip(df, p, info):
    """Pionex Smart Trade with trailing take-profit, entered on a 12% drawdown from the 30-day high."""
    c = df["close"]
    return _out(df, c <= 0.88 * highest(c, 30), _false(df))


@signal("dynamic_breakout2")
def dynamic_breakout2(df, p, info):
    """QuantConnect Strategy Library: Dynamic Breakout II (long side)."""
    c, h = df["close"].to_numpy(), df["high"].to_numpy()
    n_arr = np.full(len(c), np.nan)
    up = np.full(len(c), np.nan)
    hh = np.full(len(c), np.nan)
    sm = np.full(len(c), np.nan)
    n = 20
    for i in range(31, len(c)):
        tv = np.std(c[i - 29:i + 1])
        yv = np.std(c[i - 30:i])
        if tv > 0:
            n = int(round(n * (1 + (tv - yv) / tv)))
        n = max(20, min(60, n))
        n_arr[i] = n
        if i >= n:
            w = c[i - n + 1:i + 1]
            alpha = 2 / (n + 1)
            e = w[0]
            for x in w[1:]:
                e = alpha * x + (1 - alpha) * e
            up[i] = e + 2 * np.std(w)
            hh[i] = np.max(h[i - n:i])
            sm[i] = np.mean(w)
    cs = df["close"]
    entry = (cs > pd.Series(up, index=df.index)) & (cs > pd.Series(hh, index=df.index))
    return _out(df, entry, cs < pd.Series(sm, index=df.index))


@signal("dual_thrust")
def dual_thrust(df, p, info):
    """QuantConnect Strategy Library: Dual Thrust (N=4, K=0.5), long side; the day starts at 00:00 UTC."""
    d1 = info.get("1d")
    if d1 is None or not len(d1):
        return _out(df, _false(df), _false(df))
    d = d1.copy()
    n = int(p.get("n", 4))
    hh, ll = d["high"].rolling(n).max(), d["low"].rolling(n).min()
    hc, lc = d["close"].rolling(n).max(), d["close"].rolling(n).min()
    d["range"] = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
    m = merge_informative(df[[]], d[["range"]], TF["15m"], TF["1d"])
    day_open = df["open"].groupby(df.index.floor("D")).transform("first")
    k = float(p.get("k", 0.5))
    trig = day_open + k * m["range_inf"]
    floor = day_open - k * m["range_inf"]
    o = _out(df, df["close"] < trig, df["close"] < floor)
    o["entry_price"] = trig
    return o


@signal("linreg_channel")
def linreg_channel(df, p, info):
    """QuantConnect Lean RegressionChannelAlgorithm: RC(30, 2) on daily data."""
    c = df["close"]
    mid = linreg(c, 30)
    sd = stdev(c, 30)
    return _out(df, c < mid - 2 * sd, c > mid + 2 * sd)


@signal("regression_slope")
def regression_slope(df, p, info):
    """QuantConnect Lean ScikitLearnLinearRegressionAlgorithm: slope of the last 30 daily opens
    (relative to the intercept); buy above 0.001, liquidate below -0.001."""
    o_ = df["open"].to_numpy()
    n = 30
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()
    rel = np.full(len(o_), np.nan)
    for i in range(n - 1, len(o_)):
        w = o_[i - n + 1:i + 1]
        b = ((x - xm) * (w - w.mean())).sum() / den
        a = w.mean() - b * xm
        if a != 0:
            rel[i] = b / a
    r = pd.Series(rel, index=df.index)
    return _out(df, r > 0.001, r < -0.001)


@signal("ichimoku_chikou")
def ichimoku_chikou(df, p, info):
    """QuantConnect Strategy Library (Gurrib 2020): the lagging span against the cloud it is plotted
    on; long when it moves above the cloud, out when below."""
    _, _, a, b = ichimoku(df, 9, 26, 52, 26)
    a26, b26 = a.shift(26), b.shift(26)
    c = df["close"]
    loc = pd.Series(np.where(c > np.maximum(a26, b26), 1, np.where(c < np.minimum(a26, b26), -1, 0)), index=df.index)
    loc[a26.isna() | b26.isna()] = 0
    return _out(df, (loc == 1) & (loc.shift(1) != 1), loc == -1)


def _hurst_aihf(prices: np.ndarray, max_lag: int = 20) -> float:
    """ai-hedge-fund's Hurst estimate (kept as published: it returns about H/2)."""
    lags = range(2, max_lag)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])))) for lag in lags]
    try:
        return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])
    except Exception:
        return 0.5


@signal("ai_hedge_fund_technicals")
def ai_hedge_fund_technicals(df, p, info):
    """virattt/ai-hedge-fund technicals agent (deterministic part), daily bars."""
    from ..lib import adx
    c, v = df["close"], df["volume"]
    e8, e21, e55 = ema(c, 8), ema(c, 21), ema(c, 55)
    a = adx(df, 14) / 100
    trend = np.where((e8 > e21) & (e21 > e55), 1, np.where((e8 < e21) & (e21 < e55), -1, 0))
    trend_c = np.where(trend != 0, a.fillna(0.5), 0.5)
    m50, s50 = sma(c, 50), c.rolling(50).std()
    z = (c - m50) / s50
    bbm = sma(c, 20)
    bbs = c.rolling(20).std()
    pb = (c - (bbm - 2 * bbs)) / (4 * bbs)
    mr = np.where((z < -2) & (pb < 0.2), 1, np.where((z > 2) & (pb > 0.8), -1, 0))
    mr_c = np.where(mr != 0, np.minimum(z.abs() / 4, 1), 0.5)
    ret = c.pct_change()
    score = 0.4 * ret.rolling(21).sum() + 0.3 * ret.rolling(63).sum() + 0.3 * ret.rolling(126).sum()
    volr = v / v.rolling(21).mean()
    mo = np.where((score > 0.05) & (volr > 1), 1, np.where((score < -0.05) & (volr > 1), -1, 0))
    mo_c = np.where(mo != 0, np.minimum(5 * score.abs(), 1), 0.5)
    hv = ret.rolling(21).std() * np.sqrt(252)
    reg = hv / hv.rolling(63).mean()
    vz = (hv - hv.rolling(63).mean()) / hv.rolling(63).std()
    vo = np.where((reg < 0.8) & (vz < -1), 1, np.where((reg > 1.2) & (vz > 1), -1, 0))
    vo_c = np.where(vo != 0, np.minimum(vz.abs() / 3, 1), 0.5)
    cv = c.to_numpy()
    hurst = np.full(len(cv), np.nan)
    from .ml import _skip_before
    for i in range(max(126, _skip_before(df.index, info.get("_from")) - 2), len(cv)):
        hurst[i] = _hurst_aihf(cv[i - 126:i + 1])
    sk = ret.rolling(63).skew()
    sa = np.where((hurst < 0.4) & (sk > 1), 1, np.where((hurst < 0.4) & (sk < -1), -1, 0))
    sa_c = np.where(sa != 0, np.clip((0.5 - np.nan_to_num(hurst, nan=0.5)) * 2, 0, 1), 0.5)
    w = {"t": 0.25, "m": 0.20, "o": 0.25, "v": 0.15, "s": 0.15}
    num = w["t"] * trend * trend_c + w["m"] * mr * mr_c + w["o"] * mo * mo_c + w["v"] * vo * vo_c + w["s"] * sa * sa_c
    den = w["t"] * trend_c + w["m"] * mr_c + w["o"] * mo_c + w["v"] * vo_c + w["s"] * sa_c
    s = pd.Series(np.asarray(num, dtype=float) / np.asarray(den, dtype=float), index=df.index)
    s[c.rolling(126).count() < 126] = np.nan
    return _out(df, s > 0.2, s < -0.2)


def _hurst_fixed(x: np.ndarray) -> float:
    lags = np.arange(2, 20)
    tau = [max(1e-12, np.std(x[lag:] - x[:-lag])) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


@signal("hurst_regime")
def hurst_regime(df, p, info):
    """Hurst-exponent regime switch: trend mode (H > 0.55) holds above EMA21; mean-reversion mode
    (H < 0.45) buys a 2-sigma dip below the 50-day mean and exits at the mean or after 10 days."""
    c = df["close"]
    lc = np.log(c.to_numpy())
    h = np.full(len(lc), np.nan)
    from .ml import _skip_before
    for i in range(max(100, _skip_before(df.index, info.get("_from")) - 2), len(lc)):
        h[i] = _hurst_fixed(lc[i - 100:i + 1])
    H = pd.Series(h, index=df.index)
    e21 = ema(c, 21)
    z = (c - sma(c, 50)) / c.rolling(50).std()
    trend_in = (H > 0.55) & (c > e21)
    mr_in = (H < 0.45) & (z < -2)
    exit_ = ((H > 0.55) & (c < e21)) | ((H <= 0.55) & (z > 0)) | ((H >= 0.45) & (H <= 0.55))
    return _out(df, trend_in | mr_in, exit_ & ~(trend_in | mr_in))
