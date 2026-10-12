"""US ETF strategies: daily rules decided at 15:55 New York time (EtfDailyBot / EtfWeightBot,
whose frames end with today's bar so far) and intraday rules on 5-minute candles."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..generic import signal, weights
from ..lib import rsi, sma
from .crypto_ft import _out


def _f(df):
    return pd.Series(False, index=df.index)


# ============================================================================== Connors daily rules
@signal("etf_rsi2")
def etf_rsi2(df, p, info):
    """Connors & Alvarez (2008): above the 200-day average, buy RSI(2) <= 5; sell above the 5-day average."""
    c = df["close"]
    return _out(df, (c > sma(c, 200)) & (rsi(c, 2) <= float(p.get("level", 5))), c > sma(c, 5))


@signal("etf_cumulative_rsi")
def etf_cumulative_rsi(df, p, info):
    c = df["close"]
    r = rsi(c, 2)
    cum = r + r.shift(1)
    return _out(df, (c > sma(c, 200)) & (cum < 35), cum > 65)


@signal("etf_r3")
def etf_r3(df, p, info):
    c = df["close"]
    r = rsi(c, 2)
    entry = (c > sma(c, 200)) & (r.shift(2) < 60) & (r.shift(1) < r.shift(2)) & (r < r.shift(1)) & (r < 10)
    return _out(df, entry, r > 70)


@signal("etf_pct_b")
def etf_pct_b(df, p, info):
    c = df["close"]
    m, sd = sma(c, 5), c.rolling(5).std(ddof=0)
    pb = (c - (m - sd)) / (2 * sd)
    low3 = (pb < 0.2) & (pb.shift(1) < 0.2) & (pb.shift(2) < 0.2)
    return _out(df, (c > sma(c, 200)) & low3, pb > 0.8)


@signal("etf_tps")
def etf_tps(df, p, info):
    c = df["close"]
    r = rsi(c, 2)
    return _out(df, (c > sma(c, 200)) & (r < 25) & (r.shift(1) < 25), r > 70)


@signal("etf_3day_high_low")
def etf_3day_hl(df, p, info):
    c, h, lo = df["close"], df["high"], df["low"]
    three = (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3)) & \
            (lo < lo.shift(1)) & (lo.shift(1) < lo.shift(2)) & (lo.shift(2) < lo.shift(3))
    return _out(df, (c > sma(c, 200)) & (c < sma(c, 5)) & three, c > sma(c, 5))


@signal("etf_double_7s")
def etf_double_7s(df, p, info):
    c = df["close"]
    return _out(df, (c > sma(c, 200)) & (c <= c.rolling(7).min()), c >= c.rolling(7).max())


@signal("etf_multiple_days_down")
def etf_multi_down(df, p, info):
    c = df["close"]
    down = (c < c.shift(1)).astype(int).rolling(5).sum()
    return _out(df, (down >= 4) & (c > sma(c, 200)) & (c < sma(c, 5)), c > sma(c, 5))


@signal("etf_vix_stretch")
def etf_vix_stretch(df, p, info):
    """Connors & Alvarez (2008) VIX stretch: VIX at least 5% above its 10-day average for 3 days
    (prior closes; intraday VIX is not used) and the ETF above its 200-day average."""
    c = df["close"]
    vix = (info.get("daily") or {}).get("^VIX")
    if vix is None or vix.empty:
        return _out(df, _f(df), rsi(c, 2) > 65)
    v = vix["close"].reindex(df.index).ffill().shift(1)
    st = v >= 1.05 * v.rolling(10).mean()
    three = st & st.shift(1, fill_value=False) & st.shift(2, fill_value=False)
    return _out(df, three & (c > sma(c, 200)), rsi(c, 2) > 65)


# ============================================================================== allocation models
def _mom13612(c: pd.DataFrame) -> pd.DataFrame:
    return 12 * (c / c.shift(21) - 1) + 4 * (c / c.shift(63) - 1) + 2 * (c / c.shift(126) - 1) + (c / c.shift(252) - 1)


def _sma12m(c: pd.DataFrame) -> pd.DataFrame:
    """Price / average of the last 13 month-end prices (approximated with 21-day steps) - 1."""
    avg = sum(c.shift(21 * k) for k in range(13)) / 13
    return c / avg - 1


def _w(panel):
    return pd.DataFrame(0.0, index=panel.index, columns=panel.names)


def _best(row: pd.Series, k: int = 1) -> list:
    r = row.dropna().sort_values(ascending=False)
    return list(r.index[:k])


def _last_only(panel, fn):
    """Allocation rules only need the latest row (the bot trades on it)."""
    w = _w(panel)
    if len(panel.index):
        w.iloc[-1] = pd.Series(fn(len(panel.index) - 1)).reindex(panel.names).fillna(0.0)
    return w


@weights("gtaa")
def gtaa(panel, p, extra):
    """Faber (2007) GTAA, daily-SMA version (Concretum 2026): 20% each in SPY, EFA, IEF, DBC, VNQ when
    above the 200-day average, otherwise that 20% in T-bills (BIL)."""
    c = panel.close
    assets = [a for a in ("SPY", "EFA", "IEF", "DBC", "VNQ") if a in c]

    def f(i):
        out = {}
        for a in assets:
            ok = c[a].iloc[i] > c[a].iloc[max(0, i - 199):i + 1].mean() if i >= 199 else False
            if ok:
                out[a] = out.get(a, 0) + 0.2
            else:
                out["BIL"] = out.get("BIL", 0) + 0.2
        return out
    return _last_only(panel, f)


@weights("gem")
def gem(panel, p, extra):
    """Antonacci Global Equities Momentum: 12-month returns; US vs T-bills, then US vs ex-US; else bonds."""
    c = panel.close
    r12 = c / c.shift(252) - 1

    def f(i):
        row = r12.iloc[i]
        if row.isna()[["SPY", "BIL"]].any():
            return {}
        if row["SPY"] > row["BIL"]:
            return {"SPY" if row["SPY"] >= row.get("VEU", -9) else "VEU": 1.0}
        return {"AGG": 1.0}
    return _last_only(panel, f)


@weights("vaa_g4")
def vaa_g4(panel, p, extra):
    """Keller & Keuning (2017) VAA-G4: all four risky assets with positive 13612W momentum -> best
    risky asset; otherwise the best cash asset."""
    m = _mom13612(panel.close)
    risky, cash = ["SPY", "EFA", "EEM", "AGG"], ["SHY", "IEF", "LQD"]

    def f(i):
        row = m.iloc[i]
        if row[risky].isna().any():
            return {}
        if (row[risky] > 0).all():
            return {_best(row[risky])[0]: 1.0}
        return {_best(row[cash])[0]: 1.0}
    return _last_only(panel, f)


@weights("daa_g12")
def daa_g12(panel, p, extra):
    """Keller & Keuning (2018) DAA-G12: canaries VWO and BND set the cash share (0/50/100%); the
    rest is split equally among the top 6 risky assets by 13612W momentum."""
    m = _mom13612(panel.close)
    risky = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "DBC", "GLD", "TLT", "HYG", "LQD"]
    cash = ["SHY", "IEF", "LQD"]

    def f(i):
        row = m.iloc[i]
        if row[["VWO", "BND"]].isna().any():
            return {}
        b = int((row[["VWO", "BND"]] < 0).sum())
        cs = b / 2
        out = {}
        if cs > 0:
            out[_best(row[cash])[0]] = cs
        if cs < 1:
            top = _best(row[risky], 6)
            for a in top:
                out[a] = out.get(a, 0) + (1 - cs) / len(top)
        return out
    return _last_only(panel, f)


@weights("baa_g4")
def baa_g4(panel, p, extra):
    """Keller (2022) Bold Asset Allocation, aggressive G4."""
    c = panel.close
    m = _mom13612(c)
    s12 = _sma12m(c)
    canary = ["SPY", "VWO", "VEA", "BND"]
    off = ["QQQ", "VWO", "VEA", "BND"]
    deff = ["TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND"]

    def f(i):
        row = m.iloc[i]
        if row[canary].isna().any():
            return {}
        if (row[canary] < 0).any():
            srow = s12.iloc[i]
            top = _best(srow[deff], 3)
            out = {}
            for a in top:
                tgt = a if srow[a] >= srow["BIL"] else "BIL"
                out[tgt] = out.get(tgt, 0) + 1 / 3
            return out
        return {_best(s12.iloc[i][off])[0]: 1.0}
    return _last_only(panel, f)


@weights("paa")
def paa(panel, p, extra):
    """Keller & Keuning (2016) Protective Asset Allocation (a=2, top 6, 12-month SMA momentum)."""
    c = panel.close
    risky = ["SPY", "QQQ", "IWM", "VGK", "EWJ", "EEM", "VNQ", "DBC", "GLD", "TLT", "HYG", "LQD"]
    mom = c / c.rolling(252).mean() - 1

    def f(i):
        row = mom.iloc[i][risky]
        if row.isna().any():
            return {}
        N, n1 = len(risky), 6
        n = int((row > 0).sum())
        bf = min(1.0, (N - n) / (N - n1))
        out = {"IEF": bf} if bf > 0 else {}
        pos = row[row > 0].sort_values(ascending=False)[:6]
        for a in pos.index:
            out[a] = out.get(a, 0) + (1 - bf) / len(pos)
        return out
    return _last_only(panel, f)


@weights("sector_momentum")
def sector_momentum(panel, p, extra):
    """Faber (2010) sector relative strength: top 3 US sector ETFs by 12-month return."""
    r = panel.close / panel.close.shift(252) - 1

    def f(i):
        top = _best(r.iloc[i], int(p.get("k", 3)))
        return {a: 1 / len(top) for a in top} if top else {}
    return _last_only(panel, f)


@weights("industry_trend")
def industry_trend(panel, p, extra):
    """Zarattini & Antonacci (2024) 'A Century of Profitable Industry Trends', on sector/industry
    ETFs: enter when the close reaches the upper band (min of the 20-day Donchian high and a Keltner
    band), trail a ratcheting stop (max of Donchian and Keltner lower), size each position to 1%
    daily volatility (max 25%), total exposure capped at 100%."""
    n, k = 20, 2.0
    w = _w(panel)
    for a in panel.names:
        c, h, lo = panel.close[a], panel.high[a], panel.low[a]
        e = c.ewm(span=n, adjust=False).mean()
        mad = c.diff().abs().rolling(n).mean()
        up = np.minimum(h.rolling(n).max(), e + 1.4 * k * mad).to_numpy()
        dn = np.maximum(lo.rolling(n).min(), e - 1.4 * k * mad).to_numpy()
        cv = c.to_numpy()
        vol = c.pct_change().rolling(14).std().to_numpy()
        state, stop = False, -np.inf
        out = np.zeros(len(cv))
        for i in range(1, len(cv)):
            if np.isnan(up[i - 1]) or np.isnan(cv[i]):
                continue
            if state:
                stop = max(stop, dn[i - 1])
                if cv[i] < stop:
                    state = False
            elif cv[i] >= up[i - 1]:
                state, stop = True, dn[i - 1]
            if state and vol[i] > 0:
                out[i] = min(0.25, 0.01 / vol[i])
        w[a] = out
    s = w.sum(axis=1)
    return w.div(s.where(s > 1, 1.0), axis=0)


@weights("ath_trend")
def ath_trend(panel, p, extra):
    """Zarattini, Pagani & Wilcox (2025) all-time-high trend following on large US stocks: buy at a
    new all-time closing high, exit on a close below (highest close since entry - 10 x ATR42); at
    most 10 positions of 10% (new entries ranked by 12-month return)."""
    C, H, L = panel.close, panel.high, panel.low
    names = panel.names
    tr = pd.concat({a: pd.concat([H[a] - L[a], (H[a] - C[a].shift()).abs(), (L[a] - C[a].shift()).abs()],
                                 axis=1).max(axis=1) for a in names}, axis=1)
    atr42 = tr.rolling(42).mean()
    ath = C.cummax()
    r12 = C / C.shift(252) - 1
    held: dict[str, float] = {}
    w = _w(panel)
    Cv = C.to_numpy()
    for i in range(1, len(C.index)):
        for j, a in enumerate(names):
            if a in held:
                held[a] = max(held[a], Cv[i, j])
                if Cv[i, j] < held[a] - 10 * atr42.iat[i, j]:
                    del held[a]
        cands = [a for j, a in enumerate(names) if a not in held and not np.isnan(Cv[i, j])
                 and Cv[i, j] >= ath.iat[i, j] and C[a].iloc[:i].count() > 252]
        cands.sort(key=lambda a: -(r12[a].iat[i] if not np.isnan(r12[a].iat[i]) else -9))
        for a in cands[: max(0, 10 - len(held))]:
            held[a] = Cv[i, names.index(a)]
        for a in held:
            w.iat[i, names.index(a)] = 0.1
    return w


@weights("volatility_edge")
def volatility_edge(panel, p, extra):
    """Zarattini, Aziz & Mele (2025) 'The Volatility Edge': regime from the volatility risk premium
    (VIX - 10-day SPY realised vol) and the VIX term structure (VIX vs VIX3M); long SVXY (short vol)
    in contango, long VXX in backwardation with a negative premium; weight = VIX/100 (SVXY doubled)."""
    c = panel.close
    if not {"SPY", "^VIX", "^VIX3M"} <= set(c.columns):
        return _w(panel)
    vix, vix3 = c["^VIX"].ffill(), c["^VIX3M"].ffill()
    rv = c["SPY"].pct_change().rolling(10).std() * math.sqrt(252) * 100
    vrp = vix - rv
    cont = vix < vix3
    base = (vix / 100).clip(upper=0.5)
    w = _w(panel)
    w["SVXY"] = np.where(cont & (vrp > 0), 2 * base, np.where(cont & (vrp <= 0), base, 0.0))
    w["VXX"] = np.where(~cont & (vrp <= 0), base, 0.0)
    return w.clip(upper=1.0)


@weights("vol_managed_spy")
def vol_managed_spy(panel, p, extra):
    """Moreira & Muir (2017) / Harvey et al. (2018) volatility targeting: SPY weight = min(1, 15% /
    20-day realised volatility); the rest in T-bills (BIL)."""
    c = panel.close
    vol = c["SPY"].pct_change().rolling(20).std() * math.sqrt(252)
    w = _w(panel)
    w["SPY"] = (0.15 / vol).clip(upper=1.0).fillna(0)
    w["BIL"] = 1 - w["SPY"]
    return w


@weights("lowvol_leveraged")
def lowvol_leveraged(panel, p, extra):
    """Quantpedia (2025) leveraged ETFs in low-volatility environments: hold the 3x S&P ETF while the
    VIX's moving average (prior closes) is above SPY's realised volatility."""
    c = panel.close
    if "^VIX" not in c or "SPY" not in c:
        return _w(panel)
    vix = c["^VIX"].ffill().shift(1).rolling(int(p.get("vix_sma", 60))).mean()
    rv = c["SPY"].pct_change().rolling(int(p.get("rv", 10))).std() * math.sqrt(252) * 100
    w = _w(panel)
    w[p.get("asset", "SPXL")] = (vix > rv).astype(float)
    return w


# ============================================================================== calendar rules
def _calendar_weight(panel, value: float, ticker: str):
    w = _w(panel)
    if len(w):
        w.iloc[-1, list(panel.names).index(ticker)] = value
    return w


@weights("treasury_month_end")
def treasury_month_end(panel, p, extra):
    """Beyond Passive / Robot Wealth (2025): long TLT over the last 5 trading days of the month."""
    nth, left = extra["pos"]
    return _calendar_weight(panel, 1.0 if 1 <= left <= 5 else 0.0, "TLT")


@weights("month_end_reversal")
def month_end_reversal(panel, p, extra):
    """Beyond Passive (2025): if bonds beat stocks over the month's first 15 trading days, hold VTI
    over the last 5 days; if stocks led, hold TBF (short Treasuries) over the next month's first 5 days."""
    nth, left = extra["pos"]
    bot = extra["bot"]
    day = pd.Timestamp(extra["day"])
    c = panel.close
    w = _w(panel)
    if not len(w):
        return w

    def led(month_end_day: pd.Timestamp):
        idx = bot.sessions.index
        ms = idx[(idx.year == month_end_day.year) & (idx.month == month_end_day.month)]
        prev = idx[idx < ms[0]]
        if len(ms) < 15 or not len(prev):
            return None
        a, b = prev[-1], ms[14]
        if a not in c.index or b not in c.index:
            return None
        rv = c.loc[b, "VTI"] / c.loc[a, "VTI"] - 1
        rt = c.loc[b, "TLT"] / c.loc[a, "TLT"] - 1
        return "bonds" if rt > rv else "stocks"

    i = -1
    if 1 <= left <= 5 and nth >= 15:
        if led(day) == "bonds":
            w.iloc[i, list(panel.names).index("VTI")] = 1.0
    if left == 0 and nth >= 15 and led(day) == "stocks":
        w.iloc[i, list(panel.names).index("TBF")] = 1.0
    if 1 <= nth <= 4:
        idx = bot.sessions.index
        prev_month_last = idx[idx < day.replace(day=1)]
        if len(prev_month_last) and led(prev_month_last[-1]) == "stocks":
            w.iloc[i, list(panel.names).index("TBF")] = 1.0
    return w


def _thanksgiving(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=11, day=1)
    first_thu = d + pd.Timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + pd.Timedelta(weeks=3)


@weights("holiday_seasonality")
def holiday_seasonality(panel, p, extra):
    """Quantpedia (2025): AMZN from 5 trading days before Thanksgiving to 3 trading days after; GLD
    from 23 December into the new year."""
    day = pd.Timestamp(extra["day"])
    idx = extra["sessions"].index
    w = _w(panel)
    if not len(w):
        return w
    tg = _thanksgiving(day.year)
    before = idx[idx < tg]
    after = idx[idx > tg]
    if len(before) >= 5 and len(after) >= 3 and before[-5] <= day < after[2]:
        w.iloc[-1, list(panel.names).index("AMZN")] = 1.0
    xmas_start = idx[idx >= pd.Timestamp(year=day.year, month=12, day=23)]
    if len(xmas_start) and day >= xmas_start[0]:
        w.iloc[-1, list(panel.names).index("GLD")] = 1.0
    return w


@weights("pre_holiday")
def pre_holiday(panel, p, extra):
    """QuantConnect Strategy Library: hold SPY when an exchange holiday (a weekday without a
    session) falls within the next 2 calendar days."""
    day = pd.Timestamp(extra["day"])
    idx = set(extra["sessions"].index)
    hol = any((day + pd.Timedelta(days=k)).weekday() < 5 and (day + pd.Timedelta(days=k)) not in idx for k in (1, 2))
    return _calendar_weight(panel, 1.0 if hol else 0.0, "SPY")


def _third_friday(y, m) -> pd.Timestamp:
    d = pd.Timestamp(year=y, month=m, day=1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(weeks=2)


@weights("opex_week")
def opex_week(panel, p, extra):
    """QuantConnect Strategy Library: hold OEF from the Monday of option-expiration week until the
    expiration day (sold at 15:55 that day)."""
    day = pd.Timestamp(extra["day"])
    idx = extra["sessions"].index
    exp = _third_friday(day.year, day.month)
    if exp not in idx:
        prior = idx[idx < exp]
        exp = prior[-1] if len(prior) else exp
    week_start = exp - pd.Timedelta(days=exp.weekday())
    hold = week_start <= day < exp
    return _calendar_weight(panel, 1.0 if hold else 0.0, panel.names[0])


def moon_phase(day: pd.Timestamp) -> float:
    """Fraction of the synodic month since a new moon (0 new, 0.25 first quarter, 0.5 full, 0.75 last quarter)."""
    ref = pd.Timestamp("2000-01-06 18:14")
    return ((day - ref).total_seconds() / 86400.0 / 29.530588853) % 1.0


@weights("lunar_cycle")
def lunar_cycle(panel, p, extra):
    """QuantConnect Strategy Library (lunar cycle): long from the last quarter to the first quarter
    (around the new moon), cash around the full moon."""
    ph = moon_phase(pd.Timestamp(extra["day"]) + pd.Timedelta(hours=16))
    return _calendar_weight(panel, 1.0 if (ph >= 0.75 or ph < 0.25) else 0.0, panel.names[0])


# ============================================================================== intraday (5-minute) rules
def _session_frame(df, sessions):
    ny = df.index.tz_convert("America/New_York")
    day = ny.normalize().tz_localize(None)
    return day


@signal("noise_area_momentum")
def noise_area_momentum(df, p, info):
    """Zarattini, Aziz & Barbon (2024) 'Beat the Market': long above the noise area's upper bound
    (open or prior close x (1 + 14-day average move from the open at that time of day)), checked
    at :00 and :30; exit below max(upper bound, VWAP); flat at the close."""
    day = _session_frame(df, info.get("sessions"))
    ny = df.index.tz_convert("America/New_York")
    end = ny + pd.Timedelta(minutes=5)
    slot = end.hour * 60 + end.minute
    c = df["close"]
    dopen = df["open"].groupby(day).transform("first")
    move = (c / dopen - 1).abs()
    tab = pd.DataFrame({"day": day, "slot": slot, "move": move.to_numpy()}, index=df.index)
    piv = tab.pivot_table(index="day", columns="slot", values="move", aggfunc="last")
    sig = piv.shift(1).rolling(14, min_periods=10).mean()
    sigma = pd.Series([sig.at[d, s] if (d in sig.index and s in sig.columns) else np.nan
                       for d, s in zip(day, slot)], index=df.index)
    last_close = c.groupby(day).last().shift(1)
    prev_close = pd.Series(day, index=df.index).map(last_close)
    ub = np.maximum(dopen, prev_close) * (1 + sigma)
    tp = (df["high"] + df["low"] + c) / 3
    vwap = (tp * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)
    check = pd.Series((end.minute == 0) | (end.minute == 30), index=df.index)
    entry = check & (c > ub)
    exit_ = check & (c < np.maximum(ub, vwap))
    return _out(df, entry, exit_)


@signal("vwap_trend")
def vwap_trend(df, p, info):
    """Zarattini & Aziz (2023) VWAP trend: long while the 5-minute close is above the session VWAP."""
    day = _session_frame(df, info.get("sessions"))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)
    above = df["close"] > vwap
    return _out(df, above, ~above)


@signal("intraday_momentum_gao")
def intraday_momentum_gao(df, p, info):
    """Gao, Han, Li & Zhou (2018) market intraday momentum: if the return from the prior close to
    10:00 is positive, buy for the last half hour (enter 15:30, flat at the close)."""
    day = _session_frame(df, info.get("sessions"))
    ny = df.index.tz_convert("America/New_York")
    end = ny + pd.Timedelta(minutes=5)
    c = df["close"]
    prev_close = pd.Series(day, index=df.index).map(c.groupby(day).last().shift(1))
    at10 = pd.Series(np.where((end.hour == 10) & (end.minute == 0), c, np.nan), index=df.index)
    r1 = (at10.groupby(day).transform("max") / prev_close - 1)
    known = pd.Series((end.hour > 10) | ((end.hour == 10) & (end.minute >= 0)), index=df.index)
    entry = pd.Series((end.hour == 15) & (end.minute == 30), index=df.index) & known & (r1 > 0)
    return _out(df, entry, _f(df))
