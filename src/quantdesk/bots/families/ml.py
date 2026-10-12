"""Small machine-learning strategies from public repositories, retrained on a fixed weekly calendar
(Mondays 00:00 UTC) using only candles that were complete before the retrain time, so the
signal for a candle never depends on later data. Models train in well under a second."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data import TF
from ..generic import signal, weights
from ..lib import sma
from .crypto_ft import _out


def _retrain_points(index: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the first candle at or after each Monday 00:00 UTC."""
    mondays = pd.date_range(index[0].normalize() - pd.Timedelta(days=7), index[-1] + pd.Timedelta(days=7),
                            freq="W-MON")
    pos = np.searchsorted(index.values, mondays.values)
    return np.unique(pos[(pos > 0) & (pos < len(index))])


def bt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set of backtesting.py's 'Trading with Machine Learning' example (minus its date dummy)."""
    c = df["close"]
    X = pd.DataFrame(index=df.index)
    s10, s20, s50, s100 = sma(c, 10), sma(c, 20), sma(c, 50), sma(c, 100)
    X["x_sma10"] = (c - s10) / c
    X["x_sma20"] = (c - s20) / c
    X["x_sma50"] = (c - s50) / c
    X["x_sma100"] = (c - s100) / c
    X["x_d10_20"] = (s10 - s20) / c
    X["x_d20_50"] = (s20 - s50) / c
    X["x_d50_100"] = (s50 - s100) / c
    X["x_chg"] = c.pct_change(2)
    hlc3 = (df["high"] + df["low"] + c) / 3
    m, sd = hlc3.rolling(20).mean(), hlc3.rolling(20).std()
    X["x_bbu"] = (m + 2 * sd - c) / c
    X["x_bbl"] = (m - 2 * sd - c) / c
    X["x_bbw"] = 4 * sd / c
    X["dow"] = df.index.dayofweek
    X["hour"] = df.index.hour
    return X


def _skip_before(index: pd.DatetimeIndex, frm) -> int:
    """Position from which outputs are needed (the runner passes the first unprocessed candle);
    model blocks that end before it are not fitted, which only saves time."""
    if frm is None:
        return 0
    return int(np.searchsorted(index.values, pd.Timestamp(frm).to_datetime64()))


def _walk_forward(df, X: pd.DataFrame, y: pd.Series, horizon: int, window: int, make_model, predict, frm=None):
    out = pd.Series(np.nan, index=df.index)
    pts = _retrain_points(df.index)
    pts = np.r_[pts, len(df)]
    need = _skip_before(df.index, frm)
    Xv = X.to_numpy()
    yv = y.to_numpy()
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= need:
            continue
        last_known = a - horizon                       # labels up to here are fully observed before candle a
        lo = max(0, last_known - window)
        tr = np.arange(lo, max(lo, last_known))
        tr = tr[~np.isnan(Xv[tr]).any(axis=1) & ~np.isnan(yv[tr])] if len(tr) else tr
        if len(tr) < window * 0.6:
            continue
        model = make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xv[tr], yv[tr])
            rows = np.arange(a, b)
            ok = ~np.isnan(Xv[rows]).any(axis=1)
            if ok.any():
                out.iloc[rows[ok]] = predict(model, Xv[rows[ok]])
    return out


@signal("gbm_stumps")
def gbm_stumps(df, p, info):
    """QuantConnect 'Gradient Boosting Model': 20 depth-1 trees predicting the 10-bar return; trade
    only when the prediction clears costs; hold 10 bars. Features from backtesting.py's example."""
    from sklearn.ensemble import GradientBoostingRegressor
    X = bt_features(df)
    y = df["close"].shift(-10) / df["close"] - 1
    pred = _walk_forward(df, X, y, 10, int(p.get("window", 672)),
                         lambda: GradientBoostingRegressor(n_estimators=20, max_depth=1, random_state=0),
                         lambda m, x: m.predict(x), info.get("_from"))
    return _out(df, pred > float(p.get("threshold", 0.006)), pd.Series(False, index=df.index))


@signal("knn_direction")
def knn_direction(df, p, info):
    """backtesting.py 'Trading with Machine Learning': 7-nearest-neighbours on the 48-bar direction
    (|r| < 0.4% labelled 0), trained on the last 400 bars."""
    from sklearn.neighbors import KNeighborsClassifier
    X = bt_features(df)
    fwd = df["close"].shift(-48) / df["close"] - 1
    y = pd.Series(np.where(fwd > 0.004, 1, np.where(fwd < -0.004, -1, 0)), index=df.index).where(fwd.notna())
    pred = _walk_forward(df, X, y, 48, int(p.get("window", 400)),
                         lambda: KNeighborsClassifier(7), lambda m, x: m.predict(x), info.get("_from"))
    o = _out(df, pred == 1, pd.Series(False, index=df.index))
    o["sl_pct"], o["tp_pct"] = 0.015, 0.015          # the example's 0.4% bracket widened to clear fees
    return o


def _forward_filter(model, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities P(state_t | x_1..x_t) (no smoothing, so no look-ahead)."""
    logl = model._compute_log_likelihood(X)
    n, k = logl.shape
    alpha = np.zeros((n, k))
    prev = model.startprob_
    for t in range(n):
        pr = prev if t == 0 else alpha[t - 1] @ model.transmat_
        a = pr * np.exp(logl[t] - logl[t].max())
        alpha[t] = a / a.sum()
    return alpha


@signal("hmm_regime")
def hmm_regime(df, p, info):
    """Machine Learning for Trading (S. Jansen) 2-state Gaussian HMM on [daily log return x100,
    21-day volatility]; refit weekly on up to 3 years; hold while P(calm) > 0.7, out below 0.3."""
    from hmmlearn.hmm import GaussianHMM
    lr = np.log(df["close"]).diff() * 100
    vol = lr.rolling(21).std()
    X = pd.concat([lr, vol], axis=1).to_numpy()
    prob = np.full(len(df), np.nan)
    pts = np.r_[_retrain_points(df.index), len(df)]
    need = _skip_before(df.index, info.get("_from"))
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= need:
            continue
        lo = max(0, a - int(p.get("window", 1095)))
        tr = X[lo:a]
        tr = tr[~np.isnan(tr).any(axis=1)]
        if len(tr) < 300:
            continue
        m = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, tol=1e-4, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m.fit(tr)
            except Exception:
                continue
        calm = int(np.argmin([np.sqrt(np.diag(c))[1] if c.ndim == 2 else c[1] for c in m.covars_]))
        seq_lo = lo
        seq = X[seq_lo:b]
        ok = ~np.isnan(seq).any(axis=1)
        first = int(np.argmax(ok))
        alpha = _forward_filter(m, seq[first:])
        idx = seq_lo + first + np.arange(len(alpha))
        sel = idx >= a
        prob[idx[sel]] = alpha[sel, calm]
    pc = pd.Series(prob, index=df.index)
    return _out(df, pc > 0.7, pc < 0.3)


@weights("gnb_cross_section")
def gnb_cross_section(panel, p, extra):
    """QuantConnect 'Gaussian Naive Bayes Model' (Lu 2016), on the crypto universe: features are the
    last 4 daily open-to-close returns of every coin; one model per coin predicts whether the next
    day is up; hold the coins predicted up, equal weight; retrained weekly on 100 days."""
    from sklearn.naive_bayes import GaussianNB
    oc = (panel.close / panel.open - 1)
    feats = pd.concat([oc.shift(k).add_suffix(f"_l{k}") for k in range(4)], axis=1)
    fwd = (panel.close.shift(-1) / panel.open.shift(-1) - 1)       # next day's open-to-close
    idx = panel.index
    w = pd.DataFrame(0.0, index=idx, columns=panel.names)
    pts = np.r_[_retrain_points(idx), len(idx)]
    Xall = feats.to_numpy()
    need = _skip_before(idx, extra.get("_from"))
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= need:
            continue
        last_known = a - 1
        lo = max(0, last_known - int(p.get("window", 100)))
        rows = np.arange(lo, last_known)
        rows = rows[~np.isnan(Xall[rows]).any(axis=1)] if len(rows) else rows
        if len(rows) < 60:
            continue
        preds = {}
        for coin in panel.names:
            y = (fwd[coin].to_numpy()[rows] > 0).astype(int)
            if len(set(y)) < 2:
                continue
            m = GaussianNB().fit(Xall[rows], y)
            r_ = np.arange(a, b)
            ok = ~np.isnan(Xall[r_]).any(axis=1)
            pr = np.zeros(len(r_))
            if ok.any():
                pr[ok] = m.predict(Xall[r_][ok])
            preds[coin] = pr
        if preds:
            up = pd.DataFrame(preds, index=idx[a:b]).reindex(columns=panel.names, fill_value=0)
            cnt = up.sum(axis=1).replace(0, np.nan)
            w.iloc[a:b] = up.div(cnt, axis=0).fillna(0.0).to_numpy()
    return w
