"""Performance metrics, Probabilistic / Deflated Sharpe Ratio (Bailey & Lopez de Prado), bootstrap CIs."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def perf_stats(equity: pd.Series, ppy: float) -> dict:
    equity = equity.dropna()
    r = returns(equity)
    if len(r) < 5:
        return {"n_obs": len(r)}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    sd = float(r.std(ddof=1))
    sr_pp = float(r.mean() / sd) if sd > 0 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    dd = equity / equity.cummax() - 1
    mdd = float(dd.min())
    active = r[r != 0]
    sk = float(skew(r, bias=False)) if len(r) > 3 else 0.0
    ku = float(kurtosis(r, fisher=False, bias=False)) if len(r) > 4 else 3.0
    yearly = yearly_returns(equity)
    return {
        "start": str(equity.index[0].date()), "end": str(equity.index[-1].date()), "years": round(years, 2),
        "n_obs": int(len(r)), "total_return": total, "cagr": cagr, "vol": sd * math.sqrt(ppy),
        "sharpe": sr_pp * math.sqrt(ppy), "sharpe_pp": sr_pp,
        "sortino": float(r.mean() / downside * math.sqrt(ppy)) if downside > 0 else float("nan"),
        "max_drawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "hit_rate_days": float((active > 0).mean()) if len(active) else float("nan"),
        "skew": sk, "kurtosis": ku, "time_in_market": float(len(active) / len(r)),
        "psr_0": psr(sr_pp, len(r), sk, ku, 0.0),
        "best_year": float(yearly.max()) if len(yearly) else float("nan"),
        "worst_year": float(yearly.min()) if len(yearly) else float("nan"),
        "ex_best_year_cagr": ex_best_year_cagr(yearly),
        "yearly": {str(k): float(v) for k, v in yearly.items()},
        "longest_dd_days": longest_drawdown_days(equity),
    }


def yearly_returns(equity: pd.Series) -> pd.Series:
    ye = equity.groupby(equity.index.year).last()
    first = equity.iloc[0]
    prev = pd.concat([pd.Series([first], index=[ye.index[0] - 1]), ye.iloc[:-1]])
    out = pd.Series(ye.values / prev.values - 1, index=ye.index)
    return out


def ex_best_year_cagr(yearly: pd.Series) -> float:
    if len(yearly) < 3:
        return float("nan")
    rest = yearly.drop(yearly.idxmax())
    growth = float(np.prod(1 + rest.values))
    return growth ** (1 / len(rest)) - 1 if growth > 0 else -1.0


def longest_drawdown_days(equity: pd.Series) -> int:
    peak_date = equity.index[0]
    peak = equity.iloc[0]
    longest = 0
    for d, v in equity.items():
        if v >= peak:
            peak, peak_date = v, d
        else:
            longest = max(longest, (d - peak_date).days)
    return int(longest)


def psr(sr_pp: float, n: int, sk: float, ku: float, sr_star_pp: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star) given sample SR, length, skew, kurtosis."""
    if n < 3:
        return float("nan")
    denom = 1.0 - sk * sr_pp + (ku - 1.0) / 4.0 * sr_pp ** 2
    if denom <= 0:
        return float("nan")
    z = (sr_pp - sr_star_pp) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe among n independent trials with true SR 0 (False Strategy Theorem)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
                                + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(sr_pp: float, n: int, sk: float, ku: float, trial_sharpes_annual: list[float],
                    ppy: float) -> tuple[float, float]:
    """DSR = PSR against the expected max Sharpe of all trials. Returns (dsr, sr_star_annual)."""
    trials = [s for s in trial_sharpes_annual if s is not None and math.isfinite(s)]
    if len(trials) < 2:
        return psr(sr_pp, n, sk, ku, 0.0), 0.0
    sr_star_ann = expected_max_sharpe(len(trials), float(np.var(trials, ddof=1)))
    return psr(sr_pp, n, sk, ku, sr_star_ann / math.sqrt(ppy)), sr_star_ann


def bootstrap_sharpe_ci(equity: pd.Series, ppy: float, n_boot: int = 1000, block: int = 20,
                        seed: int = 7, alpha: float = 0.10) -> tuple[float, float]:
    """Stationary block bootstrap CI for the annualised Sharpe ratio."""
    r = returns(equity).values
    n = len(r)
    if n < 50:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    resets = rng.random((n_boot, n)) < 1.0 / block
    resets[:, 0] = True
    starts = rng.integers(n, size=(n_boot, n))
    last = np.maximum.accumulate(np.where(resets, t, 0), axis=1)
    idx = (np.take_along_axis(starts, last, axis=1) + (t - last)) % n
    s = r[idx]
    sd = s.std(axis=1, ddof=1)
    out = np.where(sd > 0, s.mean(axis=1) / np.where(sd > 0, sd, 1.0) * math.sqrt(ppy), 0.0)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def relative_stats(equity: pd.Series, bench: pd.Series, ppy: float) -> dict:
    df = pd.concat([equity, bench], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {}
    r = df.pct_change().dropna()
    a, b = r.iloc[:, 0], r.iloc[:, 1]
    ex = a - b
    beta = float(np.cov(a, b)[0, 1] / np.var(b, ddof=1)) if np.var(b) > 0 else float("nan")
    te = float(ex.std(ddof=1) * math.sqrt(ppy))
    bench_total = float(df.iloc[-1, 1] / df.iloc[0, 1] - 1)
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    return {
        "bench_total_return": bench_total,
        "bench_cagr": float((1 + bench_total) ** (1 / years) - 1),
        "excess_cagr": float((df.iloc[-1, 0] / df.iloc[0, 0]) ** (1 / years) - (1 + bench_total) ** (1 / years)),
        "info_ratio": float(ex.mean() / ex.std(ddof=1) * math.sqrt(ppy)) if ex.std() > 0 else float("nan"),
        "tracking_error": te, "beta": beta, "correlation": float(a.corr(b)),
    }


def trade_stats(result) -> dict:
    eq = result.equity
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    avg_eq = float(eq.mean())
    traded = float(result.fills["value_eur"].sum()) if len(result.fills) else 0.0
    costs = result.costs
    explicit = costs.get("commissions", 0.0) + costs.get("spread_slippage", 0.0) + costs.get("fx", 0.0)
    real = [x["pnl_eur"] for x in result.ledger.realized]
    return {
        "n_fills": int(len(result.fills)), "n_rejects": int(len(result.rejects)),
        "turnover_per_year": traded / avg_eq / years if avg_eq > 0 else float("nan"),
        "cost_drag_per_year": explicit / avg_eq / years if avg_eq > 0 else float("nan"),
        "commissions_eur": costs.get("commissions", 0.0), "spread_slippage_eur": costs.get("spread_slippage", 0.0),
        "fx_eur": costs.get("fx", 0.0), "funding_eur": costs.get("funding", 0.0),
        "inference_eur": costs.get("inference", 0.0),
        "closed_trades": len(real), "trade_hit_rate": float(np.mean([p > 0 for p in real])) if real else float("nan"),
        "avg_exposure": float(result.exposure.mean()) if len(result.exposure) else float("nan"),
    }
