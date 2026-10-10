import math

import numpy as np
import pandas as pd
import pytest

from quantdesk.backtest import metrics as M


def test_sharpe_and_drawdown_on_known_series():
    idx = pd.bdate_range("2020-01-01", periods=505)
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, len(idx) - 1)
    eq = pd.Series(1000 * np.cumprod(np.r_[1, 1 + r]), index=idx)
    st = M.perf_stats(eq, 252)
    assert st["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252), rel=1e-6)
    assert st["max_drawdown"] == pytest.approx(float((eq / eq.cummax() - 1).min()))
    assert 0 <= st["psr_0"] <= 1


def test_psr_monotonic_in_sharpe_and_length():
    assert M.psr(0.05, 1000, 0, 3) > M.psr(0.02, 1000, 0, 3)
    assert M.psr(0.05, 2000, 0, 3) > M.psr(0.05, 500, 0, 3)
    assert M.psr(0.0, 1000, 0, 3) == pytest.approx(0.5)


def test_expected_max_sharpe_grows_with_trials():
    assert M.expected_max_sharpe(1, 0.25) == 0.0
    assert M.expected_max_sharpe(100, 0.25) > M.expected_max_sharpe(10, 0.25) > 0


def test_deflated_sharpe_is_lower_with_more_trials():
    d1, _ = M.deflated_sharpe(0.06, 2500, 0, 3, [1.0], 252)
    d2, star = M.deflated_sharpe(0.06, 2500, 0, 3, [1.0, 0.2, -0.3, 0.6, 0.9, 0.1], 252)
    assert star > 0 and d2 < d1


def test_yearly_and_ex_best_year():
    idx = pd.to_datetime(["2020-01-01", "2020-12-31", "2021-12-31", "2022-12-30", "2023-12-29"])
    eq = pd.Series([100, 110, 99, 198, 198], index=idx)
    y = M.yearly_returns(eq)
    assert y.loc[2020] == pytest.approx(0.10)
    assert y.loc[2022] == pytest.approx(1.0)
    assert M.ex_best_year_cagr(y) < 0.05


def test_bootstrap_ci_brackets_point_estimate():
    idx = pd.bdate_range("2015-01-01", periods=2000)
    rng = np.random.default_rng(1)
    eq = pd.Series(1000 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    lo, hi = M.bootstrap_sharpe_ci(eq, 252, n_boot=300)
    sr = M.perf_stats(eq, 252)["sharpe"]
    assert lo < sr < hi
