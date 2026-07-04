"""The evaluation shim keeps the honest-backtester guarantees."""

import numpy as np
import pandas as pd
import pytest

from regimes.allocation import evaluate, exposure_from_prob, summarize


def random_walk(n=1000, seed=7) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, n)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)),
                     index=pd.bdate_range("2015-01-01", periods=n))


def test_same_bar_execution_refused():
    close = random_walk()
    flat = pd.Series(0.5, index=close.index)
    with pytest.raises(ValueError, match="same-bar"):
        evaluate(close, flat, execution_lag=0)


def test_cost_accounting_hand_example():
    close = pd.Series([100.0, 100.0, 110.0, 110.0],
                      index=pd.bdate_range("2024-01-01", periods=4))
    signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=close.index)
    result = evaluate(close, signal, cost_bps=10.0)
    assert result.held.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result.net_return.tolist() == pytest.approx([0.0, -0.001, 0.10, -0.001])


def test_exposure_clipping():
    prob = pd.Series([0.0, 0.4, 1.0])
    exposure = exposure_from_prob(prob, floor=0.2, cap=0.9)
    assert exposure.tolist() == [0.2, 0.4, 0.9]
    with pytest.raises(ValueError, match="floor"):
        exposure_from_prob(prob, floor=0.9, cap=0.2)


def test_summarize_matches_hand_computation():
    returns = pd.Series([0.10, -0.50, 0.25],
                        index=pd.bdate_range("2024-01-01", periods=3))
    stats = summarize(returns)
    assert stats["max_drawdown"] == pytest.approx(-0.5)
