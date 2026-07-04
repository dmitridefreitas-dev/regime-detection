"""Regime-aware exposure, evaluated with the same discipline as honest-backtester.

The rule is deliberately simple: hold exposure equal to the filtered
probability of the calm state (clipped to [floor, cap]). The evaluation
mirrors the honest-backtester engine's guarantees in miniature: positions
are lagged at least one bar and turnover is charged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def exposure_from_prob(prob_calm: pd.Series, floor: float = 0.0, cap: float = 1.0) -> pd.Series:
    """Map P(calm) directly to target exposure."""
    if not 0.0 <= floor <= cap <= 1.5:
        raise ValueError(f"need 0 <= floor <= cap, got floor={floor}, cap={cap}")
    return prob_calm.clip(lower=floor, upper=cap)


def evaluate(
    close: pd.Series,
    positions: pd.Series,
    cost_bps: float = 2.0,
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Lagged, cost-aware daily evaluation. Returns a per-day frame.

    Same honesty rules as the honest-backtester engine: execution_lag >= 1
    is enforced (a position cannot earn the return of the bar its signal
    was computed on) and every unit of turnover pays cost_bps.
    """
    if execution_lag < 1:
        raise ValueError("execution_lag must be >= 1 (no same-bar fills)")
    if not close.index.equals(positions.index):
        raise ValueError("close and positions must share the same index")

    asset_returns = close.pct_change().fillna(0.0)
    held = positions.shift(execution_lag).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    net = held * asset_returns - turnover * (cost_bps / 1e4)
    return pd.DataFrame({
        "asset_return": asset_returns,
        "held": held,
        "turnover": turnover,
        "net_return": net,
    })


def summarize(returns: pd.Series) -> dict[str, float]:
    """Annualised summary of a daily net-return series."""
    equity = float((1.0 + returns).cumprod().iloc[-1])
    years = len(returns) / TRADING_DAYS
    vol = returns.std(ddof=1)
    downside = (1.0 + returns).cumprod()
    return {
        "annual_return": equity ** (1.0 / years) - 1.0 if equity > 0 else -1.0,
        "annual_vol": float(vol * math.sqrt(TRADING_DAYS)),
        "sharpe": float(returns.mean() / vol * math.sqrt(TRADING_DAYS))
        if vol > 0 else float("nan"),
        "max_drawdown": float((downside / downside.cummax() - 1.0).min()),
    }
