"""Tests for performance analytics (performance.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.performance import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    avg_win_loss_ratio,
)


class TestAnnualisedReturn:
    """Tests for CAGR calculation."""

    def test_flat_returns_zero(self):
        equity = np.array([100.0] * 252)
        assert annualised_return(equity) == pytest.approx(0.0)

    def test_doubling_in_one_year(self):
        """$100 → $200 in 252 days ≈ 100% annualised return."""
        equity = np.linspace(100, 200, 252)
        ret = annualised_return(equity)
        assert ret == pytest.approx(1.0, abs=0.02)

    def test_insufficient_data(self):
        assert annualised_return(np.array([100.0])) == 0.0
        assert annualised_return(np.array([])) == 0.0


class TestAnnualisedVolatility:
    """Tests for volatility calculation."""

    def test_zero_vol_for_flat(self):
        equity = np.array([100.0] * 100)
        assert annualised_volatility(equity) == 0.0

    def test_positive_for_volatile(self):
        np.random.seed(42)
        equity = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 252))
        vol = annualised_volatility(equity)
        assert vol > 0.10  # should be roughly 30%+ annualised


class TestSharpeRatio:
    """Tests for Sharpe ratio."""

    def test_zero_vol_returns_zero(self):
        equity = np.array([100.0] * 100)
        assert sharpe_ratio(equity) == 0.0

    def test_positive_sharpe_for_uptrend(self):
        equity = np.linspace(100, 130, 252)
        assert sharpe_ratio(equity, risk_free_rate=0.02) > 0


class TestSortinoRatio:
    """Tests for Sortino ratio."""

    def test_insufficient_data(self):
        assert sortino_ratio(np.array([100.0])) == 0.0

    def test_positive_for_uptrend(self):
        # Use a noisy uptrend that has some negative daily returns
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.01, 252)  # slight upward bias
        equity = 100 * np.cumprod(1 + daily_returns)
        assert sortino_ratio(equity) > 0


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown_for_uptrend(self):
        equity = np.linspace(100, 200, 100)
        result = max_drawdown(equity)
        assert result.max_drawdown_pct == 0.0
        assert result.max_drawdown_duration == 0

    def test_known_drawdown(self):
        """$100 → $80 → $120  →  max DD = -20%."""
        equity = np.array([100.0, 90.0, 80.0, 90.0, 100.0, 120.0])
        result = max_drawdown(equity)
        assert result.max_drawdown_pct == pytest.approx(-0.20)
        assert result.max_drawdown_duration == 3  # bars 1, 2, 3 are below peak

    def test_single_point(self):
        result = max_drawdown(np.array([100.0]))
        assert result.max_drawdown_pct == 0.0


class TestCalmarRatio:
    """Tests for Calmar ratio."""

    def test_no_drawdown_returns_zero(self):
        equity = np.linspace(100, 200, 252)
        assert calmar_ratio(equity) == 0.0  # no drawdown → 0

    def test_positive_for_mixed(self):
        equity = np.array([100, 90, 80, 110, 120, 130, 140, 150])
        cr = calmar_ratio(equity)
        assert cr != 0.0


class TestTradeMetrics:
    """Tests for win rate, profit factor, avg win/loss."""

    def test_win_rate_all_winners(self):
        assert win_rate([10, 20, 5]) == pytest.approx(1.0)

    def test_win_rate_half(self):
        assert win_rate([10, -5, 20, -10]) == pytest.approx(0.5)

    def test_win_rate_empty(self):
        assert win_rate([]) == 0.0

    def test_profit_factor_no_losses(self):
        assert profit_factor([10, 20]) == float("inf")

    def test_profit_factor_known(self):
        """Gross profit 30, gross loss 10 → PF = 3.0."""
        assert profit_factor([10, 20, -5, -5]) == pytest.approx(3.0)

    def test_avg_win_loss_known(self):
        """Avg win = 15, avg loss = 5 → ratio = 3.0."""
        assert avg_win_loss_ratio([10, 20, -5, -5]) == pytest.approx(3.0)

    def test_avg_win_loss_no_losses(self):
        assert avg_win_loss_ratio([10, 20]) == 0.0
