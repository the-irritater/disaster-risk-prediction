"""
Tests for the Advanced Evaluation module.

Tests:
1. Decision Curve Analysis produces valid net benefit curves
2. Cost-sensitive analysis finds an optimal threshold
3. Regression assumption checks return complete diagnostic results
4. Spatial validation returns Moran's I results
"""

import numpy as np
import pandas as pd
import pytest
from src.advanced_evaluation import (
    decision_curve_analysis,
    cost_sensitive_analysis,
    regression_assumption_checks,
    spatial_validation_residuals,
)


@pytest.fixture
def binary_predictions():
    """Generate synthetic binary classification predictions."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.binomial(1, 0.15, size=n)
    # Create probabilities that correlate with true labels
    y_prob = np.clip(
        y_true * 0.5 + rng.uniform(0, 0.5, size=n),
        0, 1
    )
    return y_true, y_prob


@pytest.fixture
def regression_predictions():
    """Generate synthetic regression predictions."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.gamma(2.0, 100.0, size=n)
    y_pred = y_true * rng.uniform(0.5, 1.5, size=n) + rng.normal(0, 50, size=n)
    return y_true, y_pred


class TestDecisionCurveAnalysis:
    """Tests for Decision Curve Analysis."""

    def test_output_keys(self, binary_predictions):
        """Output must contain required keys."""
        y_true, y_prob = binary_predictions
        result = decision_curve_analysis(y_true, y_prob)
        assert "thresholds" in result
        assert "model_net_benefit" in result
        assert "treat_all_net_benefit" in result
        assert "treat_none_net_benefit" in result
        assert "useful_threshold_range" in result

    def test_treat_none_is_zero(self, binary_predictions):
        """Treat-none strategy should always have zero net benefit."""
        y_true, y_prob = binary_predictions
        result = decision_curve_analysis(y_true, y_prob)
        assert all(nb == 0.0 for nb in result["treat_none_net_benefit"])

    def test_lengths_match(self, binary_predictions):
        """All curve arrays must have the same length as thresholds."""
        y_true, y_prob = binary_predictions
        result = decision_curve_analysis(y_true, y_prob)
        n = len(result["thresholds"])
        assert len(result["model_net_benefit"]) == n
        assert len(result["treat_all_net_benefit"]) == n
        assert len(result["treat_none_net_benefit"]) == n

    def test_prevalence_correct(self, binary_predictions):
        """Reported prevalence should match actual prevalence."""
        y_true, y_prob = binary_predictions
        result = decision_curve_analysis(y_true, y_prob)
        expected = round(float(np.mean(y_true)), 4)
        assert abs(result["prevalence"] - expected) < 0.001


class TestCostSensitiveAnalysis:
    """Tests for cost-sensitive analysis."""

    def test_output_structure(self, binary_predictions):
        """Output must contain required keys."""
        y_true, y_prob = binary_predictions
        result = cost_sensitive_analysis(y_true, y_prob)
        assert "optimal_threshold" in result
        assert "minimum_expected_cost" in result
        assert "expected_costs" in result
        assert "interpretation" in result

    def test_optimal_threshold_in_range(self, binary_predictions):
        """Optimal threshold must be between 0 and 1."""
        y_true, y_prob = binary_predictions
        result = cost_sensitive_analysis(y_true, y_prob)
        assert 0.0 < result["optimal_threshold"] < 1.0

    def test_cost_asymmetry_effect(self, binary_predictions):
        """Higher FN cost should shift optimal threshold lower (more predictions positive)."""
        y_true, y_prob = binary_predictions
        result_low = cost_sensitive_analysis(y_true, y_prob, cost_fp=1.0, cost_fn=10.0)
        result_high = cost_sensitive_analysis(y_true, y_prob, cost_fp=1.0, cost_fn=1000.0)
        assert result_high["optimal_threshold"] <= result_low["optimal_threshold"]


class TestRegressionAssumptionChecks:
    """Tests for regression diagnostic checks."""

    def test_output_structure(self, regression_predictions):
        """Output must contain all diagnostic sections."""
        y_true, y_pred = regression_predictions
        result = regression_assumption_checks(y_true, y_pred)
        assert "normality" in result
        assert "homoscedasticity" in result
        assert "influence_diagnostics" in result
        assert "residual_distribution" in result

    def test_shapiro_wilk_reported(self, regression_predictions):
        """Shapiro-Wilk test must produce statistic and p-value."""
        y_true, y_pred = regression_predictions
        result = regression_assumption_checks(y_true, y_pred)
        assert "statistic" in result["normality"]
        assert "p_value" in result["normality"]
        assert 0.0 <= result["normality"]["statistic"] <= 1.0

    def test_influential_points_nonnegative(self, regression_predictions):
        """Number of influential points must be non-negative."""
        y_true, y_pred = regression_predictions
        result = regression_assumption_checks(y_true, y_pred)
        assert result["influence_diagnostics"]["n_influential_points"] >= 0


class TestSpatialValidation:
    """Tests for spatial validation of residuals."""

    def test_output_structure(self):
        """Output must contain Moran's I results for each k."""
        rng = np.random.default_rng(42)
        n = 50
        y_true = rng.binomial(1, 0.15, size=n)
        y_prob = rng.uniform(0, 0.3, size=n)
        districts = [f"D_{i:03d}" for i in range(n)]
        lats = rng.uniform(20, 25, size=n)
        lons = rng.uniform(75, 80, size=n)

        result = spatial_validation_residuals(
            y_true, y_prob, districts, lats, lons, k_values=[3, 5]
        )
        assert "k=3" in result
        assert "k=5" in result
        assert "morans_i" in result["k=3"]
        assert "p_value_permutation" in result["k=3"]

    def test_morans_i_bounded(self):
        """Moran's I should be between -1 and 1."""
        rng = np.random.default_rng(42)
        n = 30
        y_true = rng.binomial(1, 0.2, size=n)
        y_prob = rng.uniform(0, 0.3, size=n)
        districts = [f"D_{i:03d}" for i in range(n)]
        lats = rng.uniform(20, 25, size=n)
        lons = rng.uniform(75, 80, size=n)

        result = spatial_validation_residuals(
            y_true, y_prob, districts, lats, lons, k_values=[3]
        )
        mi = result["k=3"]["morans_i"]
        assert -1.5 <= mi <= 1.5  # Theoretical bounds with some tolerance
