"""
Tests for the Uncertainty Quantification module.

Tests:
1. Dirichlet weight samples sum to 1.0 and are properly distributed
2. Monte Carlo risk index produces valid output structure
3. Convergence diagnostics show stabilising estimates at increasing N
4. Bootstrap interval bounds are properly ordered
"""

import numpy as np
import pandas as pd
import pytest
from src.uncertainty import (
    dirichlet_weight_samples,
    monte_carlo_risk_index,
)


@pytest.fixture
def sample_risk_df():
    """Create a small synthetic DataFrame with risk score components."""
    rng = np.random.default_rng(42)
    n_districts = 10
    n_months = 12
    records = []
    for d in range(n_districts):
        for m in range(n_months):
            records.append({
                "District": f"D_{d+1:03d}",
                "Hazard_Score": rng.uniform(10, 80),
                "Exposure_Score": rng.uniform(10, 80),
                "Vulnerability_Score": rng.uniform(10, 80),
                "Preparedness_Deficit_Score": rng.uniform(10, 80),
                "Disaster_Risk_Score": rng.uniform(20, 60),
            })
    return pd.DataFrame(records)


class TestDirichletWeightSamples:
    """Tests for Dirichlet weight sampling."""

    def test_samples_sum_to_one(self):
        """Each weight vector must sum to 1.0."""
        expert = [0.30, 0.25, 0.25, 0.20]
        samples = dirichlet_weight_samples(expert, n_samples=100, seed=42)
        sums = samples.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_correct_shape(self):
        """Output shape must be (n_samples, n_weights)."""
        expert = [0.30, 0.25, 0.25, 0.20]
        samples = dirichlet_weight_samples(expert, n_samples=200, seed=42)
        assert samples.shape == (200, 4)

    def test_means_close_to_expert(self):
        """With high concentration, means should be close to expert weights."""
        expert = [0.30, 0.25, 0.25, 0.20]
        samples = dirichlet_weight_samples(expert, n_samples=5000,
                                           concentration=100, seed=42)
        means = samples.mean(axis=0)
        np.testing.assert_allclose(means, expert, atol=0.02)

    def test_all_positive(self):
        """All weights must be positive (Dirichlet property)."""
        expert = [0.30, 0.25, 0.25, 0.20]
        samples = dirichlet_weight_samples(expert, n_samples=1000, seed=42)
        assert (samples > 0).all()


class TestMonteCarloRiskIndex:
    """Tests for Monte Carlo risk index uncertainty analysis."""

    def test_output_structure(self, sample_risk_df):
        """Output must contain required keys."""
        results = monte_carlo_risk_index(sample_risk_df, n_simulations=50, seed=42)
        assert "district_summaries" in results
        assert "convergence_diagnostics" in results
        assert "rank_stability" in results
        assert "weight_sample_stats" in results

    def test_district_count_matches(self, sample_risk_df):
        """Number of district summaries must match unique districts."""
        results = monte_carlo_risk_index(sample_risk_df, n_simulations=50, seed=42)
        n_districts = sample_risk_df["District"].nunique()
        assert len(results["district_summaries"]) == n_districts

    def test_ci_ordering(self, sample_risk_df):
        """Lower CI bound must be <= upper CI bound for every district."""
        results = monte_carlo_risk_index(sample_risk_df, n_simulations=100, seed=42)
        for d in results["district_summaries"]:
            assert d["ci_lower_2.5"] <= d["ci_upper_97.5"], \
                f"CI ordering violated for {d['District']}"
            assert d["ci_lower_5"] <= d["ci_upper_95"]

    def test_convergence_diagnostics_increasing_n(self, sample_risk_df):
        """Convergence diagnostics should have increasing n_iterations."""
        results = monte_carlo_risk_index(sample_risk_df, n_simulations=500, seed=42)
        ns = [c["n_iterations"] for c in results["convergence_diagnostics"]]
        assert ns == sorted(ns), "Convergence checkpoints must be in ascending order"

    def test_convergence_stabilisation(self, sample_risk_df):
        """Grand mean risk should stabilise as N increases (std decreases)."""
        results = monte_carlo_risk_index(sample_risk_df, n_simulations=1000, seed=42)
        diagnostics = results["convergence_diagnostics"]
        if len(diagnostics) >= 2:
            # Standard deviation across simulations should decrease or remain stable
            first_std = diagnostics[0]["grand_std_across_sims"]
            last_std = diagnostics[-1]["grand_std_across_sims"]
            # Allow some tolerance — convergence doesn't require monotone decrease
            assert last_std < first_std * 2.0, \
                "Grand std should not dramatically increase with more iterations"

    def test_rank_stability_bounded(self, sample_risk_df):
        """Rank stability should be between 0 and 1."""
        results = monte_carlo_risk_index(sample_risk_df, n_simulations=100, seed=42)
        stability = results["rank_stability"]["overall_rank_stability_within_5"]
        assert 0.0 <= stability <= 1.0
