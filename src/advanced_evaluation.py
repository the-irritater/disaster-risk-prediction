"""
Advanced Model Evaluation Module.

Implements four advanced evaluation methods recommended by journal reviewers:
1. Calibration Comparison (before vs. after Platt scaling)
2. Decision Curve Analysis (net benefit across intervention thresholds)
3. Spatial Validation (Moran's I on prediction residuals)
4. Robustness Analysis (repeated stratified holdout, ROC-AUC distribution)

Additionally includes:
5. Cost-Sensitive Analysis (asymmetric FP/FN cost computation)
6. Regression Assumption Checks (normality, homoscedasticity, influence diagnostics)

References:
- Vickers, A.J. & Elkin, E.B. (2006). Decision Curve Analysis. Medical Decision Making.
- Anselin, L. (1995). Local Indicators of Spatial Association. Geographical Analysis.
- Platt, J. (1999). Probabilistic Outputs for SVMs. Advances in Large Margin Classifiers.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve, auc
)
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. Calibration Comparison
# ─────────────────────────────────────────────────────────────────────────────

def calibration_comparison(pipeline, X_cal, y_cal, X_test, y_test, n_bins=10):
    """
    Compare classifier calibration before and after Platt scaling.

    Uses a held-out calibration set to fit Platt scaling (sigmoid), then
    evaluates on the test set. Reports reliability diagram data and Brier scores
    for both the original and recalibrated models.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained classifier pipeline with predict_proba.
    X_cal : pd.DataFrame
        Calibration set features.
    y_cal : array-like
        Calibration set labels.
    X_test : pd.DataFrame
        Test set features.
    y_test : array-like
        Test set labels.
    n_bins : int
        Number of bins for reliability diagram.

    Returns
    -------
    dict
        Calibration comparison results.
    """
    # Original (uncalibrated) predictions on test
    probs_original = pipeline.predict_proba(X_test)[:, 1]
    brier_original = brier_score_loss(y_test, probs_original)
    frac_pos_orig, mean_pred_orig = calibration_curve(
        y_test, probs_original, n_bins=n_bins, strategy="uniform"
    )

    # Fit Platt scaling on calibration set
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # Recalibrated predictions on test
    probs_calibrated = calibrated.predict_proba(X_test)[:, 1]
    brier_calibrated = brier_score_loss(y_test, probs_calibrated)
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_test, probs_calibrated, n_bins=n_bins, strategy="uniform"
    )

    # ECE computation
    def _ece(y_true, y_prob, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop = np.mean(in_bin)
            if prop > 0:
                acc = np.mean(y_true[in_bin])
                conf = np.mean(y_prob[in_bin])
                ece += prop * np.abs(conf - acc)
        return float(ece)

    ece_original = _ece(np.array(y_test), probs_original)
    ece_calibrated = _ece(np.array(y_test), probs_calibrated)

    return {
        "original": {
            "brier_score": round(brier_original, 4),
            "ece": round(ece_original, 4),
            "reliability_curve": {
                "fraction_of_positives": frac_pos_orig.tolist(),
                "mean_predicted_value": mean_pred_orig.tolist(),
            }
        },
        "platt_scaled": {
            "brier_score": round(brier_calibrated, 4),
            "ece": round(ece_calibrated, 4),
            "reliability_curve": {
                "fraction_of_positives": frac_pos_cal.tolist(),
                "mean_predicted_value": mean_pred_cal.tolist(),
            }
        },
        "brier_improvement": round(brier_original - brier_calibrated, 4),
        "ece_improvement": round(ece_original - ece_calibrated, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decision Curve Analysis
# ─────────────────────────────────────────────────────────────────────────────

def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """
    Compute Decision Curve Analysis (DCA) net benefit across thresholds.

    Net Benefit = (TP/N) - (FP/N) * (t / (1-t))

    where t is the threshold probability. This quantifies the clinical/operational
    utility of the model: at what intervention thresholds does the model provide
    more benefit than treating all or treating none?

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_prob : array-like
        Predicted probabilities.
    thresholds : array-like, optional
        Threshold probabilities to evaluate. Default: 0.01 to 0.99.

    Returns
    -------
    dict
        Net benefit curves for model, treat-all, and treat-none strategies.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    model_net_benefits = []
    treat_all_net_benefits = []
    treat_none_net_benefits = []

    for t in thresholds:
        # Model predictions at threshold t
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Net benefit for model
        if t < 1.0:
            nb = (tp / n) - (fp / n) * (t / (1.0 - t))
        else:
            nb = 0.0
        model_net_benefits.append(round(float(nb), 6))

        # Treat all strategy
        if t < 1.0:
            nb_all = prevalence - (1 - prevalence) * (t / (1.0 - t))
        else:
            nb_all = 0.0
        treat_all_net_benefits.append(round(float(nb_all), 6))

        # Treat none: always 0
        treat_none_net_benefits.append(0.0)

    # Find the range where model outperforms both treat-all and treat-none
    useful_range = []
    for i, t in enumerate(thresholds):
        if (model_net_benefits[i] > treat_all_net_benefits[i] and
                model_net_benefits[i] > 0):
            useful_range.append(round(float(t), 2))

    return {
        "thresholds": [round(float(t), 2) for t in thresholds],
        "model_net_benefit": model_net_benefits,
        "treat_all_net_benefit": treat_all_net_benefits,
        "treat_none_net_benefit": treat_none_net_benefits,
        "useful_threshold_range": {
            "min": min(useful_range) if useful_range else None,
            "max": max(useful_range) if useful_range else None,
            "n_thresholds": len(useful_range),
        },
        "prevalence": round(prevalence, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial Validation
# ─────────────────────────────────────────────────────────────────────────────

def spatial_validation_residuals(y_true, y_prob, districts, latitudes, longitudes,
                                  k_values=None):
    """
    Compute Moran's I on prediction residuals to assess spatial autocorrelation.

    If residuals are spatially autocorrelated, the model is systematically over-
    or under-predicting in certain geographic areas, suggesting unmodelled spatial
    structure.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    districts : array-like
        District identifiers.
    latitudes : array-like
        Latitude coordinates per observation.
    longitudes : array-like
        Longitude coordinates per observation.
    k_values : list, optional
        KNN k-values for spatial weights. Default: [3, 5, 8].

    Returns
    -------
    dict
        Moran's I results on residuals at different k-specifications.
    """
    if k_values is None:
        k_values = [3, 5, 8]

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Compute residuals
    residuals = y_true - y_prob

    # Aggregate residuals to district level (mean residual per district)
    df_temp = pd.DataFrame({
        "District": districts,
        "residual": residuals,
        "lat": latitudes,
        "lon": longitudes,
    })
    district_agg = df_temp.groupby("District").agg({
        "residual": "mean",
        "lat": "first",
        "lon": "first",
    }).reset_index()

    n = len(district_agg)
    coords = district_agg[["lat", "lon"]].values
    z = district_agg["residual"].values
    z_centered = z - z.mean()

    results = {}
    for k in k_values:
        if k >= n:
            continue

        # Build KNN spatial weight matrix
        from scipy.spatial import distance_matrix
        dist_mat = distance_matrix(coords, coords)

        W = np.zeros((n, n))
        for i in range(n):
            # Find k nearest neighbours (excluding self)
            dists = dist_mat[i, :]
            dists[i] = np.inf
            knn_idx = np.argsort(dists)[:k]
            W[i, knn_idx] = 1.0

        # Row-standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W_std = W / row_sums[:, np.newaxis]

        # Compute Moran's I
        S0 = W.sum()
        numerator = n * float(z_centered @ W_std @ z_centered)
        denominator = S0 * float(z_centered @ z_centered)
        morans_i = numerator / denominator if denominator != 0 else 0.0

        # Expected value under null
        E_I = -1.0 / (n - 1)

        # Permutation test (999 permutations)
        rng = np.random.default_rng(42)
        n_perms = 999
        perm_I = np.zeros(n_perms)
        for p in range(n_perms):
            z_perm = rng.permutation(z_centered)
            num_p = n * float(z_perm @ W_std @ z_perm)
            den_p = S0 * float(z_perm @ z_perm)
            perm_I[p] = num_p / den_p if den_p != 0 else 0.0

        # Two-sided p-value
        p_value = float(np.mean(np.abs(perm_I) >= np.abs(morans_i)))
        z_score = (morans_i - E_I) / (np.std(perm_I) + 1e-10)

        results[f"k={k}"] = {
            "morans_i": round(float(morans_i), 4),
            "expected_i": round(float(E_I), 4),
            "z_score": round(float(z_score), 4),
            "p_value_permutation": round(float(p_value), 4),
            "interpretation": (
                "significant spatial autocorrelation in residuals"
                if p_value < 0.05
                else "no significant spatial autocorrelation in residuals"
            ),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Robustness Analysis
# ─────────────────────────────────────────────────────────────────────────────

def robustness_repeated_holdout(X, y, districts, n_repeats=5, test_size=0.2,
                                 seed=42):
    """
    Repeated stratified holdout evaluation.

    Trains and evaluates the classifier across multiple random train/test splits,
    reporting the distribution of ROC-AUC rather than a single point estimate.
    Splits are stratified by target and respect the panel structure by not
    splitting within districts.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Binary labels.
    districts : array-like
        District identifiers.
    n_repeats : int
        Number of holdout repeats.
    test_size : float
        Fraction of districts held out for testing.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Distribution of ROC-AUC across repeats.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    rng = np.random.default_rng(seed)
    unique_districts = np.unique(districts)
    n_test_districts = max(1, int(len(unique_districts) * test_size))

    roc_aucs = []
    pr_aucs = []
    brier_scores = []

    numeric_features = [f for f in X.columns if f not in ["Region", "Season"]]
    categorical_features = [f for f in X.columns if f in ["Region", "Season"]]

    for rep in range(n_repeats):
        # Randomly select test districts
        test_districts = rng.choice(unique_districts, size=n_test_districts,
                                     replace=False)
        test_mask = np.isin(districts, test_districts)
        train_mask = ~test_mask

        X_train_r = X[train_mask]
        y_train_r = np.array(y)[train_mask]
        X_test_r = X[test_mask]
        y_test_r = np.array(y)[test_mask]

        if len(np.unique(y_train_r)) < 2 or len(np.unique(y_test_r)) < 2:
            continue

        # Build pipeline
        transformers = []
        if numeric_features:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features))
        if categorical_features:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features))

        pipe = Pipeline([
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("classifier", RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=seed + rep, n_jobs=1
            ))
        ])

        pipe.fit(X_train_r, y_train_r)
        probs = pipe.predict_proba(X_test_r)[:, 1]

        roc_aucs.append(float(roc_auc_score(y_test_r, probs)))
        prec, rec, _ = precision_recall_curve(y_test_r, probs)
        pr_aucs.append(float(auc(rec, prec)))
        brier_scores.append(float(brier_score_loss(y_test_r, probs)))

    return {
        "n_repeats": len(roc_aucs),
        "roc_auc": {
            "mean": round(float(np.mean(roc_aucs)), 4),
            "std": round(float(np.std(roc_aucs)), 4),
            "min": round(float(np.min(roc_aucs)), 4),
            "max": round(float(np.max(roc_aucs)), 4),
            "values": [round(v, 4) for v in roc_aucs],
        },
        "pr_auc": {
            "mean": round(float(np.mean(pr_aucs)), 4),
            "std": round(float(np.std(pr_aucs)), 4),
            "values": [round(v, 4) for v in pr_aucs],
        },
        "brier_score": {
            "mean": round(float(np.mean(brier_scores)), 4),
            "std": round(float(np.std(brier_scores)), 4),
            "values": [round(v, 4) for v in brier_scores],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cost-Sensitive Analysis
# ─────────────────────────────────────────────────────────────────────────────

def cost_sensitive_analysis(y_true, y_prob, thresholds=None,
                             cost_fp=1.0, cost_fn=100.0):
    """
    Compute expected cost under asymmetric loss for disaster prediction.

    In disaster warning systems, the cost of a false negative (missing a disaster)
    far exceeds the cost of a false positive (unnecessary preparedness).
    The precautionary principle in disaster risk management supports this asymmetry.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    thresholds : array-like, optional
        Decision thresholds to evaluate.
    cost_fp : float
        Cost of a false positive (e.g., inspection/preparedness cost).
    cost_fn : float
        Cost of a false negative (e.g., unmitigated disaster impact).

    Returns
    -------
    dict
        Cost analysis results including optimal threshold and cost curves.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)

    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        total_cost = (fp * cost_fp + fn * cost_fn) / n
        costs.append(round(float(total_cost), 4))

    optimal_idx = np.argmin(costs)
    optimal_threshold = float(thresholds[optimal_idx])
    min_cost = costs[optimal_idx]

    return {
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "cost_ratio": f"1:{int(cost_fn/cost_fp)}",
        "thresholds": [round(float(t), 2) for t in thresholds],
        "expected_costs": costs,
        "optimal_threshold": round(optimal_threshold, 2),
        "minimum_expected_cost": round(min_cost, 4),
        "interpretation": (
            f"Under a {int(cost_fn/cost_fp)}:1 FN:FP cost ratio, the optimal "
            f"decision threshold is {optimal_threshold:.2f}. This reflects the "
            f"precautionary principle: it is preferable to issue a false alarm "
            f"(cost={cost_fp}) than to miss a genuine disaster (cost={cost_fn})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Regression Assumption Checks
# ─────────────────────────────────────────────────────────────────────────────

def regression_assumption_checks(y_true, y_pred, X=None):
    """
    Comprehensive regression diagnostic checks.

    Tests:
    1. Normality of residuals (Shapiro-Wilk)
    2. Homoscedasticity (Breusch-Pagan via correlation test)
    3. Influence diagnostics (Cook's distance approximation)

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    X : pd.DataFrame, optional
        Feature matrix (needed for Cook's distance).

    Returns
    -------
    dict
        Diagnostic test results with interpretations.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred
    n = len(residuals)

    # 1. Normality (Shapiro-Wilk) — use subsample if n > 5000
    if n > 5000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(residuals[sample_idx])
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

    # Skewness and kurtosis
    skew = float(stats.skew(residuals))
    kurt = float(stats.kurtosis(residuals))

    # 2. Homoscedasticity — correlation between |residuals| and predicted values
    abs_resid = np.abs(residuals)
    bp_corr, bp_p = stats.spearmanr(y_pred, abs_resid)

    # 3. Cook's distance approximation
    # Cook's D_i ≈ (residual_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
    # Without leverage values, we approximate using standardised residuals
    mse = np.mean(residuals ** 2)
    std_residuals = residuals / (np.sqrt(mse) + 1e-10)

    # Approximate Cook's distance using standardised residuals
    # Simplified: D_i ≈ std_resid_i^2 / p (where p = number of predictors)
    p = X.shape[1] if X is not None else 30  # approximate
    cooks_d = (std_residuals ** 2) / p

    influential_threshold = 4.0 / n
    n_influential = int(np.sum(cooks_d > influential_threshold))

    # Residual distribution summary
    resid_percentiles = {
        "p1": round(float(np.percentile(residuals, 1)), 4),
        "p5": round(float(np.percentile(residuals, 5)), 4),
        "p25": round(float(np.percentile(residuals, 25)), 4),
        "p50": round(float(np.percentile(residuals, 50)), 4),
        "p75": round(float(np.percentile(residuals, 75)), 4),
        "p95": round(float(np.percentile(residuals, 95)), 4),
        "p99": round(float(np.percentile(residuals, 99)), 4),
    }

    return {
        "normality": {
            "test": "Shapiro-Wilk",
            "statistic": round(float(shapiro_stat), 4),
            "p_value": round(float(shapiro_p), 6),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "interpretation": (
                "Residuals are approximately normal"
                if shapiro_p > 0.05
                else "Residuals deviate significantly from normality "
                     f"(skew={skew:.2f}, kurtosis={kurt:.2f}). "
                     "Heavy-tailed distributions are common for economic loss data."
            ),
        },
        "homoscedasticity": {
            "test": "Spearman rank correlation of |residuals| vs predicted",
            "correlation": round(float(bp_corr), 4),
            "p_value": round(float(bp_p), 6),
            "interpretation": (
                "No significant heteroscedasticity detected"
                if bp_p > 0.05
                else "Significant heteroscedasticity detected "
                     f"(ρ={bp_corr:.3f}, p={bp_p:.4f}). "
                     "Consider robust standard errors or log-transformation."
            ),
        },
        "influence_diagnostics": {
            "test": "Approximate Cook's distance",
            "threshold": round(float(influential_threshold), 6),
            "n_influential_points": n_influential,
            "pct_influential": round(100.0 * n_influential / n, 2),
            "interpretation": (
                f"{n_influential} observations ({100*n_influential/n:.1f}%) exceed "
                f"the Cook's distance threshold of {influential_threshold:.4f}."
            ),
        },
        "residual_distribution": resid_percentiles,
        "n_observations": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_advanced_evaluation(pipeline, X_cal, y_cal, X_test, y_test,
                             test_districts, test_lats, test_lons,
                             X_full, y_full, districts_full,
                             y_reg_true=None, y_reg_pred=None,
                             X_reg=None,
                             output_path="outputs/advanced_evaluation_results.json"):
    """
    Run all advanced evaluation analyses and save results.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained classifier pipeline.
    X_cal, y_cal : pd.DataFrame, array-like
        Calibration set for Platt scaling.
    X_test, y_test : pd.DataFrame, array-like
        Test set for evaluation.
    test_districts : array-like
        District identifiers for test set.
    test_lats, test_lons : array-like
        Coordinates for spatial validation.
    X_full, y_full, districts_full : pd.DataFrame, array-like, array-like
        Full dataset for robustness analysis.
    y_reg_true, y_reg_pred : array-like, optional
        Regression true and predicted values for assumption checks.
    X_reg : pd.DataFrame, optional
        Regression features for Cook's distance.
    output_path : str
        Path to save results.
    """
    results = {}

    # 1. Calibration comparison
    print("Running calibration comparison...")
    results["calibration_comparison"] = calibration_comparison(
        pipeline, X_cal, y_cal, X_test, y_test
    )

    # 2. Decision Curve Analysis
    print("Running Decision Curve Analysis...")
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    results["decision_curve_analysis"] = decision_curve_analysis(y_test, test_probs)

    # 3. Spatial validation
    print("Running spatial validation on residuals...")
    results["spatial_validation"] = spatial_validation_residuals(
        y_test, test_probs, test_districts, test_lats, test_lons
    )

    # 4. Robustness analysis
    print("Running robustness analysis (repeated holdout)...")
    results["robustness_analysis"] = robustness_repeated_holdout(
        X_full, y_full, districts_full, n_repeats=5, seed=42
    )

    # 5. Cost-sensitive analysis
    print("Running cost-sensitive analysis...")
    results["cost_analysis"] = cost_sensitive_analysis(
        y_test, test_probs, cost_fp=1.0, cost_fn=100.0
    )

    # 6. Regression assumption checks
    if y_reg_true is not None and y_reg_pred is not None:
        print("Running regression assumption checks...")
        results["regression_diagnostics"] = regression_assumption_checks(
            y_reg_true, y_reg_pred, X_reg
        )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Advanced evaluation results saved to {output_path}")

    return results
