"""
Ablation Study and Permutation Importance Module.

Implements feature-group ablation analysis to quantify the contribution of each
feature family to the classifier's predictive performance. Also includes
permutation importance for individual feature-level analysis.

Feature Groups:
- Environmental: Rainfall_Anomaly, Temperature_Anomaly, Wind_Speed_kmph, etc.
- Exposure: Population_Density, Urbanisation_Rate, Infrastructure_Density
- Vulnerability: Poverty_Rate, Housing_Quality_Index, Healthcare_Access_Index, etc.
- Preparedness: Shelter/Hospital/Rescue rates, EWS, Evacuation, Response Time, DPI
- Temporal/Lag: Previous_Month_Disaster_Occurred, Previous_Month_Hazard_Severity, Rolling count

Methodology:
- For each group, retrain the classifier with that group's features removed.
- Compare ROC-AUC and PR-AUC on validation set against the full-feature baseline.
- Delta metrics indicate each group's marginal contribution.
- Permutation importance provides individual feature-level ranking.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.inspection import permutation_importance


# Feature group definitions
FEATURE_GROUPS = {
    "Environmental": [
        "Rainfall_Anomaly", "Temperature_Anomaly", "Wind_Speed_kmph",
        "River_Level_Metres", "Soil_Moisture", "Drought_Index", "Vegetation_Index",
        "Elevation_Metres", "Slope_Degrees", "Distance_From_Coast_km",
        "Distance_From_River_km", "Seismic_Activity_Index",
    ],
    "Exposure": [
        "Population_Density", "Urbanisation_Rate", "Infrastructure_Density",
    ],
    "Vulnerability": [
        "Poverty_Rate", "Elderly_Population_Percentage", "Child_Population_Percentage",
        "Housing_Quality_Index", "Healthcare_Access_Index", "Road_Access_Index",
        "Communication_Access_Index",
    ],
    "Preparedness": [
        "Shelter_Rate_per_100k", "Hospital_Rate_per_100k", "Rescue_Team_Rate_per_100k",
        "Early_Warning_System", "Evacuation_Plan_Available",
        "Emergency_Response_Time_Minutes", "Disaster_Preparedness_Index",
    ],
    "Temporal_Lag": [
        "Previous_Month_Disaster_Occurred", "Previous_Month_Hazard_Severity",
        "Rolling_12_Month_Disaster_Count",
    ],
}


def _compute_metrics(y_true, y_prob):
    """Compute ROC-AUC and PR-AUC from true labels and predicted probabilities."""
    roc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr = auc(rec, prec)
    return {"roc_auc": round(float(roc), 4), "pr_auc": round(float(pr), 4)}


def run_ablation_study(train_classifier_fn, X_train, y_train, X_val, y_val,
                       all_features, categorical_features=None, seed=42):
    """
    Run feature-group ablation study.

    For each feature group, removes that group from the feature set,
    retrains the classifier, and measures the performance delta.

    Parameters
    ----------
    train_classifier_fn : callable
        Function that takes (X_train, y_train) and returns a fitted pipeline.
        Should accept a 'feature_subset' parameter or handle missing columns.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series or np.ndarray
        Validation labels.
    all_features : list
        Complete list of feature names.
    categorical_features : list, optional
        Names of categorical features (excluded from ablation groups).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Ablation results with baseline and per-group delta metrics.
    """
    if categorical_features is None:
        categorical_features = ["Region", "Season"]

    # Baseline: train with all features
    from src.train_models import build_preprocessing_pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    print("  Training baseline model (all features)...")
    baseline_pipeline = _train_ablation_model(
        X_train[all_features], y_train, all_features, seed
    )
    baseline_probs = baseline_pipeline.predict_proba(X_val[all_features])[:, 1]
    baseline_metrics = _compute_metrics(y_val, baseline_probs)
    print(f"  Baseline: ROC-AUC={baseline_metrics['roc_auc']}, PR-AUC={baseline_metrics['pr_auc']}")

    # Ablation: remove each group
    ablation_results = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        # Features remaining after removing this group
        remaining = [f for f in all_features if f not in group_features]
        if len(remaining) == 0:
            continue

        print(f"  Ablating group: {group_name} ({len(group_features)} features removed)...")
        ablated_pipeline = _train_ablation_model(
            X_train[remaining], y_train, remaining, seed
        )
        ablated_probs = ablated_pipeline.predict_proba(X_val[remaining])[:, 1]
        ablated_metrics = _compute_metrics(y_val, ablated_probs)

        delta_roc = round(baseline_metrics["roc_auc"] - ablated_metrics["roc_auc"], 4)
        delta_pr = round(baseline_metrics["pr_auc"] - ablated_metrics["pr_auc"], 4)

        ablation_results[group_name] = {
            "features_removed": group_features,
            "n_features_removed": len(group_features),
            "n_features_remaining": len(remaining),
            "roc_auc": ablated_metrics["roc_auc"],
            "pr_auc": ablated_metrics["pr_auc"],
            "delta_roc_auc": delta_roc,
            "delta_pr_auc": delta_pr,
        }
        print(f"    Result: ROC-AUC={ablated_metrics['roc_auc']} (Δ={delta_roc}), "
              f"PR-AUC={ablated_metrics['pr_auc']} (Δ={delta_pr})")

    return {
        "baseline": baseline_metrics,
        "ablation": ablation_results,
    }


def _train_ablation_model(X_train, y_train, features, seed):
    """Train a Random Forest classifier for ablation with the given feature subset."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier

    numeric_features = [f for f in features if f not in ["Region", "Season"]]
    categorical_features = [f for f in features if f in ["Region", "Season"]]

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def run_permutation_importance(pipeline, X_val, y_val, feature_names,
                                n_repeats=10, seed=42):
    """
    Compute permutation importance for all features.

    Permutation importance measures the decrease in model performance when a single
    feature's values are randomly shuffled, breaking the relationship between the
    feature and the target.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained classifier pipeline.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series or np.ndarray
        Validation labels.
    feature_names : list
        Names of features (for labelling).
    n_repeats : int
        Number of permutation repeats.
    seed : int
        Random seed.

    Returns
    -------
    list of dict
        Feature importance results sorted by mean importance (descending).
    """
    print("  Computing permutation importance...")
    result = permutation_importance(
        pipeline, X_val, y_val,
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=seed,
        n_jobs=1
    )

    importance_results = []
    for i, name in enumerate(feature_names):
        importance_results.append({
            "feature": name,
            "importance_mean": round(float(result.importances_mean[i]), 6),
            "importance_std": round(float(result.importances_std[i]), 6),
        })

    # Sort by mean importance descending
    importance_results.sort(key=lambda x: x["importance_mean"], reverse=True)
    return importance_results


def run_full_ablation_analysis(train_classifier_fn, X_train, y_train,
                                X_val, y_val, pipeline, all_features,
                                output_path="outputs/ablation_results.json",
                                seed=42):
    """
    Run complete ablation analysis: group ablation + permutation importance.

    Parameters
    ----------
    train_classifier_fn : callable
        Function to train a classifier.
    X_train, y_train : pd.DataFrame, array-like
        Training data.
    X_val, y_val : pd.DataFrame, array-like
        Validation data.
    pipeline : sklearn Pipeline
        Pre-trained full-feature pipeline.
    all_features : list
        Complete feature list.
    output_path : str
        Path to save results.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Combined ablation and permutation importance results.
    """
    print("Running ablation study...")
    ablation = run_ablation_study(
        train_classifier_fn, X_train, y_train, X_val, y_val,
        all_features, seed=seed
    )

    print("Running permutation importance...")
    perm_importance = run_permutation_importance(
        pipeline, X_val, y_val, all_features,
        n_repeats=10, seed=seed
    )

    results = {
        "ablation_study": ablation,
        "permutation_importance": perm_importance,
        "feature_groups": {k: v for k, v in FEATURE_GROUPS.items()},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved to {output_path}")

    return results
