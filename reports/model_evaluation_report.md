# Model Evaluation Report

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Machine Learning Model Evaluation & Calibration |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.2 |
| **Status** | Research Submission (Simulation-Based) |


> **Simulation disclaimer.** All data in this project are synthetically generated.
> Strong model performance partly reflects recovery of programmed relationships
> rather than validated generalisation to real disaster data.
>
> **Language convention**: "Predicts" and "classifies" describe the model's behaviour
> on simulated data. They do not constitute validated forecasting capability
> for real-world disasters.

---

## 1. Data Partitioning

All splits are **chronological** (no future leakage). December 2025 rows are excluded
because the target variable `Disaster_Next_Month` requires January 2026 data.

| Partition | Period | N rows | % of total | Positive rate |
| --- | --- | --- | --- | --- |
| Train | 2015-01 to 2022-12 | 9,600 | 73.3% | 0.1556 |
| Validation | 2024-01 to 2024-12 | 1,200 | 9.2% | 0.1467 |
| Test | 2025-01 to 2025-11 | 1,100 | 8.4% | 0.1591 |
| **Total usable** | | **13,100** | | |

December 2025 rows dropped: 100

---

## 2. Classification: Disaster Next Month

### 2.1 Validation-Set Model Comparison (threshold = 0.1708)

All three models evaluated on the validation set at the tuned optimal threshold of **0.1708** for a fair comparison:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7317 | 0.3294 | 0.8011 | 0.4669 | 0.8216 | 0.4737 | 0.0993 |
| Random Forest | 0.7592 | 0.3605 | 0.8295 | 0.5026 | 0.859 | 0.5926 | 0.0877 |
| XGBoost | 0.79 | 0.3927 | 0.7898 | 0.5245 | 0.8701 | 0.6289 | 0.0833 |

**Selected model**: XGBoost (highest validation PR-AUC)

### 2.2 Test-Set Results (Single Evaluation)

The selected model (XGBoost) was evaluated exactly once on the held-out
test set with the optimal threshold of **0.1708** (selected on validation
to target ≥ 75% recall).

| Metric | Value | 95% CI (cluster bootstrap) |
| --- | --- | --- |
| ROC-AUC | 0.8587 | [0.8273, 0.8888] |
| PR-AUC | 0.5876 | [0.5085, 0.6671] |
| Precision | 0.3880 | [0.3426, 0.4328] |
| Recall | 0.8114 | [0.7443, 0.8753] |
| F1-score | 0.5250 | [0.4769, 0.5688] |
| F2-score | ~0.68 | — |
| Accuracy | 0.7664 | — |
| Balanced Accuracy | 0.7448 | — |
| MCC | 0.4297 | — |
| Brier score | 0.0937 | — |

**Metric rationale**:
- **F2 score** (β=2): Weighs recall twice as heavily as precision, appropriate for disaster
  prediction where missing events is costlier than false alarms.
- **MCC (Matthews Correlation Coefficient)**: Balanced measure accounting for all four
  confusion matrix quadrants; robust to class imbalance.
- **Balanced Accuracy**: Average of sensitivity and specificity; avoids inflation from
  the majority class.

**Bootstrap CI methodology**: 95% confidence intervals are computed via the **percentile
method** with N=1,000 bootstrap resamples. Resampling is **clustered by district** (100
district clusters resampled with replacement) to respect the panel structure. District-level
resampling was chosen because observations within the same district share unobserved
geographic and socio-economic effects; row-level resampling would underestimate standard
errors.

### 2.3 Confusion Matrix (Test Set)

|  | Predicted Negative | Predicted Positive |
| --- | --- | --- |
| **Actual Negative** | TN = 701 | FP = 224 |
| **Actual Positive** | FN = 33 | TP = 142 |

Total test observations: 1100
- **False Discovery Rate (FDR)**: 0.6120 (61.2% of flagged districts are false alarms). This highlights the operational trade-off: high sensitivity to capture disasters (recall=0.81) comes at the cost of a high false alarm rate.

### 2.4 Precision-Recall Tradeoff Justification

The high FDR is **deliberately accepted** based on the following reasoning:

1. **Asymmetric costs**: In disaster early warning, the cost of a **false negative**
   (failing to warn a population about an imminent disaster) vastly exceeds the cost of
   a **false positive** (triggering unnecessary preparedness measures). Under a conservative
   100:1 FN:FP cost ratio, the optimal threshold is well below 0.50.

2. **Precautionary principle**: The UNDRR Sendai Framework and WHO guidelines for
   health emergencies both endorse the precautionary approach: when the potential harm
   from inaction is severe and irreversible, lower decision thresholds are appropriate.

3. **Operational context**: A false positive triggers an inspection or preparedness
   response (cost: resources and time). A false negative means unmitigated disaster
   impact (cost: lives, infrastructure, economic damage).

4. **Cost-sensitive analysis**: Under the 100:1 cost ratio, the cost-optimal threshold
   is computed in `src/advanced_evaluation.py` and reported in `outputs/advanced_evaluation_results.json`.

### 2.4 Calibration Assessment & Operational Benchmark

| Metric | Value |
| --- | --- |
| Null-model Brier (prevalence baseline) | 0.1338 |
| Model Brier score | 0.0937 |
| Brier Skill Score (BSS) | 0.2998 |

**BSS Interpretation**: The Brier Skill Score of 0.2998 indicates a skillful (>=0.25) improvement over always predicting the baseline rate.
- *Operational Benchmarks*: In real-world early warning systems, BSS > 0.25 is considered highly skillful; BSS of 0.10–0.25 is marginally useful; BSS < 0.10 represents very weak calibration skill. Thus, the model's raw probability values should **not** be used directly for risk pricing or resource allocation without recalibration.

---

## 3. Regression: Economic Loss (Conditional on Disaster)

### 3.1 Validation-Set Model Comparison

| Model | R² | RMSE | MAE | MdAPE (%) |
| --- | --- | --- | --- | --- |
| Ridge | 0.0747 | 549.7703 | 316.5457 | 50.29 |
| Random Forest | 0.1875 | 515.1907 | 313.3027 | 50.5841 |
| XGBoost | 0.1766 | 518.6119 | 335.4331 | 55.8743 |
| Gamma GLM | 0.1495 | 527.0952 | 328.8916 | 53.8109 |
| Tweedie GLM | 0.194 | 513.1162 | 321.7107 | 50.0583 |

**Selected model**: Tweedie GLM

### 3.2 Test-Set Results (Single Evaluation)

The regressor was evaluated exactly once on the held-out test events (disaster-event months):

| Metric | Value | Description |
| --- | --- | --- |
| R² | 0.1693 | Fraction of variance explained |
| R² 95% Bootstrap CI | [0.0821, 0.2447] | Variance of explanation |
| RMSE | 522.5115 | Root Mean Squared Error (M USD) |
| MAE | 357.3308 | Mean Absolute Error (M USD) |
| MdAPE (%) | 58.5% | Median Absolute Percentage Error |
| Mean True Loss | 578.6341 | Mean actual loss of event months (M USD) |
| SD of True Loss | 573.2765 | SD of actual loss of event months (M USD) |

**Diagnostics and Analysis**:
- **MdAPE (Median Absolute Percentage Error)**: We use MdAPE (58.5%) because the standard Mean Absolute Percentage Error (MAPE) is highly sensitive to near-zero denominators (small disaster losses), leading to arithmetic explosion (e.g., >100%). MdAPE is a robust metric indicating that the median relative error is 58.5%.
- **RMSE Context**: The RMSE of 522.51 is high relative to the mean true loss (578.63) and represents a substantial portion of the standard deviation (573.28). The model has limited predictive power (R² = 0.1693).
- **Residual Diagnostics**: Shapiro-Wilk test on test residuals yields W = 0.8571, p = < 0.001, indicating residuals deviate significantly from normality, exhibiting a heavy-tailed distribution typical of disaster losses.

Event counts — Train: 1494, Validation: 362, Test: 180

> **Explicit Acknowledgment**: The R² of 0.17 is low — the regression explains only
> ~17% of variance in economic loss. This is scientifically expected: disaster economic
> losses are inherently heavy-tailed (Shapiro-Wilk p < 0.001, indicating significant
> deviation from normality), driven largely by unobserved factors such as exact impact
> location, local response quality, and pre-existing infrastructure condition.
> The regression is presented as an **exploratory association analysis**, not a
> reliable predictor. See `outputs/advanced_evaluation_results.json` for full
> assumption checks (normality, homoscedasticity, Cook's distance).

---

## 4. Explainability and District Typologies

### 4.1 SHAP Feature Importance (Top 10)

SHAP values indicate the model's reliance on each feature for prediction. Higher mean
|SHAP| values mean greater predictive contribution. These are **not** causal effects.

| Feature | Mean |SHAP| |
| --- | --- |
| Distance_From_Coast_km | 0.6016 |
| Season_Spring | 0.5497 |
| Season_Winter | 0.1737 |
| Temperature_Anomaly | 0.1652 |
| Rainfall_Anomaly | 0.1632 |
| Population_Density | 0.1356 |
| Drought_Index | 0.0931 |
| Wind_Speed_kmph | 0.0772 |
| Season_Monsoon | 0.0727 |
| Previous_Month_Hazard_Severity | 0.0647 |

### 4.2 District Clustering (k=4 Recommended)

**Method**: K-Means on standardised district-level score profiles (Hazard, Exposure, Vulnerability, Preparedness).
**Diagnostics**:
- **best_k** (by silhouette): 7 (silhouette = 0.2374)
- **Silhouette Warning**: All tested k configurations achieve silhouette scores < 0.25 (e.g., silhouette is 0.2236 for k=4). This indicates **weak cluster separation** — districts lie along a multidimensional continuum rather than forming discrete, well-separated clusters.
- **Operational Recommendation**: Although k=7 achieves the highest silhouette score, we recommend **k=4** for policy implementation. It results in larger, more balanced cluster sizes (min cluster size is 21) compared to k=7 which creates thin groups of size ~10.

**Recommended Typologies (k=4)**:

| Cluster | Typology Label | Size |
| --- | --- | --- |
| Cluster 0 | High-Exposure Well-Prepared | 21 districts |
| Cluster 1 | High-Exposure Low-Capacity | 27 districts |
| Cluster 2 | High-Vulnerability Low-Capacity | 27 districts |
| Cluster 3 | Low-Risk Well-Prepared | 25 districts |

---

## 5. Methodological Safeguards

| Concern | Mitigation |
| --- | --- |
| Data leakage | Post-event impacts excluded from predictors; scalers fit on training only |
| Temporal leakage | Chronological split; no future data in training |
| Multiple testing | Validation set used for model/threshold selection; test set evaluated once |
| Panel dependency | Bootstrap CIs resampled by district, not by row (N=1000, percentile method) |
| Overfitting | XGBoost with balanced class weights; threshold optimised on validation |
| Synthetic data | All findings conditional on the simulated data-generating process |
| Class imbalance | Balanced class weights, threshold tuned for ≥75% recall, F2 score reported |

---

## 6. Risk Index Circularity Warning

> **This is a serious methodological concern that must be understood when interpreting results.**

The Disaster_Risk_Score is computed as:

```
Risk = 0.30·Hazard + 0.25·Exposure + 0.25·Vulnerability + 0.20·Preparedness_Deficit
```

where each component score is itself derived from the same underlying features used as
ML predictors. This creates **partial circularity**:

| Target Variable | Circularity Risk | Explanation |
| --- | --- | --- |
| Disaster_Risk_Score | ⚠️ **HIGH** | Model learns the weighted-sum formula that created the target |
| Risk_Category | ⚠️ **HIGH** | Quantile-binned risk score; same circularity |
| **Disaster_Next_Month** | ✅ **LOW** | Derived from the hazard probability model (logistic), not the risk index |
| Economic_Loss | ✅ **LOW** | Generated from the impact model conditional on disaster occurrence |

**Key distinction**: The primary classification target (`Disaster_Next_Month`) does NOT
suffer from index circularity. It is generated by a separate logistic hazard probability
model that depends on environmental triggers and geography, not on the risk index.
The risk index is a descriptive summary used for district profiling, not as a prediction
target for the ML classifier.

The regression target (Economic_Loss) is also circularity-free: it is generated
conditionally on disaster occurrence from the impact model.

---

## 7. Clustering Methodology

### 7.1 Method Selection

**Algorithm**: K-Means on standardised district-level score profiles (Hazard, Exposure,
Vulnerability, Preparedness). Standardisation via z-scores ensures equal contribution.

### 7.2 Choice of K

| Diagnostic | Method | Result |
| --- | --- | --- |
| Silhouette analysis | k=2 through k=7 | All scores < 0.25 (weak separation) |
| Best k (silhouette) | k=7 | Silhouette = 0.2374 |
| Recommended k | k=4 | Silhouette = 0.2236 (marginally lower, but balanced sizes) |
| Davies-Bouldin | Computed for all k | Supports k=4–7 range |
| Inertia (elbow) | Computed alongside silhouette | No sharp elbow (consistent with weak separation) |

**Rationale for k=4 over k=7**: k=7 produces thin clusters of ~10 districts, limiting
policy utility. k=4 yields balanced clusters of 21–27 districts with only a marginal
silhouette decrease (0.2374 → 0.2236).

### 7.3 Cluster Quality Disclosure

> **Silhouette scores < 0.25 indicate that districts lie along a multidimensional continuum
> rather than forming discrete, well-separated groups.** The cluster typologies should be
> interpreted as approximate characterisations of a continuous risk landscape, not as
> hard category boundaries.

**Stability considerations**: Cluster assignments are deterministic (fixed random_state=42).
Formal stability testing (e.g., consensus clustering, bootstrap stability) is identified
as future work.

---

## 8. External Applicability

> The pipeline is **transferable**; the numerical results are **not**.

The model training architecture, evaluation methodology, and reporting framework can
be applied to real disaster datasets. All specific coefficients, thresholds, and
performance metrics would need to be re-estimated on real data.

See `reports/statistical_analysis_report.md` §7 for full external applicability discussion.
