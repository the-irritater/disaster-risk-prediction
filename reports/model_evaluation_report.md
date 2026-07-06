# Model Evaluation Report

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Machine Learning Model Evaluation & Calibration |
| **Project** | Disaster Risk Prediction Dashboard |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.0 |
| **Status** | Research Submission (Simulation-Based) |


> **Simulation disclaimer.** All data in this project are synthetically generated.
> Strong model performance partly reflects recovery of programmed relationships
> rather than validated generalisation to real disaster data.

---

## 1. Data Partitioning

All splits are **chronological** (no future leakage). December 2025 rows are excluded
because the target variable `Disaster_Next_Month` requires January 2026 data.

| Partition | Period | N rows | % of total | Positive rate |
| --- | --- | --- | --- | --- |
| Train | 2015-01 to 2022-12 | 9,600 | 73.3% | 0.2083 |
| Validation | 2023-01 to 2024-12 | 2,400 | 18.3% | 0.2100 |
| Test | 2025-01 to 2025-11 | 1,100 | 8.4% | 0.2273 |
| **Total usable** | | **13,100** | | |

December 2025 rows dropped: 100

---

## 2. Classification: Disaster Next Month

### 2.1 Validation-Set Model Comparison (threshold = 0.1799)

All three models evaluated on the validation set at the tuned optimal threshold of **0.1799** for a fair comparison:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.3696 | 0.2438 | 0.9524 | 0.3882 | 0.7747 | 0.4639 | 0.1913 |
| Random Forest | 0.6792 | 0.3686 | 0.7401 | 0.4921 | 0.767 | 0.5239 | 0.1353 |
| XGBoost | 0.6575 | 0.3522 | 0.752 | 0.4797 | 0.7649 | 0.5258 | 0.1526 |

**Selected model**: XGBoost (highest validation PR-AUC)

### 2.2 Test-Set Results (Single Evaluation)

The selected model (XGBoost) was evaluated exactly once on the held-out
test set with the optimal threshold of **0.1799** (selected on validation
to target ≥ 75% recall).

| Metric | Value | 95% CI (cluster bootstrap) |
| --- | --- | --- |
| ROC-AUC | 0.7537 | [0.7239, 0.7852] |
| PR-AUC | 0.5289 | [0.4741, 0.5833] |
| Precision | 0.3556 | [0.3229, 0.3916] |
| Recall | 0.7680 | [0.7255, 0.8161] |
| F1-score | 0.4861 | [0.4518, 0.5216] |
| Accuracy | 0.6309 | — |
| Brier score | 0.1621 | — |

### 2.3 Confusion Matrix (Test Set)

|  | Predicted Negative | Predicted Positive |
| --- | --- | --- |
| **Actual Negative** | TN = 502 | FP = 348 |
| **Actual Positive** | FN = 58 | TP = 192 |

Total test observations: 1100
- **False Discovery Rate (FDR)**: 0.6444 (64.4% of flagged districts are false alarms). This highlights the operational trade-off: high sensitivity to capture disasters (recall=0.77) comes at the cost of a high false alarm rate.

### 2.4 Calibration Assessment & Operational Benchmark

| Metric | Value |
| --- | --- |
| Null-model Brier (prevalence baseline) | 0.1756 |
| Model Brier score | 0.1621 |
| Brier Skill Score (BSS) | 0.0771 |

**BSS Interpretation**: The Brier Skill Score of 0.0771 indicates a weak (<0.10) improvement over always predicting the baseline rate.
- *Operational Benchmarks*: In real-world early warning systems, BSS > 0.25 is considered highly skillful; BSS of 0.10–0.25 is marginally useful; BSS < 0.10 represents very weak calibration skill. Thus, the model's raw probability values should **not** be used directly for risk pricing or resource allocation without recalibration.

---

## 3. Regression: Economic Loss (Conditional on Disaster)

### 3.1 Validation-Set Model Comparison

| Model | R² | RMSE | MAE | MdAPE (%) |
| --- | --- | --- | --- | --- |
| Ridge | 0.1956 | 516.3848 | 347.2861 | 54.8216 |
| Random Forest | 0.1699 | 524.583 | 350.2203 | 55.0355 |
| XGBoost | 0.0601 | 558.1783 | 378.6319 | 59.7945 |

**Selected model**: Ridge

### 3.2 Test-Set Results (Single Evaluation)

The regressor was evaluated exactly once on the held-out test events (disaster-event months):

| Metric | Value | Description |
| --- | --- | --- |
| R² | 0.1568 | Fraction of variance explained |
| R² 95% Bootstrap CI | [0.0649, 0.2183] | Variance of explanation |
| RMSE | 543.6151 | Root Mean Squared Error (M USD) |
| MAE | 374.5071 | Mean Absolute Error (M USD) |
| MdAPE (%) | 61.9% | Median Absolute Percentage Error |
| Mean True Loss | 580.9105 | Mean actual loss of event months (M USD) |
| SD of True Loss | 592.0063 | SD of actual loss of event months (M USD) |

**Diagnostics and Analysis**:
- **MdAPE (Median Absolute Percentage Error)**: We use MdAPE (61.9%) because the standard Mean Absolute Percentage Error (MAPE) is highly sensitive to near-zero denominators (small disaster losses), leading to arithmetic explosion (e.g., >100%). MdAPE is a robust metric indicating that the median relative error is 61.9%.
- **RMSE Context**: The RMSE of 543.62 is high relative to the mean true loss (580.91) and represents a substantial portion of the standard deviation (592.01). The model has limited predictive power (R² = 0.1568).
- **Residual Diagnostics**: Shapiro-Wilk test on test residuals yields W = 0.8206, p = < 0.001, indicating residuals deviate significantly from normality, exhibiting a heavy-tailed distribution typical of disaster losses.

Event counts — Train: 2008, Validation: 499, Test: 254

---

## 4. Explainability and District Typologies

### 4.1 SHAP Feature Importance (Top 10)

SHAP values indicate the model's reliance on each feature for prediction. Higher mean
|SHAP| values mean greater predictive contribution. These are **not** causal effects.

| Feature | Mean |SHAP| |
| --- | --- |
| Season_Spring | 0.6791 |
| Distance_From_Coast_km | 0.6514 |
| Season_Winter | 0.4068 |
| River_Level_Metres | 0.3574 |
| Previous_Month_Hazard_Severity | 0.3330 |
| Wind_Speed_kmph | 0.2990 |
| Rainfall_Anomaly | 0.2777 |
| Vegetation_Index | 0.2279 |
| Drought_Index | 0.2045 |
| Population_Density | 0.1983 |

### 4.2 District Clustering (k=4 Recommended)

**Method**: K-Means on standardised district-level score profiles (Hazard, Exposure, Vulnerability, Preparedness).
**Diagnostics**:
- **best_k** (by silhouette): 7 (silhouette = 0.2409)
- **Silhouette Warning**: All tested k configurations achieve silhouette scores < 0.25 (e.g., silhouette is 0.2249 for k=4). This indicates **weak cluster separation** — districts lie along a multidimensional continuum rather than forming discrete, well-separated clusters.
- **Operational Recommendation**: Although k=7 achieves the highest silhouette score, we recommend **k=4** for policy implementation. It results in larger, more balanced cluster sizes (min cluster size is 21) compared to k=7 which creates thin groups of size ~10.

**Recommended Typologies (k=4)**:

| Cluster | Typology Label | Size |
| --- | --- | --- |
| Cluster 0 | High-Exposure Well-Prepared | 27 districts |
| Cluster 1 | High-Exposure Well-Prepared | 21 districts |
| Cluster 2 | Low-Risk Well-Prepared | 27 districts |
| Cluster 3 | High-Vulnerability Low-Capacity | 25 districts |

---

## 5. Methodological Safeguards

| Concern | Mitigation |
| --- | --- |
| Data leakage | Post-event impacts excluded from predictors; scalers fit on training only |
| Temporal leakage | Chronological split; no future data in training |
| Multiple testing | Validation set used for model/threshold selection; test set evaluated once |
| Panel dependency | Bootstrap CIs resampled by district, not by row |
| Overfitting | XGBoost with balanced class weights; threshold optimised on validation |
| Synthetic data | All findings conditional on the simulated data-generating process |
