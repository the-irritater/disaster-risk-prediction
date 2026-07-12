# Disaster Risk Prediction Analytics Framework — Final Project Report

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Project Summary & Analytical Report |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.2 |
| **Status** | Research Submission (Simulation-Based) |


> **Simulation-Based Analytics Framework Prototype**
>
> This project uses entirely synthetic data generated with a fixed random seed (42).
> All numerical findings demonstrate an analytical and modelling workflow. They must
> not be interpreted as evidence about real geographical areas, real disaster patterns,
> or real causal relationships.
>
> **Language convention**: Throughout this report, phrases such as "associated with",
> "differs across", or "predicts" refer to relationships within the simulated data.
> Statistically significant results confirm recovery of programmed simulation
> parameters, not empirical discoveries.

---

## Executive Summary

This report documents a complete disaster risk analytics pipeline covering synthetic
data generation, data validation, spatial analysis, descriptive risk indexing,
statistical hypothesis testing, predictive modelling (classification and regression),
model explainability (SHAP), and district clustering. All analytical outputs are
exported for Power BI star-schema analysis.

**Key findings** (all conditional on the synthetic data-generating process):

- Under the simulation assumptions, regional risk scores do **not** differ across regions (F(4, 95) = 0.65, p = 0.632, η² = 0.026)
- Annual disaster counts show a **decreasing** simulated trend (S = -18, p = 0.184)
- The XGBoost classifier achieves ROC-AUC = 0.8587 [0.8273, 0.8888] on the held-out test set
- Economic loss prediction achieves R² = 0.1693 on disaster-event test data (low; explicitly acknowledged)
- Districts cluster into 4 recommended risk typologies (operational k=4, silhouette = 0.2236)

---

## 1. Introduction

### 1.1 Research Questions

1. Which geographical regions and districts are associated with higher disaster risk scores?
2. What environmental and socio-economic variables are most strongly associated with disaster occurrence?
3. Can future disaster occurrence be predicted from current-month environmental indicators?
4. Given that a disaster occurs, what pre-event factors are associated with economic loss magnitude?
5. How can districts be grouped into actionable risk typologies for targeted intervention?

### 1.2 Scope and Limitations

- **Simulation-based**: All data are synthetic. Statistical significance reflects recovery
  of programmed relationships, not empirical discoveries.
- **No causal claims**: Associations between variables do not imply causation. Language
  throughout this report uses "associated with", not "causes" or "determines".
- **Temporal coverage**: 2015–2025 (11 years × 12 months × 100 districts = 13,200 rows).
- **Spatial resolution**: District centroids with KNN weights (no boundary polygons).

---

## 2. Data Description

### 2.1 Panel Structure

The dataset is a balanced district-month panel:

| Dimension | Value |
| --- | --- |
| Districts | 100 |
| Time span | January 2015 – December 2025 |
| Rows | 13,200 |
| Raw Variables | 61 |
| Cleaned Variables | 72 |
| Disaster Prevalence (contemporaneous) | 15.42% |
| Target Prevalence (Disaster_Next_Month) | 15.56% (Training) |

### 2.2 Variable Categories

| Category | Examples | Role |
| --- | --- | --- |
| Identifiers | District, Year, Month, Event_Date | Panel keys |
| Environmental hazards | Rainfall_Anomaly, Wind_Speed, Seismic_Activity | Pre-event predictors |
| Exposure | Population_Density, Infrastructure_Density | Pre-event predictors |
| Vulnerability | Poverty_Rate, Literacy_Rate | Pre-event predictors |
| Preparedness | Hospitals_per_100k, Disaster_Preparedness_Index | Pre-event predictors |
| Target (classification) | Disaster_Next_Month (lead variable) | Prediction target |
| Post-event impacts | Deaths, Injuries, Economic_Loss_Million | Regression target (conditional) |

### 2.3 Data Quality

We calculate data quality metrics dynamically from the dataset:
- **Missing values**: 0 null entries detected.
- **Duplicate records**: 0 duplicate rows found.

IQR-flagged outliers in `Hazard_Severity` and `Wind_Speed_kmph` represent genuine extreme events and are retained.

---

## 3. Descriptive Risk Index

### 3.1 Construction

The composite Disaster Risk Score uses assumed expert weights:

> Risk = 0.30 × Hazard + 0.25 × Exposure + 0.25 × Vulnerability + 0.20 × Preparedness Deficit

Component scores are min-max normalised to [0, 100]. Districts are categorised into
Low / Medium / High / Critical risk based on quartile boundaries.

### 3.2 Sensitivity Analysis

Comparing expert weights to equal weights (0.25 each):
- Spearman rank correlation: ρ = 0.9892 (p = 1.37e-83)
- Interpretation: District risk rankings are highly stable across weight specifications.

> **Note**: High correlation between the risk score and its components is expected because
> the score is a weighted sum of those components. This is a contribution analysis
> of the index's internal structure, not independent evidence.

### 3.3 Literature Justification

The risk index follows the internationally recognised **Hazard-Exposure-Vulnerability-Capacity
(HEVC)** paradigm, as codified in:

- **UNDRR Sendai Framework** (2015–2030): Risk = f(Hazard, Exposure, Vulnerability, Capacity)
- **INFORM Risk Index** (European Commission JRC, 2023): Three-dimensional model with
  Hazard & Exposure, Vulnerability, and Lack of Coping Capacity
- **IPCC SREX Report** (Cardona et al., 2012): Determinants of risk framework

**Weight rationale**:
- Hazard (0.30): Primary driver; without a hazard trigger, no disaster occurs.
- Exposure (0.25): Population and infrastructure at risk amplify impact.
- Vulnerability (0.25): Socio-economic fragility determines damage severity.
- Preparedness Deficit (0.20): Lower weight because preparedness mitigates (rather than causes) risk.

### 3.4 Component Knockout Analysis

Each component is removed one at a time, with remaining weights re-normalised. Rank
correlation with the full-index ranking indicates component importance:

| Knocked-Out Component | Spearman ρ | Interpretation |
| --- | --- | --- |
| Hazard_Score | See outputs | Removing hazard most/least affects rankings |
| Exposure_Score | See outputs | Quantifies exposure contribution |
| Vulnerability_Score | See outputs | Quantifies vulnerability contribution |
| Preparedness_Deficit | See outputs | Quantifies preparedness gap contribution |

*Full results are computed dynamically by `src/feature_engineering.py` and saved in `outputs/`.*

### 3.5 Weight Sweep Analysis

Each component weight is varied from 0% to 200% of its expert value (21 steps),
with remaining weights proportionally adjusted. The resulting Spearman ρ curves
form a **weight sensitivity heatmap** showing which components most strongly
influence district risk rankings.

---

## 4. Statistical Analysis

### 4.1 ANOVA: Regional Risk Differences

Aggregated to district means (N = 100):
- **Response variable**: District-level mean Disaster_Risk_Score
- **Factor**: Region (5 levels: Central, East, North, South, West)
- **Assumptions**: Independence (✅ district-level aggregation), normality (✅ Shapiro-Wilk p > 0.05), homogeneity (✅ Levene p = 0.997), equal groups (✅ 20 per region)
- F(4, 95) = 0.6452, p = 0.632, η² = 0.0264
- Under the simulation assumptions, regional risk scores do **not** differ across regions.

### 4.2 Chi-Square: Region × Risk Category (Permutation Test)

Aggregated to district-level modal categories (N = 100):
- χ²(12) = 7.2341, permutation p = 0.8532
- Cramér’s V = 0.1553 (small to moderate association)
- Under the simulation assumptions, region and risk category are **not** associated.

### 4.3 Mann–Kendall Trend Test

Annual disaster counts (N = 11 years):
- S = -18, τ = -0.3273, p = 0.184
- Sen’s slope = -1.3333 events/year → **decreasing** simulated trend
- *Note*: Test power is low (~30-40%) due to small annual sample size (N=11).

### 4.4 OLS Regression for Economic Loss

Event-only months (N = 2036), cluster-robust SEs by district:
- R² = 0.3015, Adjusted R² = 0.2998
- Standard errors: cluster-robust (by district)
- **VIF**: All < 1.25 (no multicollinearity)
- **Diagnostics**: Shapiro-Wilk (non-normal residuals expected), heteroscedasticity (mitigated by Tweedie GLM and cluster-robust SEs), Cook’s distance

### 4.5 Spatial Autocorrelation

Moran’s I on district-level disaster rates (N = 100 districts) is consistently positive
across KNN specifications, confirming spatial clustering within the simulated data.

### 4.6 Time Series Dependence

Temporal structure is addressed via:
- Lag features (t-1 disaster occurrence and severity)
- Rolling 12-month disaster count
- Seasonal categorical encoding
- Chronological train/test split
- Cluster-robust bootstrap CIs

*Limitation*: No formal ACF/PACF analysis; dedicated time series models (ARIMA, LSTM) are future work.

---

## 5. Predictive Modelling

### 5.1 Classification: Disaster Next Month

**Target**: Binary indicator, shifted one month forward.
**December 2025 handling**: 100 rows dropped (no January 2026 target).

**Data split**:

| Partition | Period | N | % | Prevalence |
| --- | --- | --- | --- | --- |
| Train | 2015-01 to 2022-12 | 9,600 | 73.3% | 0.1556 |
| Validation | 2024-01 to 2024-12 | 1,200 | 9.2% | 0.1467 |
| Test | 2025-01 to 2025-11 | 1,100 | 8.4% | 0.1591 |

**Selected model**: XGBoost (best validation PR-AUC)
**Threshold**: 0.1708 (optimised for ≥ 75% recall on validation)

**Test-set performance**:

| Metric | Value | 95% CI |
| --- | --- | --- |
| ROC-AUC | 0.8587 | [0.8273, 0.8888] |
| PR-AUC | 0.5876 | [0.5085, 0.6671] |
| Recall | 0.8114 | [0.7443, 0.8753] |
| Precision | 0.3880 | [0.3426, 0.4328] |
| F1 | 0.5250 | [0.4769, 0.5688] |

**Confusion matrix** (test set):

|  | Pred. Neg | Pred. Pos |
| --- | --- | --- |
| **Act. Neg** | 701 | 224 |
| **Act. Pos** | 33 | 142 |

**Calibration**: Brier = 0.0937 vs null = 0.1338 → BSS = 0.2998 (Interpretation: skillful (>=0.25) calibration skill).

---

## 5.3 Ablation Study: Feature Group Contributions

Feature-group ablation quantifies the contribution of each variable family by
removing it from the feature set and measuring performance degradation:

| Group Removed | Features Removed | Δ ROC-AUC | Δ PR-AUC |
| --- | --- | --- | --- |
| Environmental | 12 | See outputs | See outputs |
| Exposure | 3 | See outputs | See outputs |
| Vulnerability | 7 | See outputs | See outputs |
| Preparedness | 7 | See outputs | See outputs |
| Temporal/Lag | 3 | See outputs | See outputs |

*Results computed by `src/ablation.py` and saved to `outputs/ablation_results.json`.*

Permutation importance provides individual feature-level ranking, complementing
the group-level ablation.

---

## 5.4 Uncertainty Quantification

### 5.4.1 Monte Carlo Risk Index (Dirichlet Weights)

Risk index component weights are perturbed using a **Dirichlet distribution**
centered on expert weights (concentration=50, N=1000 simulations). This produces
per-district risk score credible intervals.

- **Convergence diagnostics** at N=100, 200, 500, 1000 verify estimate stability.
- **Rank stability**: Proportion of simulations where each district's rank stays
  within ±5 of its median rank.

### 5.4.2 Bootstrap Prediction Intervals

Cluster-resampled (by district) bootstrap produces:
- 95% prediction intervals for classifier probabilities
- 95% prediction intervals for economic loss regression
- Coverage metrics validating interval calibration

---

## 5.5 Advanced Evaluation

### 5.5.1 Calibration Comparison

Classifier probabilities are compared before and after **Platt scaling**
(sigmoid calibration fitted on the calibration set):

| Metric | Before | After |
| --- | --- | --- |
| Brier Score | See outputs | See outputs |
| ECE | See outputs | See outputs |

### 5.5.2 Decision Curve Analysis

**Net benefit** is computed across intervention thresholds (0.01–0.99):
- Identifies the threshold range where the model provides more benefit than
  treating all districts or treating none.
- Demonstrates operational utility beyond discrimination metrics.

### 5.5.3 Cost-Sensitive Analysis

Under asymmetric loss (FN cost = 100× FP cost, reflecting the precautionary
principle in disaster management):
- Optimal threshold shifts lower than the prevalence-based threshold.
- Justifies the high false positive rate as an acceptable operational cost
  when false negatives mean unmitigated disaster impact.

### 5.5.4 Spatial Validation

Moran's I on prediction residuals tests whether the model systematically
over- or under-predicts in geographic clusters, indicating unmodelled spatial structure.

### 5.5.5 Robustness Analysis

Repeated stratified holdout (5 repeats, 20% district holdout) reports the
**distribution** of ROC-AUC rather than a single point estimate.

---

## 5.6 Regression: Conditional Economic Loss

**Selected model**: Tweedie GLM
**Event counts**: Train 1494, Val 362, Test 180

| Metric | Validation | Test |
| --- | --- | --- |
| R² | 0.1940 | 0.1693 |
| RMSE | 513.1162 | 522.5115 |
| MAE | 321.7107 | 357.3308 |
| MdAPE (%) | 50.1% | 58.5% |

> **Explicit Acknowledgment**: The R² of 0.17 is low. The regression explains
> only ~17% of variance in economic loss. This is scientifically expected:
> economic losses are inherently noisy, heavy-tailed (Shapiro-Wilk p < 0.001),
> and driven by unobserved factors (exact location of impact, local response
> quality, pre-existing infrastructure condition). The regression is presented
> as an **exploratory association analysis**, not a reliable predictor.

---

## 6. Explainability and Typology

### 6.1 SHAP Analysis

Top predictive features by mean |SHAP| value (model reliance, not causal effects):

| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | Distance_From_Coast_km | 0.6016 |
| 2 | Season_Spring | 0.5497 |
| 3 | Season_Winter | 0.1737 |
| 4 | Temperature_Anomaly | 0.1652 |
| 5 | Rainfall_Anomaly | 0.1632 |
| 6 | Population_Density | 0.1356 |
| 7 | Drought_Index | 0.0931 |
| 8 | Wind_Speed_kmph | 0.0772 |
| 9 | Season_Monsoon | 0.0727 |
| 10 | Previous_Month_Hazard_Severity | 0.0647 |

### 6.2 District Clustering (k=4 Recommended)

K-Means clustering on standardised district profiles (Hazard, Exposure, Vulnerability, Preparedness):
- Best k (silhouette): 7 (silhouette = 0.2374)
- Recommended operational k: 4 (more balanced cluster sizes)
- Silhouette warning: All configurations have silhouette < 0.25, indicating weak separation.

---

## 7. Power BI Integration Structure

The Power BI star schema includes:

| Table | Type | Rows | Purpose |
| --- | --- | --- | --- |
| DimDate | Dimension | 132 | Calendar months |
| DimGeography | Dimension | 100 | District attributes (includes cluster labels) |
| DimDisasterType | Dimension | ~6 | Hazard taxonomy |
| FactDistrictMonthRisk | Fact | 13,200 | Risk scores, predictions |
| FactDisasterEvents | Fact | 2036 | Post-event impacts (event-months only) |

**Recommended report views**:
1. **Regional Risk Overview** — Choropleth map, risk distribution, regional comparisons
2. **Temporal Trends** — Annual/seasonal disaster patterns, Mann–Kendall trend overlay
3. **Prediction Performance** — Model metrics, confusion matrix, calibration plot
4. **District Profiles** — Drill-through to individual district risk decomposition and typology

---

## 8. Reproducibility

### 8.1 Execution

```bash
# Generate notebooks, execute pipeline, and generate reports
python src/build_notebooks.py
python src/execute_notebooks.py
python src/generate_reports.py
```

### 8.2 Single Source of Truth

All statistics reported in this document are read from machine-readable JSON files
in `outputs/`. No numbers are manually typed into reports.

### 8.3 Environment

- Python 3.11+
- Key packages: scikit-learn, xgboost, shap, statsmodels, scipy, pandas, numpy
- Full dependency list: `requirements.txt`
- Random seed: 42 (all stochastic operations)

## 8a. Risk Index Circularity

> **This is a serious methodological concern that is explicitly disclosed.**

The Disaster_Risk_Score = 0.30·Hazard + 0.25·Exposure + 0.25·Vulnerability + 0.20·Preparedness_Deficit.

**Circularity analysis**:

| Target | Circularity | Reason |
| --- | --- | --- |
| Disaster_Risk_Score | ⚠️ HIGH | Target IS a function of the predictors |
| Risk_Category | ⚠️ HIGH | Quantile-binned risk score |
| **Disaster_Next_Month** | ✅ LOW | Generated by logistic hazard model, NOT the risk index |
| **Economic_Loss** | ✅ LOW | Generated by impact model conditional on occurrence |

**Key distinction**: The primary ML target (`Disaster_Next_Month`) is NOT circular.
It comes from the hazard probability logistic model. The risk index is used for
descriptive profiling only. Classifier performance should not be conflated with
risk index prediction. See `reports/model_evaluation_report.md` §6.

---

## 9. Limitations

1. **Synthetic data**: No real-world validation is possible. Results demonstrate methodology only.
2. **Spatial simplification**: KNN-based weights on centroids; no district boundary polygons.
3. **No causal inference**: All relationships are correlational associations.
4. **Leakage-free by design**: The simulation ensures no post-event data enter pre-event predictors,
   but this guarantee depends on the data generator, not on defensive modelling.
5. **Small test set**: 11 months × 100 districts = 1,100 rows limits test-set power.
6. **Index circularity**: The risk score correlates strongly with its components because
   it is computed from them — this is mathematical, not empirical.
7. **Low regression R²**: Economic loss prediction explains ~17% of variance. This is
   expected for heavy-tailed disaster loss data and is acknowledged explicitly.
8. **No spatiotemporal models**: The project uses feature engineering to capture spatial
   and temporal structure rather than dedicated spatiotemporal architectures (LSTM, GNN).

---

## 10. Ethical Considerations

See [`reports/ethical_considerations.md`](reports/ethical_considerations.md) for full discussion.

Key concerns:
1. **Deployment risk**: Models trained on synthetic data must not be deployed for real
   disaster warnings without real-world validation and recalibration.
2. **Simulation bias**: Embedded correlations between poverty and adverse outcomes
   may perpetuate structural disadvantage if used for resource allocation.
3. **Responsible AI**: Predictions should support human decision-makers, not replace them.
   The high false discovery rate (~61%) requires human triage of flagged alerts.
4. **Model governance**: Version-controlled artefacts, deterministic execution, and
   audit trail via JSON outputs support auditability requirements.
5. **"Cry wolf" problem**: Repeated false alarms may erode public trust. Mitigation
   strategies include two-tier alert systems and probabilistic communication.

---

## 11. Responding to Reviewer Questions

| Question | Response |
| --- | --- |
| Why synthetic data? | Real datasets lack district-month granularity with complete covariates. Synthetic data enables controlled methodology validation. |
| How were simulation parameters chosen? | Calibrated against IMD, Census India, EM-DAT reference ranges. See `reports/simulation_design.md`. |
| Is the risk index validated? | Follows UNDRR Sendai / INFORM HEVC paradigm. Robustness confirmed via Dirichlet MC, knockout, and weight sweep analyses. |
| How does the model generalise? | It does not — explicitly stated. Generalisation requires real-world validation. |
| Why not spatiotemporal models? | Identified as future work. Current approach uses feature engineering (lags, rolling counts, geographic features) within classical ML. |
| How are dependencies handled? | Cluster-robust SEs, cluster bootstrap, temporal lags, AR(1) persistence in data generation. |

---

## 12. External Applicability

> The analytical pipeline is **transferable**; the numerical results are **not**.

**What transfers**: Feature engineering framework (HEVC), model training architecture
(chronological split, threshold optimisation, cluster bootstrap), evaluation methodology
(calibration, DCA, cost-sensitive analysis, spatial validation), reporting framework.

**What does NOT transfer**: Specific coefficients, p-values, effect sizes, thresholds,
hyperparameters, risk index weights. All require re-estimation on real data.

**Recommended pathway**: (1) Obtain real district-month data, (2) retrain all models,
(3) recalibrate thresholds, (4) validate via spatial and temporal holdout, (5) pilot
in advisory mode. See `reports/statistical_analysis_report.md` §7 for full discussion.

---

*Report generated programmatically from `outputs/` JSON files by `src/generate_reports.py`.*
