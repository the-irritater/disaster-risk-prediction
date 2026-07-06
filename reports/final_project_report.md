# Disaster Risk Prediction Analytics Framework — Final Project Report

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Project Summary & Analytical Report |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.1 |
| **Status** | Research Submission (Simulation-Based) |


> **Simulation-Based Analytics Framework Prototype**
>
> This project uses entirely synthetic data generated with a fixed random seed (42).
> All numerical findings demonstrate an analytical and modelling workflow. They must
> not be interpreted as evidence about real geographical areas, real disaster patterns,
> or real causal relationships.

---

## Executive Summary

This report documents a complete disaster risk analytics pipeline covering synthetic
data generation, data validation, spatial analysis, descriptive risk indexing,
statistical hypothesis testing, predictive modelling (classification and regression),
model explainability (SHAP), and district clustering. All analytical outputs are
exported for Power BI star-schema analysis.

**Key findings** (all conditional on the synthetic data-generating process):

- Regional risk scores do **not** differ significantly across regions (F(4, 95) = 0.65, p = 0.632, η² = 0.026)
- Annual disaster counts show a **decreasing** trend (S = -18, p = 0.184)
- The XGBoost classifier achieves ROC-AUC = 0.8587 [0.8273, 0.8888] on the held-out test set
- Economic loss prediction achieves R² = 0.1693 on disaster-event test data
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

---

## 4. Statistical Analysis

### 4.1 ANOVA: Regional Risk Differences

Aggregated to district means (N = 100):
- F(4, 95) = 0.6452, p = 0.632
- η² = 0.0264 (2.6% of between-district variance)
- Regional risk scores do **not** differ significantly across regions.

### 4.2 Chi-Square: Region × Risk Category (Permutation Test)

Aggregated to district-level modal categories (N = 100):
- χ²(12) = 7.2341, permutation p = 0.8532
- Cramér's V = 0.1553 (small to moderate association)
- Region and risk category are **not** significantly associated.

### 4.3 Mann–Kendall Trend Test

Annual disaster counts (N = 11 years):
- S = -18, τ = -0.3273, p = 0.184
- Sen's slope = -1.3333 events/year → **decreasing** trend
- *Note*: Test power is low (~30-40%) due to small annual sample size (N=11).

### 4.4 OLS Regression for Economic Loss

Event-only months (N = 2036), cluster-robust SEs by district:
- R² = 0.3015, Adjusted R² = 0.2998
- Standard errors: cluster-robust (by district)

### 4.5 Spatial Autocorrelation

Moran's I on district-level disaster rates (N = 100 districts) is consistently positive
across KNN specifications, confirming spatial clustering. This validates the simulation's
spatial structure.

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

### 5.2 Regression: Conditional Economic Loss

**Selected model**: Tweedie GLM (simple linear Ridge model selected over tree ensembles to prevent validation overfitting)
**Event counts**: Train 1494, Val 362, Test 180

| Metric | Validation | Test |
| --- | --- | --- |
| R² | 0.1940 | 0.1693 |
| RMSE | 513.1162 | 522.5115 |
| MAE | 321.7107 | 357.3308 |
| MdAPE (%) | 50.1% | 58.5% |

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
in `outputs/`. No numbers are manually typed into reports. The mapping:

| Output file | Report sections |
| --- | --- |
| `outputs/statistical_results.json` | §4 Statistical Analysis |
| `outputs/spatial_results.json` | §4.5 Spatial Autocorrelation |
| `outputs/classification_metrics.json` | §5.1 Classification |
| `outputs/regression_metrics.json` | §5.2 Regression |
| `outputs/cluster_summary.json` | §6 Explainability |
| `outputs/sensitivity_results.json` | §3.2 Sensitivity Analysis |

### 8.3 Environment

- Python 3.11+
- Key packages: scikit-learn, xgboost, shap, statsmodels, scipy, pandas, numpy
- Full dependency list: `requirements.txt`
- Random seed: 42 (all stochastic operations)

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

---

*Report generated programmatically from `outputs/` JSON files by `src/generate_reports.py`.*
