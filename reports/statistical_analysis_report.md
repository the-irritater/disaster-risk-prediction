# Statistical Analysis Report

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Statistical Analysis & Hypothesis Testing |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.2 |
| **Status** | Research Submission (Simulation-Based) |


> **Simulation Disclaimer.** All data in this project are synthetically generated.
> Statistical findings demonstrate that the analytical pipeline is correctly implemented;
> they are **not** empirical evidence about real geographical areas or populations.
>
> **Language convention**: Throughout this report, phrases such as "associated with" or
> "differs across" refer to relationships **within the simulated data-generating process**.
> They do not imply real-world causal or associative relationships. Where a test is
> statistically significant, it confirms recovery of a programmed simulation parameter,
> not an empirical discovery.

---

## 1. Regional Differences in Disaster Risk Score (ANOVA)

**Unit of analysis**: District-level means (N = 100 districts).
Monthly panel observations (N = 13,200) are aggregated to district means to avoid
violating the independence assumption that ANOVA requires.

**Hypotheses**:
- H₀: μ₁ = μ₂ = … = μ₅ (all regions have the same mean district risk score)
- H₁: At least one region differs

**Levene's test for homogeneity of variance**: W = 0.0414, p = 0.997

**ANOVA type**: Standard

| Statistic | Value |
| --- | --- |
| F(4, 95) | 0.6452 |
| p-value | 0.632 |
| η² (eta-squared) | 0.0264 |

**Group statistics**:

| Region | N districts | Mean risk score | SD |
| --- | --- | --- | --- |
| Central | 20 | 35.30 | 6.95 |
| East | 20 | 34.99 | 6.41 |
| North | 20 | 36.48 | 6.39 |
| South | 20 | 35.39 | 6.70 |
| West | 20 | 37.89 | 6.74 |

**Interpretation**: Under the simulation assumptions, regional risk scores do **not** differ
across regions (p = 0.632). The effect size η² = 0.0264 indicates that approximately
2.6% of the between-district variance in simulated risk scores is associated with region
membership. This non-significant result is genuine: the data generator does not embed
strong regional differentiation in the risk index.

**Post-hoc comparison**: Since the overall F-test is non-significant, no post-hoc comparisons (e.g., Tukey’s HSD) are warranted.

**ANOVA Assumptions Check**:

| Assumption | Test | Result | Verdict |
| --- | --- | --- | --- |
| Independence | District-level aggregation | N=100 independent district means | ✅ Satisfied |
| Normality of residuals | Shapiro-Wilk on ANOVA residuals | W ≈ 0.99, p > 0.05 | ✅ Satisfied (N=100 is also robust to mild non-normality) |
| Homogeneity of variance | Levene’s test | W = 0.041, p = 0.997 | ✅ Satisfied |
| Equal group sizes | By design | 20 per region | ✅ Balanced design |

> **Response variable**: District-level mean Disaster_Risk_Score (averaged across 132 months).
> **Factor**: Region (5 levels: Central, East, North, South, West).
> **Unit of analysis**: District (N=100), not individual monthly observations (N=13,200).
> Aggregation to district means avoids violating the independence assumption that would
> arise from treating correlated panel rows as independent.

---

## 2. Association Between Region and Risk Category (Chi-Square Permutation Test)

**Unit of analysis**: District-level modal risk category (N = 100 districts).
Each district is assigned the risk category that occurs most frequently across its
132 monthly observations.

**Hypotheses**:
- H₀: Region and modal risk category are independent
- H₁: They are associated

**Contingency table**:

| Region | Critical | High | Low | Moderate |
| --- | --- | --- | --- | --- |
| Central | 4 | 4 | 6 | 6 |
| East | 3 | 5 | 6 | 6 |
| North | 5 | 6 | 5 | 4 |
| South | 5 | 5 | 7 | 3 |
| West | 8 | 3 | 3 | 6 |

| Statistic | Value |
| --- | --- |
| χ²(12) | 7.2341 |
| Permutation p-value (9999 shuffles) | 0.8532 |
| Cramér's V | 0.1553 (small to moderate) |
| Min expected frequency | 4.60 |
| Cells with expected < 5 | 25.0% |

**Interpretation**: The association is not significant (permutation p = 0.8532). Cramér's V = 0.1553 indicates a small to moderate association.
**Methodological Safeguard Warning**: 25.0% of expected cell frequencies are below 5 (minimum expected frequency is 4.6), violating standard chi-square asymptotic assumptions. Therefore, the traditional chi-square approximation is **invalid**. The reported p-value is computed via Monte Carlo permutation (shuffling risk categories across regions), which is distribution-free and statistically rigorous under small expected cell frequencies.

---

## 3. Temporal Trend in Annual Disaster Counts (Mann–Kendall)

**Unit of analysis**: Annual totals (N = 11 years, 2015–2025).

| Year | Disaster count |
| --- | --- |
| 2015 | 189 |
| 2016 | 205 |
| 2017 | 171 |
| 2018 | 188 |
| 2019 | 182 |
| 2020 | 193 |
| 2021 | 179 |
| 2022 | 187 |
| 2023 | 187 |
| 2024 | 175 |
| 2025 | 180 |

| Statistic | Value |
| --- | --- |
| Mann–Kendall S | -18 |
| Kendall τ | -0.3273 |
| z-statistic | -1.3275 |
| p-value (two-sided) | 0.184 |
| Sen's slope | -1.3333 events/year |
| Direction | **decreasing** |

**Interpretation**: The Mann–Kendall S = -18 is negative, indicating a decreasing monotonic trend.
Sen's slope of -1.3333 events/year estimates the median annual change in disaster count.
The trend is not statistically significant at the 0.05 level (p = 0.184).

> **Limitation & Power Warning**: With only 11 annual observations, this test is highly underpowered (~30–40% power to detect moderate trends). A non-significant result should not be interpreted as definitive evidence that no underlying trend exists.

---

## 4. Predictors of Economic Loss (Cluster-Robust OLS Regression)

**Unit of analysis**: District-months where a disaster occurred (N = 2036 events).
Standard errors are clustered by district to account for within-district correlation.

**Formula**: `Economic_Loss_Million ~ Hazard_Severity + Population_Density + Infrastructure_Density + Poverty_Rate + Preparedness_Score`

**Model fit**:

| Statistic | Value |
| --- | --- |
| R² | 0.3015 |
| Adjusted R² | 0.2998 |
| F-statistic | 53.5850 |
| F p-value | < 0.001 |
| SE type | cluster-robust (by district) |

**Coefficients**:

| Variable | β | SE (cluster) | 95% CI | p |
| --- | --- | --- | --- | --- |
| Intercept | -340.0198 | 78.3667 | [-493.6158, -186.4239] | < 0.001 |
| Hazard_Severity | 72.0135 | 5.1608 | [61.8985, 82.1285] | < 0.001 |
| Population_Density | -0.0117 | 0.0094 | [-0.0301, 0.0068] | 0.214 |
| Infrastructure_Density | 12.1214 | 1.0658 | [10.0325, 14.2103] | < 0.001 |
| Poverty_Rate | -9.7827 | 128.7602 | [-262.1481, 242.5827] | 0.939 |
| Preparedness_Score | -3.3509 | 1.0149 | [-5.3401, -1.3617] | < 0.001 |

**Variance Inflation Factors (VIF)**:

| Variable | VIF |
| --- | --- |
| Hazard_Severity | 1.06 |
| Population_Density | 1.20 |
| Infrastructure_Density | 1.08 |
| Poverty_Rate | 1.08 |
| Preparedness_Score | 1.08 |

**Interpretation**: Under the simulation design, the model explains 30.1% of the variance in
economic loss among disaster-event months. The association between hazard severity and
economic loss (β = 72.0, p < 0.001) reflects a relationship programmed into the
data-generating process. Cluster-robust standard errors account for the non-independence
of repeated events within the same district.

> VIF values above 5 indicate moderate multicollinearity; above 10 indicates severe.
> All VIF values are below 1.25, confirming **no multicollinearity** among predictors.

**Regression Assumption Diagnostics**:

| Assumption | Test | Result | Interpretation |
| --- | --- | --- | --- |
| Multicollinearity | VIF | All VIF < 1.25 | No collinearity concern |
| Residual normality | Shapiro-Wilk | W ≈ 0.857, p < 0.001 | Non-normal; heavy-tailed distribution typical of disaster loss data |
| Homoscedasticity | Spearman rank correlation of |residuals| vs predicted | See `outputs/advanced_evaluation_results.json` | Heteroscedasticity expected for loss data; cluster-robust SEs mitigate |
| Influential observations | Cook's distance | See `outputs/advanced_evaluation_results.json` | Large loss events are influential but represent genuine tail risk |
| Independence | Cluster-robust SEs by district | 100 district clusters | Accounts for within-district correlation |

> **Note on heteroscedasticity**: Economic loss data is inherently heteroscedastic (variance
> increases with predicted loss magnitude). The Tweedie GLM partially addresses this through
> its variance function, and cluster-robust standard errors provide consistent inference
> regardless of heteroscedasticity. Residual plots are generated in the master analysis
> notebook (predicted vs actual scatter, Q-Q plot).

---

## 5. Spatial Autocorrelation (Moran's I)

**Unit of analysis**: District-level disaster rates (N = 100 districts).
Weights: K-nearest neighbours with varying k, row-standardised.
Inference: Permutation test (999 randomisations).

| Specification | Moran's I | z-score | p (permutation) |
| --- | --- | --- | --- |
| k=3 | 0.7087 | 9.70 | < 0.001 |
| k=4 | 0.6767 | 10.49 | < 0.001 |
| k=5 | 0.6673 | 11.91 | < 0.001 |
| k=8 | 0.6590 | 14.56 | < 0.001 |

**Interpretation**: Under the simulation design, Moran’s I is consistently positive across
k-specifications, indicating positive spatial autocorrelation (nearby simulated districts
tend to have similar disaster rates). This spatial pattern was embedded in the synthetic
data generator through the grid-based geographic assignment, and its recovery validates
the simulation design.

---

## 6. Time Series Dependence

**Concern**: Monthly observations from the same district are temporally correlated.
Chronological train/test splitting prevents future leakage, but does not fully
address temporal structure.

**How temporal dependence is addressed in this project**:

| Approach | Implementation | Purpose |
| --- | --- | --- |
| Lag features | `Previous_Month_Disaster_Occurred`, `Previous_Month_Hazard_Severity` | Explicitly models AR(1) structure |
| Rolling window | `Rolling_12_Month_Disaster_Count` | Captures long-term persistence |
| Seasonal encoding | `Season` (categorical: Monsoon, Winter, Summer, Spring) | Controls for seasonality |
| Chronological split | Train ≤ 2022, Val = 2024, Test = 2025 | No future leakage |
| Cluster-robust SEs | Bootstrap and regression cluster by district | Accounts for within-district serial correlation |
| AR(1) in data generation | Autoregressive coefficient ~0.55 | Simulates month-to-month persistence |

**Limitations acknowledged**:
- No formal autocorrelation function (ACF/PACF) analysis is performed on residuals.
- The lag structure (t-1 only) may not capture higher-order temporal dependencies.
- Dedicated time series models (ARIMA, LSTM) are identified as future work.
- Seasonal effects are captured via categorical encoding rather than harmonic terms.

---

## 7. External Applicability

> **Critical caveat**: This project demonstrates a complete analytical pipeline
> using synthetic data. The pipeline itself is **transferable**; the numerical
> results are **not**.

**What transfers to real-world data**:
1. The feature engineering pipeline (HEVC framework, lag construction, rate computation)
2. The model training architecture (chronological split, threshold optimisation, cluster bootstrap)
3. The evaluation methodology (calibration, DCA, cost-sensitive analysis, spatial validation)
4. The reporting framework (JSON-driven, reproducible)

**What does NOT transfer**:
1. Specific coefficients, p-values, and effect sizes
2. Model hyperparameters (would need retuning)
3. Risk index weights (would need domain expert recalibration)
4. Threshold values (depend on base rate in the target population)

**Recommended pathway for real-world application**:
1. Obtain district-month panel data from national disaster agencies (e.g., DesInventar India, EM-DAT)
2. Map variables to the existing feature schema
3. Retrain all models on real data; recalibrate thresholds
4. Validate using spatial and temporal hold-out sets
5. Conduct pilot deployment in advisory (non-operational) mode

> If even a small real-world dataset were available (e.g., 20 districts × 60 months),
> running the pipeline on it would provide a preliminary external validation.
> This is identified as the highest-priority future work.

---

## Methodological Notes

1. **Panel-data dependency**: All hypothesis tests aggregate to the unit of analysis (district
   or year) rather than treating the 13,200 monthly rows as independent observations.
2. **Cluster-robust SEs**: The regression clusters by district to account for repeated
   measurements within districts.
3. **Synthetic data caveat**: All statistical relationships in these data were programmed
   into the data-generating process. Significant test results confirm recovery of the
   simulation design, not empirical discoveries about real disasters.
4. **Bootstrap methodology**: Confidence intervals use the **percentile method** with
   N=1,000 bootstrap resamples, clustered by district (100 district clusters resampled
   with replacement). District-level resampling was chosen because observations within
   the same district share unobserved district-level effects (geography, demographics,
   infrastructure). Row-level resampling would underestimate standard errors by treating
   correlated within-district observations as independent.
5. **Clustering methodology**: K-Means clustering uses k=4 (recommended operationally)
   justified by: (a) silhouette analysis across k=2–7 (all silhouette < 0.25, indicating
   a continuum rather than discrete groups), (b) Davies-Bouldin index, (c) operational
   requirement for balanced cluster sizes (min 20 districts per cluster). Inertia (elbow)
   analysis is computed alongside silhouette. Cluster stability is partially addressed
   via the silhouette analysis; formal stability testing (e.g., consensus clustering) is
   identified as future work.
6. **Risk index circularity warning**: The Disaster_Risk_Score is a weighted sum of
   Hazard_Score, Exposure_Score, Vulnerability_Score, and Preparedness_Deficit_Score.
   These component scores are themselves derived from the same features used as ML
   predictors. Therefore: (a) regression of risk score on its components is tautological;
   (b) ML models predicting risk category are partially recovering the index formula;
   (c) the classification target `Disaster_Next_Month` does NOT suffer from this
   circularity because it is derived from the hazard probability model, not the risk
   index. **The classifier predicts disaster events, not the risk score.**
7. **Multiple testing**: No correction (e.g., Bonferroni) is applied because each
   hypothesis test addresses a distinct research question. The tests are pre-specified,
   not exploratory data dredging.
