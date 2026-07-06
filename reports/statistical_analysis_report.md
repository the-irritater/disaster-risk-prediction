# Statistical Analysis Report

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Statistical Analysis & Hypothesis Testing |
| **Project** | Disaster Risk Prediction Dashboard |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.0 |
| **Status** | Research Submission (Simulation-Based) |


> **Simulation disclaimer.** All data in this project are synthetically generated.
> Statistical findings demonstrate an analytical workflow; they are not empirical evidence
> about real geographical areas.

---

## 1. Regional Differences in Disaster Risk Score (ANOVA)

**Unit of analysis**: District-level means (N = 100 districts).
Monthly panel observations (N = 13,200) are aggregated to district means to avoid
violating the independence assumption that ANOVA requires.

**Hypotheses**:
- H₀: μ₁ = μ₂ = … = μ₅ (all regions have the same mean district risk score)
- H₁: At least one region differs

**Levene's test for homogeneity of variance**: W = 0.0693, p = 0.991

**ANOVA type**: Standard

| Statistic | Value |
| --- | --- |
| F(4, 95) | 0.7166 |
| p-value | 0.583 |
| η² (eta-squared) | 0.0293 |

**Group statistics**:

| Region | N districts | Mean risk score | SD |
| --- | --- | --- | --- |
| Central | 20 | 35.04 | 6.84 |
| East | 20 | 34.76 | 6.32 |
| North | 20 | 36.38 | 6.41 |
| South | 20 | 35.08 | 6.66 |
| West | 20 | 37.73 | 6.81 |

**Interpretation**: Regional risk scores do **not** differ significantly across regions (p = 0.583). The effect size η² = 0.0293 indicates that approximately 2.9% of the between-district variance in risk scores is associated with region membership.
**Post-hoc comparison**: Since the overall F-test is non-significant, no post-hoc comparisons (e.g., Tukey's HSD) are warranted.

---

## 2. Association Between Region and Risk Category (Chi-Square Permutation Test)

**Unit of analysis**: District-level modal risk category (N = 100 districts).
Each district is assigned the risk category that occurs most frequently across its
132 monthly observations.

**Hypotheses**:
- H₀: Region and modal risk category are independent
- H₁: They are associated

**Contingency table**:

| Region | High | Low | Moderate |
| --- | --- | --- | --- |
| Central | 0 | 11 | 9 |
| East | 0 | 12 | 8 |
| North | 0 | 9 | 11 |
| South | 0 | 9 | 11 |
| West | 1 | 9 | 10 |

| Statistic | Value |
| --- | --- |
| χ²(8) | 5.4939 |
| Permutation p-value (9999 shuffles) | 0.8578 |
| Cramér's V | 0.1657 (small to moderate) |
| Min expected frequency | 0.20 |
| Cells with expected < 5 | 33.3% |

**Interpretation**: The association is not significant (permutation p = 0.8578). Cramér's V = 0.1657 indicates a small to moderate association.
**Methodological Safeguard Warning**: 33.3% of expected cell frequencies are below 5 (minimum expected frequency is 0.2), violating standard chi-square asymptotic assumptions. Therefore, the traditional chi-square approximation is **invalid**. The reported p-value is computed via Monte Carlo permutation (shuffling risk categories across regions), which is distribution-free and statistically rigorous under small expected cell frequencies.

---

## 3. Temporal Trend in Annual Disaster Counts (Mann–Kendall)

**Unit of analysis**: Annual totals (N = 11 years, 2015–2025).

| Year | Disaster count |
| --- | --- |
| 2015 | 260 |
| 2016 | 267 |
| 2017 | 254 |
| 2018 | 263 |
| 2019 | 245 |
| 2020 | 227 |
| 2021 | 245 |
| 2022 | 247 |
| 2023 | 248 |
| 2024 | 251 |
| 2025 | 257 |

| Statistic | Value |
| --- | --- |
| Mann–Kendall S | -8 |
| Kendall τ | -0.1455 |
| z-statistic | -0.5449 |
| p-value (two-sided) | 0.586 |
| Sen's slope | -1.0000 events/year |
| Direction | **decreasing** |

**Interpretation**: The Mann–Kendall S = -8 is negative, indicating a decreasing monotonic trend.
Sen's slope of -1.0000 events/year estimates the median annual change in disaster count.
The trend is not statistically significant at the 0.05 level (p = 0.586).

> **Limitation & Power Warning**: With only 11 annual observations, this test is highly underpowered (~30–40% power to detect moderate trends). A non-significant result should not be interpreted as definitive evidence that no underlying trend exists.

---

## 4. Predictors of Economic Loss (Cluster-Robust OLS Regression)

**Unit of analysis**: District-months where a disaster occurred (N = 2764 events).
Standard errors are clustered by district to account for within-district correlation.

**Formula**: `Economic_Loss_Million ~ Hazard_Severity + Population_Density + Infrastructure_Density + Poverty_Rate + Preparedness_Score`

**Model fit**:

| Statistic | Value |
| --- | --- |
| R² | 0.2468 |
| Adjusted R² | 0.2455 |
| F-statistic | 56.7315 |
| F p-value | < 0.001 |
| SE type | cluster-robust (by district) |

**Coefficients**:

| Variable | β | SE (cluster) | 95% CI | p |
| --- | --- | --- | --- | --- |
| Intercept | -363.7209 | 83.7612 | [-527.8898, -199.5520] | < 0.001 |
| Hazard_Severity | 70.2233 | 4.6498 | [61.1098, 79.3368] | < 0.001 |
| Population_Density | -0.0021 | 0.0132 | [-0.0280, 0.0238] | 0.874 |
| Infrastructure_Density | 12.2549 | 0.9435 | [10.4057, 14.1041] | < 0.001 |
| Poverty_Rate | 43.5536 | 166.2425 | [-282.2757, 369.3829] | 0.793 |
| Preparedness_Score | -3.3736 | 1.0826 | [-5.4956, -1.2517] | 0.0018 |

**Variance Inflation Factors (VIF)**:

| Variable | VIF |
| --- | --- |
| Hazard_Severity | 1.02 |
| Population_Density | 1.11 |
| Infrastructure_Density | 1.09 |
| Poverty_Rate | 1.07 |
| Preparedness_Score | 1.10 |

**Interpretation**: The model explains 24.7% of the variance in economic loss
among disaster-event months. Cluster-robust standard errors account for the non-independence
of repeated events within the same district.

> VIF values above 5 indicate moderate multicollinearity; above 10 indicates severe.

---

## 5. Spatial Autocorrelation (Moran's I)

**Unit of analysis**: District-level disaster rates (N = 100 districts).
Weights: K-nearest neighbours with varying k, row-standardised.
Inference: Permutation test (999 randomisations).

| Specification | Moran's I | z-score | p (permutation) |
| --- | --- | --- | --- |
| k=3 | 0.8031 | 10.81 | < 0.001 |
| k=4 | 0.7824 | 11.81 | < 0.001 |
| k=5 | 0.7926 | 13.81 | < 0.001 |
| k=8 | 0.7662 | 16.67 | < 0.001 |

**Interpretation**: Moran's I is consistently positive across k-specifications, indicating positive spatial autocorrelation (nearby districts tend to have similar disaster rates). This spatial pattern was embedded in the synthetic data generator and its recovery validates the simulation design.

---

## Methodological Notes

1. **Panel-data dependency**: All hypothesis tests aggregate to the unit of analysis (district
   or year) rather than treating the 13,200 monthly rows as independent observations.
2. **Cluster-robust SEs**: The regression clusters by district to account for repeated
   measurements within districts.
3. **Synthetic data caveat**: All statistical relationships in these data were programmed
   into the data-generating process. Significant test results confirm recovery of the
   simulation design, not empirical discoveries about real disasters.
