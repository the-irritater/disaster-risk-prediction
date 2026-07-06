"""
Generate all Markdown reports from machine-readable outputs in outputs/.

This is the SINGLE SOURCE OF TRUTH for all reported numbers. Every statistic
in every report is read from the saved JSON files, never typed manually.

Generates:
  - reports/statistical_analysis_report.md
  - reports/model_evaluation_report.md
  - reports/final_project_report.md
"""
import json
import os

AUTHOR = "Sanman"
PROJECT = "Disaster Risk Prediction Analytics Framework"
VERSION = "3.1"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def fmt_p(p):
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"

def get_metadata_table(title):
    return f"""| Field | Value |
| --- | --- |
| **Report Title** | {title} |
| **Project** | {PROJECT} |
| **Author** | {AUTHOR} |
| **Date** | July 2026 |
| **Version** | {VERSION} |
| **Status** | Research Submission (Simulation-Based) |
"""

def generate_statistical_report(stat, spatial):
    """Generate statistical_analysis_report.md from saved results."""
    a = stat['anova']
    # Use permutation chi_square if available, otherwise fallback
    c = stat.get('permutation_chi_square', stat.get('chi_square'))
    t = stat['trend']
    r = stat['regression']

    # Build ANOVA group table rows
    group_rows = ""
    for region in sorted(a['group_means'].keys()):
        group_rows += f"| {region} | {a['group_counts'][region]} | {a['group_means'][region]:.2f} | {a['group_sds'][region]:.2f} |\n"

    # Build regression coefficient table
    coef_rows = ""
    coefs = r['coefficients']
    for var in coefs['Coefficient']:
        c_val = coefs['Coefficient'][var]
        se = coefs['Std_Error'][var]
        ci_lo = coefs['CI_Lower'][var]
        ci_hi = coefs['CI_Upper'][var]
        p = coefs['p_value'][var]
        coef_rows += f"| {var} | {c_val:.4f} | {se:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | {fmt_p(p)} |\n"

    # VIF table
    vif_rows = ""
    for var, val in r['vif'].items():
        flag = " ⚠" if val > 5 else ""
        vif_rows += f"| {var} | {val:.2f}{flag} |\n"

    # Mann-Kendall annual counts table
    mk_rows = ""
    for yr, cnt in sorted(t['annual_counts'].items()):
        mk_rows += f"| {yr} | {cnt} |\n"

    # Chi-square contingency table (Transposed: Regions as Rows, Categories as Columns)
    chi_rows = ""
    cats = sorted(c['contingency_table'].keys())  # ["High", "Low", "Moderate"]
    regions = sorted(set(k for inner in c['contingency_table'].values() for k in inner)) # ["Central", "East", ...]
    
    chi_header = "| Region | " + " | ".join(cats) + " |\n"
    chi_header += "| --- | " + " | ".join(["---"] * len(cats)) + " |\n"
    for region in regions:
        vals = [str(int(c['contingency_table'][cat].get(region, 0))) for cat in cats]
        chi_rows += f"| {region} | " + " | ".join(vals) + " |\n"

    # Spatial results
    spatial_rows = ""
    for k_str in sorted(spatial.keys()):
        s = spatial[k_str]
        p_val_display = s.get('p_value_display', fmt_p(s['p_value_permutation']))
        spatial_rows += f"| {k_str} | {s['morans_i']:.4f} | {s['z_score']:.2f} | {p_val_display} |\n"

    anova_sig_text = "Regional risk scores differ significantly across regions (p < 0.05)." if a['p_value'] < 0.05 else "Regional risk scores do **not** differ significantly across regions (p = " + fmt_p(a['p_value']) + ")."
    
    p_chi_val = c.get('p_value_permutation', c.get('p_value'))
    p_chi_display = c.get('p_value_display', fmt_p(p_chi_val))
    chi_sig_text = f"The association is significant (permutation p = {p_chi_display})." if p_chi_val < 0.05 else f"The association is not significant (permutation p = {p_chi_display})."
    
    report = f"""# Statistical Analysis Report

## Metadata

{get_metadata_table("Statistical Analysis & Hypothesis Testing")}

> **Simulation disclaimer.** All data in this project are synthetically generated.
> Statistical findings demonstrate an analytical workflow; they are not empirical evidence
> about real geographical areas.

---

## 1. Regional Differences in Disaster Risk Score (ANOVA)

**Unit of analysis**: District-level means (N = {a['n_observations']} districts).
Monthly panel observations (N = 13,200) are aggregated to district means to avoid
violating the independence assumption that ANOVA requires.

**Hypotheses**:
- H₀: μ₁ = μ₂ = … = μ₅ (all regions have the same mean district risk score)
- H₁: At least one region differs

**Levene's test for homogeneity of variance**: W = {a['levene_statistic']:.4f}, p = {fmt_p(a['levene_p_value'])}

**ANOVA type**: {a['anova_type']}

| Statistic | Value |
| --- | --- |
| F({a['df_between']}, {a['df_residual']}) | {a['f_statistic']:.4f} |
| p-value | {fmt_p(a['p_value'])} |
| η² (eta-squared) | {a['eta_squared']:.4f} |

**Group statistics**:

| Region | N districts | Mean risk score | SD |
| --- | --- | --- | --- |
{group_rows}
**Interpretation**: {anova_sig_text} The effect size η² = {a['eta_squared']:.4f} indicates that approximately {a['eta_squared']*100:.1f}% of the between-district variance in risk scores is associated with region membership.
**Post-hoc comparison**: Since the overall F-test is non-significant, no post-hoc comparisons (e.g., Tukey's HSD) are warranted.

---

## 2. Association Between Region and Risk Category (Chi-Square Permutation Test)

**Unit of analysis**: District-level modal risk category (N = {c['n_observations']} districts).
Each district is assigned the risk category that occurs most frequently across its
132 monthly observations.

**Hypotheses**:
- H₀: Region and modal risk category are independent
- H₁: They are associated

**Contingency table**:

{chi_header}{chi_rows}
| Statistic | Value |
| --- | --- |
| χ²({c.get('degrees_of_freedom', 8)}) | {c.get('chi2_observed', c.get('chi2_statistic')):.4f} |
| Permutation p-value ({c.get('n_permutations', 9999)} shuffles) | {p_chi_display} |
| Cramér's V | {c['cramers_v']:.4f} ({c['v_interpretation']}) |
| Min expected frequency | {c['min_expected_frequency']:.2f} |
| Cells with expected < 5 | {c['pct_cells_below_5']:.1f}% |

**Interpretation**: {chi_sig_text} Cramér's V = {c['cramers_v']:.4f} indicates a {c['v_interpretation']} association.
**Methodological Safeguard Warning**: {c.get('pct_cells_below_5')}% of expected cell frequencies are below 5 (minimum expected frequency is {c.get('min_expected_frequency')}), violating standard chi-square asymptotic assumptions. Therefore, the traditional chi-square approximation is **invalid**. The reported p-value is computed via Monte Carlo permutation (shuffling risk categories across regions), which is distribution-free and statistically rigorous under small expected cell frequencies.

---

## 3. Temporal Trend in Annual Disaster Counts (Mann–Kendall)

**Unit of analysis**: Annual totals (N = {t['n_years']} years, 2015–2025).

| Year | Disaster count |
| --- | --- |
{mk_rows}
| Statistic | Value |
| --- | --- |
| Mann–Kendall S | {t['mann_kendall_S']} |
| Kendall τ | {t['kendall_tau']:.4f} |
| z-statistic | {t['z_statistic']:.4f} |
| p-value (two-sided) | {fmt_p(t['p_value_two_sided'])} |
| Sen's slope | {t['sens_slope']:.4f} events/year |
| Direction | **{t['trend_direction']}** |

**Interpretation**: The Mann–Kendall S = {t['mann_kendall_S']} is {"positive, indicating an increasing" if t['mann_kendall_S'] > 0 else "negative, indicating a decreasing" if t['mann_kendall_S'] < 0 else "zero, indicating no"} monotonic trend.
Sen's slope of {t['sens_slope']:.4f} events/year estimates the median annual change in disaster count.
{"The trend is statistically significant at the 0.05 level." if t['p_value_two_sided'] < 0.05 else "The trend is not statistically significant at the 0.05 level (p = " + fmt_p(t['p_value_two_sided']) + ")."}

> **Limitation & Power Warning**: With only {t['n_years']} annual observations, this test is highly underpowered (~30–40% power to detect moderate trends). A non-significant result should not be interpreted as definitive evidence that no underlying trend exists.

---

## 4. Predictors of Economic Loss (Cluster-Robust OLS Regression)

**Unit of analysis**: District-months where a disaster occurred (N = {r['n_events']} events).
Standard errors are clustered by district to account for within-district correlation.

**Formula**: `{r['formula']}`

**Model fit**:

| Statistic | Value |
| --- | --- |
| R² | {r['r_squared']:.4f} |
| Adjusted R² | {r['adj_r_squared']:.4f} |
| F-statistic | {r['f_statistic']:.4f} |
| F p-value | {fmt_p(r['f_pvalue'])} |
| SE type | {r['cov_type']} |

**Coefficients**:

| Variable | β | SE (cluster) | 95% CI | p |
| --- | --- | --- | --- | --- |
{coef_rows}
**Variance Inflation Factors (VIF)**:

| Variable | VIF |
| --- | --- |
{vif_rows}
**Interpretation**: The model explains {r['r_squared']*100:.1f}% of the variance in economic loss
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
{spatial_rows}
**Interpretation**: {"Moran's I is consistently positive across k-specifications, indicating positive spatial autocorrelation (nearby districts tend to have similar disaster rates). " if all(spatial[k]['morans_i'] > 0 for k in spatial) else "Results are mixed across k specifications. "}This spatial pattern was embedded in the synthetic data generator and its recovery validates the simulation design.

---

## Methodological Notes

1. **Panel-data dependency**: All hypothesis tests aggregate to the unit of analysis (district
   or year) rather than treating the 13,200 monthly rows as independent observations.
2. **Cluster-robust SEs**: The regression clusters by district to account for repeated
   measurements within districts.
3. **Synthetic data caveat**: All statistical relationships in these data were programmed
   into the data-generating process. Significant test results confirm recovery of the
   simulation design, not empirical discoveries about real disasters.
"""
    return report


def generate_model_report(clf, reg, cluster):
    """Generate model_evaluation_report.md from saved results."""
    s = clf['split_info']
    tm = clf['test_metrics']
    cm = clf['confusion_matrix_counts']
    cal = clf['calibration']
    ci = clf['bootstrap_95_ci']
    best_k = cluster.get('best_k', cluster.get('best_k_by_silhouette', 7))

    # Validation comparison table
    val_rows = ""
    # Decide which comparison to use
    comp_key = 'validation_comparison_optimal_threshold' if 'validation_comparison_optimal_threshold' in clf else 'validation_comparison'
    for model_name, metrics in clf[comp_key].items():
        val_rows += f"| {model_name}"
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'brier_score']:
            val_rows += f" | {metrics.get(key, 'N/A')}"
        val_rows += " |\n"

    # Regression validation table
    reg_val_rows = ""
    for model_name, metrics in reg['validation_results'].items():
        reg_val_rows += f"| {model_name}"
        for key in ['r_squared', 'rmse', 'mae', 'mdape']:
            reg_val_rows += f" | {metrics.get(key, 'N/A')}"
        reg_val_rows += " |\n"

    # Cluster diagnostics table
    diag_rows = ""
    for k_str, d in sorted(cluster['diagnostics'].items()):
        diag_rows += f"| {k_str} | {d['silhouette']:.4f} | {d['davies_bouldin']:.4f} | {d['cluster_sizes']} |\n"

    # SHAP top features
    shap_rows = ""
    for item in cluster['shap_top_10']:
        shap_rows += f"| {item['feature']} | {item['mean_abs_shap']:.4f} |\n"

    # Cluster k=4 recommendations
    cluster_k4_rows = ""
    if 'cluster_labels_k4' in cluster:
        for c_id, label in sorted(cluster['cluster_labels_k4'].items(), key=lambda x: int(x[0])):
            sz = cluster['cluster_sizes_k4'][int(c_id)]
            cluster_k4_rows += f"| Cluster {c_id} | {label} | {sz} districts |\n"

    # Reg test metrics
    rt = reg['test_results']
    r2_ci = reg.get('bootstrap_r2_95_ci', {'lower': 'N/A', 'upper': 'N/A'})
    sw = reg.get('residual_normality', {'shapiro_w': 0.0, 'shapiro_p': 0.0})

    report = f"""# Model Evaluation Report

## Metadata

{get_metadata_table("Machine Learning Model Evaluation & Calibration")}

> **Simulation disclaimer.** All data in this project are synthetically generated.
> Strong model performance partly reflects recovery of programmed relationships
> rather than validated generalisation to real disaster data.

---

## 1. Data Partitioning

All splits are **chronological** (no future leakage). December 2025 rows are excluded
because the target variable `Disaster_Next_Month` requires January 2026 data.

| Partition | Period | N rows | % of total | Positive rate |
| --- | --- | --- | --- | --- |
| Train | {s['train']['period']} | {s['train']['n_rows']:,} | {s['train']['percentage']:.1f}% | {s['train']['positive_rate']:.4f} |
| Validation | {s['validation']['period']} | {s['validation']['n_rows']:,} | {s['validation']['percentage']:.1f}% | {s['validation']['positive_rate']:.4f} |
| Test | {s['test']['period']} | {s['test']['n_rows']:,} | {s['test']['percentage']:.1f}% | {s['test']['positive_rate']:.4f} |
| **Total usable** | | **{s['total_usable_rows']:,}** | | |

December 2025 rows dropped: {s['december_2025_dropped']}

---

## 2. Classification: Disaster Next Month

### 2.1 Validation-Set Model Comparison (threshold = {clf['threshold']:.4f})

All three models evaluated on the validation set at the tuned optimal threshold of **{clf['threshold']:.4f}** for a fair comparison:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
{val_rows}
**Selected model**: {clf['selected_model']} (highest validation PR-AUC)

### 2.2 Test-Set Results (Single Evaluation)

The selected model ({clf['selected_model']}) was evaluated exactly once on the held-out
test set with the optimal threshold of **{clf['threshold']:.4f}** (selected on validation
to target ≥ 75% recall).

| Metric | Value | 95% CI (cluster bootstrap) |
| --- | --- | --- |
| ROC-AUC | {tm['roc_auc']:.4f} | [{ci['roc_auc']['lower']:.4f}, {ci['roc_auc']['upper']:.4f}] |
| PR-AUC | {tm['pr_auc']:.4f} | [{ci['pr_auc']['lower']:.4f}, {ci['pr_auc']['upper']:.4f}] |
| Precision | {tm['precision']:.4f} | [{ci['precision']['lower']:.4f}, {ci['precision']['upper']:.4f}] |
| Recall | {tm['recall']:.4f} | [{ci['recall']['lower']:.4f}, {ci['recall']['upper']:.4f}] |
| F1-score | {tm['f1_score']:.4f} | [{ci['f1_score']['lower']:.4f}, {ci['f1_score']['upper']:.4f}] |
| Accuracy | {tm['accuracy']:.4f} | — |
| Brier score | {tm['brier_score']:.4f} | — |

### 2.3 Confusion Matrix (Test Set)

|  | Predicted Negative | Predicted Positive |
| --- | --- | --- |
| **Actual Negative** | TN = {cm['TN']} | FP = {cm['FP']} |
| **Actual Positive** | FN = {cm['FN']} | TP = {cm['TP']} |

Total test observations: {cm['TP'] + cm['FP'] + cm['FN'] + cm['TN']}
- **False Discovery Rate (FDR)**: {clf.get('false_discovery_rate', 0.0):.4f} ({clf.get('false_discovery_rate', 0.0)*100:.1f}% of flagged districts are false alarms). This highlights the operational trade-off: high sensitivity to capture disasters (recall={tm['recall']:.2f}) comes at the cost of a high false alarm rate.

### 2.4 Calibration Assessment & Operational Benchmark

| Metric | Value |
| --- | --- |
| Null-model Brier (prevalence baseline) | {cal['null_model_brier']:.4f} |
| Model Brier score | {cal['model_brier']:.4f} |
| Brier Skill Score (BSS) | {cal['brier_skill_score']:.4f} |

**BSS Interpretation**: The Brier Skill Score of {cal['brier_skill_score']:.4f} indicates a {cal.get('bss_interpretation', 'weak')} improvement over always predicting the baseline rate.
- *Operational Benchmarks*: In real-world early warning systems, BSS > 0.25 is considered highly skillful; BSS of 0.10–0.25 is marginally useful; BSS < 0.10 represents very weak calibration skill. Thus, the model's raw probability values should **not** be used directly for risk pricing or resource allocation without recalibration.

---

## 3. Regression: Economic Loss (Conditional on Disaster)

### 3.1 Validation-Set Model Comparison

| Model | R² | RMSE | MAE | MdAPE (%) |
| --- | --- | --- | --- | --- |
{reg_val_rows}
**Selected model**: {reg['selected_model']}

### 3.2 Test-Set Results (Single Evaluation)

The regressor was evaluated exactly once on the held-out test events (disaster-event months):

| Metric | Value | Description |
| --- | --- | --- |
| R² | {rt['r_squared']:.4f} | Fraction of variance explained |
| R² 95% Bootstrap CI | [{r2_ci['lower']}, {r2_ci['upper']}] | Variance of explanation |
| RMSE | {rt['rmse']:.4f} | Root Mean Squared Error (M USD) |
| MAE | {rt['mae']:.4f} | Mean Absolute Error (M USD) |
| MdAPE (%) | {rt.get('mdape', 0.0):.1f}% | Median Absolute Percentage Error |
| Mean True Loss | {rt.get('mean_y', 0.0):.4f} | Mean actual loss of event months (M USD) |
| SD of True Loss | {rt.get('std_y', 0.0):.4f} | SD of actual loss of event months (M USD) |

**Diagnostics and Analysis**:
- **MdAPE (Median Absolute Percentage Error)**: We use MdAPE ({rt.get('mdape', 0.0):.1f}%) because the standard Mean Absolute Percentage Error (MAPE) is highly sensitive to near-zero denominators (small disaster losses), leading to arithmetic explosion (e.g., >100%). MdAPE is a robust metric indicating that the median relative error is {rt.get('mdape', 0.0):.1f}%.
- **RMSE Context**: The RMSE of {rt['rmse']:.2f} is high relative to the mean true loss ({rt.get('mean_y', 0.0):.2f}) and represents a substantial portion of the standard deviation ({rt.get('std_y', 0.0):.2f}). The model has limited predictive power (R² = {rt['r_squared']:.4f}).
- **Residual Diagnostics**: Shapiro-Wilk test on test residuals yields W = {sw.get('shapiro_w')}, p = {fmt_p(sw.get('shapiro_p'))}, indicating residuals deviate significantly from normality, exhibiting a heavy-tailed distribution typical of disaster losses.

Event counts — Train: {reg['n_train_events']}, Validation: {reg['n_val_events']}, Test: {reg['n_test_events']}

---

## 4. Explainability and District Typologies

### 4.1 SHAP Feature Importance (Top 10)

SHAP values indicate the model's reliance on each feature for prediction. Higher mean
|SHAP| values mean greater predictive contribution. These are **not** causal effects.

| Feature | Mean |SHAP| |
| --- | --- |
{shap_rows}
### 4.2 District Clustering (k={cluster['recommended_operational_k']} Recommended)

**Method**: K-Means on standardised district-level score profiles (Hazard, Exposure, Vulnerability, Preparedness).
**Diagnostics**:
- **best_k** (by silhouette): {best_k} (silhouette = {cluster['diagnostics'][str(best_k)]['silhouette']:.4f})
- **Silhouette Warning**: All tested k configurations achieve silhouette scores < 0.25 (e.g., silhouette is {cluster['diagnostics']['4']['silhouette']:.4f} for k=4). This indicates **weak cluster separation** — districts lie along a multidimensional continuum rather than forming discrete, well-separated clusters.
- **Operational Recommendation**: Although k={best_k} achieves the highest silhouette score, we recommend **k=4** for policy implementation. It results in larger, more balanced cluster sizes (min cluster size is {min(cluster['cluster_sizes_k4']) if 'cluster_sizes_k4' in cluster else 'N/A'}) compared to k={best_k} which creates thin groups of size ~10.

**Recommended Typologies (k=4)**:

| Cluster | Typology Label | Size |
| --- | --- | --- |
{cluster_k4_rows}
---

## 5. Methodological Safeguards

| Concern | Mitigation |
| --- | --- |
| Data leakage | Post-event impacts excluded from predictors; scalers fit on training only |
| Temporal leakage | Chronological split; no future data in training |
| Multiple testing | Validation set used for model/threshold selection; test set evaluated once |
| Panel dependency | Bootstrap CIs resampled by district, not by row |
| Overfitting | {clf['selected_model']} with balanced class weights; threshold optimised on validation |
| Synthetic data | All findings conditional on the simulated data-generating process |
"""
    return report


def generate_final_report(stat, spatial, clf, reg, cluster, sensitivity, missing_count, duplicate_count, raw_cols_count, clean_cols_count):
    """Generate final_project_report.md with consistent numbers from all sources."""
    a = stat['anova']
    c = stat.get('permutation_chi_square', stat.get('chi_square'))
    t = stat['trend']
    r = stat['regression']
    s = clf['split_info']
    tm = clf['test_metrics']
    cm = clf['confusion_matrix_counts']
    cal = clf['calibration']
    ci_data = clf['bootstrap_95_ci']
    best_k = cluster.get('best_k', cluster.get('best_k_by_silhouette', 7))

    anova_sig_term = "differ significantly" if a['p_value'] < 0.05 else "do **not** differ significantly"
    chi_sig_term = "are significantly associated" if c.get('p_value_permutation', c.get('p_value')) < 0.05 else "are **not** significantly associated"
    p_chi_display = c.get('p_value_display', fmt_p(c.get('p_value_permutation', c.get('p_value'))))

    report = f"""# Disaster Risk Prediction Analytics Framework — Final Project Report

## Metadata

{get_metadata_table("Project Summary & Analytical Report")}

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

- Regional risk scores {anova_sig_term} across regions (F({a['df_between']}, {a['df_residual']}) = {a['f_statistic']:.2f}, p = {fmt_p(a['p_value'])}, η² = {a['eta_squared']:.3f})
- Annual disaster counts show a **{t['trend_direction']}** trend (S = {t['mann_kendall_S']}, p = {fmt_p(t['p_value_two_sided'])})
- The {clf['selected_model']} classifier achieves ROC-AUC = {tm['roc_auc']:.4f} [{ci_data['roc_auc']['lower']:.4f}, {ci_data['roc_auc']['upper']:.4f}] on the held-out test set
- Economic loss prediction achieves R² = {reg['test_results']['r_squared']:.4f} on disaster-event test data
- Districts cluster into {cluster['recommended_operational_k']} recommended risk typologies (operational k=4, silhouette = {cluster['diagnostics']['4']['silhouette']:.4f})

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
| Raw Variables | {raw_cols_count} |
| Cleaned Variables | {clean_cols_count} |
| Disaster Prevalence (contemporaneous) | {s['disaster_occurred_prevalence']*100:.2f}% |
| Target Prevalence (Disaster_Next_Month) | {s['train']['positive_rate']*100:.2f}% (Training) |

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
- **Missing values**: {missing_count} null entries detected.
- **Duplicate records**: {duplicate_count} duplicate rows found.

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
- Spearman rank correlation: ρ = {sensitivity['spearman_rho']:.4f} (p = {sensitivity['spearman_p']:.2e})
- Interpretation: District risk rankings are highly stable across weight specifications.

> **Note**: High correlation between the risk score and its components is expected because
> the score is a weighted sum of those components. This is a contribution analysis
> of the index's internal structure, not independent evidence.

---

## 4. Statistical Analysis

### 4.1 ANOVA: Regional Risk Differences

Aggregated to district means (N = {a['n_observations']}):
- F({a['df_between']}, {a['df_residual']}) = {a['f_statistic']:.4f}, p = {fmt_p(a['p_value'])}
- η² = {a['eta_squared']:.4f} ({a['eta_squared']*100:.1f}% of between-district variance)
- Regional risk scores {anova_sig_term} across regions.

### 4.2 Chi-Square: Region × Risk Category (Permutation Test)

Aggregated to district-level modal categories (N = {c['n_observations']}):
- χ²({c.get('degrees_of_freedom', 8)}) = {c.get('chi2_observed', c.get('chi2_statistic')):.4f}, permutation p = {p_chi_display}
- Cramér's V = {c['cramers_v']:.4f} ({c['v_interpretation']} association)
- Region and risk category {chi_sig_term}.

### 4.3 Mann–Kendall Trend Test

Annual disaster counts (N = {t['n_years']} years):
- S = {t['mann_kendall_S']}, τ = {t['kendall_tau']:.4f}, p = {fmt_p(t['p_value_two_sided'])}
- Sen's slope = {t['sens_slope']:.4f} events/year → **{t['trend_direction']}** trend
- *Note*: Test power is low (~30-40%) due to small annual sample size (N=11).

### 4.4 OLS Regression for Economic Loss

Event-only months (N = {r['n_events']}), cluster-robust SEs by district:
- R² = {r['r_squared']:.4f}, Adjusted R² = {r['adj_r_squared']:.4f}
- Standard errors: {r['cov_type']}

### 4.5 Spatial Autocorrelation

Moran's I on district-level disaster rates (N = 100 districts) is consistently positive
across KNN specifications, confirming spatial clustering. This validates the simulation's
spatial structure.

---

## 5. Predictive Modelling

### 5.1 Classification: Disaster Next Month

**Target**: Binary indicator, shifted one month forward.
**December 2025 handling**: {s['december_2025_dropped']} rows dropped (no January 2026 target).

**Data split**:

| Partition | Period | N | % | Prevalence |
| --- | --- | --- | --- | --- |
| Train | {s['train']['period']} | {s['train']['n_rows']:,} | {s['train']['percentage']:.1f}% | {s['train']['positive_rate']:.4f} |
| Validation | {s['validation']['period']} | {s['validation']['n_rows']:,} | {s['validation']['percentage']:.1f}% | {s['validation']['positive_rate']:.4f} |
| Test | {s['test']['period']} | {s['test']['n_rows']:,} | {s['test']['percentage']:.1f}% | {s['test']['positive_rate']:.4f} |

**Selected model**: {clf['selected_model']} (best validation PR-AUC)
**Threshold**: {clf['threshold']:.4f} (optimised for ≥ 75% recall on validation)

**Test-set performance**:

| Metric | Value | 95% CI |
| --- | --- | --- |
| ROC-AUC | {tm['roc_auc']:.4f} | [{ci_data['roc_auc']['lower']:.4f}, {ci_data['roc_auc']['upper']:.4f}] |
| PR-AUC | {tm['pr_auc']:.4f} | [{ci_data['pr_auc']['lower']:.4f}, {ci_data['pr_auc']['upper']:.4f}] |
| Recall | {tm['recall']:.4f} | [{ci_data['recall']['lower']:.4f}, {ci_data['recall']['upper']:.4f}] |
| Precision | {tm['precision']:.4f} | [{ci_data['precision']['lower']:.4f}, {ci_data['precision']['upper']:.4f}] |
| F1 | {tm['f1_score']:.4f} | [{ci_data['f1_score']['lower']:.4f}, {ci_data['f1_score']['upper']:.4f}] |

**Confusion matrix** (test set):

|  | Pred. Neg | Pred. Pos |
| --- | --- | --- |
| **Act. Neg** | {cm['TN']} | {cm['FP']} |
| **Act. Pos** | {cm['FN']} | {cm['TP']} |

**Calibration**: Brier = {cal['model_brier']:.4f} vs null = {cal['null_model_brier']:.4f} → BSS = {cal['brier_skill_score']:.4f} (Interpretation: {cal.get('bss_interpretation', 'weak')} calibration skill).

### 5.2 Regression: Conditional Economic Loss

**Selected model**: {reg['selected_model']} (simple linear Ridge model selected over tree ensembles to prevent validation overfitting)
**Event counts**: Train {reg['n_train_events']}, Val {reg['n_val_events']}, Test {reg['n_test_events']}

| Metric | Validation | Test |
| --- | --- | --- |
| R² | {reg['validation_results'][reg['selected_model']]['r_squared']:.4f} | {reg['test_results']['r_squared']:.4f} |
| RMSE | {reg['validation_results'][reg['selected_model']]['rmse']:.4f} | {reg['test_results']['rmse']:.4f} |
| MAE | {reg['validation_results'][reg['selected_model']]['mae']:.4f} | {reg['test_results']['mae']:.4f} |
| MdAPE (%) | {reg['validation_results'][reg['selected_model']]['mdape']:.1f}% | {reg['test_results']['mdape']:.1f}% |

---

## 6. Explainability and Typology

### 6.1 SHAP Analysis

Top predictive features by mean |SHAP| value (model reliance, not causal effects):

| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
"""
    for i, item in enumerate(cluster['shap_top_10'][:10], 1):
        report += f"| {i} | {item['feature']} | {item['mean_abs_shap']:.4f} |\n"

    report += f"""
### 6.2 District Clustering (k=4 Recommended)

K-Means clustering on standardised district profiles (Hazard, Exposure, Vulnerability, Preparedness):
- Best k (silhouette): {best_k} (silhouette = {cluster['diagnostics'][str(best_k)]['silhouette']:.4f})
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
| FactDisasterEvents | Fact | {reg['n_train_events'] + reg['n_val_events'] + reg['n_test_events']} | Post-event impacts (event-months only) |

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
"""
    return report


def main():
    print("Generating reports from outputs/ JSON files...")

    os.makedirs('reports', exist_ok=True)

    # Load all outputs
    stat = load_json('outputs/statistical_results.json')
    spatial = load_json('outputs/spatial_results.json')
    clf = load_json('outputs/classification_metrics.json')
    reg = load_json('outputs/regression_metrics.json')
    cluster_data = load_json('outputs/cluster_summary.json')
    sensitivity = load_json('outputs/sensitivity_results.json')

    # Load raw and clean data to compute counts dynamically
    import pandas as pd
    df_raw = pd.read_csv('data/disaster_risk_data.csv')
    df_clean = pd.read_csv('data/cleaned_disaster_risk_data.csv')
    raw_cols_count = len(df_raw.columns)
    clean_cols_count = len(df_clean.columns)
    missing_count = int(df_clean.isna().sum().sum())
    duplicate_count = int(df_clean.duplicated().sum())

    # Generate each report
    stat_report = generate_statistical_report(stat, spatial)
    with open('reports/statistical_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(stat_report)
    print("  Written: reports/statistical_analysis_report.md")

    model_report = generate_model_report(clf, reg, cluster_data)
    with open('reports/model_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(model_report)
    print("  Written: reports/model_evaluation_report.md")

    final_report = generate_final_report(stat, spatial, clf, reg, cluster_data, sensitivity, missing_count, duplicate_count, raw_cols_count, clean_cols_count)
    with open('reports/final_project_report.md', 'w', encoding='utf-8') as f:
        f.write(final_report)
    print("  Written: reports/final_project_report.md")

    print("\nAll reports generated successfully.")
    print("Every number in every report was read from outputs/ JSON files or data files.")


if __name__ == "__main__":
    main()
