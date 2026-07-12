# Disaster Risk Prediction Analytics Framework

| Field | Value |
| --- | --- |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.2 |
| **Status** | Research Submission (Simulation-Based) |

> **Simulation-based analytics prototype.** This project uses entirely synthetic data
> generated with a fixed random seed. All numerical findings demonstrate an analytical
> workflow and must not be interpreted as evidence about real geographical areas.

A complete, reproducible, and statistically rigorous end-to-end framework for disaster
risk analysis, modelling, and visualisation. The pipeline covers data generation, cleaning,
spatial analysis, risk indexing, statistical hypothesis testing, predictive modelling
(classification and regression), model explainability (SHAP), district clustering,
and Power BI star-schema export.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build notebooks, execute pipeline, and generate reports (one command)
python src/build_notebooks.py
python src/execute_notebooks.py
python src/generate_reports.py

# 3. Verify with tests
python -m pytest tests/ -v
```

All outputs (data, models, images, reports) are regenerated deterministically.

---

## Project Structure

```text
disaster-risk-prediction/
│
├── data/
│   ├── disaster_risk_data.csv                 # Raw synthetic panel (13,200 rows)
│   ├── cleaned_disaster_risk_data.csv         # Scored and feature-engineered
│   ├── data_dictionary.csv                    # Column definitions and roles
│   ├── district_typologies.csv                # K-Means cluster assignments
│   ├── DimDate.csv                            # Power BI dimension table
│   ├── DimGeography.csv                       # Power BI dimension table
│   ├── DimDisasterType.csv                    # Power BI dimension table
│   ├── FactDistrictMonthRisk.csv              # Power BI fact: all district-months
│   └── FactDisasterEvents.csv                 # Power BI fact: disaster events only
│
├── notebooks/
│   ├── 01_data_generation.ipynb               # Panel data generator (100×132)
│   └── 02_master_analysis.ipynb               # Master analysis & Power BI export (Consolidated)
│
├── src/
│   ├── data_generation.py                     # DisasterDataGenerator engine
│   ├── preprocessing.py                       # Cleaning and structural checks
│   ├── feature_engineering.py                 # Rates, scores, lag construction
│   ├── train_models.py                        # Classifier and regressor pipelines (RF, XGB, GBM, LightGBM)
│   ├── evaluate_models.py                     # Metrics, calibration, curves
│   ├── predict_risk.py                        # CLI inference utility
│   ├── uncertainty.py                         # Monte Carlo (Dirichlet), bootstrap prediction intervals
│   ├── ablation.py                            # Feature-group ablation & permutation importance
│   ├── advanced_evaluation.py                 # Calibration comparison, DCA, spatial validation, robustness
│   ├── build_notebooks.py                     # Notebook generator (with student interpretations)
│   ├── execute_notebooks.py                   # Automated notebook execution
│   └── generate_reports.py                    # Report generator (from outputs/)
│
├── models/
│   └── disaster_risk_pipeline.joblib          # Trained classifier pipeline
│
├── reports/
│   ├── statistical_analysis_report.md         # Full hypothesis testing report
│   ├── model_evaluation_report.md             # Classification and regression results
│   ├── final_project_report.md                # Comprehensive project report
│   ├── simulation_design.md                   # DGP documentation & validation
│   └── ethical_considerations.md              # Responsible AI & deployment ethics
│
├── images/                                    # Generated visualisations
│
├── tests/                                     # Automated test suite
│   ├── test_data_generation.py
│   ├── test_data_quality.py
│   ├── test_feature_engineering.py
│   ├── test_no_leakage.py
│   ├── test_model_pipeline.py
│   ├── test_prediction_schema.py
│   ├── test_csv_roundtrip.py
│   ├── test_uncertainty.py                    # Dirichlet MC, convergence, CI ordering
│   └── test_advanced_evaluation.py            # DCA, cost analysis, spatial validation
│
├── requirements.txt                           # Python dependencies
├── README.md                                  # This document
└── LICENSE                                    # MIT License
```

> **Note**: The `outputs/` folder (saved metrics JSONs) and `power_bi/` folder (DAX files and build guides) are excluded from the Git repository (ignored via `.gitignore`) as per configuration guidelines, but are fully generated when running the pipeline locally.

---

## Reproducibility

### Environment

- **Python**: 3.11+ recommended
- **OS**: Windows, macOS, or Linux
- **Random seed**: 42 (all stochastic operations)

### Step-by-Step Execution

```bash
# Step 1: Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Generate notebook source files
python src/build_notebooks.py

# Step 4: Execute all notebooks in sequence (01 and 02)
python src/execute_notebooks.py

# Step 5: Generate reports from saved JSON outputs
python src/generate_reports.py

# Step 6: Run the test suite
python -m pytest tests/ -v
```

Every execution produces identical outputs because:
1. The synthetic data generator uses `np.random.default_rng(42)`
2. All model training uses `random_state=42`
3. All reports are generated programmatically from `outputs/` JSON files

### Single Source of Truth

All statistics in the reports are read from the JSON files in `outputs/`. No numbers
are manually typed into report templates. This eliminates numerical inconsistency
across documents.

| Output file | Used by |
| --- | --- |
| `outputs/statistical_results.json` | Statistical analysis report, final report |
| `outputs/spatial_results.json` | Statistical analysis report |
| `outputs/classification_metrics.json` | Model evaluation report, final report |
| `outputs/regression_metrics.json` | Model evaluation report, final report |
| `outputs/cluster_summary.json` | Model evaluation report, final report |
| `outputs/sensitivity_results.json` | Final report |
| `outputs/uncertainty_results.json` | Uncertainty quantification results |
| `outputs/ablation_results.json` | Feature-group ablation & permutation importance |
| `outputs/advanced_evaluation_results.json` | Calibration, DCA, spatial, robustness |

---

## Key Methodological Decisions

| Decision | Rationale |
| --- | --- |
| District-level aggregation for ANOVA (N=100) | Panel rows are not independent; aggregation respects the data hierarchy |
| Cluster-robust SEs in regression | Accounts for within-district correlation |
| Mann–Kendall sign-aware interpretation | S < 0 → decreasing trend; S > 0 → increasing |
| Chronological train/val/test split | Prevents temporal leakage |
| Post-event impacts excluded from predictors | Prevents information leakage |
| December 2025 rows dropped | Target requires January 2026 (unavailable) |
| Bootstrap CIs by district (cluster bootstrap) | Respects within-district dependence |
| Brier Skill Score against prevalence baseline | Context for calibration claims |
| Dirichlet weight perturbation for risk index MC | Statistically principled (weights sum to 1) |
| Risk index based on UNDRR Sendai / INFORM framework | Literature-justified HEVC paradigm |
| 5 classifier baselines (LR, RF, XGB, GBM, LightGBM) | Comprehensive model comparison |
| Feature-group ablation study | Quantifies contribution of each variable family |
| Decision Curve Analysis | Assesses operational utility beyond discrimination |
| Cost-sensitive FP/FN analysis | Precautionary principle for disaster warnings |

---

## Limitations

1. **Synthetic data only** — all results reflect the simulation design, not real-world evidence.
2. **No causal claims** — associations only; no experimental or quasi-experimental design.
3. **Simplified spatial weights** — KNN on centroids; no district boundary polygons.
4. **Small test set** — 1,100 rows (11 months × 100 districts).
5. **Index circularity** — the risk score correlates with its own components by construction.
6. **Regression R² is low** — economic loss prediction explains limited variance; this is expected for heavy-tailed disaster data and is explicitly acknowledged.

---

## Uncertainty Quantification

The project includes three uncertainty quantification methods (`src/uncertainty.py`):

1. **Dirichlet Monte Carlo** — Risk index weights are perturbed via Dirichlet distribution (concentration=50, N=1000), producing per-district risk score credible intervals and rank stability analysis.
2. **Bootstrap prediction intervals** — Cluster-resampled (by district) intervals for classifier probabilities.
3. **Bootstrap regression intervals** — Prediction intervals for economic loss estimates.

Convergence diagnostics verify estimate stability at N=100, 200, 500, 1000.

---

## Simulation Design Rationale

Full documentation of the data-generating process is in [`reports/simulation_design.md`](reports/simulation_design.md), including:
- Causal DAG and distribution rationale for each variable family
- Logistic hazard probability coefficients and calibration targets
- Marginal distribution validation against reference datasets (IMD, Census India, EM-DAT)
- Known simplifications and their implications

---

## Ethical Considerations

See [`reports/ethical_considerations.md`](reports/ethical_considerations.md) for discussion of:
- Risks of deploying synthetic-data models
- Socio-economic bias in the simulation design
- Responsible AI principles (transparency, accountability, fairness, human oversight)
- Model governance and auditability
- The "cry wolf" problem with high false positive rates

---

## Tests

```bash
python -m pytest tests/ -v
```

The test suite validates:
- Data generation produces the expected panel shape and column set
- No missing values or impossible values in cleaned data
- Feature engineering preserves data integrity
- No leakage of post-event outcomes into pre-event predictors
- Model pipeline produces valid probability outputs
- Prediction schema matches expected format
- CSV round-trip integrity
- Dirichlet weight samples sum to 1.0 and converge to expert weights
- Monte Carlo risk score CIs are properly ordered
- Monte Carlo convergence diagnostics show stabilisation
- Decision Curve Analysis produces valid net benefit curves
- Cost-sensitive analysis finds optimal thresholds under asymmetric loss
- Regression assumption checks return complete diagnostics
- Spatial validation returns valid Moran's I statistics

---

## Frequently Asked Questions

### Why synthetic data instead of real disaster records?

Real-world disaster datasets (EM-DAT, DesInventar, SHELDUS) lack the district-month granularity and complete covariate coverage needed for panel modelling. Synthetic data allows controlled experimentation where the ground-truth data-generating process is known. See `reports/simulation_design.md` for full justification.

### How were simulation parameters chosen?

Distribution parameters are calibrated against reference ranges from IMD (climate), Census India (demographics), and DesInventar (disaster frequency). Logistic coefficients in the hazard probability models are tuned to produce ~15% disaster prevalence. AR(1) persistence coefficients (~0.55) reflect empirical month-to-month climate autocorrelation.

### Is the risk index validated against established frameworks?

The risk index follows the UNDRR Sendai Framework and INFORM Risk Index methodology (HEVC paradigm). Weight sensitivity is assessed via: (1) equal vs expert weight Spearman correlation, (2) Dirichlet Monte Carlo perturbation, (3) component knockout analysis, (4) 0%–200% weight sweep.

### How does the model generalise beyond simulated scenarios?

It does not — and this is explicitly stated throughout the reports. The project demonstrates a complete analytical pipeline; generalisation claims would require validation against real disaster records.

### Why Random Forest / XGBoost rather than spatiotemporal models?

The project compares 5 classifiers (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, LightGBM). Temporal structure is captured via lag features and rolling windows. Spatial structure is encoded via geographic features. Full spatiotemporal models (e.g., LSTM, GNNs) are identified as future work.

### How are spatial and temporal dependencies handled?

Spatial: geographic features (elevation, coastal distance, river distance, coordinates). Temporal: lag features (t-1 disaster occurrence and severity), rolling 12-month disaster count, seasonal encoding. Panel structure is respected via cluster-robust standard errors and cluster bootstrap.

---

## License

MIT License — see [LICENSE](LICENSE).
