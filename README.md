# Disaster Risk Prediction

| Field | Value |
| --- | --- |
| **Project** | Disaster Risk Prediction Dashboard |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.0 |
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
│   ├── train_models.py                        # Classifier and regressor pipelines
│   ├── evaluate_models.py                     # Metrics, calibration, curves
│   ├── predict_risk.py                        # CLI inference utility
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
│   └── final_project_report.md                # Comprehensive project report
│
├── images/                                    # Generated visualisations
│   ├── disaster_distribution.png
│   ├── bivariate_boxplots.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── calibration_plot.png
│   └── shap_summary.png
│
├── tests/                                     # Automated test suite (11 tests)
│   ├── test_data_generation.py
│   ├── test_data_quality.py
│   ├── test_feature_engineering.py
│   ├── test_no_leakage.py
│   ├── test_model_pipeline.py
│   └── test_prediction_schema.py
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

---

## Limitations

1. **Synthetic data only** — all results reflect the simulation design, not real-world evidence.
2. **No causal claims** — associations only; no experimental or quasi-experimental design.
3. **Simplified spatial weights** — KNN on centroids; no district boundary polygons.
4. **Small test set** — 1,100 rows (11 months × 100 districts).
5. **Index circularity** — the risk score correlates with its own components by construction.

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

---

## License

MIT License — see [LICENSE](LICENSE).
