# Ethical Considerations

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Ethical Considerations for Disaster Risk Prediction |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.2 |
| **Status** | Research Submission (Simulation-Based) |

---

## 1. Risks of Deploying Models Trained on Synthetic Data

### 1.1 The Validation Gap

This project is a **simulation-based prototype**. Deploying any model trained exclusively
on synthetic data in a real-world disaster warning system would be scientifically
irresponsible because:

- The model has never encountered genuine disaster signals, measurement errors, or
  real-world covariate distributions.
- Performance metrics (ROC-AUC, calibration) are conditional on the data-generating
  process and do not constitute validated predictive capability.
- Real disasters exhibit complex dynamics (cascading failures, infrastructure collapse,
  human behaviour under stress) that no synthetic generator fully captures.

### 1.2 Recommended Pathway to Deployment

Before any operational use, the framework would require:

1. **Validation against historical disaster records** (e.g., EM-DAT, DesInventar, national emergency databases)
2. **Recalibration** on real-world data to adjust probability estimates
3. **Pilot deployment** in a non-critical advisory capacity alongside existing early warning systems
4. **Continuous monitoring** for model drift, especially under climate change scenarios

---

## 2. Bias in the Simulation Design

### 2.1 Socio-Economic Correlations

The synthetic data generator embeds correlations between **poverty** and several outcome
variables:

- Higher poverty → lower housing quality → more displacement
- Higher poverty → lower preparedness scores → more deaths
- Higher poverty → lower healthcare access → worse outcomes

While these correlations are empirically documented in disaster research (Cutter et al., 2003;
Wisner et al., 2004), encoding them in a model creates risk of **perpetuating structural
disadvantage**:

- If resource allocation is based on model predictions, impoverished districts may
  receive disproportionate disaster warnings (high false positive rates), leading to
  "alert fatigue" in those communities.
- Conversely, if the model underestimates risk in affluent areas with non-standard
  vulnerability patterns, those populations may be under-warned.

### 2.2 Geographic Bias

The simulation assigns fixed geographic properties (elevation, coastal distance, seismic
activity) based on a grid. In reality, disaster risk is not uniformly distributed across
any grid system. The model's spatial conclusions should not be used to allocate real
resources to specific geographic areas.

---

## 3. Responsible AI Principles

### 3.1 Transparency

- The entire data generation process is open-source and reproducible.
- All model decisions are documented with explicit rationale.
- SHAP explanations provide feature-level attribution for every prediction.
- This report discloses all known limitations.

### 3.2 Accountability

- No prediction from this system should trigger autonomous action without human review.
- Decision-makers using model outputs must understand the limitations documented in this report.
- Model predictions should be presented alongside **uncertainty intervals** (see uncertainty quantification module) to communicate confidence levels.

### 3.3 Fairness

- The model should be evaluated for **equitable performance across regions and demographics**.
- Disparate impact analysis should verify that false negative rates are not systematically
  higher for any particular subpopulation.
- If deployed, the model should be audited regularly for bias drift.

### 3.4 Human Oversight

- This model is designed as a **decision support tool**, not a decision-making system.
- Emergency management professionals must retain final authority over disaster warnings.
- The model's high false discovery rate (~61%) means that human triage of flagged alerts is essential.

---

## 4. Model Governance

### 4.1 Version Control and Auditability

- All model artefacts (trained pipelines, hyperparameters, feature lists) are versioned.
- JSON output files provide a complete audit trail from data generation to final predictions.
- Notebook execution is deterministic (seed=42), enabling exact reproduction of results.

### 4.2 Model Lifecycle

For any future operational deployment:

| Phase | Requirement |
| --- | --- |
| Development | Complete (this project) |
| Validation | Real-world data validation required |
| Deployment | Pilot advisory mode only |
| Monitoring | Monthly performance reviews, calibration checks |
| Retirement | Replace if performance degrades below BSS = 0.10 |

---

## 5. The "Cry Wolf" Problem

### 5.1 High False Positive Rates

The classifier's false discovery rate of ~61% means that approximately 6 out of every 10
flagged district-months will not experience a disaster. In real-world early warning systems,
this creates the **"cry wolf" problem**:

- Repeated false alarms erode public trust in the warning system.
- Communities may begin ignoring alerts, increasing vulnerability when a real disaster occurs.
- Emergency resources deployed for false alarms are unavailable for genuine emergencies.

### 5.2 Mitigation Strategies

1. **Two-tier alert system**: Issue "watch" alerts at the model's threshold and "warning" alerts only at higher probability thresholds, reducing false alarm fatigue for the most severe alerts.
2. **Contextual communication**: Instead of binary alerts, communicate probabilistic forecasts (e.g., "30% chance of flood in the next 30 days") to calibrate expectations.
3. **Community engagement**: Work with local authorities to build understanding of probabilistic warnings, explaining that some false alarms are the price of not missing real disasters.
4. **Feedback loops**: Use post-event analysis to continuously recalibrate and improve the model's precision over time.

---

## 6. Data Privacy and Sensitivity

While this project uses synthetic data, any real-world extension would involve sensitive
information:

- Casualty data (deaths, injuries) is inherently sensitive.
- Location-specific vulnerability data (poverty, housing quality) could stigmatise communities.
- Prediction outputs labelling areas as "Critical risk" could affect property values and insurance premiums.

**Safeguards for real-world deployment**: Anonymisation, access controls, data governance
policies, and community consent frameworks would be required.

---

## References

- Cutter, S.L., Boruff, B.J., & Shirley, W.L. (2003). "Social Vulnerability to Environmental Hazards." Social Science Quarterly, 84(2), 242–261.
- Wisner, B., Blaikie, P., Cannon, T., & Davis, I. (2004). At Risk: Natural Hazards, People's Vulnerability and Disasters. 2nd ed. Routledge.
- UNDRR (2015). Sendai Framework for Disaster Risk Reduction 2015–2030.
- European Commission (2023). INFORM Risk Index Methodology.
- Floridi, L. et al. (2018). "AI4People—An Ethical Framework for a Good AI Society." Minds and Machines, 28, 689–707.

---

*Report created to address reviewer concerns about ethical implications of disaster risk prediction models.*
