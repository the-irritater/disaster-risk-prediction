# Simulation Design Documentation

## Metadata

| Field | Value |
| --- | --- |
| **Report Title** | Synthetic Data Simulation Design & Validation |
| **Project** | Disaster Risk Prediction Analytics Framework |
| **Author** | Sanman |
| **Date** | July 2026 |
| **Version** | 3.2 |
| **Status** | Research Submission (Simulation-Based) |

> **Purpose.** This document provides full methodological transparency about the synthetic
> data generation process, addressing the scientific concern that statistical conclusions
> from simulated data are conditional on simulation assumptions.

---

## 1. Design Philosophy

The data generator (`src/data_generation.py`) creates a **district-month panel dataset**
(100 districts × 132 months = 13,200 observations) that simulates key features of a
multi-hazard disaster risk environment. The design follows a **structural simulation**
approach: physical and socio-economic relationships are encoded via parameterised
statistical models rather than being sampled independently.

### 1.1 Why Synthetic Data?

1. **Real-world disaster datasets** (e.g., EM-DAT, SHELDUS, DesInventar) lack the
   granularity needed for district-month panel modelling with complete covariate coverage.
2. Synthetic data allows **controlled experimentation**: we know the ground truth
   data-generating process (DGP), enabling us to verify whether analytical methods
   recover known relationships.
3. **Ethical considerations**: Real disaster data involves sensitive information about
   casualties and displacement; synthetic data avoids privacy and sensitivity concerns.

### 1.2 Limitations of This Approach

- Statistical findings demonstrate methodology, not empirical discoveries.
- High model performance may reflect recovery of programmed relationships.
- Marginal distributions are calibrated to plausible ranges but not formally fitted to real data.

---

## 2. Causal Structure (Directed Acyclic Graph)

The simulation embeds the following causal DAG:

```
Geography (Grid Position, Elevation, Coastal Distance)
    ↓
Environmental Variables (Rainfall, Temperature, Wind, Soil Moisture)
    ↓                    ↓
Seasonal Patterns    Climate Trends (linear year effect)
    ↓                    ↓
Hazard Probabilities (via logistic link functions)
    ↓
Disaster Occurrence (Bernoulli trials)
    ↓
Impact Variables (Deaths, Injuries, Economic Loss — conditional on occurrence)
```

**Key structural relationships:**
- Geography is exogenous (fixed grid assignment)
- Environmental variables depend on geography + season + climate trend + autoregressive persistence
- Hazard probabilities are logistic functions of environmental triggers and geography
- Impact outcomes are generated ONLY when a disaster occurs (no leakage)

---

## 3. Distribution Rationale by Variable Family

### 3.1 Environmental Variables

| Variable | Distribution | Rationale |
| --- | --- | --- |
| Monthly Rainfall | Gaussian noise around seasonal baseline | Tropical monsoon patterns: high June–Sep, low Dec–Feb. Coastal and elevation boosts. Reference: IMD monsoon statistics. |
| Temperature | Gaussian with lapse rate | Elevation-dependent cooling (6°C/1000m lapse rate). Summer peak in April–June. Reference: Standard atmospheric lapse rate. |
| Wind Speed | Uniform noise + autoregression | Higher near coast during cyclone season (May, Oct, Nov). Reference: IMD cyclone statistics. |
| River Level | Gaussian noise + rain-driven rise | Dependent on upstream rainfall anomaly and elevation. Reference: CWC flood monitoring thresholds. |
| Soil Moisture | Gaussian noise, clipped [0.05, 0.95] | Driven by rainfall and temperature. Reference: NASA SMAP soil moisture range 0.05–0.50 m³/m³. |
| Drought Index | Gaussian noise, clipped [0, 100] | Inverse of soil moisture plus temperature stress. Reference: Palmer Drought Severity Index conceptual basis. |
| NDVI (Vegetation) | Gaussian noise, clipped [0.1, 0.9] | Lag response to soil moisture; lower in urban areas. Reference: MODIS NDVI range 0.1–0.9. |
| Seismic Activity | Rare Bernoulli shocks | Background index + rare high-magnitude events (p ≈ 0.015 per month). Reference: USGS seismic hazard zonation. |

**Autoregressive persistence**: Environmental variables use AR(1) with coefficient ≈ 0.55, reflecting month-to-month persistence in climate systems.

### 3.2 Hazard Probability Models

Each hazard type uses a **logistic link function** with domain-specific coefficients:

| Hazard | Logistic z-formula | Key Drivers |
| --- | --- | --- |
| Flood | z = -5.8 + 2.8·rain_anomaly − 0.008·elevation − 0.015·dist_river + 0.5·river_level | Rain, low elevation, river proximity |
| Cyclone | z = -7.5 + 0.12·wind − 0.015·dist_coast | Wind speed, coastal proximity; active May, Oct, Nov only |
| Landslide | z = -7.2 + 2.2·rain_anomaly + 0.14·slope − 3.5·NDVI | Rain, steep terrain, low vegetation; elevation > 400m only |
| Drought | z = -5.5 + 1.8·temp_anomaly − 2.5·rain_anomaly − 4.5·soil_moisture | Heat, low rain, dry soil; active Mar–May, Oct–Nov |
| Earthquake | z = -9.0 + 3.2·seismic_val | Seismic index; location-independent timing |
| Heatwave | z = -5.2 + 3.0·temp_anomaly + 0.002·pop_density | Heat, urban density; active Apr–Jun |
| Wildfire | z = -6.5 + 2.2·temp_anomaly − 3.8·soil_moisture + 1.2·NDVI | Heat, dry soil, vegetation fuel; rural areas only |
| Severe Storm | z = -6.2 + 0.06·wind + 1.2·rain_anomaly | Wind and rain; year-round |

**Coefficient justification**: Intercepts are calibrated to produce ~15% overall disaster prevalence, consistent with the typical disaster event rate in high-risk South Asian districts (reference: DesInventar India database, 2000–2020).

### 3.3 Impact Variables

| Variable | Distribution | Conditioning |
| --- | --- | --- |
| Deaths | Poisson(λ) where log(λ) depends on severity, density, poverty, preparedness | Only generated when disaster_occurred = 1 |
| Injuries | deaths × Uniform(2, 10) + Uniform(1, 15) | Injuries are typically 2–10× deaths in disaster events |
| Economic Loss | Gamma(shape=2, scale=f(severity, infrastructure, preparedness)) | Heavy-tailed; reference: EM-DAT loss distributions |
| Displacement | affected × f(housing_quality) | Poor housing increases displacement |
| Crop Loss | Severity × agricultural percentage + noise | Only for Flood, Cyclone, Drought |

### 3.4 Demographic and Socio-Economic Variables

These are **district-level fixed effects** (constant across time):

| Variable | Distribution | Range | Reference |
| --- | --- | --- | --- |
| Population | Uniform(200k, 2.5M) | Indian district population range (Census 2011) |
| Poverty Rate | Uniform(0.08, 0.45) | Indian district poverty range (NITI Aayog, 2015) |
| Urbanisation | Uniform(0.10, 0.85) | India urban–rural spectrum |
| Literacy | Uniform(0.55, 0.95) | Indian district literacy range (Census 2011) |

---

## 4. Validation Against Reference Distributions

### 4.1 Summary Statistics Comparison

The following table compares key simulated variable ranges against real-world reference values:

| Variable | Simulated Range | Simulated Mean | Reference Range | Source |
| --- | --- | --- | --- | --- |
| Monthly Rainfall (mm) | 0 – 950 | ~180 | 0 – 1000+ | IMD India |
| Temperature (°C) | 5 – 45 | ~27 | 5 – 48 | IMD India |
| Wind Speed (km/h) | 5 – 80 | ~25 | 5 – 200 (cyclones) | IMD |
| Disaster prevalence | ~15% | — | 10–20% (event-months) | DesInventar India |
| Population Density | 70 – 5000/km² | ~700 | 50 – 11,000/km² | Census India 2011 |
| Poverty Rate | 8% – 45% | ~26% | 5% – 50% | NITI Aayog |

### 4.2 Known Simplifications

1. **No cascading failures**: A flood does not increase the probability of a landslide in the same district-month (independent Bernoulli trials).
2. **No inter-district contagion**: A disaster in one district does not affect neighbouring districts.
3. **No policy dynamics**: Preparedness levels are fixed across time (no learning from past events).
4. **No population growth**: Demographic variables are time-invariant.
5. **Single hazard resolution**: Only the dominant hazard type is recorded per district-month.
6. **Linear climate trend**: Climate change effect is a simple linear trend, not step-function or accelerating.

---

## 5. Implications for Statistical Inference

1. **Significant findings confirm recovery of programmed relationships**, not empirical discoveries.
2. **Model performance is necessarily bounded above** by the information content of the generating process.
3. **Non-significant findings** (e.g., ANOVA p = 0.63) are genuine: the simulation does not embed all possible relationships.
4. **External validity requires real-world data validation**: These results establish that the analytical pipeline is correctly implemented, not that the models will generalise to real disaster scenarios.

---

*Document created to address reviewer concerns about simulation design transparency and methodological justification.*
