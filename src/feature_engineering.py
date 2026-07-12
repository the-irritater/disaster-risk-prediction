import numpy as np
import pandas as pd
import json
import os
from scipy.stats import spearmanr

def calculate_rates(df):
    """
    Converts raw counts of shelters, hospitals, and rescue teams into rates per 100,000 population.
    """
    df_rates = df.copy()
    # Avoid division by zero
    pop = df_rates["Population"].clip(lower=1)
    
    df_rates["Shelter_Rate_per_100k"] = (df_rates["Raw_Shelter_Count"] / pop) * 100000.0
    df_rates["Hospital_Rate_per_100k"] = (df_rates["Raw_Hospital_Count"] / pop) * 100000.0
    df_rates["Rescue_Team_Rate_per_100k"] = (df_rates["Raw_Rescue_Team_Count"] / pop) * 100000.0
    
    return df_rates

def min_max_scale_series(series, custom_min=None, custom_max=None):
    """
    Min-max normalizes a series to a 0-100 range.
    """
    s_min = custom_min if custom_min is not None else series.min()
    s_max = custom_max if custom_max is not None else series.max()
    if s_max - s_min == 0:
        return series * 0.0
    return 100.0 * (series - s_min) / (s_max - s_min)

def construct_risk_index(df, fit_scalers=None):
    """
    Constructs standardized component scores and the overall Disaster Risk Score
    using the hazard-exposure-vulnerability-preparedness framework.

    Literature Justification:
    -------------------------
    The composite risk index follows the internationally recognised
    Hazard-Exposure-Vulnerability-Capacity (HEVC) paradigm:

    - UNDRR Sendai Framework for Disaster Risk Reduction 2015–2030:
      Risk = f(Hazard, Exposure, Vulnerability, Capacity)
      https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030

    - INFORM Risk Index (European Commission JRC, 2023):
      Uses a three-dimensional model (Hazard & Exposure, Vulnerability, Lack of Coping Capacity)
      with geometric aggregation. Our additive weighted approach is a simplification.
      https://drmkc.jrc.ec.europa.eu/inform-index

    - Cardona, O.D. et al. (2012). "Determinants of risk: exposure and vulnerability."
      In IPCC SREX Report, Ch. 2. Cambridge University Press.

    Weight Rationale (Expert-specified, default):
      - Hazard (0.30): Primary driver; without a hazard trigger, no disaster occurs.
      - Exposure (0.25): Population and infrastructure at risk amplify impact.
      - Vulnerability (0.25): Socio-economic fragility determines damage severity.
      - Preparedness Deficit (0.20): Slightly lower weight because preparedness
        mitigates (rather than causes) risk.

    These weights are assessed for robustness via:
      1. Rank-correlation sensitivity analysis (equal vs expert weights)
      2. Dirichlet Monte Carlo perturbation (see src/uncertainty.py)
      3. Component knockout analysis (see sensitivity_analysis_component_knockout)
      4. Weight sweep analysis (see sensitivity_analysis_weight_sweep)

    If fit_scalers is provided (a dict of min/max values), it uses them to prevent leakage.
    Returns the dataframe with scores and the scaler parameters if fit_scalers was None.
    """
    df_scores = df.copy()
    scalers = {} if fit_scalers is None else fit_scalers
    
    # 1. Hazard Score components
    # We combine absolute deviations of rain and temperature anomalies, wind speed, and seismic index
    rain_dev = df_scores["Rainfall_Anomaly"].abs()
    temp_dev = df_scores["Temperature_Anomaly"].abs()
    wind = df_scores["Wind_Speed_kmph"]
    seismic = df_scores["Seismic_Activity_Index"]
    
    # Scale components
    components_to_scale = {
        "rain_dev": rain_dev,
        "temp_dev": temp_dev,
        "wind": wind,
        "seismic": seismic,
        "pop_density": df_scores["Population_Density"],
        "urban_rate": df_scores["Urbanisation_Rate"],
        "infra_density": df_scores["Infrastructure_Density"],
        "poverty": df_scores["Poverty_Rate"],
        "elderly": df_scores["Elderly_Population_Percentage"],
        "child": df_scores["Child_Population_Percentage"],
        "shelter_rate": df_scores["Shelter_Rate_per_100k"],
        "hospital_rate": df_scores["Hospital_Rate_per_100k"],
        "rescue_rate": df_scores["Rescue_Team_Rate_per_100k"]
    }
    
    scaled_vals = {}
    for name, series in components_to_scale.items():
        if fit_scalers is None:
            s_min = float(series.min())
            s_max = float(series.max())
            scalers[name] = {"min": s_min, "max": s_max}
        else:
            s_min = fit_scalers[name]["min"]
            s_max = fit_scalers[name]["max"]
        
        scaled_vals[name] = min_max_scale_series(series, s_min, s_max)
        
    # Construct Hazard Score (equal average of scaled factors)
    df_scores["Hazard_Score"] = (scaled_vals["rain_dev"] + scaled_vals["temp_dev"] + scaled_vals["wind"] + scaled_vals["seismic"]) / 4.0
    
    # Construct Exposure Score
    df_scores["Exposure_Score"] = (scaled_vals["pop_density"] + scaled_vals["urban_rate"] + scaled_vals["infra_density"]) / 3.0
    
    # Construct Vulnerability Score
    # Note: Housing Quality and Healthcare Access are reverse coded (lower index is worse)
    rev_housing = 100.0 - df_scores["Housing_Quality_Index"]
    rev_healthcare = 100.0 - df_scores["Healthcare_Access_Index"]
    
    # Scale reverse indices
    for name, series in [("rev_housing", rev_housing), ("rev_healthcare", rev_healthcare)]:
        if fit_scalers is None:
            s_min = float(series.min())
            s_max = float(series.max())
            scalers[name] = {"min": s_min, "max": s_max}
        else:
            s_min = fit_scalers[name]["min"]
            s_max = fit_scalers[name]["max"]
        scaled_vals[name] = min_max_scale_series(series, s_min, s_max)
        
    df_scores["Vulnerability_Score"] = (scaled_vals["poverty"] + scaled_vals["elderly"] + scaled_vals["child"] + scaled_vals["rev_housing"] + scaled_vals["rev_healthcare"]) / 5.0
    
    # Construct Preparedness Score
    # We combine facility rates and binary indicators
    ews_val = df_scores["Early_Warning_System"] * 100.0
    evac_val = df_scores["Evacuation_Plan_Available"] * 100.0
    
    df_scores["Preparedness_Score"] = (scaled_vals["shelter_rate"] + scaled_vals["hospital_rate"] + scaled_vals["rescue_rate"] + ews_val + evac_val) / 5.0
    df_scores["Preparedness_Deficit_Score"] = 100.0 - df_scores["Preparedness_Score"]
    
    # Construct Overall Descriptive Risk Score (expert weights)
    df_scores["Disaster_Risk_Score"] = (
        0.30 * df_scores["Hazard_Score"] +
        0.25 * df_scores["Exposure_Score"] +
        0.25 * df_scores["Vulnerability_Score"] +
        0.20 * df_scores["Preparedness_Deficit_Score"]
    )

    # Equal-weighted alternative for sensitivity comparison
    df_scores["Equal_Weighted_Risk"] = (
        0.25 * df_scores["Hazard_Score"] +
        0.25 * df_scores["Exposure_Score"] +
        0.25 * df_scores["Vulnerability_Score"] +
        0.25 * df_scores["Preparedness_Deficit_Score"]
    )
    
    # Define Risk Categories based on quantile thresholds
    if fit_scalers is None or "risk_thresholds" not in fit_scalers:
        q25 = float(df_scores["Disaster_Risk_Score"].quantile(0.25))
        q50 = float(df_scores["Disaster_Risk_Score"].quantile(0.50))
        q75 = float(df_scores["Disaster_Risk_Score"].quantile(0.75))
        scalers["risk_thresholds"] = {"q25": q25, "q50": q50, "q75": q75}
    else:
        q25 = fit_scalers["risk_thresholds"]["q25"]
        q50 = fit_scalers["risk_thresholds"]["q50"]
        q75 = fit_scalers["risk_thresholds"]["q75"]
        
    def assign_category(score):
        if score <= q25:
            return "Low"
        elif score <= q50:
            return "Moderate"
        elif score <= q75:
            return "High"
        else:
            return "Critical"
            
    df_scores["Risk_Category"] = df_scores["Disaster_Risk_Score"].apply(assign_category)
    
    if fit_scalers is None:
        return df_scores, scalers
    return df_scores


def sensitivity_analysis_component_knockout(df):
    """
    Component-knockout sensitivity analysis for the Disaster Risk Index.

    Removes each component one at a time, re-weights the remaining components
    proportionally, and measures rank correlation (Spearman's rho) with the
    full-index ranking. High rho indicates robustness; low rho indicates that
    the knocked-out component materially affects district rankings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Hazard_Score, Exposure_Score, Vulnerability_Score,
        Preparedness_Deficit_Score, Disaster_Risk_Score, District.

    Returns
    -------
    dict
        Per-component knockout results.
    """
    component_cols = [
        "Hazard_Score", "Exposure_Score",
        "Vulnerability_Score", "Preparedness_Deficit_Score"
    ]
    expert_weights = {"Hazard_Score": 0.30, "Exposure_Score": 0.25,
                      "Vulnerability_Score": 0.25, "Preparedness_Deficit_Score": 0.20}

    # Aggregate to district-level means
    district_means = df.groupby("District")[component_cols + ["Disaster_Risk_Score"]].mean()
    full_ranks = district_means["Disaster_Risk_Score"].rank(ascending=False)

    results = {}
    for knockout_col in component_cols:
        remaining = [c for c in component_cols if c != knockout_col]
        remaining_weights = {c: expert_weights[c] for c in remaining}
        # Re-normalise weights to sum to 1
        w_sum = sum(remaining_weights.values())
        remaining_weights = {c: w / w_sum for c, w in remaining_weights.items()}

        # Compute knockout risk score
        knockout_score = sum(
            remaining_weights[c] * district_means[c] for c in remaining
        )
        knockout_ranks = knockout_score.rank(ascending=False)

        rho, p_val = spearmanr(full_ranks, knockout_ranks)
        results[knockout_col] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(p_val),
            "renormalised_weights": {c: round(w, 4) for c, w in remaining_weights.items()},
            "interpretation": (
                f"Removing {knockout_col} {'minimally' if rho > 0.95 else 'moderately' if rho > 0.85 else 'substantially'} "
                f"affects district rankings (ρ={rho:.3f})."
            ),
        }

    return results


def sensitivity_analysis_weight_sweep(df, sweep_range=None, n_steps=21):
    """
    Weight sweep sensitivity analysis.

    Varies each component weight from 0% to 200% of its expert value
    (in steps), holding others proportionally adjusted, and computes
    rank correlation with the baseline ranking.

    Produces data suitable for a weight sensitivity heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain component scores and Disaster_Risk_Score.
    sweep_range : tuple, optional
        (min_multiplier, max_multiplier). Default (0.0, 2.0).
    n_steps : int
        Number of steps in the sweep.

    Returns
    -------
    dict
        Sweep results with multipliers and rank correlations per component.
    """
    if sweep_range is None:
        sweep_range = (0.0, 2.0)

    component_cols = [
        "Hazard_Score", "Exposure_Score",
        "Vulnerability_Score", "Preparedness_Deficit_Score"
    ]
    expert_weights = {"Hazard_Score": 0.30, "Exposure_Score": 0.25,
                      "Vulnerability_Score": 0.25, "Preparedness_Deficit_Score": 0.20}

    district_means = df.groupby("District")[component_cols + ["Disaster_Risk_Score"]].mean()
    full_ranks = district_means["Disaster_Risk_Score"].rank(ascending=False)

    multipliers = np.linspace(sweep_range[0], sweep_range[1], n_steps)

    sweep_results = {}
    for target_col in component_cols:
        rhos = []
        for mult in multipliers:
            # Adjust the target weight
            adjusted_weights = dict(expert_weights)
            adjusted_weights[target_col] = expert_weights[target_col] * mult

            # Re-normalise so weights sum to 1
            w_sum = sum(adjusted_weights.values())
            if w_sum == 0:
                rhos.append(0.0)
                continue
            adjusted_weights = {c: w / w_sum for c, w in adjusted_weights.items()}

            # Compute adjusted risk score
            adj_score = sum(
                adjusted_weights[c] * district_means[c] for c in component_cols
            )
            adj_ranks = adj_score.rank(ascending=False)
            rho, _ = spearmanr(full_ranks, adj_ranks)
            rhos.append(round(float(rho), 4))

        sweep_results[target_col] = {
            "multipliers": [round(float(m), 2) for m in multipliers],
            "spearman_rhos": rhos,
            "min_rho": min(rhos),
            "most_sensitive_multiplier": round(float(multipliers[np.argmin(rhos)]), 2),
        }

    return sweep_results

def engineer_features(df):
    """
    Engineers seasonal, coastal, density flags, rolling disaster counts,
    and the lead target variable 'Disaster_Next_Month'.
    """
    df_eng = df.copy()
    
    # 1. Date parts
    df_eng["Event_Date"] = pd.to_datetime(df_eng["Event_Date"])
    df_eng["Month_Name"] = df_eng["Event_Date"].dt.strftime("%B")
    df_eng["Quarter"] = df_eng["Event_Date"].dt.to_period("Q").astype(str)
    
    # Season
    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Autumn"
            
    df_eng["Season"] = df_eng["Month"].apply(get_season)
    
    # Flags
    df_eng["Coastal_Region_Flag"] = (df_eng["Distance_From_Coast_km"] < 100.0).astype(int)
    
    median_density = df_eng["Population_Density"].median()
    df_eng["High_Population_Density_Flag"] = (df_eng["Population_Density"] > median_density).astype(int)
    df_eng["Extreme_Rainfall_Flag"] = (df_eng["Rainfall_Anomaly"] > 1.5).astype(int)
    df_eng["Extreme_Temperature_Flag"] = (df_eng["Temperature_Anomaly"] > 1.5).astype(int)
    
    # Gaps
    if "Disaster_Risk_Score" in df_eng.columns:
        df_eng["Risk_Preparedness_Gap"] = df_eng["Disaster_Risk_Score"] - df_eng["Preparedness_Score"]
        
    # 2. Time-series features (Lags and Rolling count per District)
    # Ensure sorted order for grouping shifts
    df_eng = df_eng.sort_values(by=["District", "Year", "Month"]).reset_index(drop=True)
    
    # Lags (t-1)
    df_eng["Previous_Month_Disaster_Occurred"] = df_eng.groupby("District")["Disaster_Occurred"].shift(1)
    df_eng["Previous_Month_Hazard_Severity"] = df_eng.groupby("District")["Hazard_Severity"].shift(1)
    
    # Fill first month lag with 0
    df_eng["Previous_Month_Disaster_Occurred"] = df_eng["Previous_Month_Disaster_Occurred"].fillna(0).astype(int)
    df_eng["Previous_Month_Hazard_Severity"] = df_eng["Previous_Month_Hazard_Severity"].fillna(0.0)
    
    # Rolling 12-month disaster count (excl. current month to avoid leakage)
    df_eng["Rolling_12_Month_Disaster_Count"] = (
        df_eng.groupby("District")["Disaster_Occurred"]
        .rolling(window=12, closed="left")
        .sum()
        .reset_index(level=0, drop=True)
    )
    df_eng["Rolling_12_Month_Disaster_Count"] = df_eng["Rolling_12_Month_Disaster_Count"].fillna(0).astype(int)
    
    # 3. Create the Target variable: Disaster in Next Month (t+1)
    df_eng["Disaster_Next_Month"] = df_eng.groupby("District")["Disaster_Occurred"].shift(-1)
    
    # We will have NaNs for Dec 2025. We will drop Dec 2025 rows when modeling or fill appropriately.
    # Note: Keep the NaNs here so that modeling functions can drop them explicitly.
    
    return df_eng
