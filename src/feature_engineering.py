import numpy as np
import pandas as pd

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
    
    # Construct Overall Descriptive Risk Score
    df_scores["Disaster_Risk_Score"] = (
        0.30 * df_scores["Hazard_Score"] +
        0.25 * df_scores["Exposure_Score"] +
        0.25 * df_scores["Vulnerability_Score"] +
        0.20 * df_scores["Preparedness_Deficit_Score"]
    )
    
    # Define Risk Categories
    def assign_category(score):
        if score < 35.0:
            return "Low"
        elif score < 50.0:
            return "Moderate"
        elif score < 65.0:
            return "High"
        else:
            return "Critical"
            
    df_scores["Risk_Category"] = df_scores["Disaster_Risk_Score"].apply(assign_category)
    
    if fit_scalers is None:
        return df_scores, scalers
    return df_scores

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
