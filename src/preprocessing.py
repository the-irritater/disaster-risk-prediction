import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def check_data_structure(df):
    """
    Checks and returns the basic dimensions, columns, data types, and null counts of the dataset.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_records": df.duplicated().sum(),
        "duplicate_event_ids": df["Record_ID"].duplicated().sum() if "Record_ID" in df.columns else 0
    }
    return summary

def handle_missing_values(df, numeric_strategy="median", categorical_strategy="most_frequent"):
    """
    Detects and imputes missing values for numeric and categorical columns.
    Excludes the target columns and identifiers from imputation.
    """
    df_imputed = df.copy()
    
    # Exclude ID and target variables from automatic imputation
    exclude_cols = ["Record_ID", "Event_Date", "Disaster_Type", "Disaster_Occurred"]
    
    num_cols = df_imputed.select_dtypes(include=[np.number]).columns
    cat_cols = df_imputed.select_dtypes(exclude=[np.number]).columns
    
    num_cols = [c for c in num_cols if c not in exclude_cols]
    cat_cols = [c for c in cat_cols if c not in exclude_cols]
    
    # Impute numeric columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy=numeric_strategy)
        df_imputed[num_cols] = imputer_num.fit_transform(df_imputed[num_cols])
        
    # Impute categorical columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy=categorical_strategy)
        df_imputed[cat_cols] = imputer_cat.fit_transform(df_imputed[cat_cols])
        
    return df_imputed

def check_outliers_and_anomalies(df):
    """
    Identifies outliers using Interquartile Range (IQR) on select continuous variables
    and reports impossible values (like negative deaths or losses).
    """
    anomaly_report = {}
    
    # Check for impossible values
    impossible_checks = {
        "negative_deaths": (df["Number_of_Deaths"] < 0).sum() if "Number_of_Deaths" in df.columns else 0,
        "negative_injuries": (df["Number_of_Injuries"] < 0).sum() if "Number_of_Injuries" in df.columns else 0,
        "negative_economic_loss": (df["Economic_Loss_Million"] < 0).sum() if "Economic_Loss_Million" in df.columns else 0,
        "invalid_latitude": ((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum() if "Latitude" in df.columns else 0,
        "invalid_longitude": ((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum() if "Longitude" in df.columns else 0,
    }
    anomaly_report["impossible_values"] = impossible_checks
    
    # Check for extreme outlier events (using IQR method)
    outlier_cols = ["Annual_Rainfall_mm", "Wind_Speed_kmph", "Seismic_Activity_Index", "Hazard_Severity"]
    iqr_outliers = {}
    for col in outlier_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            iqr_outliers[col] = {
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outliers_count": int(outliers_count)
            }
    anomaly_report["iqr_outliers"] = iqr_outliers
    
    return anomaly_report

def clean_data(df):
    """
    Applies data-type corrections, ensures chronological sorting, handles duplicates,
    and removes impossible values.
    """
    df_clean = df.copy()
    
    # 1. Convert Event_Date to proper datetime
    df_clean["Event_Date"] = pd.to_datetime(df_clean["Event_Date"])
    
    # 2. Sort chronologically
    df_clean = df_clean.sort_values(by=["Year", "Month", "District"]).reset_index(drop=True)
    
    # 3. Drop exact duplicates if any
    df_clean = df_clean.drop_duplicates()
    
    # 4. Correct impossible values (e.g. clip negative deaths to 0)
    for col in ["Number_of_Deaths", "Number_of_Injuries", "Number_of_People_Affected", "Displacement_Count", "Houses_Damaged", "Economic_Loss_Million"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0)
            
    return df_clean
