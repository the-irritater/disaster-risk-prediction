import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# List of pre-event predictors (Features at month t to predict disaster at month t+1)
PRE_EVENT_PREDICTORS = [
    # Environmental
    "Rainfall_Anomaly", "Temperature_Anomaly", "Wind_Speed_kmph", "River_Level_Metres",
    "Soil_Moisture", "Drought_Index", "Vegetation_Index", "Elevation_Metres", 
    "Slope_Degrees", "Distance_From_Coast_km", "Distance_From_River_km", "Seismic_Activity_Index",
    # Exposure
    "Population_Density", "Urbanisation_Rate", "Infrastructure_Density",
    # Vulnerability
    "Poverty_Rate", "Elderly_Population_Percentage", "Child_Population_Percentage",
    "Housing_Quality_Index", "Healthcare_Access_Index", "Road_Access_Index", "Communication_Access_Index",
    # Preparedness
    "Shelter_Rate_per_100k", "Hospital_Rate_per_100k", "Rescue_Team_Rate_per_100k",
    "Early_Warning_System", "Evacuation_Plan_Available", "Emergency_Response_Time_Minutes", "Disaster_Preparedness_Index",
    # Lags and History
    "Previous_Month_Disaster_Occurred", "Previous_Month_Hazard_Severity", "Rolling_12_Month_Disaster_Count",
    # Categorical
    "Region", "Season"
]

# Post-event impact variables (must NEVER be used as features for pre-event prediction)
POST_EVENT_VARIABLES = [
    "Number_of_Deaths", "Number_of_Injuries", "Number_of_People_Affected",
    "Displacement_Count", "Houses_Damaged", "Infrastructure_Damage_Score",
    "Crop_Loss_Percentage", "Economic_Loss_Million"
]

def split_chronologically(df, train_end_year=2022, val_end_year=2024):
    """
    Splits the district-month panel dataset chronologically.
    Training: Jan 2015 - Dec 2022
    Validation: Jan 2023 - Dec 2024
    Testing: Jan 2025 - Dec 2025
    """
    # Drop rows where target is NaN (e.g. Dec 2025 since there is no t+1)
    df_model = df.dropna(subset=["Disaster_Next_Month"]).copy()
    
    train_df = df_model[df_model["Year"] <= train_end_year]
    val_df = df_model[(df_model["Year"] > train_end_year) & (df_model["Year"] <= val_end_year)]
    test_df = df_model[df_model["Year"] > val_end_year]
    
    return train_df, val_df, test_df

def build_preprocessing_pipeline():
    """
    Constructs a ColumnTransformer for preprocessing numeric and categorical variables.
    """
    numeric_features = [f for f in PRE_EVENT_PREDICTORS if f not in ["Region", "Season"]]
    categorical_features = ["Region", "Season"]
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def train_classifier(X_train, y_train, model_type="random_forest", class_weights=None):
    """
    Trains a classifier for future disaster occurrence (Disaster_Next_Month).
    """
    preprocessor = build_preprocessing_pipeline()
    
    # Select classifier
    if model_type == "logistic_regression":
        classifier = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42, n_jobs=-1)
    elif model_type == "xgboost":
        # Compute scale_pos_weight if needed for imbalanced classes
        scale_pos = 1.0
        if class_weights == "balanced":
            pos_count = (y_train == 1).sum()
            neg_count = (y_train == 0).sum()
            scale_pos = neg_count / max(1, pos_count)
        classifier = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def train_regressor(X_train, y_train, model_type="random_forest"):
    """
    Trains a regressor for conditional disaster impact (e.g. Economic Loss).
    """
    preprocessor = build_preprocessing_pipeline()
    
    if model_type == "linear":
        regressor = Ridge(alpha=1.0, random_state=42)
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == "xgboost":
        regressor = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def save_pipeline(pipeline, filename="disaster_risk_pipeline.joblib"):
    """
    Saves the complete fitted scikit-learn pipeline to disk.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(pipeline, filename)
    print(f"Pipeline saved to {filename}")
