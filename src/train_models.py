import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, GammaRegressor, TweedieRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

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
    df_model = df.dropna(subset=["Disaster_Next_Month"]).copy()
    train_df = df_model[df_model["Year"] <= train_end_year]
    val_df = df_model[(df_model["Year"] > train_end_year) & (df_model["Year"] <= val_end_year)]
    test_df = df_model[df_model["Year"] > val_end_year]
    return train_df, val_df, test_df

def split_classification_4stage(df, train_end_year=2022, cal_end_year=2023, val_end_year=2024):
    """
    Splits the dataset into 4 chronological stages for classification.
    Train: 2015 - 2022
    Calibration: 2023
    Validation: 2024
    Test: 2025 (excluding Dec 2025)
    """
    df_model = df.dropna(subset=["Disaster_Next_Month"]).copy()
    train_df = df_model[df_model["Year"] <= train_end_year]
    cal_df = df_model[(df_model["Year"] > train_end_year) & (df_model["Year"] <= cal_end_year)]
    val_df = df_model[(df_model["Year"] > cal_end_year) & (df_model["Year"] <= val_end_year)]
    test_df = df_model[df_model["Year"] > val_end_year]
    return train_df, cal_df, val_df, test_df

def split_regression_chronologically(df, train_end_year=2022, val_end_year=2024):
    """
    Splits the dataset chronologically for regression (does not drop based on Disaster_Next_Month).
    Train: <= 2022
    Validation: 2023 - 2024
    Test: 2025
    """
    train_df = df[df["Year"] <= train_end_year]
    val_df = df[(df["Year"] > train_end_year) & (df["Year"] <= val_end_year)]
    test_df = df[df["Year"] > val_end_year]
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
        classifier = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42, n_jobs=1)
    elif model_type == "xgboost":
        # Compute scale_pos_weight if needed for imbalanced classes
        scale_pos = 1.0
        if class_weights == "balanced":
            pos_count = (y_train == 1).sum()
            neg_count = (y_train == 0).sum()
            scale_pos = neg_count / max(1, pos_count)
        classifier = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=1)
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
    Ridge, RandomForest, and XGBoost are fitted on log-transformed targets (log1p).
    Gamma and Tweedie regressors are fitted on the original scale (requiring y > 0).
    """
    preprocessor = build_preprocessing_pipeline()
    
    if model_type == "linear" or model_type == "ridge":
        regressor = Ridge(alpha=1.0, random_state=42)
        fit_log = True
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        fit_log = True
    elif model_type == "xgboost":
        regressor = XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)
        fit_log = True
    elif model_type == "gamma":
        regressor = GammaRegressor(max_iter=10000, alpha=1.0)
        fit_log = False
    elif model_type == "tweedie":
        regressor = TweedieRegressor(power=1.5, max_iter=10000, link="log")
        fit_log = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    pipeline.fit_log_ = fit_log
    
    if fit_log:
        y_fit = np.log1p(y_train)
    else:
        y_fit = y_train
        
    pipeline.fit(X_train, y_fit)
    return pipeline

def tune_classifier(X_train, y_train, model_type="random_forest", class_weights=None):
    """
    Performs hyperparameter tuning for the classifier using TimeSeriesSplit (chronological folds).
    """
    preprocessor = build_preprocessing_pipeline()
    
    if model_type == "random_forest":
        base_estimator = RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=1)
        param_distributions = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10]
        }
    elif model_type == "xgboost":
        scale_pos = 1.0
        if class_weights == "balanced":
            pos_count = (y_train == 1).sum()
            neg_count = (y_train == 0).sum()
            scale_pos = neg_count / max(1, pos_count)
        base_estimator = XGBClassifier(scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=1)
        param_distributions = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2]
        }
    else:
        raise ValueError(f"Tuning not supported for model_type: {model_type}")
        
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_estimator)
    ])
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring="average_precision",
        random_state=42,
        n_jobs=1
    )
    
    search.fit(X_train, y_train)
    print(f"Best parameters for {model_type}: {search.best_params_}")
    return search.best_estimator_

def save_pipeline(pipeline, filename="disaster_risk_pipeline.joblib"):
    """
    Saves the complete fitted scikit-learn pipeline to disk.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(pipeline, filename)
    print(f"Pipeline saved to {filename}")
