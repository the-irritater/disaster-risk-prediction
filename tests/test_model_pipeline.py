import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.data_generation import DisasterDataGenerator
from src.feature_engineering import calculate_rates, construct_risk_index, engineer_features
from src.train_models import PRE_EVENT_PREDICTORS, split_chronologically, train_classifier, save_pipeline

def test_pipeline_fitting_and_serialization(tmp_path):
    """
    Fits the preprocessing and classifier pipeline on training data,
    saves it to a temporary path, loads it, and compares predictions to ensure equality.
    """
    # 1. Generate data
    generator = DisasterDataGenerator(num_districts=5, num_years=5, seed=42)
    df = generator.generate_panel_data()
    df = calculate_rates(df)
    df, _ = construct_risk_index(df)
    df = engineer_features(df)
    
    train, val, _ = split_chronologically(df, train_end_year=2018, val_end_year=2019)
    
    X_train = train[PRE_EVENT_PREDICTORS]
    y_train = train["Disaster_Next_Month"]
    
    X_val = val[PRE_EVENT_PREDICTORS]
    
    # 2. Train classifier (Random Forest)
    pipeline = train_classifier(X_train, y_train, model_type="random_forest")
    
    assert isinstance(pipeline, Pipeline)
    
    # Make original predictions
    probs_orig = pipeline.predict_proba(X_val)[:, 1]
    
    # 3. Save to disk (temp file)
    temp_model_path = os.path.join(tmp_path, "test_pipeline.joblib")
    save_pipeline(pipeline, temp_model_path)
    
    assert os.path.exists(temp_model_path)
    
    # 4. Load from disk
    loaded_pipeline = joblib.load(temp_model_path)
    probs_loaded = loaded_pipeline.predict_proba(X_val)[:, 1]
    
    # Compare
    np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5)
