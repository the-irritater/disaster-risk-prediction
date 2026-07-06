import os
import joblib
import pandas as pd
from src.data_generation import DisasterDataGenerator
from src.feature_engineering import calculate_rates, construct_risk_index, engineer_features
from src.train_models import PRE_EVENT_PREDICTORS, split_chronologically, train_classifier, save_pipeline
from src.predict_risk import predict_disaster_risk

def test_prediction_output_schema(tmp_path):
    """
    Fits a model pipeline on mock data, runs prediction, and verifies the output schema and value ranges.
    """
    # 1. Generate and split data
    generator = DisasterDataGenerator(num_districts=5, num_years=5, seed=42)
    df = generator.generate_panel_data()
    df = calculate_rates(df)
    df, _ = construct_risk_index(df)
    df = engineer_features(df)
    
    train, val, _ = split_chronologically(df, train_end_year=2018, val_end_year=2019)
    
    X_train = train[PRE_EVENT_PREDICTORS]
    y_train = train["Disaster_Next_Month"]
    
    # Train and save pipeline
    pipeline = train_classifier(X_train, y_train, model_type="random_forest")
    temp_model_path = os.path.join(tmp_path, "test_pipeline.joblib")
    save_pipeline(pipeline, temp_model_path)
    
    # Save input csv for prediction
    temp_input_csv = os.path.join(tmp_path, "input.csv")
    val.to_csv(temp_input_csv, index=False)
    
    # 2. Run prediction
    results = predict_disaster_risk(temp_input_csv, temp_model_path, threshold=0.40)
    
    # Assert columns
    expected_cols = ["Record_ID", "District", "Year", "Month", 
                     "Predicted_Disaster_Probability", "Predicted_Disaster_Next_Month", "Decision_Threshold_Used"]
    for col in expected_cols:
        assert col in results.columns, f"Output columns missing: {col}"
        
    # Assert ranges
    assert results["Predicted_Disaster_Probability"].between(0.0, 1.0).all(), "Predicted probability out of bounds"
    assert results["Predicted_Disaster_Next_Month"].isin([0, 1]).all(), "Predictions must be binary"
    assert (results["Decision_Threshold_Used"] == 0.40).all()
