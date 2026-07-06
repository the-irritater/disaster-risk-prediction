import os
import argparse
import joblib
import pandas as pd

def predict_disaster_risk(input_data_path, model_path="models/disaster_risk_pipeline.joblib", threshold=0.35):
    """
    Loads the saved model pipeline and predicts disaster probability and binary occurrence
    for the next month.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline not found at {model_path}. Please run training first.")
        
    print(f"Loading pipeline from {model_path}...")
    pipeline = joblib.load(model_path)
    
    print(f"Reading input data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    
    # Identify record keys
    record_keys = ["Record_ID", "District", "Year", "Month"]
    missing_keys = [k for k in record_keys if k not in df.columns]
    if missing_keys:
        raise ValueError(f"Input data is missing record identifiers: {missing_keys}")
        
    # Check predictor availability
    from src.train_models import PRE_EVENT_PREDICTORS
    missing_predictors = [p for p in PRE_EVENT_PREDICTORS if p not in df.columns]
    if missing_predictors:
        print(f"Warning: Missing columns from input. Preprocessing imputer will attempt to fill: {missing_predictors}")
        
    # Make prediction
    # Predict probability of 1 (Yes)
    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Construct output DataFrame
    output = df[record_keys].copy()
    output["Predicted_Disaster_Probability"] = probabilities.round(4)
    output["Predicted_Disaster_Next_Month"] = predictions

    output["Decision_Threshold_Used"] = threshold
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict future disaster risk using the trained pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV containing district-month predictors.")
    parser.add_argument("--model", type=str, default="models/disaster_risk_pipeline.joblib", help="Path to the serialized joblib pipeline.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification probability decision threshold.")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Path to save output predictions.")
    
    args = parser.parse_args()
    
    try:
        results = predict_disaster_risk(args.input, args.model, args.threshold)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Predictions successfully written to {args.output}")
        print(results.head(10))
    except Exception as e:
        print(f"Error during prediction: {e}")
