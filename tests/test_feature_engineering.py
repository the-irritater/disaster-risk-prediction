import numpy as np
import pandas as pd
from src.data_generation import DisasterDataGenerator
from src.feature_engineering import calculate_rates, construct_risk_index, engineer_features

def test_facility_rates_calculation():
    """
    Verifies that raw counts of emergency facilities are successfully normalized to rates per 100,000.
    """
    test_df = pd.DataFrame({
        "Population": [100000, 200000, 50000],
        "Raw_Shelter_Count": [5, 10, 2],
        "Raw_Hospital_Count": [2, 4, 1],
        "Raw_Rescue_Team_Count": [3, 6, 0]
    })
    
    df_rates = calculate_rates(test_df)
    
    # Assert rates
    assert np.allclose(df_rates["Shelter_Rate_per_100k"], [5.0, 5.0, 4.0])
    assert np.allclose(df_rates["Hospital_Rate_per_100k"], [2.0, 2.0, 2.0])
    assert df_rates.loc[2, "Rescue_Team_Rate_per_100k"] == 0.0

def test_rolling_and_lag_features():
    """
    Verifies that lag and rolling disaster features are created correctly per district
    without temporal leakage.
    """
    # Create simple mock district panel data
    # District D1 has disasters in months 1 and 2, none in month 3.
    mock_data = pd.DataFrame({
        "District": ["D1", "D1", "D1", "D1", "D2", "D2"],
        "Year": [2020, 2020, 2020, 2020, 2020, 2020],
        "Month": [1, 2, 3, 4, 1, 2],
        "Event_Date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-01-01", "2020-02-01"],
        "Disaster_Occurred": [1, 1, 0, 1, 0, 1],
        "Hazard_Severity": [5.0, 7.0, 0.0, 4.0, 0.0, 6.0],
        "Distance_From_Coast_km": [200.0, 200.0, 200.0, 200.0, 50.0, 50.0],
        "Population_Density": [100.0, 100.0, 100.0, 100.0, 300.0, 300.0],
        "Rainfall_Anomaly": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "Temperature_Anomaly": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    df_eng = engineer_features(mock_data)
    
    # Assert lags for D1
    d1_eng = df_eng[df_eng["District"] == "D1"].sort_values("Month").reset_index(drop=True)
    
    # Month 1 lag should be 0 (no history)
    assert d1_eng.loc[0, "Previous_Month_Disaster_Occurred"] == 0
    # Month 2 lag should be 1 (Disaster occurred in Month 1)
    assert d1_eng.loc[1, "Previous_Month_Disaster_Occurred"] == 1
    # Month 3 lag should be 1 (Disaster occurred in Month 2)
    assert d1_eng.loc[2, "Previous_Month_Disaster_Occurred"] == 1
    # Month 4 lag should be 0 (No disaster occurred in Month 3)
    assert d1_eng.loc[3, "Previous_Month_Disaster_Occurred"] == 0
    
    # Assert target (lead variable: Disaster in month t+1)
    assert d1_eng.loc[0, "Disaster_Next_Month"] == 1  # Month 2
    assert d1_eng.loc[1, "Disaster_Next_Month"] == 0  # Month 3
    assert d1_eng.loc[2, "Disaster_Next_Month"] == 1  # Month 4
    assert pd.isna(d1_eng.loc[3, "Disaster_Next_Month"])  # Month 5 doesn't exist
