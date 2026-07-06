import numpy as np
import pandas as pd
from src.data_generation import DisasterDataGenerator

def test_bounds_and_validity():
    """
    Verifies that spatial coordinates, probabilities, and indices lie within valid mathematical and physical bounds.
    """
    generator = DisasterDataGenerator(num_districts=10, num_years=2, seed=42)
    df = generator.generate_panel_data()
    
    # Coordinates check
    assert df["Latitude"].between(-90, 90).all(), "Latitudes out of bounds"
    assert df["Longitude"].between(-180, 180).all(), "Longitudes out of bounds"
    
    # Index values check
    assert df["Soil_Moisture"].between(0, 1).all(), "Soil moisture must be between 0 and 1"
    assert df["Vegetation_Index"].between(0, 1).all(), "Vegetation index must be between 0 and 1"
    assert df["Drought_Index"].between(0, 100).all(), "Drought index must be between 0 and 100"
    assert df["Disaster_Preparedness_Index"].between(10, 100).all(), "Preparedness index out of bounds"
    
    # Binary variables check
    assert df["Disaster_Occurred"].isin([0, 1]).all(), "Disaster_Occurred must be binary (0 or 1)"
    assert df["Early_Warning_System"].isin([0, 1]).all(), "EWS must be binary"
    assert df["Evacuation_Plan_Available"].isin([0, 1]).all(), "Evacuation plan must be binary"

def test_nonnegative_impact_metrics():
    """
    Verifies that deaths, injuries, affected people, and economic losses are non-negative.
    """
    generator = DisasterDataGenerator(num_districts=10, num_years=2, seed=42)
    df = generator.generate_panel_data()
    
    impact_cols = ["Number_of_Deaths", "Number_of_Injuries", "Number_of_People_Affected", 
                   "Displacement_Count", "Houses_Damaged", "Infrastructure_Damage_Score", 
                   "Crop_Loss_Percentage", "Economic_Loss_Million"]
                   
    for col in impact_cols:
        assert (df[col] >= 0).all(), f"Found negative values in impact column: {col}"

def test_zero_impact_on_non_event():
    """
    Verifies that all post-event impact variables are exactly 0 in district-months where no disaster occurred.
    """
    generator = DisasterDataGenerator(num_districts=20, num_years=3, seed=42)
    df = generator.generate_panel_data()
    
    non_event_df = df[df["Disaster_Occurred"] == 0]
    
    impact_cols = ["Number_of_Deaths", "Number_of_Injuries", "Number_of_People_Affected", 
                   "Displacement_Count", "Houses_Damaged", "Infrastructure_Damage_Score", 
                   "Crop_Loss_Percentage", "Economic_Loss_Million"]
                   
    for col in impact_cols:
        assert (non_event_df[col] == 0).all(), f"Found non-zero values in {col} when no disaster occurred"

def test_no_disaster_type_missing_values():
    """
    Verifies that Disaster_Type has no missing values and contains 'No Disaster'.
    """
    generator = DisasterDataGenerator(seed=42)
    df = generator.generate_panel_data()
    assert df["Disaster_Type"].notna().all(), "Disaster_Type should not contain NaN values"
    assert "No Disaster" in df["Disaster_Type"].unique(), "'No Disaster' should be a category in Disaster_Type"
