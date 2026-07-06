import numpy as np
import pandas as pd
from src.data_generation import DisasterDataGenerator

def test_data_generation_shape_and_columns():
    """
    Verifies that the generated panel dataset has exactly 13,200 rows
    (100 districts * 11 years * 12 months) and all expected columns.
    """
    generator = DisasterDataGenerator(num_districts=100, num_years=11, seed=42)
    df = generator.generate_panel_data()
    
    # Assert dimensions
    assert df.shape[0] == 13200, f"Expected 13,200 rows, got {df.shape[0]}"
    
    # Assert existence of key columns
    expected_cols = ["Record_ID", "District", "Year", "Month", "Disaster_Type", "Disaster_Occurred", 
                     "Rainfall_Anomaly", "Temperature_Anomaly", "Population_Density", 
                     "Poverty_Rate", "Disaster_Preparedness_Index", "Economic_Loss_Million"]
    
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} was not generated"

def test_reproducibility():
    """
    Verifies that generating the data twice with the same seed yields identical results.
    """
    generator1 = DisasterDataGenerator(num_districts=5, num_years=2, seed=42)
    df1 = generator1.generate_panel_data()
    
    generator2 = DisasterDataGenerator(num_districts=5, num_years=2, seed=42)
    df2 = generator2.generate_panel_data()
    
    pd.testing.assert_frame_equal(df1, df2)
