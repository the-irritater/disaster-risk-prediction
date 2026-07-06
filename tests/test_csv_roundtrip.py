import os
import pandas as pd
import numpy as np
import pytest
from src.data_generation import DisasterDataGenerator
from src.feature_engineering import construct_risk_index, engineer_features, calculate_rates

def test_disaster_type_survives_csv_roundtrip(tmp_path):
    """
    Verifies that the disaster type column does not introduce NaN/missing values 
    when written to CSV and read back using pandas.
    """
    generator = DisasterDataGenerator(seed=42)
    df = generator.generate_panel_data()

    path = tmp_path / "disaster_data.csv"
    df.to_csv(path, index=False)
    loaded = pd.read_csv(path)

    assert loaded["Disaster_Type"].notna().all(), "Disaster_Type contains missing values after roundtrip"
    assert "No Disaster" in loaded["Disaster_Type"].unique(), "No Disaster label missing after roundtrip"

def test_unique_district_month_keys():
    """
    Checks that every row has a unique district-month panel key combination.
    """
    generator = DisasterDataGenerator(num_districts=20, num_years=3, seed=42)
    df = generator.generate_panel_data()
    keys = df["District"] + "_" + df["Year"].astype(str) + "_" + df["Month"].astype(str)
    assert not keys.duplicated().any(), "Duplicate district-month keys detected"

def test_unique_dimension_labels():
    """
    Verifies that the generated DimDisasterType has unique keys and no empty entries.
    """
    generator = DisasterDataGenerator(num_districts=20, num_years=3, seed=42)
    df = generator.generate_panel_data()
    
    dt_vals = [t for t in df["Disaster_Type"].unique() if t != "No Disaster"]
    dt_vals.insert(0, "No Disaster")
    dim_dt = pd.DataFrame({"DisasterTypeKey": range(1, len(dt_vals) + 1), "Disaster_Type": dt_vals})

    assert not dim_dt["DisasterTypeKey"].duplicated().any()
    assert not dim_dt["Disaster_Type"].duplicated().any()
    assert dim_dt["Disaster_Type"].notna().all()
    assert (dim_dt["Disaster_Type"] != "").all()

def test_correct_event_prevalence():
    """
    Verifies that the simulated data-generating process maintains the expected 
    disaster occurrence prevalence (~15.4%).
    """
    generator = DisasterDataGenerator(num_districts=100, num_years=11, seed=42)
    df = generator.generate_panel_data()
    prevalence = df["Disaster_Occurred"].mean()
    assert 0.14 <= prevalence <= 0.17, f"Unexpected disaster prevalence: {prevalence:.4f}"

def test_risk_category_quartile_method():
    """
    Verifies that the risk categories are mapped correctly using saved quantile boundaries 
    to prevent target leakage.
    """
    generator = DisasterDataGenerator(num_districts=20, num_years=3, seed=42)
    df = generator.generate_panel_data()
    df = calculate_rates(df)
    df = engineer_features(df)
    
    # 1. Fit on the first half
    half_idx = len(df) // 2
    df_fit, scalers = construct_risk_index(df.iloc[:half_idx])
    
    # 2. Transform the second half using fit scalers
    df_transform = construct_risk_index(df.iloc[half_idx:], fit_scalers=scalers)
    
    assert "risk_thresholds" in scalers
    assert "Risk_Category" in df_transform.columns
    assert set(df_transform["Risk_Category"].unique()).issubset({"Low", "Moderate", "High", "Critical"})
