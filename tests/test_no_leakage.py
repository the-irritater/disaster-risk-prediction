import pandas as pd
from src.train_models import PRE_EVENT_PREDICTORS, POST_EVENT_VARIABLES, split_chronologically
from src.data_generation import DisasterDataGenerator
from src.feature_engineering import calculate_rates, construct_risk_index, engineer_features

def test_feature_list_disjointness():
    """
    Asserts that pre-event predictors and post-event impact variables are completely disjoint
    to guarantee zero lookahead target leakage in machine learning.
    """
    predictors_set = set(PRE_EVENT_PREDICTORS)
    post_event_set = set(POST_EVENT_VARIABLES)
    
    intersection = predictors_set.intersection(post_event_set)
    assert len(intersection) == 0, f"TARGET LEAKAGE DETECTED! Features in both lists: {intersection}"

def test_strict_chronological_splits():
    """
    Verifies that the chronological splits do not contain overlapping dates or temporal leakage.
    """
    # Generate mock panel dataset
    generator = DisasterDataGenerator(num_districts=5, num_years=11, seed=42)
    df = generator.generate_panel_data()
    df = calculate_rates(df)
    df, _ = construct_risk_index(df)
    df = engineer_features(df)
    
    train, val, test = split_chronologically(df, train_end_year=2022, val_end_year=2024)
    
    # Train is <= 2022
    assert (train["Year"] <= 2022).all()
    
    # Val is 2023-2024
    assert ((val["Year"] >= 2023) & (val["Year"] <= 2024)).all()
    
    # Test is 2025
    assert (test["Year"] == 2025).all()
    
    # Assert sample sizes
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
