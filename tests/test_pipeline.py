import numpy as np
import pandas as pd
import pytest

from house_price_pipeline.pipeline import (
    validate_training_data, validate_incoming_data
)


def sample():
    return pd.DataFrame({
        "House_ID": ["H001", "H002"],
        "Area_sqft": [1000, 1200],
        "Bedrooms": [2, 3],
        "Bathrooms": [1, 2],
        "Age_Years": [10.0, None],
        "Distance_to_Center_km": [5.0, 7.0],
        "Neighborhood_Score": [7.5, 8.0],
        "Has_Parking": [1, 0],
        "Property_Condition": ["Good", "Fair"],
        "City_Zone": ["Central", "West"],
        "Price_Lakhs": [80.0, 95.0],
    })


def test_valid_data_and_feature_nulls_pass():
    assert validate_training_data(sample(), "Price_Lakhs", "House_ID")


def test_missing_column_fails():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_training_data(sample().drop(columns="Area_sqft"), "Price_Lakhs", "House_ID")


def test_missing_target_fails():
    df = sample()
    df.loc[0, "Price_Lakhs"] = None
    with pytest.raises(ValueError, match="contains missing values"):
        validate_training_data(df, "Price_Lakhs", "House_ID")


def test_duplicate_id_fails():
    df = sample()
    df.loc[1, "House_ID"] = "H001"
    with pytest.raises(ValueError, match="missing or duplicate"):
        validate_training_data(df, "Price_Lakhs", "House_ID")


def test_invalid_numeric_text_fails():
    df = sample()
    df["Area_sqft"] = df["Area_sqft"].astype(object)
    df.loc[0, "Area_sqft"] = "bad-value"
    with pytest.raises(ValueError, match="non-numeric"):
        validate_training_data(df, "Price_Lakhs", "House_ID")


def test_infinity_fails():
    df = sample()
    df["Area_sqft"] = df["Area_sqft"].astype(float)
    df.loc[0, "Area_sqft"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_training_data(df, "Price_Lakhs", "House_ID")


def test_impossible_score_fails():
    df = sample()
    df.loc[0, "Neighborhood_Score"] = 15
    with pytest.raises(ValueError, match="outside allowed range"):
        validate_training_data(df, "Price_Lakhs", "House_ID")


def test_textual_null_in_category_is_normalized():
    df = sample()
    df.loc[0, "Property_Condition"] = " null "
    from house_price_pipeline.pipeline import validate_schema
    result = validate_schema(df, "Price_Lakhs", "House_ID")
    assert pd.isna(result.loc[0, "Property_Condition"])


def test_incoming_data_requires_batch_but_not_target():
    df = sample().drop(columns="Price_Lakhs")
    df["Batch"] = [1, 1]
    result = validate_incoming_data(df, "Price_Lakhs", "House_ID")
    assert "Batch" in result.columns
    assert "Price_Lakhs" not in result.columns


def test_incoming_missing_batch_fails():
    df = sample().drop(columns="Price_Lakhs")
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_incoming_data(df, "Price_Lakhs", "House_ID")
