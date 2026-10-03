import pytest
import pandas as pd
from src.data.validation import validate_schema, validate_no_empty_df, validate_target_present
from src.preprocessing.cleaning import clean_column_names, drop_duplicates_and_unneeded


@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame({
        "Customer ID": [1, 2, 2, 3],
        "Age": [25, 30, 30, 45],
        "Income": [50000.0, 60000.0, 60000.0, 80000.0],
        "target": [0, 1, 1, 0],
    })


def test_clean_column_names(sample_raw_dataframe):
    df_clean = clean_column_names(sample_raw_dataframe)
    assert "customer_id" in df_clean.columns
    assert "age" in df_clean.columns
    assert "income" in df_clean.columns


def test_drop_duplicates(sample_raw_dataframe):
    df_clean = drop_duplicates_and_unneeded(sample_raw_dataframe, drop_cols=["Customer ID"])
    assert len(df_clean) == 3
    assert "Customer ID" not in df_clean.columns


def test_validation_schema(sample_raw_dataframe):
    assert validate_schema(sample_raw_dataframe, ["Age", "Income"]) is True
    with pytest.raises(ValueError):
        validate_schema(sample_raw_dataframe, ["non_existent_column"])


def test_target_validation(sample_raw_dataframe):
    assert validate_target_present(sample_raw_dataframe, "target") is True
    with pytest.raises(ValueError):
        validate_target_present(sample_raw_dataframe, "missing_target")
