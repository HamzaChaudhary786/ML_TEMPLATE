from typing import List, Optional
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Verifies that all required columns exist in the DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"Data validation failed! Missing required columns: {missing}"
        logger.error(msg)
        raise ValueError(msg)
    logger.info("Schema validation passed. All required columns present.")
    return True


def validate_no_empty_df(df: pd.DataFrame) -> bool:
    """Checks if the dataframe is not empty."""
    if df.empty or len(df) == 0:
        logger.error("Dataframe is empty!")
        raise ValueError("Dataframe has 0 rows.")
    return True


def validate_target_present(df: pd.DataFrame, target_col: str) -> bool:
    """Verifies target column exists and does not contain all null values."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    if df[target_col].isnull().all():
        raise ValueError(f"Target column '{target_col}' is completely empty.")
    logger.info("Target column '%s' successfully validated.", target_col)
    return True
