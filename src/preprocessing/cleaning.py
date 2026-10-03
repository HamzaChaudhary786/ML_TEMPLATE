from typing import List, Optional
import pandas as pd
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to lowercase snake_case without special characters."""
    df_clean = df.copy()

    def _to_snake(name: str) -> str:
        s = re.sub(r"[^\w\s]", "", str(name)).strip()
        s = re.sub(r"\s+", "_", s)
        return s.lower()

    df_clean.columns = [_to_snake(c) for c in df_clean.columns]
    logger.info("Column names standardized to snake_case.")
    return df_clean


def drop_duplicates_and_unneeded(
    df: pd.DataFrame, drop_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Removes duplicate rows and drops non-predictive columns (e.g. IDs)."""
    df_clean = df.copy()

    # Drop duplicate rows
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_len - len(df_clean)
    if dropped_dupes > 0:
        logger.info("Dropped %d duplicate rows.", dropped_dupes)

    # Drop unnecessary columns
    if drop_cols:
        existing_cols = [c for c in drop_cols if c in df_clean.columns]
        if existing_cols:
            df_clean = df_clean.drop(columns=existing_cols)
            logger.info("Dropped unneeded columns: %s", existing_cols)

    return df_clean
