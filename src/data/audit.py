from typing import Dict, Any
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def overall_data_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a high-level summary of the dataset: shape, memory, duplicate rows, missing rate.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dict[str, Any]: Summary metrics.
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    audit = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "total_cells": int(total_cells),
        "total_missing_cells": int(total_missing),
        "missing_percentage": float(round((total_missing / total_cells) * 100, 2)) if total_cells > 0 else 0.0,
        "duplicate_rows": int(duplicate_count),
        "memory_mb": float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
    }
    logger.info("Overall Data Audit: %s rows, %s cols, %s duplicates, %s%% missing",
                audit["num_rows"], audit["num_columns"], audit["duplicate_rows"], audit["missing_percentage"])
    return audit


def column_by_column_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Performs an exhaustive column-level audit for data types, nulls, unique values, and sample values.

    Args:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Table with per-column statistics.
    """
    records = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / len(df)) * 100, 2)
        n_unique = int(series.nunique(dropna=False))
        dtype = str(series.dtype)
        sample_values = list(series.dropna().sample(min(3, len(series.dropna())), random_state=42).values)

        records.append({
            "column": col,
            "dtype": dtype,
            "missing_count": null_count,
            "missing_pct": null_pct,
            "unique_count": n_unique,
            "sample_values": sample_values,
        })

    audit_df = pd.DataFrame(records)
    logger.info("Completed column-by-column audit for %s columns.", len(records))
    return audit_df
