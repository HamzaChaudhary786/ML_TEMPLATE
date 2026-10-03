from typing import List
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def drop_low_variance_features(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Drops numeric features with zero or near-zero variance."""
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var()
    low_var_cols = variances[variances <= threshold].index.tolist()
    if low_var_cols:
        logger.info("Dropping low variance features: %s", low_var_cols)
        return df.drop(columns=low_var_cols)
    return df


def drop_collinear_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drops features that have a pairwise Pearson correlation above the threshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        logger.info("Dropping collinear features (> %s): %s", threshold, to_drop)
        return df.drop(columns=to_drop)
    return df
