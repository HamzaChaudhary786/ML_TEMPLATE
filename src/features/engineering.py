import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extracts year, month, day, dayofweek from a datetime column."""
    df_feat = df.copy()
    if date_col in df_feat.columns:
        dt_series = pd.to_datetime(df_feat[date_col], errors="coerce")
        df_feat[f"{date_col}_year"] = dt_series.dt.year
        df_feat[f"{date_col}_month"] = dt_series.dt.month
        df_feat[f"{date_col}_day"] = dt_series.dt.day
        df_feat[f"{date_col}_dayofweek"] = dt_series.dt.dayofweek
        logger.info("Generated date features for %s", date_col)
    return df_feat


def create_ratio_features(
    df: pd.DataFrame, num_col: str, denom_col: str, new_col_name: str
) -> pd.DataFrame:
    """Creates a ratio feature safely handling division by zero."""
    df_feat = df.copy()
    if num_col in df_feat.columns and denom_col in df_feat.columns:
        df_feat[new_col_name] = df_feat[num_col] / (df_feat[denom_col] + 1e-6)
        logger.info("Created ratio feature '%s' (%s / %s)", new_col_name, num_col, denom_col)
    return df_feat
