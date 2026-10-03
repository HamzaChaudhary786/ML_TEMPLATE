from pathlib import Path
from typing import Optional
import pandas as pd
from src.utils.logger import get_logger
from src.utils.helpers import get_project_root

logger = get_logger(__name__)


def load_dataset(file_path: str, **kwargs) -> pd.DataFrame:
    """Loads a dataset from CSV, Excel, Parquet, or SQL into a pandas DataFrame.

    Args:
        file_path: Path to dataset relative to project root.
        **kwargs: Additional parameters passed to pandas reader.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    root = get_project_root()
    path = root / file_path

    if not path.exists():
        logger.error("Dataset not found at %s", path)
        raise FileNotFoundError(f"Dataset not found at {path}")

    logger.info("Loading dataset from %s", path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info("Loaded dataframe with shape: %s", df.shape)
    return df


def save_processed_data(df: pd.DataFrame, output_path: str, format: str = "parquet") -> None:
    """Saves processed dataframe to disk."""
    root = get_project_root()
    path = root / output_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported save format: {format}")

    logger.info("Saved data to %s (shape: %s)", path, df.shape)
