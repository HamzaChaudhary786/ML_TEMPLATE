from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.utils.helpers import load_artifact

logger = get_logger(__name__)


def predict(
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: str = "models/model_pipeline.joblib",
) -> np.ndarray:
    """Loads saved pipeline artifact and runs inference on new data.

    Args:
        input_data: pandas DataFrame or dictionary of features.
        model_path: Path to serialized pipeline.

    Returns:
        np.ndarray: Predicted labels.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()

    logger.info("Loading model pipeline from %s", model_path)
    pipeline = load_artifact(model_path)

    logger.info("Executing prediction on %d records...", len(df))
    predictions = pipeline.predict(df)
    return predictions
