from typing import Any, Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger
from src.utils.helpers import save_artifact

logger = get_logger(__name__)


def get_model_instance(model_type: str, params: Dict[str, Any]) -> Any:
    """Instantiates a scikit-learn model based on type string and parameters."""
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    model_config: Dict[str, Any],
    save_path: str = "models/model_pipeline.joblib",
) -> Pipeline:
    """Combines preprocessor and estimator into a single sklearn Pipeline, trains it, and saves it.

    Args:
        X_train: Training features DataFrame.
        y_train: Training labels Series.
        preprocessor: ColumnTransformer or transformer object.
        model_config: Dictionary containing 'type' and 'params'.
        save_path: Where to save the final pipeline.

    Returns:
        Pipeline: Fitted full machine learning pipeline.
    """
    model_type = model_config.get("type", "random_forest")
    model_params = model_config.get("params", {})

    logger.info("Initializing model: %s with params %s", model_type, model_params)
    estimator = get_model_instance(model_type, model_params)

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    logger.info("Fitting complete pipeline on training data (%d samples)...", len(X_train))
    full_pipeline.fit(X_train, y_train)
    logger.info("Pipeline training complete.")

    save_artifact(full_pipeline, save_path)
    logger.info("Saved trained pipeline artifact to %s", save_path)

    return full_pipeline
