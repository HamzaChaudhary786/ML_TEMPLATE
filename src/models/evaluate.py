import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from src.utils.logger import get_logger
from src.utils.helpers import get_project_root

logger = get_logger(__name__)


def evaluate_classifier(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_report_path: str = "reports/metrics.json",
) -> Dict[str, Any]:
    """Evaluates classification performance and saves metrics to JSON."""
    y_pred = pipeline.predict(X_test)

    # Check for probability predictions if supported
    roc_auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(round(roc_auc_score(y_test, y_proba), 4))
        except Exception:
            roc_auc = None

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
        "roc_auc": roc_auc,
    }

    logger.info("Evaluation Results: %s", metrics)

    # Save metrics JSON
    root = get_project_root()
    report_file = root / output_report_path
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved metrics to %s", report_file)

    return metrics
