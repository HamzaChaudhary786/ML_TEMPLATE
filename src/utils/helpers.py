from pathlib import Path
from typing import Any, Dict
import yaml
import joblib


def get_project_root() -> Path:
    """Returns the absolute root directory of the ML project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file safely.

    Args:
        config_path: Path to YAML file relative to project root.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    root = get_project_root()
    full_path = root / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_artifact(obj: Any, file_path: str) -> Path:
    """Serializes and saves a Python object (model, scaler, pipeline) via joblib."""
    root = get_project_root()
    target_path = root / file_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target_path)
    return target_path


def load_artifact(file_path: str) -> Any:
    """Loads a serialized object from disk."""
    root = get_project_root()
    target_path = root / file_path
    if not target_path.exists():
        raise FileNotFoundError(f"Artifact not found at: {target_path}")
    return joblib.load(target_path)
