"""End-to-end Machine Learning Pipeline Runner.

Executes the complete workflow:
Audit -> Cleaning -> Validation -> Split -> Pipeline Preprocessing -> Training -> Evaluation.
"""

from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.helpers import load_config
from src.data.ingestion import load_dataset
from src.data.audit import overall_data_audit, column_by_column_audit
from src.data.validation import validate_schema, validate_target_present
from src.preprocessing.cleaning import clean_column_names, drop_duplicates_and_unneeded
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.models.train import train_model_pipeline
from src.models.evaluate import evaluate_classifier

logger = get_logger("run_pipeline")


def main():
    logger.info("================ Starting ML Pipeline ================")
    config = load_config("configs/config.yaml")

    # 1. Ingestion
    raw_data_path = config["paths"]["raw_data"]
    try:
        df = load_dataset(raw_data_path)
    except FileNotFoundError:
        logger.warning("No raw dataset found at %s. Please place your dataset there.", raw_data_path)
        return

    # 2. Auditing
    audit_summary = overall_data_audit(df)
    col_audit = column_by_column_audit(df)

    # 3. Cleaning
    df_clean = clean_column_names(df)
    drop_cols = config["data"].get("drop_columns", [])
    df_clean = drop_duplicates_and_unneeded(df_clean, drop_cols=drop_cols)

    # 4. Validation
    target_col = config["data"]["target_column"]
    validate_target_present(df_clean, target_col)

    # 5. Train / Test Split
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if config["data"].get("stratify") else None
    )
    logger.info("Train split: %s rows, Test split: %s rows", len(X_train), len(X_test))

    # 6. Preprocessing Pipeline Construction
    num_cols = config["features"].get("numerical_cols", [])
    cat_cols = config["features"].get("categorical_cols", [])
    preprocessor = build_preprocessing_pipeline(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        scaling_strategy=config["features"].get("scaling_strategy", "standard"),
        imputation_strategy_num=config["features"].get("imputation_strategy_num", "median"),
        imputation_strategy_cat=config["features"].get("imputation_strategy_cat", "most_frequent"),
    )

    # 7. Model Training
    primary_model_config = config["models"]["primary"]
    pipeline = train_model_pipeline(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        model_config=primary_model_config,
        save_path="models/model_pipeline.joblib",
    )

    # 8. Model Evaluation
    metrics = evaluate_classifier(pipeline, X_test, y_test, output_report_path="reports/metrics.json")
    logger.info("Pipeline executed successfully! Metrics: %s", metrics)
    logger.info("======================================================")


if __name__ == "__main__":
    main()
