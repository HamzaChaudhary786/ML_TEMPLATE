from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessing_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scaling_strategy: str = "standard",
    imputation_strategy_num: str = "median",
    imputation_strategy_cat: str = "most_frequent",
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features.

    Avoids data leakage: will be fitted only on training data.
    """
    logger.info("Constructing preprocessing pipeline...")

    # Numerical Transformer
    num_scaler = StandardScaler() if scaling_strategy == "standard" else MinMaxScaler()
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputation_strategy_num)),
            ("scaler", num_scaler),
        ]
    )

    # Categorical Transformer
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputation_strategy_cat)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    logger.info(
        "Built ColumnTransformer with %d numeric and %d categorical features.",
        len(numerical_cols),
        len(categorical_cols),
    )
    return preprocessor
