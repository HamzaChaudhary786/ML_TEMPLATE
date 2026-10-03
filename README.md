# Professional Machine Learning Project Template

A modular, reproducible, and beginner-friendly template for end-to-end Machine Learning projects.

## Project Structure Overview

```text
├── configs/             <- Central configuration files (config.yaml)
├── data/
│   ├── raw/             <- Original, immutable raw data dumps
│   ├── processed/       <- Cleaned, transformed data ready for modeling
│   └── external/        <- Third-party reference data
├── notebooks/           <- Numbered Jupyter notebooks for exploration & EDA
├── src/                 <- Reusable source code modules
│   ├── data/            <- Ingestion, auditing, and validation
│   ├── preprocessing/   <- Data cleaning and sklearn pipelines
│   ├── features/        <- Feature engineering and selection
│   ├── models/          <- Training, evaluation, and inference
│   └── utils/           <- Logging and helper utilities
├── models/              <- Serialized model pipelines (.joblib)
├── artifacts/           <- Scalers, encoders, and feature column lists
├── reports/             <- Generated metrics and presentation plots
├── scripts/             <- Standalone execution scripts (run_pipeline.py)
├── tests/               <- Unit tests with pytest
├── docs/                <- Data dictionary and workflow guides
├── requirements.txt     <- Core production dependencies
├── requirements-dev.txt <- Development and testing dependencies
└── .gitignore           <- Ignores models, big data, checkpoints, venvs
```

## Quickstart

### 1. Environment Setup
```bash
# Using venv:
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies and editable package:
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Jupyter Kernel
```bash
python -m ipykernel install --user --name=ml_project --display-name="Python (ML Project)"
```

### 3. Run Pipeline
```bash
python scripts/run_pipeline.py
```

### 4. Run Unit Tests
```bash
pytest
```
