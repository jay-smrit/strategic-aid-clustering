from dataclasses import dataclass
from typing import List

from src.constants import ORIGINAL_FEATURES, ENGINEERED_FEATURES, MODEL_FEATURES

# DATASET SCHEMA
COUNTRY_COLUMN = "country"
TARGET_COLUMN = "cluster"
PRIORITY_COLUMN = "priority"

# PCA COLUMNS
PCA_COLUMNS = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]

# PREDICTION INPUT SCHEMA
PREDICTION_INPUT_SCHEMA = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "income",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp"
]

# REQUIRED RAW DATASET COLUMNS
REQUIRED_COLUMNS = [COUNTRY_COLUMN, *ORIGINAL_FEATURES]

#TRAINING DATASET COLUMNS
TRAINING_COLUMNS = [COUNTRY_COLUMN, *MODEL_FEATURES]

# PREDICTION OUTPUT COLUMNS
PREDICTION_OUTPUT_COLUMNS = [COUNTRY_COLUMN, TARGET_COLUMN, PRIORITY_COLUMN]

# NUMERICAL COLUMNS
NUMERIC_COLUMNS = MODEL_FEATURES

# DATACLASS : DATASET SCHEMA
@dataclass(frozen=True)
class DatasetSchema:
    country_column: str = COUNTRY_COLUMN
    target_column: str = TARGET_COLUMN
    priority_column: str = PRIORITY_COLUMN
    raw_features: List[str] = None
    engineered_features: List[str] = None
    model_features: List[str] = None
    pca_columns: List[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'raw_features', ORIGINAL_FEATURES)
        object.__setattr__(self, 'engineered_features', ENGINEERED_FEATURES)
        object.__setattr__(self, 'model_features', MODEL_FEATURES)
        object.__setattr__(self, 'pca_columns', PCA_COLUMNS)

# GLOBAL SCHEMA INSTANCE
SCHEMA = DatasetSchema()