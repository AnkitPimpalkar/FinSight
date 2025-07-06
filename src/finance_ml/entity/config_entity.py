from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_data_file: Path
    period: str
    interval: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data: Path
    all_schemas: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    raw_data_file: Path

    
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_name: str

@dataclass(frozen=True)
class ModelTrainingParams: # Added a data class for model parameters
    epochs: int
    batch_size: int
    lstm_units_1: int
    lstm_units_2: int
    dense_units_1: int
    dropout_rate: float    