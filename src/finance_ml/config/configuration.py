import os

from finance_ml.constants import *
from finance_ml.utils.common import read_yaml, create_directories
from finance_ml.entity.config_entity import (DataIngestionConfig,
                                             DataValidationConfig,
                                             DataTransformationConfig,
                                             ModelTrainingConfig,
                                             ModelTrainingParams,
                                             ModelEvaluationConfig,
                                             ModelPredictionConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.data_ingestion # Access data_ingestion params from params.yaml

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= Path(config.root_dir),
            raw_data_file=os.path.join(config.root_dir, config.raw_data_file), 
            period=params.period,     
            interval=params.interval  
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_Validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_Validation_config = DataValidationConfig(
            root_dir= Path(config.root_dir),
            STATUS_FILE= config.STATUS_FILE,
            data= Path(config.data),
            all_schemas= schema  
        )

        return data_Validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig: # Add this method
        config = self.config.data_transformation
        params = self.params.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            raw_data_file=Path(config.raw_data_file),
            lookback=params.lookback
        )

        return data_transformation_config
    
    def get_model_training_config(self) -> tuple[ModelTrainingConfig, ModelTrainingParams]: # Modify return type
        config = self.config.model_training
        params = self.params.model_training # Access parameters from params.yaml

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            trained_model_name=config.trained_model_name
        )

        model_training_params = ModelTrainingParams(
            epochs=params.epochs,
            batch_size=params.batch_size,
            lstm_units_1=params.lstm_units_1,
            lstm_units_2=params.lstm_units_2,
            dense_units_1=params.dense_units_1,
            dropout_rate=params.dropout_rate
        )

        return model_training_config, model_training_params # Return both 
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])
    
        evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            metrics_file_name=config.metrics_file_name,
            model_path=config.model_path,
            X_test_path=config.X_test_path,
            y_test_path=config.y_test_path
            )

        return evaluation_config
    
    def get_model_prediction_config(self) -> ModelPredictionConfig: # Add the new method
        config = self.config.model_prediction
        params = self.params.data_transformation

        create_directories([config.root_dir])

        model_prediction_config = ModelPredictionConfig(
            root_dir=config.root_dir,
            trained_model_path=Path(config.trained_model_path), # Ensure Path type
            input_data_path=Path(config.input_data_path), # Ensure Path type
            predictions_file_name=config.predictions_file_name,
            lookback=params.lookback
        )

        return model_prediction_config