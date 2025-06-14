import os

from finance_ml.constants import *
from finance_ml.utils.common import read_yaml, create_directories
from finance_ml.entity.config_entity import (DataIngestionConfig,
                                             DataValidationConfig)

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
            root_dir=config.root_dir,
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
            root_dir= config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            data= config.data,
            all_schemas= schema  
        )

        return data_Validation_config