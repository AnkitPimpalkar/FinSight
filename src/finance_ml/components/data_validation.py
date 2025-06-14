import os
from finance_ml import logger
from finance_ml.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            data = pd.read_csv(self.config.data)
            all_cols = list(data.columns)
            all_schema = self.config.all_schemas

            for col in all_schema:
                if col not in all_cols: 
                    validation_status = False
                    break

         
            for col, expected_type in all_schema.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        validation_status = False
                        break

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status
        except Exception as e:
            raise e