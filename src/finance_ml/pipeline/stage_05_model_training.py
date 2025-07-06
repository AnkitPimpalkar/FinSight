from src.finance_ml.config.configuration import ConfigurationManager
from src.finance_ml.components.model_training import ModelTraining
from src.finance_ml import logger
import pandas as pd
import os

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Get configuration and parameters
            config_manager = ConfigurationManager()
            model_training_config, model_training_params = config_manager.get_model_training_config()

            model_training = ModelTraining(config=model_training_config, params=model_training_params)
            model_training.train_model()

            logger.info("Model training completed successfully.")

        except Exception as e:
            logger.exception(e)
            raise e

# Example of how to run this stage
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
