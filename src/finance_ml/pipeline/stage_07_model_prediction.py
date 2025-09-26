from src.finance_ml.config.configuration import ConfigurationManager
from src.finance_ml.components.prediction import ModelPrediction
from src.finance_ml import logger

STAGE_NAME = "Model Prediction stage"

class ModelPredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
            # Get configuration
            config_manager = ConfigurationManager()
            model_prediction_config = config_manager.get_model_prediction_config()

            # Perform model prediction
            model_prediction = ModelPrediction(config=model_prediction_config)
            result = model_prediction.predict()
            return result

        except ValueError as e:
            if "cannot reshape array" in str(e):
                logger.error("Error: Insufficient data for prediction with current lookback window size")
                logger.error(f"Original error: {str(e)}")
                raise ValueError("Insufficient historical data for the selected ticker. Please choose a ticker with more data points.")
            else:
                logger.exception(e)
                raise e
        except Exception as e:
            logger.exception(e)
            raise e

# Example of how to run this stage
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPredictionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e