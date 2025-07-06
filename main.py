from src.finance_ml import logger
from finance_ml.pipeline.stage_01_LLMticker import TickerFinderPipeline
from finance_ml.pipeline.stage_02_data_ingestion import DataIngestionTrainingPipeline
from finance_ml.pipeline.stage_03_data_validation import DataValidationTrainingPipeline
from finance_ml.pipeline.stage_04_data_transformation import DataTransformationTrainingPipeline
from finance_ml.pipeline.stage_05_model_training import ModelTrainingPipeline

STAGE_NAME = "Ticker Finder Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    ticker_pipeline = TickerFinderPipeline()
    ticker = ticker_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<")
    ingestion = DataIngestionTrainingPipeline(ticker=ticker)
    ingestion.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e