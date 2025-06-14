from src.finance_ml import logger
from src.finance_ml.pipeline.stage_01_LLMticker import TickerFinderPipeline
from src.finance_ml.pipeline.stage_02_data_ingestion import DataIngestionTrainingPipeline
from finance_ml.pipeline.stage_03_data_validation import DataValidationTrainingPipeline

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
