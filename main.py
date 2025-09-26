from src.finance_ml import logger
from finance_ml.pipeline.stage_01_LLMticker import TickerFinderPipeline
from finance_ml.pipeline.stage_02_data_ingestion import DataIngestionTrainingPipeline
from finance_ml.pipeline.stage_03_data_validation import DataValidationTrainingPipeline
from finance_ml.pipeline.stage_04_data_transformation import DataTransformationTrainingPipeline
from finance_ml.pipeline.stage_05_model_training import ModelTrainingPipeline
from finance_ml.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from finance_ml.pipeline.stage_07_model_prediction import ModelPredictionPipeline
import argparse

def run_pipeline():
    parser = argparse.ArgumentParser(description='Run FinSight ML Pipeline')
    parser.add_argument('--choice', type=str, help='Choice of operation: 1 for manual, 2 for LLM')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol for manual choice')
    args = parser.parse_args()

    choice = args.choice
    ticker_symbol = args.ticker

    # If no args, fallback to input
    if choice is None:
        choice = input("Select ticker source:\n1. Enter manually\n2. Use LLM agent\nEnter 1 or 2: ").strip()



    STAGE_NAME = "Ticker Finder Stage"
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        ticker_pipeline = TickerFinderPipeline()
        if choice == '1':
            if ticker_symbol is None:
                ticker = ticker_pipeline.main(choice=choice)
            else:
                ticker = ticker_pipeline.main(choice=choice, ticker=ticker_symbol)
        else:
            ticker = ticker_pipeline.main(choice=choice)
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f"\n\n>>>>> {STAGE_NAME} started <<<<<")
        ingestion = DataIngestionTrainingPipeline(ticker=ticker)
        ingestion.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Data Validation stage"
    try:
        logger.info(f">>>>>>  {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>  {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f">>>>>>  {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>  {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Model Training stage"
    try:
        logger.info(f">>>>>>  {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>  {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME= "Model Evaluation stage"
    try:
        logger.info(f'>>>>>>>>>  {STAGE_NAME} started <<<<<<<<<')
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>  {STAGE_NAME} Completed <<<<<<<<<\n\nx==========x')
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Final Prediction stage"
    try:
        logger.info(f">>>>>>  {STAGE_NAME} started <<<<<<")
        obj = ModelPredictionPipeline()
        obj.main()
        logger.info(f">>>>>>  {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == '__main__':
    run_pipeline()