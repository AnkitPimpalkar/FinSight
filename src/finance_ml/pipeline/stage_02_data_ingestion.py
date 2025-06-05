from src.finance_ml.config.configuration import ConfigurationManager
from src.finance_ml.components.data_ingestion import DataIngestion
from src.finance_ml import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def main(self):
        config = ConfigurationManager()
        data_config = config.get_data_ingestion_config()

        data_ingestor = DataIngestion(config=data_config, ticker=self.ticker)
        data_ingestor.download_data()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e