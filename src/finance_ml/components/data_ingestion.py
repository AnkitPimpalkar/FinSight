import yfinance as yf
import pandas as pd
from src.finance_ml import logger
from src.finance_ml.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig, ticker: str):
        self.config = config
        self.ticker = ticker

    def download_data(self):
        logger.info(f"Downloading data for {self.ticker}")
        df = yf.download(
            tickers=self.ticker,
            period=self.config.period,
            interval=self.config.interval
        )
        df.to_csv(self.config.raw_data_file)
        logger.info(f"Data saved to {self.config.raw_data_file}")
