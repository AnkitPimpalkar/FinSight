from src.finance_ml.components.LLM_ticker import get_bullish_ticker
from src.finance_ml import logger

class TickerFinderPipeline:
    def main(self):
        logger.info("Running TickerFinderPipeline")
        ticker = get_bullish_ticker()
        logger.info(f"TickerFinderPipeline finished with ticker: {ticker}")
        print("Final Ticker selected:", ticker)
        print("Ticker Type:", type(ticker))

        return ticker


