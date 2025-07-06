from src.finance_ml.components.LLM_ticker import get_bullish_ticker, is_valid_ticker
from src.finance_ml import logger

class TickerFinderPipeline:
    def main(self):
        logger.info("Running TickerFinderPipeline")

     # Manual input setup
        choice = input("Select ticker source:\n1. Enter manually\n2. Use LLM agent\nEnter 1 or 2: ").strip()
        if choice == "1":
             while True:
                ticker = input("Enter the ticker symbol (e.g., ITC.NS): ").strip().upper()
                if is_valid_ticker(ticker):
                    break
                print("Invalid ticker. Please enter a valid NSE ticker (e.g., ITC.NS).")

     # LLM input setup
        else:
            ticker = get_bullish_ticker()
        if ticker is None:
            logger.error("No valid ticker found. Aborting pipeline.")
            raise ValueError("No valid ticker found from Ticker Finder stage.")
        logger.info(f"TickerFinderPipeline finished with ticker: {ticker}")
        print("Final Ticker selected:", ticker)
        print("Ticker Type:", type(ticker))

        return ticker
    