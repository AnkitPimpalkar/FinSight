from src.finance_ml.components.LLM_ticker import get_bullish_ticker, is_valid_ticker
from src.finance_ml import logger

class TickerFinderPipeline:
    def main(self, choice=None, ticker=None):
        logger.info("Running TickerFinderPipeline")

        if choice is None:
            choice = input("Select ticker source:\n1. Enter manually\n2. Use LLM agent\nEnter 1 or 2: ").strip()

        if choice == "1":
            if ticker is None:
                while True:
                    ticker = input("Enter the ticker symbol (e.g., ITC.NS): ").strip().upper()
                    if is_valid_ticker(ticker):
                        break
                    print("Invalid ticker. Please enter a valid NSE ticker (e.g., ITC.NS).")
            else:
                if not is_valid_ticker(ticker):
                    print(f"Invalid ticker provided: {ticker}. Please enter a valid NSE ticker.")
                    # Decide how to handle this - maybe raise an error or ask again
                    raise ValueError(f"Invalid ticker provided: {ticker}")

        elif choice == "2":
            ticker = get_bullish_ticker()
            if ticker:
                print(f"LLM Selected Ticker: {ticker}")
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return None

        if ticker is None:
            logger.error("No valid ticker found. Aborting pipeline.")
            raise ValueError("No valid ticker found from Ticker Finder stage.")
        
        logger.info(f"TickerFinderPipeline finished with ticker: {ticker}")
        return ticker
