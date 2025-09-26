from src.finance_ml.components.LLM_ticker import get_bullish_ticker, is_valid_ticker
from src.finance_ml.utils.common import normalize_input_to_ticker
from src.finance_ml import logger

class TickerFinderPipeline:
    def main(self, choice=None, ticker=None):
        logger.info("Running TickerFinderPipeline")

        if choice is None:
            choice = input("Select ticker source:\n1. Enter manually\n2. Use LLM agent\nEnter 1 or 2: ").strip()

        if choice == "1":
            if ticker is None:
                while True:
                    user_input = input("Enter company name or ticker symbol (e.g., Reliance or RELIANCE.NS): ").strip()
                    normalized_ticker = normalize_input_to_ticker(user_input)
                    if normalized_ticker and is_valid_ticker(normalized_ticker):
                        ticker = normalized_ticker
                        print(f"Using ticker: {ticker}")
                        break
                    print("Invalid input. Please enter a valid company name or NSE ticker.")
            else:
                # Convert the provided ticker/company name to a valid ticker
                normalized_ticker = normalize_input_to_ticker(ticker)
                if normalized_ticker and is_valid_ticker(normalized_ticker):
                    ticker = normalized_ticker
                    logger.info(f"Normalized input '{ticker}' to valid ticker: {normalized_ticker}")
                else:
                    print(f"Invalid input provided: {ticker}. Please enter a valid company name or NSE ticker.")
                    raise ValueError(f"Invalid input provided: {ticker}")

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
