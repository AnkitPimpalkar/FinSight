from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
import re
import time
import yfinance as yf

from datetime import datetime
from src.finance_ml import logger
from src.finance_ml.utils.exceptions import AgentExecutionError
from src.finance_ml.utils.encoding import ensure_utf8_console

ensure_utf8_console()

today = datetime.now().strftime("%B %d, %Y")

try:
    web_agent = Agent(
        name="WebAgent",
        role="Market research expert",
        model=OpenAIChat(id="gpt-4o"),   #Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        tools=[GoogleSearch(), DuckDuckGo()],
        instructions=[
            f"You are a web analyst. Search current news (as of {today}) to identify Indian stocks showing bullish trends today. Focus on companies with positive momentum, news, or sentiment. Return the NSE ticker (ending with .NS) of the most bullish stock you find, and mention it in your answer."
        ],
        show_tools_calls=True,
        markdown=False,
    )

    finance_agent = Agent(
        name="FinanceAnalyst",
        role="Finance analyst expert",
        model=OpenAIChat(id="gpt-4o"),
        tools=[
            YFinanceTools(
                company_news=True,
                technical_indicators=True,
                historical_prices=True,
                analyst_recommendations=True,
                stock_price=True,
                income_statements=True,
                key_financial_ratios=True,
                company_info=True
            )
        ],
        instructions=[
            f"You are a financial analyst. Study financial data for stocks considered bullish today ({today}) and select the single most promising Indian stock for intraday or short-term trading. Return the NSE ticker (ending with .NS) in your answer."
        ],
        show_tools_calls=True,
        markdown=False,
    )

    finsight_agent = Agent(
        team=[web_agent, finance_agent],
        model=OpenAIChat(id="gpt-4o"),
        tools=[YFinanceTools()],
        instructions=[
            f"Based on your team's research as of {today}, provide the NSE ticker (ending with .NS) of the most bullish Indian stock for today. You may include a brief explanation, but make sure the ticker is present in your answer. Do not suggest BSE or delisted stocks."
        ],
        show_tools_calls=True,
        markdown=False,
    )

except Exception as e:
    logger.error("Error while initializing agents", exc_info=True)
    raise AgentExecutionError("Failed to initialize one or more agents.") from e


def is_valid_ticker(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1d")
        return not data.empty
    except Exception:
        return False


def extract_ticker(text):
    """Extract the first valid NSE ticker (ending with .NS) from any text."""
    matches = re.findall(r"\b[A-Z0-9]{2,8}\.NS\b", text.upper())
    blacklist = {"MARKDOWN.NS", "CANNOT.NS", "THE.NS"}
    for ticker in matches:
        if ticker not in blacklist:
            return ticker
    return None


def get_bullish_ticker(max_retries=3, delay=5):
    # List of default tickers to use when LLM fails (for testing/fallback)
    fallback_tickers = [
        "RELIANCE.NS",  # Reliance Industries
        "TCS.NS",       # Tata Consultancy Services
        "INFY.NS",      # Infosys
        "HDFCBANK.NS",  # HDFC Bank
        "ITC.NS"        # ITC Limited
    ]
    
    try:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Running Finsight Agent to get bullish ticker... (Attempt {attempt})")
                result = finsight_agent.run()
                logger.info(f"Raw LLM output: {result.content}")
                ticker = extract_ticker(result.content)
                if ticker and is_valid_ticker(ticker):
                    logger.info(f"Top bullish stock ticker identified: {ticker}")
                    return ticker
                logger.warning(f"No valid ticker found in agent response. Retrying...")
            except Exception as e:
                logger.error(f"Agent execution failed on attempt {attempt}: {e}", exc_info=True)
            if attempt < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
        # If all attempts fail, use a fallback ticker
        import random
        fallback_ticker = random.choice(fallback_tickers)
        logger.warning(f"LLM selection failed, using fallback ticker: {fallback_ticker}")
        return fallback_ticker

    except Exception as e:
        logger.error("Critical error in get_bullish_ticker", exc_info=True)
        # Return a safe default ticker if everything fails
        return "RELIANCE.NS"


