from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import re
import time
import yfinance as yf

from datetime import datetime
from src.finance_ml import logger
from src.finance_ml.utils.exceptions import AgentExecutionError

today = datetime.now().strftime("%B %d, %Y")

try:
    web_agent = Agent(
        name="WebAgent",
        role="Market research expert",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        tools=[DuckDuckGo()],
        instructions=[
            f"You are a web analyst. Search current news (as of {today}) to identify Indian stocks showing bullish trends today. Focus on companies with positive momentum, news, or sentiment."
        ],
        show_tools_calls=True,
        markdown=True,
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
            f"You are a financial analyst. Study financial data for stocks considered bullish today ({today}) and select the single most promising Indian stock for intraday or short-term trading."
        ],
        show_tools_calls=True,
        markdown=True,
    )

    finsight_agent = Agent(
        team=[web_agent, finance_agent],
        tools= [YFinanceTools()],
        instructions=[
            f"Based on the information and research done by your team as of {today}, return only NSE tickers that are available on Yahoo Finance (ending with .NS). Do not suggest BSE or delisted stocks. Do NOT include any other text, explanation, or formatting. Only output the ticker symbol, nothing else."
        ],
        show_tools_calls=True,
        markdown=True,
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

def get_bullish_ticker(max_retries=3, delay=5):
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Running Finsight Agent to get bullish ticker... (Attempt {attempt})")
            result = finsight_agent.run()
            matches = re.findall(r'\b[A-Z]{2,8}(?:\.NS)?\b', result.content.upper())
            if not matches:
                raise ValueError("No valid ticker found in agent response.")
            ticker = matches[0].replace("*", "").replace(" ", "")
            if not ticker.endswith(".NS"):
                ticker = ticker + ".NS"
            # Validate ticker
            if is_valid_ticker(ticker):
                logger.info(f"Top bullish stock ticker identified: {ticker}")
                return ticker
            else:
                logger.warning(f"Ticker {ticker} is not valid on yfinance. Retrying...")
                continue
        except Exception as e:
            logger.error(f"Agent execution failed on attempt {attempt}: {e}", exc_info=True)
            if attempt < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise AgentExecutionError("Finsight agent failed to run after retries.") from e
