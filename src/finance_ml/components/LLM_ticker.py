from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

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
        instructions=[
            f"Based on the information and research done by your team as of {today}, return only the **stock ticker** of the **single best bullish Indian stock** for today. Do NOT include any other text or explanation."
        ],
        show_tools_calls=True,
        markdown=True,
    )

except Exception as e:
    logger.error("Error while initializing agents", exc_info=True)
    raise AgentExecutionError("Failed to initialize one or more agents.") from e


def get_bullish_ticker():
    try:
        logger.info("Running Finsight Agent to get bullish ticker...")
        result = finsight_agent.run()
        ticker = result.content.strip().upper()
        ticker = ticker.replace("*", "")  
        ticker = ticker.replace(" ", "")  
        logger.info(f"Top bullish stock ticker identified: {ticker}")
        return ticker
    except Exception as e:
        logger.error("Agent execution failed", exc_info=True)
        raise AgentExecutionError("Finsight agent failed to run.") from e
