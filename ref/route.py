from pydantic import BaseModel
from typing import List, Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team


class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str

class CompanyAnalysis(BaseModel):
    company_name: str
    analysis: str

stock_searcher = Agent(
    name="Stock Searcher",
    model=OpenAIChat("gpt-4o"),
    response_model=StockAnalysis,
    role="Searches for information on stocks and provides price analysis.",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
        )
    ],
)

company_info_agent = Agent(
    name="Company Info Searcher",
    model=OpenAIChat("gpt-4o"),
    role="Searches for information about companies and recent news.",
    response_model=CompanyAnalysis,
    tools=[
        YFinanceTools(
            stock_price=False,
            company_info=True,
            company_news=True,
        )
    ],
)

team = Team(
    name="Stock Research Team",
    mode="route",
    model=OpenAIChat("gpt-4o"),
    members=[stock_searcher, company_info_agent],
    markdown=True,
)

# This should route to the stock_searcher
response = team.run("What is the current stock price of NVDA?")
assert isinstance(response.content, StockAnalysis)