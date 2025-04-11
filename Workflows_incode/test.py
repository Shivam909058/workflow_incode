from agno.agent import Agent
from agno.tools.baidusearch import BaiduSearchTools
from agno.models.groq import Groq

agent = Agent(
    tools=[BaiduSearchTools()],
    model=Groq(id="llama3-70b-8192"),
    description="You are a search agent that helps users find the most relevant information using Baidu.",
    instructions=[
        "Given a topic by the user, respond with the 3 most relevant search results about that topic.",
        "Search for 5 results and select the top 3 unique items.",
        "Search in both English and Chinese.",
    ],
    show_tool_calls=True,
)

agent.print_response("What are the latest advancements in AI in 2025 name the company and the product in detail like for example : google 2.5 is most intelligent model etc etc ", markdown=True)