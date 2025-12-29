from crewai import Agent
from config.llm_config import llm_config
from config.settings import AGENT_CONFIG

def create_summarizer_agent():
    config = AGENT_CONFIG["summarizer"]

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[],
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )

    return agent

