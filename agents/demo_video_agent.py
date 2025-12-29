from crewai import Agent
from config.llm_config import llm_config
from config.settings import AGENT_CONFIG

def create_demo_video_agent():
    config = AGENT_CONFIG["demo_video_script"]

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

