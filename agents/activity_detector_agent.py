from crewai import Agent
from tools.activity_detector_tool import ActivityDetectorTool
from config.llm_config import llm_config
from config.settings import AGENT_CONFIG

def create_activity_detector_agent():
    config = AGENT_CONFIG["activity_detector"]

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[ActivityDetectorTool()],
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )

    return agent

