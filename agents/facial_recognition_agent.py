from crewai import Agent
from tools.facial_recognition_tool import FacialRecognitionTool
from config.llm_config import llm_config
from config.settings import AGENT_CONFIG

def create_facial_recognition_agent():
    config = AGENT_CONFIG["facial_recognition"]

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[FacialRecognitionTool()],
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )

    return agent

