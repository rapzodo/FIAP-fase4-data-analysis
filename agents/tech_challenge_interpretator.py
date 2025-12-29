from crewai import Agent
from tools.pdf_parser_tool import PDFParserTool
from config.llm_config import llm_config
from config.settings import AGENT_CONFIG

def create_tech_challenge_interpretator_agent():
    config = AGENT_CONFIG["tech_challenge_interpretator"]

    agent = Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        tools=[PDFParserTool()],
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )

    return agent

