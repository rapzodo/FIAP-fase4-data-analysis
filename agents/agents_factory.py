import yaml
from crewai import Agent

from config.llm_config import llm_config


class AgentsFactory:

    def __init__(self, config_path:str):
        with open(config_path,'r') as file:
            self.config = yaml.safe_load(file)

    def create_agent(self, agent_name:str, tools: dict) -> Agent:
        if agent_name not in self.config:
            raise ValueError(f'Agent name {agent_name} does not exist')

        agent_config = self.config[agent_name]
        tools_config = agent_config['tools']

        agent_tools = []
        for tool_name in tools_config:
            if tool_name in tools:
                agent_tools.append(tools[tool_name])
            else:
                print(f"⚠️  Warning: Tool '{tool_name}' not found for agent '{agent_name}'")

        agent = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            tools=agent_tools,
            backstory=agent_config['backstory'],
            verbose=agent_config['verbose'],
            llm=llm_config.get_llm(),
            allow_delegation=agent_config['allow_delegation'],

        )
        return agent

    def get_all_agents(self):
        return list(self.config.keys())
