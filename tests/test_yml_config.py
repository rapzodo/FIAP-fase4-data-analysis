import unittest

import yaml

from agents.agents_factory import AgentsFactory
from config.settings import AGENTS_CONFIG_PATH, TASKS_CONFIG_PATH
from tasks.task_factory import TaskFactory


class TestYamlConfig(unittest.TestCase):

    def test_agent_config(self):
        factory = AgentsFactory(AGENTS_CONFIG_PATH)
        agents = factory.get_all_agents()
        with open(AGENTS_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        self.assertEqual(agents[0], list(config.keys())[0])
        self.assertEqual(agents[1], list(config.keys())[1])
        self.assertEqual(agents[2], list(config.keys())[2])

    def test_tasks_config(self):
        factory = TaskFactory(TASKS_CONFIG_PATH)
        tasks = factory.get_all_tasks()
        with open(TASKS_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        print(config.keys())
        self.assertEqual(tasks[0], list(config.keys())[0])
        self.assertEqual(tasks[1], list(config.keys())[1])
        self.assertEqual(tasks[2], list(config.keys())[2])

    def test_agent_structure(self):
        factory = AgentsFactory(AGENTS_CONFIG_PATH)
        for agent in factory.config.keys():
            self.assertIsNotNone(factory.config[agent]['role'])
            self.assertIsNotNone(factory.config[agent]['goal'])
            self.assertIsNotNone(factory.config[agent]['backstory'])

    def test_task_structure(self):
        factory = TaskFactory(TASKS_CONFIG_PATH)
        for agent in factory.config.keys():
            self.assertIsNotNone(factory.config[agent]['description'])
            self.assertIsNotNone(factory.config[agent]['agent'])
            self.assertIsNotNone(factory.config[agent]['expected_output'])

    def test_task_agent_name(self):
        factory = TaskFactory(TASKS_CONFIG_PATH)
        agent = factory.get_task_agent_name("analyze_facial_recognition")
        self.assertEqual(agent, "facial_emotions_analyzer")