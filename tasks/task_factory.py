import yaml
from crewai import Task
from crewai.agents.agent_builder.base_agent import BaseAgent

from config.settings import VIDEO_PATH, FRAME_SAMPLE_RATE, POSE_MODEL


class TaskFactory:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def create_task(self, task_name: str, agent: BaseAgent, context_tasks: dict = None) -> Task:
        if task_name not in self.config:
            raise ValueError(f'Task name {task_name} does not exist')

        task_config = self.config[task_name]
        description = task_config['description'].format(
            video_path=VIDEO_PATH,
            pose_model=POSE_MODEL,
            frame_sample_rate=FRAME_SAMPLE_RATE
        )

        context = []

        if context_tasks:
            for context_name in self.config['context_tasks']:
                if context_name in context_tasks:
                    context.append(context_tasks[context_name])

        return Task(
            description=description,
            expected_output=task_config['expected_output'],
            agent=agent,
            max_retries=task_config['max_retries'],
            context=context if context else None,
        )

    def get_all_tasks(self):
        return list(self.config.keys())

    def get_task_agent_name(self, task_name):
        if task_name not in self.config:
            raise ValueError(f'Task name {task_name} does not exist')
        return self.config[task_name]['agent']
