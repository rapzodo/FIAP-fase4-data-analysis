from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, task, crew

from agents.agents_factory import AgentsFactory
from config.settings import AGENTS_CONFIG_PATH, TASKS_CONFIG_PATH
from tasks.task_factory import TaskFactory
from tools.facial_detection_tool import FacialDetectionTool

@CrewBase
class VideoAnalysisSummaryCrew:

    @agent
    def facial_emotions_analyzer_agent(self) -> Agent:
        return AgentsFactory(AGENTS_CONFIG_PATH).create_agent("facial_emotions_analyzer",
                                                              {"facial_detection_tool": FacialDetectionTool()})
    @task
    def analyze_facial_detection_task(self) -> Task:
        facial_analyzer = self.facial_emotions_analyzer_agent()
        return TaskFactory(TASKS_CONFIG_PATH).create_task("analyze_facial_detection", facial_analyzer)

    @crew
    def create(self):
        return Crew(
            agents=[
                self.facial_emotions_analyzer_agent(),
            ],
            tasks=[
                self.analyze_facial_detection_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )

