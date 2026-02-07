import os
from datetime import datetime
from pathlib import Path

from crewai import Crew, Agent, Task, Process
from crewai.hooks import before_tool_call, ToolCallHookContext, LLMCallHookContext, after_llm_call
from crewai.project import CrewBase, agent, task, crew, tool
from crewai.rag.embeddings.providers.ollama import OllamaProvider

from config.settings import AGENTS_CONFIG_PATH, TASKS_CONFIG_PATH
from guardrails.guardrails_functions import execution_error_guardrail
from tools import EmotionDetectionTool, ActivityDetectionTool
from utils import clean_detection_tools_input
from utils.helper_functions import clean_llm_response

project_root = Path(__file__).parent.parent
storage_dir = project_root / "crewai_storage"

os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

@CrewBase
class VideoAnalysisSummaryCrew:
    agents_config = AGENTS_CONFIG_PATH
    tasks_config = TASKS_CONFIG_PATH
    agents: list[Agent]
    tasks: list[Task]

    def __init__(self,llm_config=None):
        self.llm_config = llm_config
        self.llm = llm_config.get_llm()
        self.report_llm = llm_config.get_llm(model_name="gemma3n:latest")
        self.translate_gemma = llm_config.get_llm(model_name="translategemma:latest")
        self.embedding_provider = OllamaProvider(model_name="qwen3-embedding:8b")



    @tool
    def emotion_detection(self):
        return EmotionDetectionTool(result_as_answer=True)

    @tool
    def activity_detection(self):
        return ActivityDetectionTool(result_as_answer=True)

    @agent
    def emotions_detector(self) -> Agent:
        return Agent(
            config=self.agents_config['emotions_detector'],
            llm=self.llm,
            allow_delegation=False,
            max_iter=1
        )

    @agent
    def activity_detector(self) -> Agent:
        return Agent(
            config=self.agents_config['activity_detector'],
            llm=self.llm,
            allow_delegation=False,
            max_iter=1
        )

    @agent
    def emotions_report_writer(self):
        return Agent(
            config=self.agents_config['emotions_report_writer'],
            llm=self.report_llm,
            reasoning=True,
            max_reasoning_attempts=3,
        )


    @agent
    def activities_report_writer(self):
        # json_knowledge_source = JSONKnowledgeSource(file_paths=project_root / self.tasks_config['detect_activities']['output_file'])
        return Agent(
            config=self.agents_config['activities_report_writer'],
            llm=self.report_llm,
            reasoning=True,
            max_reasoning_attempts=3,
            # knowledge_sources=[json_knowledge_source]
        )

    @agent
    def translator(self):
        return Agent(
            config=self.agents_config['translator'],
            llm=self.translate_gemma,
            # reasoning=True,
            # max_reasoning_attempts=3,
        )

    @task
    def detect_emotions(self) -> Task:
        task_config = self.tasks_config['detect_emotions']
        return Task(
            config=task_config,
            async_execution=True,
            guardrail=execution_error_guardrail
        )

    @task
    def detect_activities(self) -> Task:
        task_config = self.tasks_config['detect_activities']
        return Task(
            config=task_config,
            async_execution=True,
            guardrail=execution_error_guardrail
        )

    @task
    def generate_emotions_report(self):
        task_config = self.tasks_config['generate_emotions_report'].copy()
        task_config['output_file'] = task_config['output_file'].format(date_time=datetime.now())
        return Task(
            config=task_config,
        )

    @task
    def generate_activities_report(self):
        task_config = self.tasks_config['generate_activities_report'].copy()
        task_config['output_file'] = task_config['output_file'].format(date_time=datetime.now())
        return Task(
            config=task_config,
        )

    @task
    def translate_report(self, language: str = "pt-br"):
        task_config = self.tasks_config['translate_report'].copy()
        task_config['description'] = task_config['description'].format(language=language)
        task_config['expected_output'] = task_config['expected_output'].format(language=language)
        task_config['output_file'] = task_config['output_file'].format(language=language, date_time=datetime.now())
        return Task(
            config=task_config,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=True,
            embedder=self.embedding_provider,
        )


    @after_llm_call
    def clean_reasoning_text(self, llm_context: LLMCallHookContext):
        if llm_context:
            clean_llm_response(llm_context)
            print(f"Clean LLM response: {llm_context.response[:50]}")

    @before_tool_call
    def validate_tool_input(self, context: ToolCallHookContext):
        if context:
            clean_detection_tools_input(context)

