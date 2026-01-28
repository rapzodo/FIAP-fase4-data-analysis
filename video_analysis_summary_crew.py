from crewai import Crew, Agent, Task, Process
from crewai.hooks import before_tool_call, ToolCallHookContext, LLMCallHookContext, after_llm_call
from crewai.project import CrewBase, agent, task, crew, tool

from config.llm_config import llm_config
from config.settings import AGENTS_CONFIG_PATH, TASKS_CONFIG_PATH, VIDEO_PATH, POSE_MODEL, FRAME_SAMPLE_RATE
from guardrails.guardrails_functions import execution_error_guardrail
from tools import EmotionDetectionTool, ActivityDetectionTool


@CrewBase
class VideoAnalysisSummaryCrew:
    agents_config = AGENTS_CONFIG_PATH
    tasks_config = TASKS_CONFIG_PATH
    llm = llm_config.get_llm()
    report_llm = llm_config.get_llm(model_name="gemma3n:latest")
    translate_gemma = llm_config.get_llm(model_name="translategemma:latest")

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
        return Agent(
            config=self.agents_config['activities_report_writer'],
            llm=self.report_llm,
            reasoning=True,
            max_reasoning_attempts=3,
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
        task_config['description'] = (task_config.copy()['description'].format(
            video_path=VIDEO_PATH,
            frame_rate=FRAME_SAMPLE_RATE,
        ))
        return Task(
            config=task_config,
            async_execution=True,
            guardrail=execution_error_guardrail
        )

    @task
    def detect_activities(self) -> Task:
        task_config = self.tasks_config['detect_activities']
        task_config['description'] = (task_config.copy()['description'].format(
            video_path=VIDEO_PATH,
            pose_model=POSE_MODEL,
            frame_rate=FRAME_SAMPLE_RATE
        ))
        return Task(
            config=task_config,
            async_execution=True,
            guardrail=execution_error_guardrail
        )

    @task
    def generate_emotions_report(self):
        return Task(
            config=self.tasks_config['generate_emotions_report'],
        )

    @task
    def generate_activities_report(self):
        return Task(
            config=self.tasks_config['generate_activities_report'],
        )

    @task
    def translate_report(self, language: str = "pt-br"):
        task_config = self.tasks_config['translate_report'].copy()
        task_config['description'] = task_config['description'].format(language=language)
        task_config['expected_output'] = task_config['expected_output'].format(language=language)
        task_config['output_file'] = task_config['output_file'].format(language=language)
        return Task(
            config=task_config,
            # guardrail=
            # f"""The output is translated to {language}. If not, read the context again and translate it to {language}.
            #     The result doesn't need to be a pydantic model
            #     """,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=True,
            # memory=True
        )

    @staticmethod
    @after_llm_call
    def clean_reasoning_text(llm_context: LLMCallHookContext) -> str:
        if not llm_context:
            return str(llm_context)

        print(f"cleaning the LLM response: {llm_context.response[:100]} ...")
        reasoning_markers = [
            "Reasoning Plan:",
            "Strategic Plan:",
            "Analysis Plan:",
            "Plan:",
            "Final Answer:"
        ]
        response = llm_context.response
        for marker in reasoning_markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    clear_response = parts[1].strip()
                    print(f"clear response: {clear_response[:50]} ...")
                    return clear_response

        return response

    @before_tool_call
    def validate_tool_input(self, context: ToolCallHookContext):
        print(f"CALLING PRE HOOK before tool {context.tool_name}")
        inputs = context.tool_input
        print(f"Input {inputs}")
        if 'properties' in inputs:
            print("cleansing the input")
            inputs['video_path'] = inputs['properties']['video_path']
            inputs['frame_rate'] = inputs['properties']['frame_rate']
            if 'media_pipe_model' in inputs['properties']:
                inputs['media_pipe_model'] = inputs['properties']['media_pipe_model']
            del inputs['properties']
            print(f"input fixed : {inputs}")


