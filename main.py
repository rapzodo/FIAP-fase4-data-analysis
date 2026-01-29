from crew import VideoAnalysisSummaryCrew
from utils import reset_crew_memory
from utils.helper_functions import print_crewai_storage_path


def kickoff(reset_memory:bool=False) -> str | None:
    """Test activity detection on a video."""

    print("🏃 Starting Activity Detector Agent\n")

    print("🚀 Analyzing ...\n")
    print("⏱️  This may take 2-15 minutes\n")

    crew = VideoAnalysisSummaryCrew().crew()
    try:
        if reset_memory:
            reset_crew_memory(crew)
        # crew.test(n_iterations=3, eval_llm=llm_config.get_llm().model)
        result = crew.kickoff()
        return result.raw
    except Exception as e:
        print(e)


if __name__ == "__main__":
    print_crewai_storage_path()
    kickoff(True)