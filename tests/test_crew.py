from video_analysis_summary_crew import VideoAnalysisSummaryCrew


def test_video_analysis_crew():
    """Test activity detection on a video."""

    print("🏃 Testing Activity Detector Agent\n")

    print("🚀 Analyzing activities...\n")
    print("⏱️  This may take 2-5 minutes\n")

    crew = VideoAnalysisSummaryCrew().crew()
    try:

        # crew.test(n_iterations=3, eval_llm=llm_config.get_llm().model)
        result = crew.kickoff()
        # print("\n" + "="*70)
        # print("🏃 ACTIVITY DETECTION ANALYSIS")
        # print("="*70)
        # print(result)
        # print(f'task output: {crew.tasks[0].output.pydantic}')
        # print("="*70)

        return result.raw
    except Exception as e:
        print(e)


if __name__ == "__main__":
    print(test_video_analysis_crew())