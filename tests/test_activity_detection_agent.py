import os

from crewai import Crew

from agents.agents_factory import AgentsFactory
from tasks.task_factory import TaskFactory
from tools.activity_detection_tool import ActivityDetectionTool


def test_activity_detector_agent():
    """Test activity detection on a video."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(project_root, "tech-challenge", "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print(f"💡 Project root: {project_root}")
        print("💡 Update video_path to your actual video file")
        return

    # Set environment variables for TaskFactory
    os.environ["VIDEO_PATH"] = video_path
    os.environ["FRAME_SAMPLE_RATE"] = "1"

    print("🏃 Testing Activity Detector Agent\n")

    # Create agent
    config_path = os.path.join(project_root, "config", "agents.yml")
    config_path_task = os.path.join(project_root, "config", "tasks.yml")
    tool = ActivityDetectionTool()
    agent = AgentsFactory(config_path).create_agent("activity_detector", tools={tool.name: tool})
    print("✅ Agent created\n")

    # Create task
    task = TaskFactory(config_path_task).create_task(task_name="detect_activities", agent=agent)
    print("✅ Task created\n")

    # Execute
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    print("🚀 Analyzing activities...\n")
    print("⏱️  This may take 2-5 minutes\n")

    result = crew.kickoff()

    print("\n" + "="*70)
    print("🏃 ACTIVITY DETECTION ANALYSIS")
    print("="*70)
    print(result)
    print("="*70)

    return result


if __name__ == "__main__":
    test_activity_detector_agent()