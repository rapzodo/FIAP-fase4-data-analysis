import os
from pathlib import Path

from dotenv import load_dotenv

from tools.activity_detection_tool import MediaPipeModel

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

VIDEO_PATH = PROJECT_ROOT / os.getenv("VIDEO_PATH", "tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")
FRAME_SAMPLE_RATE = os.getenv("FRAME_SAMPLE_RATE", "30")
POSE_MODEL = MediaPipeModel.LITE.name

AGENTS_CONFIG_PATH = str(PROJECT_ROOT / os.getenv("AGENTS_CONFIG_PATH", "config/agents.yml"))
TASKS_CONFIG_PATH = str(PROJECT_ROOT / os.getenv("TASKS_CONFIG_PATH", "config/tasks.yml"))