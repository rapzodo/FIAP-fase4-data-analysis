import os

from dotenv import load_dotenv

from tools.activity_detection_tool import MediaPipeModel

load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH")

FRAME_SAMPLE_RATE = os.getenv("FRAME_SAMPLE_RATE")

POSE_MODEL = MediaPipeModel.LITE.name

OUTPUT_PATH = "output"

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
AGENTS_CONFIG_PATH = PROJECT_ROOT/os.getenv("AGENTS_CONFIG_PATH")
TASKS_CONFIG_PATH = PROJECT_ROOT/os.getenv("TASKS_CONFIG_PATH")

os.makedirs(OUTPUT_PATH, exist_ok=True)