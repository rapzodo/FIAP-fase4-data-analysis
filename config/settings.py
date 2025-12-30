import os
from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH")

FRAME_SAMPLE_RATE = os.getenv("FRAME_SAMPLE_RATE")

OUTPUT_PATH = "output"

AGENTS_CONFIG_PATH = os.getenv("AGENTS_CONFIG_PATH")
TASKS_CONFIG_PATH = os.getenv("TASKS_CONFIG_PATH")

os.makedirs(OUTPUT_PATH, exist_ok=True)