import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

AGENTS_CONFIG_PATH = str(PROJECT_ROOT / os.getenv("AGENTS_CONFIG_PATH", "config/agents.yml"))
TASKS_CONFIG_PATH = str(PROJECT_ROOT / os.getenv("TASKS_CONFIG_PATH", "config/tasks.yml"))