import datetime
import json
import os

from crewai import Crew
from crewai.hooks import LLMCallHookContext, ToolCallHookContext
from crewai.utilities.paths import db_storage_path

from models.base_models import DetectionStatistics


def format_frame_timestamp(timestamp: float) -> str:
    return str(datetime.timedelta(milliseconds=timestamp))


def capture_statistics(stats: dict[str, DetectionStatistics], detected_activity: str, timestamp: float):
    stats[detected_activity] = stats.get(detected_activity,
                                         DetectionStatistics(
                                                               detection_name=detected_activity,
                                                               total_fames_appearances=0,
                                                               timestamps=[],
                                                           ))
    stats[detected_activity].total_fames_appearances += 1
    stats[detected_activity].timestamps.append(format_frame_timestamp(timestamp))

def reset_crew_memory(crew: Crew, memory: str= "all") -> None:
    crew.reset_memories(memory)


def print_crewai_storage_path():
    # Get the base storage path
    storage_path = db_storage_path()
    print(f"CrewAI storage location: {storage_path}")

    # List all CrewAI storage directories
    if os.path.exists(storage_path):
        print("\nStored files and directories:")
        for item in os.listdir(storage_path):
            item_path = os.path.join(storage_path, item)
            if os.path.isdir(item_path):
                print(f"📁 {item}/")
                # Show ChromaDB collections
                if os.path.exists(item_path):
                    for subitem in os.listdir(item_path):
                        print(f"   └── {subitem}")
            else:
                print(f"📄 {item}")
    else:
        print("No CrewAI storage directory found yet.")

def clean_llm_response(context: LLMCallHookContext):
    if not context.response:
        return

    response_stripped = context.response.strip()

    if response_stripped.startswith("```json"):
        return

    try:
        json.loads(response_stripped)
        return
    except (json.JSONDecodeError, ValueError):
        pass

    print(f"cleaning the LLM response: {context.response[:100]} ...")
    reasoning_markers = [
        "Reasoning Plan:",
        "Strategic Plan:",
        "Analysis Plan:",
        "Plan:",
        "Final Answer:",
    ]
    for marker in reasoning_markers:
        if marker in context.response:
            parts = context.response.split(marker, 1)
            if len(parts) > 1:
                context.response = parts[1].strip()
                context.response = context.response.replace('```markdown', '')
                context.response = context.response.replace('```', '')
                print(f"clear response: {context.response[:50]} ...")

def clean_detection_tools_input(context: ToolCallHookContext):
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