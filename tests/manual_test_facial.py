import json
from pathlib import Path
from tools.facial_detection_tool import FacialDetectionTool

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_PATH = PROJECT_ROOT / "tech-challenge" / "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"

tool = FacialDetectionTool()

if not VIDEO_PATH.exists():
    print(f"❌ Video file not found: {VIDEO_PATH}")
    exit(1)

result_json = tool._run(video_path=str(VIDEO_PATH), sample_rate=5)
result = json.loads(result_json)

if "error" in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Frames Analyzed: {result['frames_analyzed']}")
    print(f"✅ Total Faces: {result['total_faces']}")
    print(f"✅ Faces Detected: {result['faces_detected'][:3]}")
