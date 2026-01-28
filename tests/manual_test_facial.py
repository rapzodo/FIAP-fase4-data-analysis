import json
from pathlib import Path

from tools import FacialDetectionTool

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_PATH = PROJECT_ROOT / "tech-challenge" / "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"

tool = FacialDetectionTool()
def test_facial_detection():
    if not VIDEO_PATH.exists():
        print(f"❌ Video file not found: {VIDEO_PATH}")
        exit(1)

    result_json = tool._run(video_path=str(VIDEO_PATH), frame_rate=5)
    result = json.loads(result_json)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Frames Analyzed: {result['frames_analyzed']}")
        print(f"✅ Total Faces: {result['total_faces']}")
        print(f"✅ Faces Detected: {result['faces_detected'][:3]}")

if __name__ == "__main__":
    test_facial_detection()