import json

from tests.manual_test_facial import VIDEO_PATH
from tools import FacialDetectionTool, EmotionDetectionTool


def test_pipeline():

    print("🧪 Testing Complete Pipeline\n")

    print("Step 1: Detecting faces...")
    facial_tool = FacialDetectionTool()
    faces_json = facial_tool._run(video_path=VIDEO_PATH, sample_rate=30)

    faces_result = json.loads(faces_json)
    print(f"✅ Found {faces_result['total_faces']} faces")

    if faces_result['total_faces'] == 0:
        print("No faces detected")

    print("\nStep 2: Analyzing emotions...")
    emotion_tool = EmotionDetectionTool()
    emotions_json = emotion_tool._run(video_path=VIDEO_PATH, face_detections=json.dumps(faces_result['faces_detected']))
    emotions_result = json.loads(emotions_json)
    print(f"✅ Analyzed {emotions_result['total_faces_analyzed']} faces")

    print(f"\n📊 Emotion Summary:")
    for emotion, count in emotions_result['emotion_summary'].items():
        if count > 0:
            print(f"   {emotion}: {count}")

    if emotions_result['anomalies']:
        for anomaly in emotions_result['anomalies']:
            print("⚠️  Anomaly Detected")
            print(anomaly)


if __name__ == "__main__":
    test_pipeline()