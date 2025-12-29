import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from typing import List, Dict, Any, Tuple
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json

class FacialRecognitionToolInput(BaseModel):
    video_path: str = Field(..., description="Path to the video file to analyze")
    frame_sample_rate: int = Field(1, description="Process every Nth frame (default: 1 = every frame)")

class FacialRecognitionTool(BaseTool):
    name: str = "facial_recognition_tool"
    description: str = "Analyzes video frames to detect faces, recognize individuals, and identify emotions. Returns structured data with timestamps, detected persons, emotions, and confidence scores."
    args_schema: type[BaseModel] = FacialRecognitionToolInput

    def _run(self, video_path: str, frame_sample_rate: int = 1) -> str:
        results = {
            "total_frames_analyzed": 0,
            "faces_detected": [],
            "emotions_detected": {},
            "anomalies": [],
            "processing_details": {}
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return json.dumps({"error": f"Failed to open video: {video_path}"})

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        analyzed_count = 0

        emotion_counts = {
            "happy": 0, "sad": 0, "angry": 0,
            "surprise": 0, "neutral": 0, "fear": 0, "disgust": 0
        }

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % frame_sample_rate != 0:
                    continue

                analyzed_count += 1
                timestamp = frame_count / fps if fps > 0 else frame_count

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                face_locations = face_recognition.face_locations(rgb_small_frame)

                if not face_locations:
                    results["anomalies"].append({
                        "frame": frame_count,
                        "timestamp": timestamp,
                        "type": "no_face_detected"
                    })
                    continue

                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    face_img = frame[top:bottom, left:right]

                    try:
                        emotion_result = DeepFace.analyze(
                            face_img,
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=True
                        )

                        if isinstance(emotion_result, list):
                            emotion_result = emotion_result[0]

                        dominant_emotion = emotion_result['dominant_emotion']
                        emotion_scores = emotion_result['emotion']

                        emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1

                        face_data = {
                            "frame": frame_count,
                            "timestamp": round(timestamp, 2),
                            "location": {
                                "top": top, "right": right,
                                "bottom": bottom, "left": left
                            },
                            "emotion": dominant_emotion,
                            "emotion_scores": {k: round(v, 2) for k, v in emotion_scores.items()},
                            "confidence": round(max(emotion_scores.values()), 2)
                        }

                        results["faces_detected"].append(face_data)

                        if face_data["confidence"] < 50:
                            results["anomalies"].append({
                                "frame": frame_count,
                                "timestamp": timestamp,
                                "type": "low_confidence",
                                "confidence": face_data["confidence"]
                            })

                    except Exception as e:
                        results["anomalies"].append({
                            "frame": frame_count,
                            "timestamp": timestamp,
                            "type": "emotion_detection_failed",
                            "error": str(e)
                        })

        finally:
            cap.release()

        results["total_frames_analyzed"] = analyzed_count
        results["emotions_detected"] = emotion_counts
        results["processing_details"] = {
            "video_fps": fps,
            "total_video_frames": total_frames,
            "frame_sample_rate": frame_sample_rate,
            "duration_seconds": total_frames / fps if fps > 0 else 0
        }

        return json.dumps(results, indent=2)

