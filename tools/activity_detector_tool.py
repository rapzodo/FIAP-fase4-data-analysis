import cv2
import mediapipe as mp
from typing import List, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json
import numpy as np

class ActivityDetectorToolInput(BaseModel):
    video_path: str = Field(..., description="Path to the video file to analyze")
    frame_sample_rate: int = Field(1, description="Process every Nth frame (default: 1 = every frame)")

class ActivityDetectorTool(BaseTool):
    name: str = "activity_detector_tool"
    description: str = "Detects human activities and poses in video frames using MediaPipe. Identifies activities like standing, sitting, walking, hand gestures, and body movements."
    args_schema: type[BaseModel] = ActivityDetectorToolInput

    def _run(self, video_path: str, frame_sample_rate: int = 1) -> str:
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands

        results = {
            "total_frames_analyzed": 0,
            "activities_detected": [],
            "activity_summary": {},
            "pose_landmarks_detected": 0,
            "hand_gestures_detected": 0,
            "anomalies": []
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return json.dumps({"error": f"Failed to open video: {video_path}"})

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        analyzed_count = 0

        activity_counts = {
            "standing": 0,
            "sitting": 0,
            "moving": 0,
            "hands_raised": 0,
            "hands_down": 0,
            "unknown": 0
        }

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose, mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            try:
                prev_pose = None

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    if frame_count % frame_sample_rate != 0:
                        continue

                    analyzed_count += 1
                    timestamp = frame_count / fps if fps > 0 else frame_count

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pose_results = pose.process(rgb_frame)
                    hand_results = hands.process(rgb_frame)

                    activity_data = {
                        "frame": frame_count,
                        "timestamp": round(timestamp, 2),
                        "activities": [],
                        "confidence": 0
                    }

                    if pose_results.pose_landmarks:
                        results["pose_landmarks_detected"] += 1
                        landmarks = pose_results.pose_landmarks.landmark

                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

                        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                        hip_y = (left_hip.y + right_hip.y) / 2
                        knee_y = (left_knee.y + right_knee.y) / 2

                        torso_angle = abs(hip_y - shoulder_y)

                        if torso_angle > 0.3:
                            activity = "standing"
                            confidence = left_shoulder.visibility * 100
                        elif knee_y - hip_y < 0.1:
                            activity = "sitting"
                            confidence = left_hip.visibility * 100
                        else:
                            activity = "moving"
                            confidence = 80

                        if prev_pose:
                            movement = np.sqrt(
                                (shoulder_y - prev_pose[0])**2 +
                                (hip_y - prev_pose[1])**2
                            )
                            if movement > 0.05:
                                activity = "moving"

                        prev_pose = (shoulder_y, hip_y)

                        activity_counts[activity] += 1
                        activity_data["activities"].append(activity)
                        activity_data["confidence"] = round(confidence, 2)

                    if hand_results.multi_hand_landmarks:
                        results["hand_gestures_detected"] += len(hand_results.multi_hand_landmarks)

                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                            if wrist.y < 0.5:
                                activity_data["activities"].append("hands_raised")
                                activity_counts["hands_raised"] += 1
                            else:
                                activity_data["activities"].append("hands_down")
                                activity_counts["hands_down"] += 1

                    if not activity_data["activities"]:
                        activity_data["activities"].append("unknown")
                        activity_counts["unknown"] += 1
                        results["anomalies"].append({
                            "frame": frame_count,
                            "timestamp": timestamp,
                            "type": "no_pose_detected"
                        })

                    results["activities_detected"].append(activity_data)

            finally:
                cap.release()

        results["total_frames_analyzed"] = analyzed_count
        results["activity_summary"] = activity_counts
        results["processing_details"] = {
            "video_fps": fps,
            "total_video_frames": total_frames,
            "frame_sample_rate": frame_sample_rate,
            "duration_seconds": total_frames / fps if fps > 0 else 0
        }

        return json.dumps(results, indent=2)

