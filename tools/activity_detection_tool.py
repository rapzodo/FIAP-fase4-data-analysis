import os
import urllib
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
from crewai.tools import BaseTool
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions, \
    PoseLandmarker

from models import ExecutionError
from models.activity_detection_models import ActivityDetectionInput, ActivityDetectionResult, BodyLandmarks, \
    ActivityDetection, ActivityAnomaly

MEDIA_PIPE_MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/"


class MediaPipeModel(Enum):
    LITE = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", "pose_landmarker_lite.task"
    FULL = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", "pose_landmarker_full.task"
    HEAVY = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", "pose_landmarker_heavy.task"
    HANDS = MEDIA_PIPE_MODEL_BASE_URL + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "hand_landmarker.task"


class ActivityDetectionTool(BaseTool):
    name: str = "activity_detection"
    description: str = "Detects human activities, poses, and gestures in videos using MediaPipe. Requires video_path (string), media_pipe_model (string: 'LITE', 'FULL', or 'HEAVY'), and sample_rate (integer)."
    args_schema: type[ActivityDetectionInput] = ActivityDetectionInput

    def detect_activity_from_pose(self, pose_landmarks: list[list[NormalizedLandmark]]):
        activities = []
        if not pose_landmarks or len(pose_landmarks) == 0:
            return activities

        landmarks = pose_landmarks[0]

        body_landmarks = BodyLandmarks(landmarks)

        activities.append(self.detect_standing_or_sitting(body_landmarks))
        activities.append(self.detect_hand_position(body_landmarks))
        activities.append(self.detect_body_movement(body_landmarks))
        return activities

    def detect_standing_or_sitting(self, body_landmarks: BodyLandmarks):
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        torso_length = abs(hip_y - shoulder_y)
        if torso_length < 0.3:
            return "standing"
        elif torso_length < 0.15:
            return "sitting"
        return "unknown"

    def detect_hand_position(self, body_landmarks: BodyLandmarks):
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        if body_landmarks.left_wrist.y < shoulder_y or body_landmarks.right_wrist.y < shoulder_y:
            return "hands_raised"
        else:
            return "hands_down"

    def detect_body_movement(self, body_landmarks: BodyLandmarks):
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        if body_landmarks.left_knee.y < hip_y or body_landmarks.right_knee.y < hip_y:
            return "moving"
        else:
            return "standing"

    @staticmethod
    def calculate_average_position(body_landmarks: BodyLandmarks):
        shoulder_y = (body_landmarks.left_shoulder.y + body_landmarks.right_shoulder.y) / 2
        hip_y = (body_landmarks.left_hip.y + body_landmarks.right_hip.y) / 2
        return shoulder_y, hip_y

    @staticmethod
    def download_model(pose_model: MediaPipeModel):
        pose_model_url = pose_model.value[0]
        pose_model_name = pose_model.value[1]

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media_pipe', 'pose_models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, pose_model_name)

        if not os.path.exists(model_path):
            print(f'Downloading model file... {pose_model_name}')
            urllib.request.urlretrieve(pose_model_url, model_path)
            print(f'Model {pose_model_name} downloaded successfully.')

        return model_path

    def _run(self, video_path, media_pipe_model: str, sample_rate: int = 5) -> str:
        # Get project root (parent of tools directory) and resolve video path
        project_root = Path(__file__).parent.parent
        video_path = str(project_root / video_path)

        pose_model = self.download_model(MediaPipeModel[media_pipe_model])
        hands_model = self.download_model(MediaPipeModel.HANDS)

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
        )
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hands_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )

        hands_landmarker = HandLandmarker.create_from_options(hand_options)

        result = ActivityDetectionResult(
            frames_analyzed=0,
            activities=[],
            activity_summary={},
            pose_detections=0,
            hands_detections=0,
            anomalies=[]
        )
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return ExecutionError(error=f"Unable to open video source {video_path}.").model_dump_json(indent=2)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        activity_counter = {
            "standing":0,
            "sitting":0,
            "hands_raised":0,
            "hands_down":0,
            "moving":0,
            "unknown":0
        }

        frame_number = 0
        analyzed_counter = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                """
                frame sampling mechanism for improved performance. 
                Useful for activity detection where analyzing every single frame may be unnecessary and processing-intensive.
                """
                if frame_number % sample_rate != 0:
                    frame_number += 1
                    continue

                analyzed_counter += 1
                timestamp = frame_number / fps if fps > 0 else frame_number

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                frame_timestamp_ms = int(timestamp * 1000)

                pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                hands_result = hands_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                frame_activities = []

                #tracking detections
                if pose_result.pose_landmarks:
                    result.pose_detections += len(pose_result.pose_landmarks)

                if hands_result.hand_landmarks:
                    result.hands_detections += len(hands_result.hand_landmarks)

                #analyze activities from pose
                detected_activities = self.detect_activity_from_pose(pose_result.pose_landmarks)

                if detected_activities:
                    frame_activities.extend(detected_activities)
                    for activity in detected_activities:
                        activity_counter[activity] += 1

                if frame_activities:
                    result.activities.append(ActivityDetection(
                        frame = frame_number,
                        timestamp = round(timestamp, 2),
                        activities=frame_activities
                    ))
                else:
                    activity_counter["unknown"] += 1
                    result.anomalies.append(ActivityAnomaly(
                        frame = frame_number,
                        timestamp = round(timestamp, 2),
                        type="No pose detected",
                        details="No human pose detected in frame"
                    ))

                frame_number += 1
        finally:
            cap.release()
            pose_landmarker.close()
            hands_landmarker.close()

        result.frames_analyzed = analyzed_counter
        result.activity_summary = activity_counter
        return result.model_dump_json(indent=2)


    @staticmethod
    def get_model_path(pose_model: MediaPipeModel):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media_pipe', 'pose_models',
                            pose_model.value[1])
