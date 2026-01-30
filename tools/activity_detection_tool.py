import os
import urllib
from enum import Enum

import cv2
import mediapipe as mp
from crewai.tools import BaseTool
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerOptions,
    PoseLandmarker)

from models import ExecutionError
from models.activity_detection_models import (
    ActivityDetectionInput, BodyLandmarks,
    Activity
)
from models.base_models import DetectionToolOutput
from utils import capture_statistics

MEDIA_PIPE_MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/"


class MediaPipeModel(Enum):
    LITE = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", "pose_landmarker_lite.task"
    FULL = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", "pose_landmarker_full.task"
    HEAVY = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", "pose_landmarker_heavy.task"
    HANDS = MEDIA_PIPE_MODEL_BASE_URL + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "hand_landmarker.task"


class ActivityDetectionTool(BaseTool):
    name: str = "activity_detection"
    description: str = "Detects human activities, poses, and gestures in videos using MediaPipe. Requires video_path (string), media_pipe_model (string: 'LITE', 'FULL', or 'HEAVY'), and frame_rate (integer). Returns JSON with frames_analyzed, detections array, activity_summary, pose_detections, hands_detections, and anomalies."
    args_schema: type[ActivityDetectionInput] = ActivityDetectionInput

    def detect_activity_from_pose(self, pose_landmarks: list[list[NormalizedLandmark]]) -> Activity | None:
        if not pose_landmarks or len(pose_landmarks) == 0:
            return None

        landmarks = pose_landmarks[0]

        body_landmarks = BodyLandmarks(landmarks)
        return Activity(
            hands_activity=self.detect_hand_position(body_landmarks),
            movement_activity=self.detect_body_movement(body_landmarks),
        )

    def detect_standing_or_sitting(self, body_landmarks: BodyLandmarks):
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        torso_length = abs(hip_y - shoulder_y)
        if torso_length < 0.15:
            return "sitting"
        return self.detect_body_movement(body_landmarks)

    @staticmethod
    def detect_hand_position(body_landmarks: BodyLandmarks):
        return body_landmarks.detect_hand_activity()

    @staticmethod
    def detect_body_movement(body_landmarks: BodyLandmarks):
        if body_landmarks.is_jumping_pose():
            return "jumping"
        elif body_landmarks.is_crouching_pose():
            return "crouching"
        elif body_landmarks.is_walking_pose():
            return "walking"
        elif body_landmarks.is_standing_still():
            return "standing"
        else:
            return "transitioning"

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

    def _run(self, video_path, media_pipe_model: str, frame_rate: int = 5) -> str:
        pose_model = self.download_model(MediaPipeModel[media_pipe_model])

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            min_pose_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            min_pose_presence_confidence=0.8,
            running_mode=vision.RunningMode.VIDEO,
        )
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return ExecutionError(error=f"Unable to open video source {video_path}.").model_dump_json(indent=2)

        activity_stats = {}

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
                if frame_number % frame_rate != 0:
                    frame_number += 1
                    continue

                analyzed_counter += 1
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                pose_result = pose_landmarker.detect_for_video(mp_image, int(timestamp))

                # analyze activities from pose
                detected_activity = self.detect_activity_from_pose(pose_result.pose_landmarks)

                if detected_activity:
                    capture_statistics(activity_stats, f"Pose activity - {detected_activity.movement_activity}",
                                       timestamp)
                    capture_statistics(activity_stats, f"Hand activity - {detected_activity.hands_activity}", timestamp)

                frame_number += 1
        except Exception as e:
            return ExecutionError(error=str(e)).model_dump_json(indent=2)
        finally:
            cap.release()
            pose_landmarker.close()

        response = DetectionToolOutput(
            statistics=list(activity_stats.values()),
        )
        return response.model_dump_json(indent=2)

    def get_hand_landmarker(self) -> HandLandmarker:
        hands_model = self.download_model(MediaPipeModel.HANDS)
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hands_model),
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            min_hand_presence_confidence=0.8,
            num_hands=2
        )

        hands_landmarker = HandLandmarker.create_from_options(hand_options)
        return hands_landmarker

    @staticmethod
    def get_model_path(pose_model: MediaPipeModel):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media_pipe', 'pose_models',
                            pose_model.value[1])
