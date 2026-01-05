import json
from typing import List, Optional, Any

import cv2
from crewai.tools import BaseTool
from deepface import DeepFace
from numpy import ndarray

from models import (
    EmotionDetectionInput,
    EmotionScores, FaceEmotion, EmotionDetectionResult,
    ExecutionError, FaceDetection, EmotionAnomaly
)
from models.emotion_detection_models import EMOTIONS


class EmotionDetectionTool(BaseTool):
    name: str = "emotion_detection"
    description: str = "Analyzes emotions in pre detected faces. Requires facial detection tools inputs"
    args_schema: type[EmotionDetectionInput] = EmotionDetectionInput

    def _run(self, video_path: str, face_detections: str) -> str:
        result = EmotionDetectionResult(
            total_faces_analyzed=0,
            emotions_detected=[],
            emotion_summary={},
            anomalies=[]
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ExecutionError(error="Unable to open video file").model_dump_json(indent=2)

        emotions_count = {}
        # ##initialize the counter
        for emotion in EMOTIONS:
            emotions_count[emotion.name.lower()] = 0

        faces_by_frame = self.group_faces_by_frame(face_detections)

        try:
            for frame_number, faces in faces_by_frame.items():
                # skipping frames without faces for efficiency
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
                ret, frame = cap.read()
                if not ret:
                    result.anomalies.extend(self.create_unknown_anomalies(faces))
                    continue

                face_emotion, emotions_count, anomalies_detected = self.detect_emotions(faces, emotions_count, frame)
                result.emotions_detected.append(face_emotion)
                result.total_faces_analyzed += 1
                result.anomalies.extend(anomalies_detected)
        finally:
            cap.release()
        result.emotion_summary = emotions_count
        return result.model_dump_json(indent=2)


    def detect_emotions(self, faces, emotions_count: dict[str, int], current_frame: ndarray[int]) -> tuple[
        FaceEmotion | None, dict[str, int], list[Any]
    ]:
        anomalies_detected = []
        face_emotion = None
        for face in faces:
            top, right, bottom, left = face.location.top, face.location.right, face.location.bottom, face.location.left
            face_image = current_frame[top:bottom, left:right]
            if face_image.size == 0:
                anomalies_detected.append(self.create_anomaly(face=face, anomaly_type="Invalid Face Region"))
                continue
            try:
                analysis = DeepFace.analyze(
                    face_image,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]

                dominant_emotion = analysis["dominant_emotion"]
                analysis_emotion = analysis["emotion"]
                confidence = max(analysis_emotion.values())
                emotion_score = EmotionScores(
                    angry=round(analysis_emotion.get(EMOTIONS.ANGRY.name.lower(), 0), 2),
                    disgust=round(analysis_emotion.get(EMOTIONS.DISGUST.name.lower(), 0), 2),
                    fear=round(analysis_emotion.get(EMOTIONS.FEAR.name.lower(), 0), 2),
                    happy=round(analysis_emotion.get(EMOTIONS.HAPPY.name.lower(), 0), 2),
                    sad=round(analysis_emotion.get(EMOTIONS.SAD.name.lower(), 0), 2),
                    surprise=round(analysis_emotion.get(EMOTIONS.SURPRISE.name.lower(), 0), 2),
                    neutral=round(analysis_emotion.get(EMOTIONS.NEUTRAL.name.lower(), 0), 2),
                )

                face_emotion = FaceEmotion(
                    frame=face.frame,
                    timestamp=face.timestamp,
                    face_id=face.face_id,
                    dominant_emotion=dominant_emotion,
                    confidence=round(confidence, 2),
                    emotion_score=emotion_score,
                )

                if dominant_emotion not in [emotion.name.lower() for emotion in EMOTIONS]:
                    anomalies_detected.append(self.create_anomaly(face=face, anomaly_type="Invalid Emotion"))

                emotions_count[dominant_emotion] += 1

                if confidence < 50:
                    anomalies_detected.append(
                        self.create_anomaly(face=face, anomaly_type="Low confidence"))
            except Exception as e:
                anomalies_detected.append(self.create_anomaly(face, "Emotion detection failed",
                                                              ExecutionError(error=str(e))))
        return face_emotion, emotions_count, anomalies_detected

    @staticmethod
    def create_anomaly(face, anomaly_type: str, error: Optional[ExecutionError] = None):
        return EmotionAnomaly(
            frame=face.frame,
            timestamp=face.timestamp,
            face_id=face.face_id,
            type=anomaly_type,
            error=error,
        )

    @staticmethod
    def create_unknown_anomalies(faces) -> List[EmotionAnomaly]:
        return list(map(lambda face: EmotionAnomaly(
            frame=face.frame,
            timestamp=face.timestamp,
            face_id=face.face_id,
            type=EMOTIONS.UNKNOWN.name
        ), faces))

    @staticmethod
    def group_faces_by_frame(face_locations: str):
        face_detections_json = json.loads(face_locations)
        face_detections = [FaceDetection(**face_detection) for face_detection in face_detections_json]
        faces_by_frame = {}
        for face in face_detections:
            frame = face.frame
            if frame not in faces_by_frame:
                faces_by_frame[frame] = []
            faces_by_frame[frame].append(face)
        return faces_by_frame
