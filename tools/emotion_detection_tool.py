import datetime

import cv2
from crewai.tools import BaseTool
from deepface import DeepFace
from pydantic.v1 import BaseModel
from tqdm import tqdm

from models import (
    ExecutionError
)
from models.base_models import BaseInputModel, DetectionStatistics, DetectionToolOutput


class EmotionDetectionTool(BaseTool):
    name: str = "emotion_detection"
    description: str = "Analyzes emotions in faces detected in video. Requires video_path (string) parameter. Returns JSON with total_faces_analyzed, emotions_detected array, emotion_summary, and anomalies."
    args_schema: type[BaseModel] = BaseInputModel

    def _run(self, video_path: str, frame_rate:int) -> str:
        statistics = DetectionToolOutput(
            emotion_statistics=[]
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ExecutionError(error="Unable to open video file").model_dump_json(indent=2)

        emotions_summary : dict[str, DetectionStatistics] = {}
        # ##initialize the counter
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_number = 0
        try:
            for _ in tqdm(range(int(total_frames)), desc="Analyzing emotions"):
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

                analysis_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                for face in analysis_result:
                    dominant_emotion = face["dominant_emotion"]
                    face_confidence = face["face_confidence"]
                    if face_confidence == 0.0:
                        continue
                    if dominant_emotion and face_confidence < 0.3:
                        emotions_summary["anomaly"] = emotions_summary.get("anomaly", DetectionStatistics(
                            emotion="anomaly",
                            total_fames_appearances=0,
                            timestamps=[],
                        ))
                        emotions_summary["anomaly"].total_fames_appearances+=1
                        emotions_summary["anomaly"].timestamps.append(format_frame_timestamp(timestamp))
                        continue


                    emotions_summary[dominant_emotion] = emotions_summary.get(dominant_emotion, DetectionStatistics(
                        emotion=dominant_emotion,
                        total_fames_appearances=0,
                        timestamps=[],
                    ))

                    emotions_summary[dominant_emotion].total_fames_appearances+=1
                    emotions_summary[dominant_emotion].timestamps.append(format_frame_timestamp(timestamp))

                frame_number += 1
        except Exception as e:
            ExecutionError(error=str(e)).model_dump_json(indent=2)
        finally:
            cap.release()
        statistics.emotion_statistics = list(emotions_summary.values())
        return statistics.model_dump_json(indent=2)


def format_frame_timestamp(timestamp: float) -> str:
    return str(datetime.timedelta(milliseconds=timestamp))
