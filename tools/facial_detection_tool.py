import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='face_recognition_models')

from typing import Tuple
from crewai.tools import BaseTool
import cv2
import numpy as np
import face_recognition

from models import (
    FaceDetectionInput,
    FaceDetection,
    FacialDetectionResult,
    FacialDetectionError,
    FaceLocation
)


class FacialDetectionTool(BaseTool):
    name: str = "Facial Detection"
    description: str = "Detects faces in videos"
    args_schema: type[FaceDetectionInput] = FaceDetectionInput

    def _run(self, video_path: str, sample_rate: int = 5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return FacialDetectionError(error=f"Unable to open video {video_path}").model_dump_json(indent=2)

        try:
            result = self.process_face_detection(cap, sample_rate)
        finally:
            cap.release()

        return result.model_dump_json(indent=2)

    def process_face_detection(self, cap: cv2.VideoCapture, sample_rate: int) -> FacialDetectionResult:
        result = FacialDetectionResult(
            frames_analyzed=0,
            faces_detected=[],
            total_faces=0,
        )

        frame_count = 0
        analyzed_count = 0
        face_id_counter = 0

        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % sample_rate != 0:
                continue

            analyzed_count += 1
            timestamp = self.calculate_timestamp(fps, frame_count)

            rgb_small_frame = self.resize_and_convert_to_rgb(frame)

            face_locations = face_recognition.face_locations(rgb_small_frame)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                top, right, bottom, left = self.scale_to_original_size(top, right, bottom, left)

                detected_face = FaceDetection(
                    frames=frame_count,
                    timestamp=round(timestamp, 2),
                    face_id=face_id_counter,
                    location=FaceLocation(
                        top=top,
                        right=right,
                        bottom=bottom,
                        left=left
                    )
                )
                result.faces_detected.append(detected_face)
                face_id_counter += 1

        result.total_faces = len(result.faces_detected)
        result.frames_analyzed = analyzed_count
        return result

    @staticmethod
    def calculate_timestamp(fps: float, frame_count: int) -> float:
        return frame_count / fps if fps > 0 else float(frame_count)

    @staticmethod
    def scale_to_original_size(top: int, right: int, bottom: int, left: int) -> Tuple[int, int, int, int]:
        multiply_factor = 4
        return top * multiply_factor, right * multiply_factor, bottom * multiply_factor, left * multiply_factor

    @staticmethod
    def resize_and_convert_to_rgb(frame: np.ndarray) -> np.ndarray:
        resize_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        return np.ascontiguousarray(small_frame[:, :, ::-1])

