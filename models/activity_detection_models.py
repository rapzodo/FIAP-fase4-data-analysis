from typing import Optional

from pydantic import BaseModel, Field

from .base_models import ExecutionError, BaseInputModel


class BodyLandmarks:
    def __init__(self, landmarks):
        self.left_shoulder = landmarks[11]
        self.right_shoulder = landmarks[12]
        self.left_hip = landmarks[23]
        self.right_hip = landmarks[24]
        self.left_wrist = landmarks[15]
        self.right_wrist = landmarks[16]
        self.left_elbow = landmarks[13]
        self.right_elbow = landmarks[14]
        self.left_ankle = landmarks[27]
        self.right_ankle = landmarks[28]
        self.left_knee = landmarks[25]
        self.right_knee = landmarks[26]


class ActivityDetectionInput(BaseInputModel):
    media_pipe_model: str = Field(description="MediaPipe model for pose detection. Valid values: 'LITE', 'FULL', 'HEAVY'")


class Activity(BaseModel):
    hands_activity: str = Field(description="hands movement activity")
    movement_activity: str = Field(description="movement activity")

class ActivityDetection(BaseModel):
    frame :int = Field(description="Frame number")
    timestamp : str = Field(description="Timestamp")
    activities : list[Activity] = Field(description="Activities")


class ActivityAnomaly(BaseModel):
    frame : int = Field(description="Frame number")
    timestamp : float = Field(description="Timestamp")
    type: str = Field(description="Anomaly type")
    details : Optional[str] = Field(None, description="Anomaly details")


class ActivityDetectionResult(BaseModel):
    frames_analyzed: int = Field(description="Frames analyzed")
    detections: list[ActivityDetection] = Field(description="Activities detected")
    activity_summary: dict[str, int] = Field(description="Activity summary")
    pose_detections : int = Field(description="Total Poses detected")
    error : Optional[ExecutionError] = Field(None, description="Error message")

    class Config:
        json_schema_extra = {
            "example": {
                "frames_analyzed": 100,
                "activities": [
                    {
                        "frame": 1,
                        "timestamp": 0.03,
                        "activities": ["standing", "hands_down"]
                    }
                ],
                "activity_summary": {
                    "standing": 60,
                    "sitting": 25,
                    "moving": 15,
                    "hands_raised": 40,
                    "hands_down": 60
                },
                "pose_detections": 98,
                "anomalies": []
            }
        }