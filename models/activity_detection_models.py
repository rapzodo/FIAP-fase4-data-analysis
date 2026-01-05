from typing import Optional

from pydantic import BaseModel, Field


class BodyLandmarks:
    def __init__(self, landmarks):
        self.left_shoulder = landmarks[11]
        self.right_shoulder = landmarks[12]
        self.left_hip = landmarks[23]
        self.right_hip = landmarks[24]
        self.left_wrist = landmarks[15]
        self.right_wrist = landmarks[16]
        self.left_ankle = landmarks[27]
        self.right_ankle = landmarks[28]
        self.left_knee = landmarks[25]
        self.right_knee = landmarks[26]


class ActivityDetectionInput(BaseModel):
    video_path: str = Field(description="Path to the video file")
    media_pipe_model: str = Field(description="MediaPipe model for pose detection. Valid values: 'LITE', 'FULL', 'HEAVY'")
    sample_rate: int = Field(description="Sample rate for frame processing")

class ActivityDetection(BaseModel):
    frame :int = Field(description="Frame number")
    timestamp : float = Field(description="Timestamp")
    activities : list[str] = Field(description="Activities", examples=["standing, hands_raised"])

class ActivityAnomaly(BaseModel):
    frame : int = Field(description="Frame number")
    timestamp : float = Field(description="Timestamp")
    type: str = Field(description="Anomaly type")
    details : Optional[str] = Field(None, description="Anomaly details")

class ActivityDetectionResult(BaseModel):
    frames_analyzed: int = Field(description="Frames analyzed")
    activities: list[ActivityDetection] = Field(description="Activities detected")
    activity_summary: dict[str, int] = Field(description="Activity summary")
    pose_detections : int = Field(description="Total Poses detected")
    hands_detections : int = Field(description="Total Hands detected")
    anomalies : list[ActivityAnomaly] = Field(description="Anomalies detected")

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
                "hand_detections": 100,
                "anomalies": []
            }
        }

##output
class ActivityData(BaseModel):
    activity: str = Field(..., title="Activity name")
    percentage: float = Field(..., title="Activity frequency percentage")
    frame_count: int = Field(..., title="Number of frames with this activity")

class PoseData(BaseModel):
    pose: str = Field(..., title="Pose name")
    percentage: float = Field(..., title="Pose frequency percentage")
    frame_count: int = Field(..., title="Number of frames with this pose")

class GestureData(BaseModel):
    gesture: str = Field(..., title="Gesture name")
    percentage: float = Field(..., title="Gesture frequency percentage")
    frame_count: int = Field(..., title="Number of frames with this gesture")