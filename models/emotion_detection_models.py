from typing import Optional

from pydantic import BaseModel, Field

from .base_models import ExecutionError


class BaseAnalyticsDataModel(BaseModel):
    timestamp: str = Field(description="Formatted Timestamp HH:mm:ss", examples=["00:00:00", "00:00:01"])


class FaceEmotion(BaseAnalyticsDataModel):
    dominant_emotion: str = Field(..., description="Dominant emotion")
    confidence: float = Field(ge=0, le=100)


class EmotionDetection(BaseAnalyticsDataModel):
    emotion: str = Field(..., description="Emotion name")


class EmotionAnomaly(BaseAnalyticsDataModel):
    type: str = Field(..., description="Type of anomaly")
    error: Optional[ExecutionError] = Field(None, description="Error message")

class Summary(BaseModel):
    type: str = Field(..., description="Type of the detection")
    appearances: int = Field(..., description="Number of appearances")


class EmotionDetectionResult(BaseModel):
    total_faces_analyzed: int = Field(0, description="Total number of faces analyzed")
    total_frames_analyzed: int = Field(0, description="Total number of frames analyzed")
    emotions_detected: list[FaceEmotion] = Field(default_factory=list, description="List of detected emotions")
    emotions_summary: dict[str, int] = Field(description="Summary of emotions")


class EmotionReportOutput(BaseModel):
    top_emotion: str = Field(..., description="Top emotion")
    total_emotions_analyzed: int = Field(0, description="Total number of emotions analyzed")
    emotion_detections: list[EmotionDetection] = Field(default_factory=list, description="List of detected emotions")
    final_thought: str = Field(default_factory=str, description="Final thought")
