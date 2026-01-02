from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

from models import FacialDetectionError, FaceDetection


class EMOTIONS(Enum):
    ANGRY = 0,
    DISGUST = 1,
    FEAR = 2,
    HAPPY = 3,
    SAD = 4,
    NEUTRAL = 5,
    SURPRISE = 6,
    UNKNOWN = 7


class EmotionDetectionInput(BaseModel):
    video_path: str = Field(..., description="Path to video file")
    face_locations: List[FaceDetection] = Field(...,
                                                description="List of pre-detected face locations")  ##TODO review this, is it a list of face locations or face detections ???


class EmotionScores(BaseModel):
    angry: float = Field(..., ge=0, le=100)
    disgust: float = Field(..., ge=0, le=100)
    fear: float = Field(..., ge=0, le=100)
    happy: float = Field(..., ge=0, le=100)
    neutral: float = Field(..., ge=0, le=100)
    surprise: float = Field(..., ge=0, le=100)
    sad: float = Field(..., ge=0, le=100)


class BaseAnalyticsDataModel(BaseModel):
    frame: int = Field(description="Frame number")
    timestamp: float = Field(description="Timestamp")
    face_id: int = Field(..., description="Face ID")


class FaceEmotion(BaseAnalyticsDataModel):
    dominant_emotion: str = Field(..., description="Dominant emotion")
    emotion_score: EmotionScores = Field(..., description="Emotion scores")
    confidence: float = Field(ge=0, le=100)


class EmotionAnomaly(BaseAnalyticsDataModel):
    type: str = Field(..., description="Type of anomaly")
    confidence: Optional[float] = Field(None, ge=0, le=100)
    error: Optional[FacialDetectionError] = Field(None, description="Error message")


class EmotionDetectionResult(BaseModel):
    total_faces_analyzed: int = Field(..., description="Total number of faces analyzed")
    emotions_detected: List[FaceEmotion] = Field(..., description="List of detected emotions")
    emotion_summary: Dict[str, int] = Field(..., description="Summary of emotions")
    anomalies: List[EmotionAnomaly] = Field(..., description="List of anomalies")


class EmotionData(BaseModel):
    emotion: str = Field(..., title="Detected Emotion")
    frequency_percentage: float = Field(..., title="Frequency of Emotion")
    frame_count: int = Field(..., title="Number of frames with this emotion")
