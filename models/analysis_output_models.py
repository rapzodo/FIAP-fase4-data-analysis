from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from models import EmotionData, ActivityData, PoseData


class BaseAnalysisOutputModel(BaseModel):
    total_fames: int = Field(0, description="Total Fames analyzed")
    anomalies_count: int = Field(0, description="Number of anomalies")
    anomalies_detected: Dict[str, int] = Field(..., description="Type of anomalies detected and their frequency")
    error: Optional[str] = Field(None, description="Error message in case the task fails")

class FacialAnalysisOutput(BaseAnalysisOutputModel):
    faces_detected: int = Field(0, description="Number of faces detected")
    emotions_distribution: List[EmotionData] = Field(default_factory=List, description="Emotions distribution")
    avg_confidence: float = Field(0.0, description="Average confidence")
    detection_rate: float = Field(0.0, description="Detection rate (percentage)")


class ActivityAnalysisOutput(BaseAnalysisOutputModel):
    activities: list[ActivityData] = Field(default_factory=list, description="Activities analyzed")
    poses: list[PoseData] = Field(default_factory=list, description="Poses analyzed")
    gestures: list[str] = Field(default_factory=list, description="Gestures analyzed")
    pose_detection_rate: float = Field(0.0, description="Detection rate (percentage)")
