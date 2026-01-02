from typing import List, Dict

from pydantic import BaseModel, Field

from models.emotion_detection_models import EmotionData

class BaseAnalysisOutputModel(BaseModel):
    total_fames: int = Field(..., description="Total Fames analyzed")
    anomalies_count: int = Field(..., description="Number of anomalies")
    anomalies_detected: Dict[str, int] = Field(..., description="Type of anomalies detected and their frequency")


class FacialAnalysisOutput(BaseAnalysisOutputModel):
    faces_detected: int = Field(..., description="Number of faces detected")
    emotions_distribution: List[EmotionData] = Field(default_factory=List, description="Emotions distribution")
    avg_confidence: float = Field(..., description="Average confidence")
    detection_rate: float = Field(..., description="Detection rate (percentage)")

class ActivityAnalysisOutput(BaseAnalysisOutputModel):
    pass