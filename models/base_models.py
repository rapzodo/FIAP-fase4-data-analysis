from typing import Optional

from pydantic import BaseModel, Field


class ExecutionError(BaseModel):
    error: str = Field(..., description="Error message")


class BaseAnalysisOutputModel(BaseModel):
    total_fames: int = Field(0, description="Total Fames analyzed")
    error: Optional[str] = Field(None, description="Error message in case the task fails")


class BaseInputModel(BaseModel):
    video_path: str = Field(description="Path to the video file")
    frame_rate: int = Field(..., description="Frame rate")
    pose_model: Optional[str] = Field(None,description="MediaPipe model for pose detection. Valid values: 'LITE', 'FULL', 'HEAVY'")

class ReportOutputModel(BaseModel):
    summary: str = Field(..., description="Summary of the analysis")
    statistics: str = Field(..., description="Statistics of the analysis")
    insight: Optional[str] = Field(None, description="Insights of the analysis")
    recommendation: Optional[str] = Field(None, description="Recommendations of the analysis")
    final_answer: str = Field(description="Final answer of the analysis")


class ProcessingStatistics(BaseModel):
    frames_analyzed: int = Field(..., description="Total frames analyzed")
    success_rate: float = Field(..., description="Success rate percentage")


class AnomalyReport(BaseModel):
    total_anomalies: int = Field(..., description="Total number of anomalies")
    types_and_frequencies: dict = Field(..., description="Anomaly types with their frequencies")



class EmotionsReportModel(BaseModel):
    overview: str = Field(..., description="Overview in 3-4 sentences")
    processing_statistics: ProcessingStatistics
    emotions_breakdown: dict = Field(..., description="Emotions with percentages")
    anomaly_report: AnomalyReport
    emotions_timeline: list[dict] = Field(..., description="Timeline of emotion detections")
    top_emotions: list[str] = Field(..., description="Top 3 emotions detected")


class ActivitiesReportModel(BaseModel):
    overview: str = Field(..., description="Overview in 3-4 sentences")
    processing_statistics: ProcessingStatistics
    activity_breakdown: dict = Field(..., description="Activities with percentages")
    anomaly_report: AnomalyReport
    activities_timeline: list[dict] = Field(..., description="Timeline of activities")
    top_activities: list[str] = Field(..., description="Top 3 activities")

class DetectionStatistics(BaseModel):
    detection_name: str = Field(..., description="Detection name")
    total_fames_appearances: int = Field(0, description="Total number of frames analyzed")
    timestamps: list[str] = Field(..., description="Timestamps analyzed")
    confidence_avg: Optional[float] = None


class DetectionToolOutput(BaseModel):
    statistics: list[DetectionStatistics] = Field(default_factory=list, description="List of detection statistics")