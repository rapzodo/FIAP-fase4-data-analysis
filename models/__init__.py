"""Pydantic models for multi-agent video analysis system."""

from .facial_detection_models import (
    FaceDetectionInput,
    FaceLocation,
    FaceDetection,
    Anomaly,
    FacialDetectionResult,
    FacialDetectionError
)

__all__ = [
    "FaceDetectionInput",
    "FaceLocation",
    "FaceDetection",
    "Anomaly",
    "FacialDetectionResult",
    "FacialDetectionError"
]

