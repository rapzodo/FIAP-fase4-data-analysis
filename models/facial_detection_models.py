from typing import List

from pydantic import BaseModel, Field


class FaceDetectionInput(BaseModel):
    video_path: str = Field(..., description="Path to the video file")
    sample_rate: int = Field(5, description="Sample rate of the video")


class FaceLocation(BaseModel):
    top: int = Field(..., description="Top face location")
    right: int = Field(..., description="Right face location")
    bottom: int = Field(..., description="Bottom face location")
    left: int = Field(..., description="Left face location")


class FaceDetection(BaseModel):
    face_id: int = Field(..., description="ID of the face")
    location: FaceLocation = Field(..., description="Location of the face")
    frame: int = Field(..., description="Frame number of the face")
    timestamp: float = Field(..., description="Timestamp of the face")


class FacialDetectionResult(BaseModel):
    frames_analyzed: int = Field(..., description="Number of frames analyzed")
    faces_detected: List[FaceDetection] = Field(..., description="List of detected faces")
    total_faces: int = Field(..., description="Total number of faces")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "frames_analyzed": 100,
                    "faces_detected": [
                        {
                            "location": {"top": 0, "left": 0, "right": 0, "bottom": 0},
                            "timestamp": 0.03,
                            "frame": 1,
                            "face_id": 1,
                        }
                    ],
                    "total_faces": 1,
                }
            ]
        }

