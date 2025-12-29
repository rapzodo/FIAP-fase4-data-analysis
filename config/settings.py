import os
from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH", "tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")
PDF_PATH = os.getenv("PDF_PATH", "tech-challenge/Tech Challenge - IADT - Fase 4.pdf")
FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "1"))
OUTPUT_DIR = "output"

AGENT_CONFIG = {
    "tech_challenge_interpretator": {
        "role": "Tech Challenge Requirements Analyst",
        "goal": "Extract and structure the tech challenge requirements from PDF documents",
        "backstory": "Expert in analyzing technical documentation and extracting key requirements, problems, solutions, and expectations from complex documents."
    },
    "facial_recognition": {
        "role": "Facial Recognition and Emotion Detection Specialist",
        "goal": "Identify faces and detect emotions from video frames with high accuracy",
        "backstory": "Specialized computer vision expert with deep knowledge in facial recognition systems and emotion detection algorithms."
    },
    "activity_detector": {
        "role": "Human Activity Recognition Specialist",
        "goal": "Detect and classify human activities from video sequences",
        "backstory": "Expert in computer vision and pose estimation, capable of identifying complex human activities and movements in video streams."
    },
    "summarizer": {
        "role": "Video Analysis Report Generator",
        "goal": "Generate comprehensive summary reports of video analysis results",
        "backstory": "Data analyst specialized in aggregating and presenting complex video analysis data in clear, actionable reports."
    },
    "demo_video_script": {
        "role": "Technical Demo Script Writer",
        "goal": "Create engaging demonstration scripts that showcase application capabilities",
        "backstory": "Technical writer with expertise in creating compelling demonstration scripts that highlight key features and functionalities."
    }
}

