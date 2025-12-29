# Multi-Agent Video Analysis System

A sophisticated multi-agent application built with CrewAI framework that analyzes videos for facial recognition, emotion detection, and human activity recognition. The system uses Groq API for fast LLM inference with Ollama as a fallback option.

## 🎯 Overview

This project implements a multi-agent system that:
- Interprets tech challenge requirements from PDF documents
- Performs facial recognition and emotion detection in videos
- Detects human activities and poses in video streams
- Generates comprehensive analysis reports
- Creates demonstration video scripts

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrewAI Orchestrator                       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Tech        │   │    Facial        │   │    Activity      │
│  Challenge    │──▶│  Recognition     │   │    Detector      │
│ Interpretator │   │     Agent        │   │     Agent        │
└───────────────┘   └──────────────────┘   └──────────────────┘
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                              ┌──────────────────┐
                              │   Summarizer     │
                              │     Agent        │
                              └──────────────────┘
                                         │
                                         ▼
                              ┌──────────────────┐
                              │  Demo Video      │
                              │  Script Agent    │
                              └──────────────────┘
```

## 🤖 Agents

### 1. Tech Challenge Interpretator Agent
- **Role**: Tech Challenge Requirements Analyst
- **Tools**: PDF Parser Tool
- **Function**: Extracts problem statements, solutions, and requirements from PDF documents

### 2. Facial Recognition Agent
- **Role**: Facial Recognition and Emotion Detection Specialist
- **Tools**: Facial Recognition Tool (face_recognition + DeepFace)
- **Function**: Detects faces and identifies emotions (happy, sad, angry, surprise, neutral, fear, disgust)

### 3. Activity Detector Agent
- **Role**: Human Activity Recognition Specialist
- **Tools**: Activity Detector Tool (MediaPipe Pose & Hands)
- **Function**: Identifies human activities (standing, sitting, moving, hand gestures)

### 4. Summarizer Agent
- **Role**: Video Analysis Report Generator
- **Function**: Aggregates results from all agents and generates comprehensive reports

### 5. Demo Video Script Agent
- **Role**: Technical Demo Script Writer
- **Function**: Creates demonstration scripts showcasing the application's capabilities

## 🛠️ Tech Stack

- **Framework**: CrewAI 0.70+
- **LLM**: Groq API (llama-3.3-70b-versatile) with Ollama fallback
- **Computer Vision**: OpenCV, face_recognition, DeepFace, MediaPipe
- **Language**: Python 3.10+
- **Tools**: PyPDF2, numpy, pydantic

## 📋 Prerequisites

- Python 3.10 or higher
- Groq API key (get it from [console.groq.com](https://console.groq.com))
- Optional: Ollama installed locally for fallback LLM

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd FIAP-fase4-reconhecimento-facial
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env .env
```

Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
USE_GROQ=true
VIDEO_PATH=tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4
PDF_PATH=tech-challenge/Tech Challenge - IADT - Fase 4.pdf
FRAME_SAMPLE_RATE=1
```

## 🎮 Usage

### Run the complete analysis

```bash
python main.py
```

This will:
1. Parse the tech challenge PDF
2. Analyze the video for facial recognition and emotions
3. Detect activities in the video
4. Generate a comprehensive summary report
5. Create a demo video script

### Output

All outputs are saved in the `output/` directory:
- `analysis_report_YYYYMMDD_HHMMSS.md` - Comprehensive analysis report

## 📊 Sample Output

The system provides detailed analysis including:
- **Frame Analysis**: Total frames processed from video
- **Emotion Detection**: Distribution of emotions detected (happy, sad, angry, etc.)
- **Activity Detection**: Timeline of activities (standing, sitting, moving, gestures)
- **Anomalies**: Detection failures, low confidence scores, missing faces
- **Statistics**: Confidence scores, processing details, timestamps

## 🔧 Configuration

### LLM Configuration

The system automatically uses Groq API if configured, otherwise falls back to Ollama:

```python
# config/llm_config.py
USE_GROQ=true  # Use Groq API
USE_GROQ=false  # Use Ollama
```

### Video Processing

Adjust frame sampling rate to balance speed vs. accuracy:

```env
FRAME_SAMPLE_RATE=1  # Process every frame
FRAME_SAMPLE_RATE=5  # Process every 5th frame (faster)
```

## 📁 Project Structure

```
FIAP-fase4-reconhecimento-facial/
├── agents/                      # CrewAI agents
│   ├── tech_challenge_interpretator.py
│   ├── facial_recognition_agent.py
│   ├── activity_detector_agent.py
│   ├── summarizer_agent.py
│   └── demo_video_agent.py
├── config/                      # Configuration files
│   ├── llm_config.py           # LLM setup (Groq/Ollama)
│   └── settings.py             # Application settings
├── tools/                       # Custom CrewAI tools
│   ├── facial_recognition_tool.py
│   ├── activity_detector_tool.py
│   └── pdf_parser_tool.py
├── aula1/                       # Original facial recognition code
├── tech-challenge/              # Input files (PDF & video)
├── output/                      # Generated reports
├── main.py                      # Main orchestrator
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🔍 How It Works

1. **Tech Challenge Interpretation**: The PDF parser tool extracts text from the tech challenge document, and the LLM structures it into problem/solution/expectations.

2. **Facial Recognition**: The video is processed frame-by-frame. Face_recognition library detects faces, and DeepFace analyzes emotions for each detected face.

3. **Activity Detection**: MediaPipe Pose and Hands models identify body positions and hand gestures to classify activities.

4. **Summarization**: All agent outputs are aggregated, analyzed, and formatted into a comprehensive Markdown report.

5. **Demo Script**: Based on the tech challenge requirements and analysis results, a demonstration script is generated.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is part of the FIAP Tech Challenge - Phase 4.

## 👥 Authors

- Danilo de Castro

## 🙏 Acknowledgments

- CrewAI framework for multi-agent orchestration
- Groq for fast LLM inference
- OpenCV and face_recognition for computer vision
- DeepFace for emotion detection
- MediaPipe for pose and hand tracking

