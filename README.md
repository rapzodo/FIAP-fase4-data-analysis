# Multi-Agent Video Analysis System

A sophisticated multi-agent application built with CrewAI framework that analyzes videos for emotion detection and human activity recognition. The system uses Groq/OpenAI API for fast LLM inference with Ollama as a fallback option.

## 🚀 Quick Start with Streamlit UI

Run the interactive web interface:

```bash
streamlit run app.py
```

Or use the startup script:

```bash
./run_app.sh
```

The web interface allows you to:
- Upload video files (mp4, avi, mov, mkv)
- Configure frame sample rate (1-60 fps)
- Select pose detection model (LITE, FULL, HEAVY)
- View real-time analysis progress
- Access generated reports directly in the browser

## 🎯 Overview

This project implements a multi-agent system that:
- Performs emotion detection in videos using DeepFace
- Detects human activities, poses, and hand gestures using MediaPipe
- Generates comprehensive analysis reports for emotions and activities
- Translates reports to multiple languages (PT-BR, etc.)
- Uses advanced LLM reasoning for report generation

> **⚠️ Note**: This is a Proof of Concept (POC) project developed for educational purposes as part of FIAP Tech Challenge Phase 4. While functional, it is not designed for high-precision detection in production environments. The detection accuracy and performance can be significantly improved with fine-tuning, better model selection, and optimization of processing parameters.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrewAI Orchestrator                       │
│              (Sequential Process with Memory)                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                     ┌──────────────────┐
│    Emotion       │                     │    Activity      │
│    Detector      │                     │    Detector      │
│     Agent        │                     │     Agent        │
│  (DeepFace)      │                     │  (MediaPipe)     │
└──────────────────┘                     └──────────────────┘
        │                                           │
        └──────────────────┬────────────────────────┘
                           ▼
                ┌──────────────────────┐
                │   Emotions Report    │
                │    Writer Agent      │
                │  (Reasoning LLM)     │
                └──────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Activities Report   │
                │    Writer Agent      │
                │  (Reasoning LLM)     │
                └──────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Translator        │
                │      Agent           │
                │ (TranslateGemma)     │
                └──────────────────────┘
```

## 🤖 Agents

### 1. Emotions Detector Agent
- **Role**: Face Emotion Analysis Specialist
- **Tools**: Emotion Detection Tool (DeepFace)
- **Function**: Analyzes videos frame-by-frame to detect emotions (happy, sad, angry, surprise, neutral, fear, disgust)
- **Output**: JSON file with emotion statistics, timestamps, and anomalies

### 2. Activity Detector Agent
- **Role**: Human Activity Analysis Specialist
- **Tools**: Activity Detection Tool (MediaPipe Pose & Hands)
- **Function**: Identifies human activities, poses, and hand gestures (standing, sitting, moving, hands raised, arms open, etc.)
- **Output**: JSON file with activity statistics, timestamps, and anomalies

### 3. Emotions Report Writer Agent
- **Role**: Emotion Detection Data Analysis Report Writer
- **Function**: Creates detailed markdown reports from emotion detection data including statistics, timelines, and insights
- **Features**: Uses reasoning capabilities for better analysis
- **Output**: Comprehensive emotion analysis report in Markdown

### 4. Activities Report Writer Agent
- **Role**: Human Activity Data Analysis Report Writer
- **Function**: Generates detailed markdown reports from activity detection data with movement patterns and gesture analysis
- **Features**: Uses reasoning capabilities for better analysis
- **Output**: Comprehensive activity analysis report in Markdown

### 5. Translator Agent
- **Role**: Technical Report Translator
- **Function**: Translates both emotion and activity reports to requested languages (PT-BR, etc.) while preserving formatting
- **Model**: TranslateGemma LLM
- **Output**: Translated summary report in target language

## 🛠️ Tech Stack

- **Framework**: CrewAI 0.70+
- **LLM Providers**: 
  - Groq API (primary) with models like llama-3.3-70b-versatile
  - OpenAI API (alternative) with models like o3-mini
  - Ollama (fallback) with local models
- **Reasoning Models**: Gemma3n for report generation
- **Translation**: TranslateGemma for multilingual reports
- **Embeddings**: Qwen3-Embedding for memory storage
- **Computer Vision**: 
  - DeepFace for emotion detection
  - MediaPipe (Pose & Hands) for activity detection
  - OpenCV for video processing
- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow/Keras for DeepFace backend
- **Configuration**: YAML-based agent and task configuration

## 📋 Prerequisites

- Python 3.10 or higher
- **One of the following LLM providers**:
  - Groq API key (get it from [console.groq.com](https://console.groq.com)) - Recommended
  - OpenAI API key (get it from [platform.openai.com](https://platform.openai.com))
  - Ollama installed locally (download from [ollama.ai](https://ollama.ai))
- Video file for analysis

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

Create a `.env` file in the project root with the following variables:

```env
# LLM Provider Configuration (choose one or both)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
USE_GROQ=true
USE_OPENAI=false

# Or use Ollama (leave API keys empty)
OLLAMA_BASE_URL=http://localhost:11434

# Video Processing Configuration
VIDEO_PATH=tech-challenge/video.mp4
FRAME_SAMPLE_RATE=30

# Optional Configuration
AGENTS_CONFIG_PATH=config/agents.yml
TASKS_CONFIG_PATH=config/tasks.yml
CREWAI_TRACING_ENABLED=false
```

**Note**: If you use Ollama, make sure the following models are installed:
```bash
ollama pull llama3.1:latest
ollama pull gemma3n:latest
ollama pull translategemma:latest
ollama pull qwen3-embedding:8b
```

## 🎮 Usage

### Run the complete analysis

```bash
python main.py
```

This will execute the following workflow:
1. **Detect Emotions**: Analyze the video for facial emotions using DeepFace
2. **Detect Activities**: Identify human activities, poses, and hand gestures using MediaPipe
3. **Generate Emotions Report**: Create a detailed markdown report of emotion detection results
4. **Generate Activities Report**: Create a detailed markdown report of activity detection results
5. **Translate Reports**: Combine and translate both reports to PT-BR (or other languages)

### Output Structure

All outputs are saved in organized directories:
- `knowledge/` - Raw detection data in JSON format
  - `facial_analysis_output.json` - Emotion detection statistics
  - `activity_analysis_output.json` - Activity detection statistics
- `reports/` - Individual analysis reports
  - `emotions-report_*.md` - Emotion analysis report
  - `activity-report_*.md` - Activity analysis report
- `summary/` - Translated combined reports
  - `pt-br-summary-report_*.md` - Portuguese translation of both reports

## 📊 Sample Output

The system provides detailed analysis including:
- **Frame Analysis**: Total frames processed from video
- **Emotion Detection**: Distribution of emotions detected (happy, sad, angry, etc.)
  - Emotion breakdown with percentages
  - Emotion patterns and timeline
  - Timestamps for each emotion occurrence
- **Activity Detection**: Timeline of activities (standing, sitting, moving, gestures)
  - Activity breakdown with percentages
  - Movement patterns analysis
  - Hand gestures analysis (hands raised, arms open, etc.)
- **Anomalies**: Detection failures, low confidence scores, unknown poses
- **Statistics**: Confidence scores, processing details, success rates
- **Key Insights**: Top emotions/activities and notable observations

## 🔧 Configuration

### LLM Configuration

The system automatically uses Groq or OpenAI API if configured, otherwise falls back to Ollama:

```env
# Use Groq API (recommended)
USE_GROQ=true
USE_OPENAI=false
GROQ_API_KEY=your_key_here

# Or use OpenAI API
USE_GROQ=false
USE_OPENAI=true
OPENAI_API_KEY=your_key_here

# Or use Ollama (local)
USE_GROQ=false
USE_OPENAI=false
OLLAMA_BASE_URL=http://localhost:11434
```

### Video Processing

Adjust frame sampling rate to balance speed vs. accuracy:

```env
FRAME_SAMPLE_RATE=1  # Process every frame (slowest, most accurate)
FRAME_SAMPLE_RATE=30  # Process every 30th frame (faster, good balance)
```

### Agent and Task Configuration

Agents and tasks are configured via YAML files in the `config/` directory:
- `config/agents.yml` - Agent definitions (roles, goals, backstories)
- `config/tasks.yml` - Task definitions (descriptions, expected outputs)

## 📁 Project Structure

```
FIAP-fase4-reconhecimento-facial/
├── config/                      # Configuration files
│   ├── agents.yml              # Agent definitions
│   ├── tasks.yml               # Task definitions
│   ├── llm_config.py           # LLM setup (Groq/OpenAI/Ollama)
│   └── settings.py             # Application settings
├── crew/                        # CrewAI crew implementation
│   └── video_analysis_summary_crew.py
├── tools/                       # Custom CrewAI tools
│   ├── emotion_detection_tool.py    # DeepFace emotion detection
│   └── activity_detection_tool.py   # MediaPipe activity detection
├── models/                      # Pydantic data models
│   ├── base_models.py
│   ├── emotion_detection_models.py
│   └── activity_detection_models.py
├── guardrails/                  # Error handling and validation
│   └── guardrails_functions.py
├── helper/                      # Helper functions
│   └── helper_functions.py
├── listeners/                   # Event listeners
│   └── knowledge_event_listener.py
├── knowledge/                   # Detection results (JSON)
│   ├── facial_analysis_output.json
│   └── activity_analysis_output.json
├── reports/                     # Individual reports (Markdown)
│   ├── emotions-report_*.md
│   └── activity-report_*.md
├── summary/                     # Translated summary reports
│   └── pt-br-summary-report_*.md
├── tech-challenge/              # Input video files
├── media_pipe/                  # MediaPipe models
│   └── pose_models/
├── crewai_storage/             # CrewAI memory storage
│   ├── long_term_memory_storage.db
│   └── latest_kickoff_task_outputs.db
├── tests/                       # Unit tests
├── main.py                      # Main orchestrator
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

## 🔍 How It Works

1. **Emotion Detection**: The video is processed frame-by-frame using DeepFace to detect facial emotions. Each frame is analyzed for dominant emotions (happy, sad, angry, fear, surprise, neutral, disgust) with confidence scores and timestamps.

2. **Activity Detection**: MediaPipe Pose and Hands models analyze body positions and hand gestures to classify activities such as standing, sitting, moving, hands raised, arms open, etc.

3. **Report Generation**: Two report writer agents use reasoning-enabled LLMs (Gemma3n) to analyze the detection data and create comprehensive markdown reports with:
   - Statistical summaries
   - Timeline of emotions/activities
   - Pattern analysis
   - Key insights and anomalies

4. **Translation**: The translator agent (TranslateGemma) combines both reports and translates them to the requested language (PT-BR by default) while preserving formatting and structure.

5. **Memory**: The system uses CrewAI's memory features with Qwen3-Embedding to maintain context across agent interactions and improve analysis quality.

## 🚧 Future Improvements

This POC can be enhanced in several ways:

### Detection Accuracy
- Fine-tune emotion detection models for specific use cases
- Implement ensemble methods for more robust predictions
- Add face tracking to maintain identity across frames
- Improve lighting normalization and face preprocessing

### Activity Recognition
- Train custom MediaPipe models for specific activities
- Add more sophisticated gesture recognition patterns
- Implement temporal smoothing for activity sequences
- Support multi-person tracking and analysis

### Performance Optimization
- Implement GPU acceleration for faster processing
- Add parallel processing for multiple videos
- Optimize frame sampling strategies dynamically
- Cache intermediate results for faster re-analysis

### Features
- Real-time video stream processing
- Web interface for video upload and analysis
- Interactive visualization dashboards
- Export to additional formats (JSON, CSV, PDF)
- Support for multiple languages in translation
- Customizable report templates

### Deployment
- Containerization with Docker
- API endpoint for remote processing
- Batch processing capabilities
- Cloud deployment options

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is part of the FIAP Tech Challenge - Phase 4.

## 👥 Authors

- Danilo de Castro

## 🙏 Acknowledgments

- CrewAI framework for multi-agent orchestration
- Groq for fast LLM inference
- OpenAI for advanced language models
- DeepFace for emotion detection
- MediaPipe for pose and hand tracking
- OpenCV for video processing

