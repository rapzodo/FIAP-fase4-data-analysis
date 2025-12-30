# 🎓 Multi-Agent Video Analysis - Coding Tutorials

**Level**: Intermediate | **Assumes**: CrewAI knowledge

## 🎯 Goal

Build a 5-agent video analysis system by coding each component yourself with step-by-step guidance.

## 📚 Modules (3-4 hours total)

### [Module 1: PDF Interpretator Agent](01-pdf-interpretator.md) - 30 min
- Build PDF parser tool
- Create agent that extracts tech requirements
- **Test**: Parse a PDF and extract structured data

### [Module 2: Facial Recognition Agent](02-facial-recognition.md) - 45 min  
- Video frame processing with OpenCV
- Face detection + emotion analysis with DeepFace
- **Test**: Detect faces and emotions in video

### [Module 3: Activity Detector Agent](03-activity-detector.md) - 45 min
- MediaPipe pose & hand tracking
- Activity classification logic
- **Test**: Identify standing/sitting/moving from video

### [Module 4: Summarizer Agent](04-summarizer.md) - 30 min
- Aggregate multi-agent outputs
- Generate statistical reports
- **Test**: Create summary from mock data

### [Module 5: Demo Script Agent](05-demo-script.md) - 20 min
- Context-aware script generation
- **Test**: Generate demo script

### [Module 6: Orchestration](06-orchestration.md) - 30 min
- Wire all agents together
- Task flow & error handling
- **Test**: Run full pipeline

## 📝 Tutorial Format

Each module has:
```
├── 🎯 What You'll Build
├── 📋 Code Structure (file to create)
├── 💻 Step-by-Step Implementation
├── 🧪 Test Your Code
└── 🐛 Common Issues
```

## 🚀 Prerequisites

Already have:
- [x] Python 3.10+
- [x] CrewAI knowledge
- [x] Groq API key

Need to set up:
```bash
# Install dependencies
pip install crewai litellm opencv-python face-recognition deepface mediapipe PyPDF2

# Configure .env
GROQ_API_KEY=your_key
CREWAI_TRACING_ENABLED=false
```

## 🎨 Code Philosophy

- **Clean & Simple**: No over-engineering
- **Type It**: Don't copy-paste, type to learn
- **Test First**: Verify each component works
- **Iterate**: Get it working, then improve

## 🏁 Start Here

**Next**: [Module 1 - PDF Interpretator](01-pdf-interpretator.md)

Build the first agent that reads PDF requirements! 📄

