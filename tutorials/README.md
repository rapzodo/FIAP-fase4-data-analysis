# 🎓 Multi-Agent Video Analysis - Coding Tutorials

**Level**: Intermediate | **Assumes**: CrewAI knowledge

## 🎯 Goal

Build a 3-agent video analysis system by coding each component yourself with step-by-step guidance. Focus on video processing with clean, testable code.

## 📚 Modules (3 hours total)

### [Module 1: YAML Configuration Setup](01-yaml-setup.md) - 25 min
- YAML-based agent and task configuration
- Pydantic output schemas for structured task outputs
- Factory pattern for creating agents/tasks
- **Test**: unittest for config validation

### [Module 2: Facial Recognition Agent](02-facial-detection.md) - 45 min  
- Video frame processing with OpenCV
- Face detection + emotion analysis with DeepFace
- **Test**: unittest for tool functionality

### [Module 3: Activity Detector Agent](03-activity-detector.md) - 45 min
- MediaPipe pose & hand tracking
- Activity classification logic
- **Test**: unittest for pose detection

### [Module 4: Summarizer Agent](04-summarizer.md) - 30 min
- Aggregate multi-agent outputs
- Generate statistical reports
- **Test**: unittest with mock data

### [Module 5: Orchestration](05-orchestration.md) - 30 min
- Wire all agents together
- Task flow & error handling
- **Test**: Integration tests

### [Module 6: Pydantic Output Best Practices](07-pydantic-outputs.md) - 15 min
- When to use Pydantic vs JSON vs Text outputs
- Decision matrix and patterns
- Tool-level vs Task-level validation
- Real-world examples from video analysis system

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

