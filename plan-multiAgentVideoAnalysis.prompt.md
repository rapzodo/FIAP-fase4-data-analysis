## Plan: Multi-Agent Video Analysis System with CrewAI Framework

Build a CrewAI-powered multi-agent application that processes tech challenge requirements from PDF, analyzes videos for facial recognition and activities, generates comprehensive summaries, creates demo scripts, and publishes to GitHub. Uses Groq for fast LLM inference with Ollama as fallback option.

### Steps

1. **Set up CrewAI framework and project structure** - Install `crewai`, `crewai-tools`, and `groq` packages, create agents directory structure ([agents/tech_challenge_interpretator.py](agents/tech_challenge_interpretator.py), [agents/facial_recognition_agent.py](agents/facial_recognition_agent.py), [agents/activity_detector_agent.py](agents/activity_detector_agent.py), [agents/summarizer_agent.py](agents/summarizer_agent.py), [agents/demo_video_agent.py](agents/demo_video_agent.py)), add [config/agents.yaml](config/agents.yaml) and [config/tasks.yaml](config/tasks.yaml) for agent/task definitions, update [requirements.txt](requirements.txt) with CrewAI, PyPDF2, opencv-python, deepface for emotions, ultralytics/mediapipe for activity detection

2. **Configure LLM integration with Groq and Ollama fallback** - Set up Groq API with llama-3.3-70b-versatile model in [config/llm_config.py](config/llm_config.py) for fast inference, add Ollama integration with llama3.2 as backup option, create environment variable management in [.env](.env) for GROQ_API_KEY, implement LLM factory pattern to switch between providers based on availability

3. **Refactor existing facial recognition into CrewAI agent** - Extract logic from [aula1/facial_recognition.py](aula1/facial_recognition.py) into `facial_recognition_agent` with custom tool for video frame processing, add emotion detection using DeepFace library to detect happy/sad/angry/surprise/neutral/fear emotions, implement frame counting and anomaly detection (face detection failures, low confidence scores), structure output as JSON with timestamps/persons/emotions/confidence scores for video at [tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4](tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4)

4. **Implement tech challenge interpretator and activity detector agents** - Create `tech_challenge_interpretator_agent` with PDFSearchTool to parse [tech-challenge/Tech Challenge - IADT - Fase 4.pdf](tech-challenge/Tech Challenge - IADT - Fase 4.pdf) and extract problem/solution/expectations using Groq LLM, build `activity_detector_agent` with custom tool using MediaPipe Pose/Hands or YOLO to detect activities (sitting/standing/walking/hand gestures) from video file, define CrewAI tasks in [config/tasks.yaml](config/tasks.yaml) with expected outputs

5. **Build summarizer and demo script generator agents** - Develop `summarizer_agent` that receives structured outputs from facial and activity agents via CrewAI task context, aggregate frame counts, detected emotions distribution, activities timeline, and anomaly statistics into markdown report, create `demo_video_agent` that generates demo script showcasing each agent's functionality and tech challenge alignment, define sequential task flow in CrewAI crew

6. **Create main orchestrator and prepare GitHub repository** - Build [main.py](main.py) with CrewAI Crew orchestrator defining all agents and tasks in execution order, add video path configuration pointing to tech-challenge folder, implement report output to [output/](output/) directory, create [.gitignore](.gitignore) excluding .venv, __pycache__, .env, *.mp4, write [README.md](README.md) with architecture diagram, setup instructions, and usage examples, initialize git repository and push to GitHub with proper commit structure

### Further Considerations

1. **Groq API Rate Limits** - Groq offers 30 requests/minute on free tier with fast inference (llama-3.3-70b-versatile is excellent choice). For this 5-agent use case, should be sufficient. If rate limits become an issue, automatically fallback to Ollama llama3.2 running locally.

2. **Video Processing Approach** - Process video at 1 FPS sampling rate to balance analysis depth with processing time. For typical video length, this provides sufficient data points while keeping Groq API calls manageable. Adjust sampling rate in configuration if needed.

3. **Emotion Detection Library** - Use DeepFace (supports multiple models: VGG-Face, Facenet, OpenFace) for robust emotion detection. Alternative: FER (lightweight but less accurate). DeepFace recommended for production quality results with your existing face_recognition setup.

