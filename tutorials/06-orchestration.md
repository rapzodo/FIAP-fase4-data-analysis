# Module 6: Orchestration - Wiring Everything Together

**Time**: 30 minutes | **Difficulty**: Medium

## 🎯 What You'll Build

The main orchestrator that:
- Coordinates all 5 agents
- Defines task execution flow
- Manages context between agents
- Handles errors gracefully
- Generates final outputs

## 📋 Files to Create

```
main_orchestrator.py
config/settings.py (update)
tests/test_full_system.py
```

## 💻 Implementation

### Step 1: Update Settings

**File**: `config/settings.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# File paths
VIDEO_PATH = os.getenv("VIDEO_PATH", "tech-challenge/video.mp4")
PDF_PATH = os.getenv("PDF_PATH", "tech-challenge/requirements.pdf")

# Processing settings
FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "10"))

# Output settings
OUTPUT_DIR = "output"
REPORT_FILENAME = "analysis_report"
DEMO_SCRIPT_FILENAME = "demo_script"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### Step 2: Create Main Orchestrator

**File**: `main_orchestrator.py`

```python
from crewai import Crew, Task, Process
from agents.pdf_interpretator import create_pdf_interpretator
from agents.facial_recognition_agent import create_facial_recognition_agent
from agents.activity_detector_agent import create_activity_detector_agent
from agents.summarizer_agent import create_summarizer_agent
from agents.demo_script_agent import create_demo_script_agent
from config.settings import VIDEO_PATH, PDF_PATH, FRAME_SAMPLE_RATE, OUTPUT_DIR
from datetime import datetime
import os


class VideoAnalysisOrchestrator:
    """Orchestrates the multi-agent video analysis system."""
    
    def __init__(self):
        """Initialize all agents."""
        print("🤖 Initializing agents...")
        
        self.pdf_agent = create_pdf_interpretator()
        self.facial_agent = create_facial_recognition_agent()
        self.activity_agent = create_activity_detector_agent()
        self.summarizer_agent = create_summarizer_agent()
        self.demo_agent = create_demo_script_agent()
        
        print("✅ All agents initialized\n")
    
    def create_tasks(self):
        """Create all tasks with proper dependencies."""
        print("📋 Creating tasks...")
        
        # Task 1: Extract PDF requirements
        task1 = Task(
            description=f"""
            Analyze the tech challenge PDF at '{PDF_PATH}'.
            
            Extract and structure:
            1. Problem Statement
            2. Proposed Solution
            3. Expected Deliverables
            
            Format as clear sections with bullet points.
            """,
            agent=self.pdf_agent,
            expected_output="Structured requirements from PDF"
        )
        
        # Task 2: Facial recognition analysis
        task2 = Task(
            description=f"""
            Analyze the video at '{VIDEO_PATH}' for faces and emotions.
            
            Use facial_recognition tool with sample_rate={FRAME_SAMPLE_RATE}.
            
            Provide:
            1. Total frames analyzed
            2. Faces detected with timestamps
            3. Emotion distribution
            4. Confidence scores
            5. Anomalies detected
            
            Format as structured data report.
            """,
            agent=self.facial_agent,
            expected_output="Facial recognition analysis with emotion data",
            context=[task1]  # Has context from PDF task
        )
        
        # Task 3: Activity detection analysis
        task3 = Task(
            description=f"""
            Analyze the video at '{VIDEO_PATH}' for human activities.
            
            Use activity_detector tool with sample_rate={FRAME_SAMPLE_RATE}.
            
            Identify:
            1. Total frames analyzed
            2. Activities detected (standing, sitting, moving)
            3. Hand gestures
            4. Activity timeline
            5. Anomalies
            
            Format as structured activity report.
            """,
            agent=self.activity_agent,
            expected_output="Activity detection analysis with pose data",
            context=[task1]  # Has context from PDF task
        )
        
        # Task 4: Generate summary report
        task4 = Task(
            description="""
            Create a comprehensive summary report aggregating all analyses.
            
            Include these sections:
            
            ## Executive Summary
            - Overview in 3-4 sentences
            - Key findings
            
            ## Processing Statistics
            - Frames analyzed per agent
            - Success rates
            - Processing metrics
            
            ## Facial Recognition Results
            - Detection rate
            - Emotion distribution
            - Confidence metrics
            
            ## Activity Detection Results
            - Activity breakdown
            - Movement patterns
            - Hand gesture analysis
            
            ## Anomaly Report
            - Total anomalies
            - Types and frequencies
            - Impact assessment
            
            ## Key Insights
            - Top 5 findings
            - Patterns and correlations
            - Notable observations
            
            ## Tech Challenge Alignment
            - Requirements coverage
            - Deliverables status
            - Success metrics
            
            Format in clean markdown with clear headings.
            """,
            agent=self.summarizer_agent,
            expected_output="Comprehensive markdown summary report",
            context=[task1, task2, task3]  # Aggregates all previous tasks
        )
        
        # Task 5: Generate demo script
        task5 = Task(
            description="""
            Create a 3-5 minute demonstration video script.
            
            Script structure:
            1. Opening hook (30s) - Problem and preview
            2. System overview (45s) - Architecture and innovation
            3. Agent demos (2-3 min) - Each agent showcased
            4. Results & impact (30s) - Metrics and use cases
            5. Closing (30s) - Call to action
            
            For each scene include:
            - Scene name and duration
            - Visual description
            - Narration text
            - Demo actions
            
            Make it engaging and showcase key capabilities.
            """,
            agent=self.demo_agent,
            expected_output="Complete video demonstration script",
            context=[task1, task4]  # Uses requirements and summary
        )
        
        print("✅ All tasks created\n")
        return [task1, task2, task3, task4, task5]
    
    def run(self):
        """Execute the entire analysis pipeline."""
        print("🚀 Starting Video Analysis System")
        print("="*70)
        print(f"📹 Video: {VIDEO_PATH}")
        print(f"📄 PDF: {PDF_PATH}")
        print(f"🎯 Frame Sample Rate: {FRAME_SAMPLE_RATE}")
        print("="*70 + "\n")
        
        # Validate inputs
        if not os.path.exists(VIDEO_PATH):
            print(f"❌ Error: Video not found at {VIDEO_PATH}")
            return None
        
        if not os.path.exists(PDF_PATH):
            print(f"❌ Error: PDF not found at {PDF_PATH}")
            return None
        
        # Create tasks
        tasks = self.create_tasks()
        
        # Create crew
        print("🎭 Assembling crew...")
        crew = Crew(
            agents=[
                self.pdf_agent,
                self.facial_agent,
                self.activity_agent,
                self.summarizer_agent,
                self.demo_agent
            ],
            tasks=tasks,
            process=Process.sequential,  # Tasks run in order
            verbose=True
        )
        print("✅ Crew assembled\n")
        
        # Execute
        print("▶️  Starting execution...")
        print("⏱️  This will take 5-15 minutes depending on video length\n")
        
        try:
            result = crew.kickoff()
            
            print("\n" + "="*70)
            print("✅ EXECUTION COMPLETE")
            print("="*70)
            
            # Save outputs
            self.save_outputs(result)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            print("\n💡 Troubleshooting:")
            print("  1. Check video and PDF paths")
            print("  2. Verify all dependencies installed")
            print("  3. Check Groq API key in .env")
            print("  4. Review error trace above")
            raise
    
    def save_outputs(self, result):
        """Save final outputs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main report
        report_file = os.path.join(OUTPUT_DIR, f"analysis_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(str(result))
        
        print(f"\n💾 Report saved to: {report_file}")
        
        # Extract and save demo script if present
        # (The last task output is the demo script)
        demo_file = os.path.join(OUTPUT_DIR, f"demo_script_{timestamp}.md")
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write("# Demo Video Script\n\n")
            f.write(str(result))
        
        print(f"💾 Demo script saved to: {demo_file}")
        print("\n" + "="*70)
        print("📊 OUTPUT FILES")
        print("="*70)
        print(f"📄 Analysis Report: {report_file}")
        print(f"🎬 Demo Script: {demo_file}")
        print("="*70)


def main():
    """Main entry point."""
    orchestrator = VideoAnalysisOrchestrator()
    result = orchestrator.run()
    return result


if __name__ == "__main__":
    main()
```

## 🔑 Key Concepts

### 1. Task Dependencies (Context)

```python
task4 = Task(
    description="Summarize everything",
    agent=summarizer,
    context=[task1, task2, task3]  # Gets outputs from these tasks
)
```

**What happens:**
- CrewAI passes outputs from task1, task2, task3 to task4
- Summarizer agent receives all previous data
- Agent can reference and analyze combined information

### 2. Sequential Process

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential  # One task at a time, in order
)
```

**Execution flow:**
```
Task 1 (PDF) → Complete
    ↓
Task 2 (Facial) → Complete
    ↓
Task 3 (Activity) → Complete
    ↓
Task 4 (Summarizer) → Complete
    ↓
Task 5 (Demo Script) → Complete
```

### 3. Error Handling

```python
try:
    result = crew.kickoff()
except Exception as e:
    # Log error
    # Provide troubleshooting
    # Save partial results if possible
```

## 🧪 Testing

### Step 3: Create Full System Test

**File**: `tests/test_full_system.py`

```python
from main_orchestrator import VideoAnalysisOrchestrator
import os


def test_full_system():
    """Test the complete multi-agent system."""
    
    print("🧪 Testing Full Video Analysis System\n")
    
    # Check prerequisites
    print("📋 Checking prerequisites...")
    
    video_path = os.getenv("VIDEO_PATH", "tech-challenge/video.mp4")
    pdf_path = os.getenv("PDF_PATH", "tech-challenge/requirements.pdf")
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("💡 Update VIDEO_PATH in .env")
        return
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("💡 Update PDF_PATH in .env")
        return
    
    print("✅ Prerequisites OK\n")
    
    # Create orchestrator
    print("🎭 Creating orchestrator...")
    orchestrator = VideoAnalysisOrchestrator()
    
    # Run full pipeline
    print("\n🚀 Running full analysis pipeline...")
    print("⏱️  Expected time: 5-15 minutes\n")
    
    result = orchestrator.run()
    
    if result:
        print("\n✅ Full system test PASSED")
        print("📊 Check output/ directory for results")
    else:
        print("\n❌ Full system test FAILED")
        print("💡 Review error messages above")


if __name__ == "__main__":
    test_full_system()
```

### Run Full System

```bash
python tests/test_full_system.py
```

Or directly:
```bash
python main_orchestrator.py
```

## 📊 Execution Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                 Start Orchestrator                   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
           ┌──────────────────┐
           │  Initialize       │
           │  All 5 Agents     │
           └─────────┬─────────┘
                     │
                     ▼
           ┌──────────────────┐
           │  Task 1: PDF      │──┐
           │  Interpretator    │  │
           └─────────┬─────────┘  │
                     │             │
                     ▼             │
           ┌──────────────────┐   │ Context
           │  Task 2: Facial   │◄─┘
           │  Recognition      │──┐
           └─────────┬─────────┘  │
                     │             │
                     ▼             │
           ┌──────────────────┐   │ Context
           │  Task 3: Activity │◄─┘
           │  Detection        │──┐
           └─────────┬─────────┘  │
                     │             │
                     ▼             │
           ┌──────────────────┐   │
           │  Task 4:          │◄─┤ Context
           │  Summarizer       │◄─┤ (All 3)
           └─────────┬─────────┘◄─┘
                     │
                     ▼
           ┌──────────────────┐
           │  Task 5: Demo     │◄─ Context
           │  Script           │   (PDF + Summary)
           └─────────┬─────────┘
                     │
                     ▼
           ┌──────────────────┐
           │  Save Outputs     │
           │  - Report.md      │
           │  - Script.md      │
           └─────────┬─────────┘
                     │
                     ▼
           ┌──────────────────┐
           │     Complete      │
           └───────────────────┘
```

## 🐛 Troubleshooting

### "Video not found"
**Fix**: Update `.env` with correct path
```env
VIDEO_PATH=path/to/your/video.mp4
```

### Very slow execution
**Normal**: 
- PDF parsing: 10-30 seconds
- Facial analysis: 2-5 minutes
- Activity analysis: 2-5 minutes
- Summarizer: 30-60 seconds
- Demo script: 30-60 seconds

**Speed up**: Increase FRAME_SAMPLE_RATE
```env
FRAME_SAMPLE_RATE=20  # Process every 20th frame
```

### Agent fails mid-execution
**Check**:
1. Which task failed? (check verbose output)
2. Tool error or LLM error?
3. Input data valid?

**Recover**: Tasks are independent, can debug/rerun specific ones

### Out of memory
**Fix**: 
- Close other applications
- Increase sample rate
- Process shorter video clips
- Use smaller video resolution

## ✅ Verification Checklist

- [ ] All agents initialize
- [ ] PDF task completes
- [ ] Facial analysis completes
- [ ] Activity analysis completes
- [ ] Summary generates
- [ ] Demo script generates
- [ ] Files save to output/
- [ ] No errors in execution

## 🎯 Optimization Tips

### For Speed
```python
# In settings.py
FRAME_SAMPLE_RATE = 30  # Faster, less accurate

# In orchestrator
# Add time limits to tasks
task2 = Task(
    ...,
    max_execution_time=300  # 5 minute limit
)
```

### For Accuracy
```python
# In settings.py
FRAME_SAMPLE_RATE = 1   # Every frame, slower

# In task descriptions
"Analyze with high precision, examine every detail"
```

### For Production
```python
# Add logging
import logging
logging.basicConfig(level=logging.INFO)

# Add progress tracking
from tqdm import tqdm

# Add retry logic
from tenacity import retry, stop_after_attempt
```

## 📝 Output Examples

### Analysis Report Structure
```markdown
# Video Analysis Report
Generated: 2024-12-30 14:30:00

## Executive Summary
...

## Processing Statistics
- Total Frames: 300
- Facial Detection: 285/300 (95%)
- Activity Detection: 295/300 (98%)
...

## Facial Recognition Results
...

## Activity Detection Results
...

## Anomaly Report
...

## Key Insights
...

## Tech Challenge Alignment
...
```

### Demo Script Structure
```markdown
# Demo Video Script
Duration: 4 minutes 30 seconds

SCENE 1: Opening
...

SCENE 2: Overview
...

[etc...]
```

## 🎉 Success!

You now have a complete multi-agent video analysis system!

## 📚 What You've Built

1. **5 Specialized Agents**
   - PDF Interpretator
   - Facial Recognition
   - Activity Detector
   - Summarizer
   - Demo Script Generator

2. **Custom Tools**
   - PDF Parser
   - Facial Recognition
   - Activity Detector

3. **Orchestration**
   - Task dependencies
   - Context sharing
   - Error handling
   - Output management

## 🚀 Next Steps

### Enhancements
- Add more emotions
- Track multiple people
- Real-time processing
- Web interface
- API endpoints
- Database storage

### Deployment
- Docker container
- Cloud deployment
- CI/CD pipeline
- Monitoring & logging

### Advanced Features
- Batch processing
- Parallel execution
- Custom agent types
- Plugin system

---

## 🎓 Congratulations!

You've coded a complete multi-agent AI system from scratch!

**Key Skills Learned:**
- CrewAI agent development
- Custom tool creation
- Multi-agent orchestration
- Video processing
- Computer vision integration
- LLM-powered analysis

**Share your project:**
- Push to GitHub
- Write a blog post
- Create a demo video
- Present at meetups

---

**You did it!** 🎉🚀

