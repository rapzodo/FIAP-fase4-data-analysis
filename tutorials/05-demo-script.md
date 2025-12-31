# Module 5: Demo Script Agent

**Time**: 20 minutes | **Difficulty**: Easy

## 🎯 What You'll Build

An agent that creates demonstration video scripts showcasing:
- System capabilities
- Each agent's functionality
- Sample outputs
- Key features and value propositions

## 📋 Files to Create

```
agents/demo_script_agent.py
tests/test_demo_script_agent.py
```

## 💻 Implementation

### Step 1: Create the Agent

**File**: `agents/demo_script_agent.py`

```python
from crewai import Agent
from config.llm_config import llm_config


def create_demo_script_agent():
    """Create agent specialized in creating demonstration scripts."""
    
    return Agent(
        role="Technical Demo Script Writer",
        goal="Create engaging demonstration scripts that showcase application capabilities",
        backstory=(
            "You are a technical content creator who excels at explaining complex "
            "systems in accessible ways. You create compelling demonstration scripts "
            "that highlight key features, show real examples, and communicate value. "
            "Your scripts are structured with scenes, narration, visual cues, and "
            "make technical concepts easy to understand for any audience."
        ),
        tools=[],  # No tools needed - creative content generation
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )
```

### Step 2: Create Test

**File**: `tests/test_demo_script_agent.py`

```python
from crewai import Crew, Task
from agents.demo_script_agent import create_demo_script_agent


def test_demo_script_agent():
    """Test demo script generation."""
    
    print("🎬 Testing Demo Script Agent\n")
    
    # System information to showcase
    system_info = """
    **Multi-Agent Video Analysis System**
    
    **Components:**
    1. PDF Interpretator - Extracts requirements from documents
    2. Facial Recognition - Detects faces and emotions (7 emotions)
    3. Activity Detector - Identifies human activities and gestures
    4. Summarizer - Generates comprehensive reports
    5. Demo Script Generator - Creates presentation content
    
    **Capabilities:**
    - 95%+ face detection accuracy
    - 7 emotion categories
    - Real-time pose estimation
    - Multi-agent coordination
    - Automated report generation
    
    **Tech Stack:**
    - CrewAI for orchestration
    - Groq LLM for intelligence
    - OpenCV + face_recognition for faces
    - DeepFace for emotions
    - MediaPipe for pose detection
    
    **Use Cases:**
    - Education (student engagement analysis)
    - Security (behavior monitoring)
    - Retail (customer sentiment analysis)
    - Healthcare (patient monitoring)
    """
    
    # Create agent
    agent = create_demo_script_agent()
    print("✅ Agent created\n")
    
    # Create task
    task = Task(
        description=f"""
        Create a demonstration video script (3-5 minutes) for our multi-agent system.
        
        **System Information:**
        {system_info}
        
        **Script Requirements:**
        
        1. **Opening (30 seconds)**
           - Hook the audience
           - State the problem being solved
           - Preview what they'll see
        
        2. **System Overview (45 seconds)**
           - Explain multi-agent architecture
           - Show visual diagram concept
           - Highlight key innovation
        
        3. **Agent Demonstrations (2-3 minutes)**
           For EACH agent, include:
           - Agent name and role
           - What it does (15 seconds)
           - Live demo description (30 seconds)
           - Sample output (visual description)
           
           Agents to demo:
           a) PDF Interpretator
           b) Facial Recognition
           c) Activity Detector
           d) Summarizer
           e) Demo Script Generator (meta!)
        
        4. **Results & Impact (30 seconds)**
           - Key metrics (accuracy, speed)
           - Real-world applications
           - Business value
        
        5. **Closing (30 seconds)**
           - Call to action
           - Next steps
           - Contact/resources
        
        **Format:**
        ```
        SCENE X: [Scene name]
        DURATION: [Time]
        
        VISUAL: [What's on screen]
        
        NARRATION:
        [What the presenter says]
        
        DEMO ACTION:
        [What happens during demo]
        ```
        
        Make it engaging, clear, and showcase our best features!
        """,
        agent=agent,
        expected_output="Complete video demonstration script with scenes, narration, and visual cues"
    )
    print("✅ Task created\n")
    
    # Execute
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    
    print("🚀 Generating demo script...\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*70)
    print("🎬 DEMO VIDEO SCRIPT")
    print("="*70)
    print(result)
    print("="*70)
    
    # Save to file
    with open("output/demo_script.md", "w") as f:
        f.write(str(result))
    print("\n💾 Script saved to: output/demo_script.md")
    
    return result


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    test_demo_script_agent()
```

## 🧪 Testing

Run the test:
```bash
python tests/test_demo_script_agent.py
```

**Expected script structure:**
```markdown
# Multi-Agent Video Analysis System - Demo Script

---

SCENE 1: Opening Hook
DURATION: 30 seconds

VISUAL: 
Montage of faces with emotion labels, people in various poses, 
overlaid with glowing AI network connections

NARRATION:
"What if your computer could understand not just what people look like, 
but how they feel and what they're doing? Today, we'll show you a 
multi-agent AI system that does exactly that - analyzing video in real-time 
to detect faces, read emotions, and identify activities. Let's dive in."

DEMO ACTION:
Video montage plays with dynamic transitions

---

SCENE 2: System Architecture
DURATION: 45 seconds

VISUAL:
Animated diagram showing 5 agents connected in a workflow:
PDF → Facial → Activity → Summarizer → Demo Script

NARRATION:
"Our system uses 5 specialized AI agents, each an expert in their domain. 
Think of it like a team: the PDF Interpretator reads requirements, 
Facial Recognition detects emotions, Activity Detector identifies movements, 
the Summarizer compiles everything, and yes - the Demo Script generator 
created this script! Each agent uses CrewAI for coordination and Groq's 
lightning-fast LLM for intelligence."

DEMO ACTION:
Diagram animates, showing data flowing between agents

---

SCENE 3A: PDF Interpretator Demo
DURATION: 45 seconds

VISUAL:
Split screen: PDF document on left, extracted requirements on right

NARRATION:
"First, meet our PDF Interpretator agent. It reads technical documents 
and extracts key information. Watch as it processes this challenge document."

DEMO ACTION:
- PDF appears
- Agent "reads" with scanning animation
- Requirements appear structured:
  * Problem: Need video analysis
  * Solution: Multi-agent system
  * Deliverables: Working prototype

NARRATION (continued):
"In seconds, it's extracted the problem, solution, and deliverables - 
perfectly structured for the team."

---

SCENE 3B: Facial Recognition Demo
DURATION: 45 seconds

VISUAL:
Video playing with real-time face boxes and emotion labels

NARRATION:
"Next, facial recognition. This agent detects faces and identifies 
7 different emotions with 95% accuracy."

DEMO ACTION:
Video plays showing:
- Green boxes around detected faces
- Emotion labels: "Happy (87%)", "Neutral (92%)"
- Confidence scores updating
- Emotion chart building on side

NARRATION (continued):
"Notice the real-time emotion detection - it's tracking happiness, 
surprise, even subtle sadness. The confidence scores show how certain 
the AI is. This runs on your laptop."

---

SCENE 3C: Activity Detector Demo
DURATION: 45 seconds

VISUAL:
Video with skeleton overlay tracking body movements

NARRATION:
"The Activity Detector uses MediaPipe to track 33 body points and 
identify what people are doing."

DEMO ACTION:
- Skeleton overlay appears on person
- Labels appear: "Standing", "Hands Raised", "Moving"
- Activity timeline builds below
- Hand gesture indicators flash

NARRATION (continued):
"See the skeleton tracking? It knows when someone's standing, sitting, 
or moving. It even tracks hand gestures - crucial for understanding 
presentations or interactions."

---

SCENE 3D: Summarizer Demo
DURATION: 30 seconds

VISUAL:
Data streams flowing into central report document

NARRATION:
"All this data goes to the Summarizer, which creates comprehensive reports."

DEMO ACTION:
- Facial data streams in (emotion percentages)
- Activity data streams in (pose breakdown)
- Report generates with:
  * Executive summary
  * Statistics table
  * Key insights
  * Visualizations

NARRATION (continued):
"Executive summaries, detailed statistics, insights - all automated. 
Ready for stakeholders in seconds."

---

SCENE 3E: Demo Script Generator (Meta!)
DURATION: 30 seconds

VISUAL:
This script appearing on screen, typing effect

NARRATION:
"And here's the fun part - the script you're hearing right now? 
Created by our Demo Script Agent. It understood the system, 
identified key features, and wrote this entire demonstration."

DEMO ACTION:
Script text types out on screen
Agent avatar appears with "I wrote this!" caption

NARRATION (continued):
"Meta, right? That's the power of multi-agent systems - agents 
working together to create something greater than the sum of their parts."

---

SCENE 4: Results & Impact
DURATION: 30 seconds

VISUAL:
Statistics and use case examples

NARRATION:
"The results speak for themselves: 95% detection accuracy, 
sub-second processing per frame, and scalable to any video length."

DEMO ACTION:
Text appears:
✓ 95%+ Face Detection Accuracy
✓ 7 Emotion Categories
✓ Real-time Processing
✓ Multi-agent Coordination

Use Cases:
• Education - Student engagement
• Security - Behavior monitoring
• Retail - Customer sentiment
• Healthcare - Patient monitoring

---

SCENE 5: Closing
DURATION: 30 seconds

VISUAL:
System logo with contact information

NARRATION:
"That's our Multi-Agent Video Analysis System. Want to see it in action 
on your videos? Check out the GitHub repository, read the documentation, 
and try it yourself. The future of video analysis is multi-agent - 
and it's here today."

DEMO ACTION:
Text appears:
📦 GitHub: github.com/your-repo
📚 Docs: your-docs-link
✉️  Contact: your-email

Fade to logo

---

TOTAL DURATION: 4 minutes 30 seconds
```

## 💡 Script Elements Explained

### Scene Structure
```
SCENE X: [Descriptive name]
DURATION: [Estimated time]
VISUAL: [What viewers see]
NARRATION: [What's spoken]
DEMO ACTION: [What happens/animations]
```

### Key Components

**1. Hook (First 10 seconds)**
- Grab attention immediately
- State the problem/opportunity
- Create curiosity

**2. Clear Sections**
- One concept per scene
- Smooth transitions
- Logical flow

**3. Show, Don't Tell**
- Describe visuals specifically
- Use concrete examples
- Demonstrate with real data

**4. Call to Action**
- Clear next steps
- Resources provided
- Easy to follow

## 🐛 Troubleshooting

### Script too technical
**Fix**: Add instruction to task
```python
"Write for a general audience. Avoid jargon. Explain technical terms."
```

### Scenes too long
**Fix**: Set duration limits
```python
"Each scene maximum 45 seconds. Total 4 minutes maximum."
```

### Missing visual descriptions
**Fix**: Emphasize visuals
```python
"For EACH scene, describe EXACTLY what's on screen. Be specific about colors, 
animations, text, diagrams."
```

### Generic content
**Fix**: Provide specific data
```python
"Use these exact statistics: 95% accuracy, 7 emotions, 33 body points tracked"
```

## ✅ Verification Checklist

- [ ] Script has clear scenes
- [ ] Each scene has visuals described
- [ ] Narration is engaging
- [ ] Technical concepts explained simply
- [ ] Demonstrates all 5 agents
- [ ] Includes call to action
- [ ] Total time reasonable (3-5 min)
- [ ] Saved to output file

## 🎯 Script Quality Tips

### Good Narration
```
✅ "Watch as it processes this document in real-time, extracting requirements 
    in under 2 seconds"
❌ "The agent processes the document"
```

### Good Visual Description
```
✅ "Green bounding boxes appear around each detected face, with emotion labels 
    'Happy (87%)' floating above in white text"
❌ "Faces are detected and labeled"
```

### Good Demo Action
```
✅ "- Video plays at 0:05
    - First face detected at 0:07 with 'Happy (92%)' label
    - Second face appears at 0:10 with 'Neutral (88%)' label
    - Emotion chart builds on right side showing distribution"
❌ "Video plays and faces are detected"
```

## 📝 Next Steps

**Module 6**: Wire everything together in the main orchestrator!

**Enhancements to try:**
- Create multiple script versions (technical vs non-technical)
- Add B-roll suggestions
- Include music/sound effect cues
- Generate shot list
- Create storyboard descriptions

---

**Ready for the final module?** Move to `06-orchestration.md` to connect all agents! 🚀

