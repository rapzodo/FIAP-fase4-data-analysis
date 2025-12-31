# Module 4: Summarizer Agent

**Time**: 30 minutes | **Difficulty**: Easy

## 🎯 What You'll Build

An agent that aggregates data from multiple agents and generates:
- Executive summary
- Statistical analysis
- Key insights and patterns
- Anomaly report
- Comprehensive markdown report

## 📋 Files to Create

```
agents/summarizer_agent.py
tests/test_summarizer_agent.py
```

## 💻 Implementation

### Step 1: Create the Agent

**File**: `agents/summarizer_agent.py`

```python
from crewai import Agent
from config.llm_config import llm_config


def create_summarizer_agent():
    """Create agent specialized in data aggregation and report generation."""
    
    return Agent(
        role="Video Analysis Report Generator",
        goal="Aggregate multi-agent analysis results and generate comprehensive summary reports",
        backstory=(
            "You are a data analyst specializing in synthesizing complex information "
            "from multiple sources. You excel at identifying patterns, calculating statistics, "
            "and presenting insights in clear, actionable reports. You transform raw data "
            "from facial recognition, activity detection, and other analyses into "
            "executive-ready summaries with key metrics and visualizations."
        ),
        tools=[],  # No tools needed - works with provided data
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )
```

**Note**: This agent doesn't need tools - it receives data through task context from previous agents.

### Step 2: Create Test with Mock Data

**File**: `tests/test_summarizer_agent.py`

```python
from crewai import Crew, Task
from agents.summarizer_agent import create_summarizer_agent


def test_summarizer_agent():
    """Test summarizer with mock data from other agents."""
    
    print("📊 Testing Summarizer Agent\n")
    
    # Mock data simulating outputs from other agents
    pdf_analysis = """
    **Problem Statement**
    - Build a video analysis system for detecting facial expressions and activities
    - Need real-time processing capabilities
    - Must handle various lighting conditions
    
    **Proposed Solution**
    - Multi-agent system with specialized components
    - Facial recognition with emotion detection
    - Activity recognition with pose estimation
    
    **Expected Deliverables**
    - Working prototype
    - Analysis reports
    - Documentation
    """
    
    facial_data = """
    **Facial Recognition Results**
    - Frames Analyzed: 100
    - Faces Detected: 95
    - Detection Rate: 95%
    
    **Emotion Distribution:**
    - Happy: 35 (37%)
    - Neutral: 40 (42%)
    - Surprise: 15 (16%)
    - Sad: 5 (5%)
    
    **Anomalies:**
    - 5 frames without faces
    - 3 low confidence detections (<50%)
    
    **Confidence:** Average 78%
    """
    
    activity_data = """
    **Activity Detection Results**
    - Frames Analyzed: 100
    - Pose Detections: 98
    
    **Activity Distribution:**
    - Standing: 60 (60%)
    - Sitting: 25 (25%)
    - Moving: 15 (15%)
    
    **Hand Gestures:**
    - Hands Raised: 40 occurrences
    - Hands Down: 58 occurrences
    
    **Anomalies:**
    - 2 frames without pose detection
    """
    
    # Create agent
    agent = create_summarizer_agent()
    print("✅ Agent created\n")
    
    # Create task with context from "previous" agents
    task = Task(
        description=f"""
        Generate a comprehensive summary report aggregating the following analyses:
        
        **Tech Challenge Requirements:**
        {pdf_analysis}
        
        **Facial Recognition Analysis:**
        {facial_data}
        
        **Activity Detection Analysis:**
        {activity_data}
        
        Create a report with these sections:
        
        ## Executive Summary
        - Brief overview of video analysis
        - Key findings in 3-4 sentences
        
        ## Processing Statistics
        - Total frames processed
        - Success rates for each analysis type
        - Processing efficiency metrics
        
        ## Emotional Analysis
        - Emotion distribution with percentages
        - Dominant emotions identified
        - Emotional trends
        
        ## Activity Analysis
        - Activity breakdown with percentages
        - Movement patterns
        - Hand gesture frequency
        
        ## Anomaly Report
        - Total anomalies detected
        - Types of anomalies
        - Impact assessment
        
        ## Key Insights
        - 5 most important findings
        - Correlations between emotions and activities
        - Patterns and trends
        
        ## Alignment with Requirements
        - How results meet tech challenge objectives
        - Deliverables status
        
        Format in clean markdown with clear headings and bullet points.
        """,
        agent=agent,
        expected_output="Comprehensive markdown report with all requested sections"
    )
    print("✅ Task created\n")
    
    # Execute
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    
    print("🚀 Generating summary report...\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*70)
    print("📊 SUMMARY REPORT")
    print("="*70)
    print(result)
    print("="*70)
    
    # Save to file
    with open("output/test_summary_report.md", "w") as f:
        f.write(str(result))
    print("\n💾 Report saved to: output/test_summary_report.md")
    
    return result


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    test_summarizer_agent()
```

## 🧪 Testing

Run the test:
```bash
python tests/test_summarizer_agent.py
```

**What happens:**
1. Mock data simulates outputs from other agents
2. Summarizer agent receives all data
3. Agent analyzes and aggregates information
4. Generates structured markdown report
5. Report saved to `output/test_summary_report.md`

**Expected report structure:**
```markdown
## Executive Summary
The video analysis processed 100 frames with high success rates across 
facial recognition (95%) and activity detection (98%). Dominant emotions 
were neutral (42%) and happy (37%), while subjects spent most time 
standing (60%).

## Processing Statistics
- Total Frames Analyzed: 100
- Facial Recognition Success Rate: 95%
- Activity Detection Success Rate: 98%
- Overall Anomaly Rate: 10%

## Emotional Analysis
**Emotion Distribution:**
- Happy: 35 detections (37%)
- Neutral: 40 detections (42%)
- Surprise: 15 detections (16%)
- Sad: 5 detections (5%)

**Dominant Emotion:** Neutral, with significant happy expressions

## Activity Analysis
**Activity Breakdown:**
- Standing: 60 frames (60%)
- Sitting: 25 frames (25%)
- Moving: 15 frames (15%)

**Hand Gestures:**
- Total gestures: 98
- Raised hands: 40 (41%)
- Lowered hands: 58 (59%)

## Anomaly Report
**Total Anomalies:** 10
- 5 frames without facial detection
- 3 low confidence facial detections
- 2 frames without pose detection

**Impact:** Minimal, 90%+ detection rate maintained

## Key Insights
1. High overall detection rates indicate good video quality
2. Neutral and happy emotions suggest positive engagement
3. Predominant standing position indicates active participation
4. Frequent hand gestures suggest communication/presentation
5. Low anomaly rate demonstrates robust detection algorithms

## Alignment with Requirements
✅ Real-time processing capabilities demonstrated
✅ Handles lighting conditions (95% detection rate)
✅ Multi-agent coordination successful
✅ Analysis reports generated
✅ Documentation complete
```

## 💡 How It Works

### Data Flow
```
PDF Agent → Context
    ↓
Facial Agent → Context  → Summarizer → Report
    ↓                         ↑
Activity Agent → Context ─────┘
```

### Key Concepts

**1. Context Passing**
```python
# In real orchestration (Module 6)
task4 = Task(
    description="Summarize all analyses",
    agent=summarizer,
    context=[task1, task2, task3]  # Previous task outputs
)
```

**2. No Tools Needed**
- Summarizer works with text data
- LLM excels at analysis and aggregation
- No external services required

**3. Structured Output**
- Clear sections with headings
- Statistics and percentages
- Bullet points for readability
- Markdown formatting

## 🐛 Troubleshooting

### Report is too generic
**Fix**: Provide more specific instructions in task description
```python
description="""
...
Include specific statistics:
- Exact percentages with 2 decimals
- Frame counts for each category
- Confidence intervals
"""
```

### Missing sections
**Fix**: Explicitly list all required sections
```python
"Create a report with EXACTLY these sections: 1. Executive Summary, 2. Statistics, ..."
```

### Numbers don't match
**Normal**: LLM may round or interpret data. For exact numbers, parse JSON directly.

### Report too long
**Fix**: Set length constraints
```python
"Limit Executive Summary to 4 sentences. Use bullet points."
```

## ✅ Verification Checklist

- [ ] Agent creates successfully
- [ ] Task processes mock data
- [ ] Report has all sections
- [ ] Statistics are present
- [ ] Insights make sense
- [ ] File saves to output/

## 🎯 Report Quality Tips

### Good Executive Summary
```
✅ "Analysis of 100 frames showed 95% facial detection with predominantly 
    neutral (42%) and happy (37%) emotions, while subjects remained 
    standing (60%) with frequent hand gestures."

❌ "The analysis was good and showed results."
```

### Good Statistics
```
✅ "Facial Recognition: 95/100 frames (95%)"
❌ "Most frames had faces"
```

### Good Insights
```
✅ "Standing position (60%) combined with raised hands (41%) suggests 
    presentation or teaching activity"

❌ "People were standing and using hands"
```

## 📊 Customization

### Add Visual Elements
```python
description="""
Include this ASCII chart for emotion distribution:
Happy    ███████████████ 37%
Neutral  ████████████████████ 42%
Surprise ███████ 16%
Sad      ██ 5%
"""
```

### Add Comparisons
```python
description="""
Compare to benchmarks:
- Industry standard: 85% detection
- Our result: 95% detection
- Performance: +10% above standard
"""
```

## 📝 Next Steps

**Module 5**: Build the Demo Script Agent for presentation content!

**Enhancements to try:**
- Export to PDF or HTML
- Add charts and graphs
- Include recommendations
- Generate executive vs technical versions

---

**Ready?** Move to `05-demo-script.md` to create presentation scripts! 🚀

