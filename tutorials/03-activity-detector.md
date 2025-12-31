# Module 3: Activity Detector Agent

**Time**: 45 minutes | **Difficulty**: Medium

## 🎯 What You'll Build

An agent that identifies human activities in videos:
- Body pose detection (standing, sitting, moving)
- Hand gesture recognition (raised, lowered)
- Activity timeline
- Movement patterns

## 📋 Files to Create

```
tools/activity_detector_tool.py
agents/activity_detector_agent.py
tests/test_activity_agent.py
```

## 💻 Implementation

### Step 1: Create Activity Detector Tool

**File**: `tools/activity_detector_tool.py`

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import mediapipe as mp
from typing import List, Dict, Optional


class ActivityDetectorInput(BaseModel):
    """Input for activity detector tool."""
    video_path: str = Field(..., description="Path to video file")
    sample_rate: int = Field(5, description="Process every Nth frame")


class ActivityDetection(BaseModel):
    """Single activity detection result."""
    frame: int
    timestamp: float
    activities: List[str]  # e.g., ["standing", "hands_raised"]


class ActivityAnomaly(BaseModel):
    """Activity detection anomaly."""
    frame: int
    timestamp: float
    type: str
    details: Optional[str] = None


class ActivityDetectorResult(BaseModel):
    """Complete activity detection analysis result."""
    frames_analyzed: int
    activities: List[ActivityDetection]
    activity_summary: Dict[str, int]
    pose_detections: int
    hand_detections: int
    anomalies: List[ActivityAnomaly]
    
    class Config:
        json_schema_extra = {
            "example": {
                "frames_analyzed": 100,
                "activities": [
                    {
                        "frame": 1,
                        "timestamp": 0.03,
                        "activities": ["standing", "hands_down"]
                    }
                ],
                "activity_summary": {
                    "standing": 60,
                    "sitting": 25,
                    "moving": 15,
                    "hands_raised": 40,
                    "hands_down": 60
                },
                "pose_detections": 98,
                "hand_detections": 100,
                "anomalies": []
            }
        }


class ActivityDetectorError(BaseModel):
    """Error response."""
    error: str


class ActivityDetectorTool(BaseTool):
    name: str = "activity_detector"
    description: str = "Detects human poses, gestures, and activities in video"
    args_schema: type[BaseModel] = ActivityDetectorInput
    
    def _run(self, video_path: str, sample_rate: int = 5) -> str:
        """Analyze video for human activities."""
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        
        # Initialize result model
        result = ActivityDetectorResult(
            frames_analyzed=0,
            activities=[],
            activity_summary={},
            pose_detections=0,
            hand_detections=0,
            anomalies=[]
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error = ActivityDetectorError(error=f"Cannot open video: {video_path}")
            return error.model_dump_json(indent=2)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        analyzed_count = 0
        
        # Activity counters
        activity_counts = {
            "standing": 0,
            "sitting": 0,
            "moving": 0,
            "hands_raised": 0,
            "hands_down": 0,
            "unknown": 0
        }
        
        with mp_pose.Pose(min_detection_confidence=0.5) as pose, \
             mp_hands.Hands(min_detection_confidence=0.5) as hands:
            
            prev_shoulder_y = None
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Sample frames
                    if frame_count % sample_rate != 0:
                        continue
                    
                    analyzed_count += 1
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    # Convert to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect pose
                    pose_results = pose.process(rgb_frame)
                    hand_results = hands.process(rgb_frame)
                    
                    frame_activities = []
                    
                    # Analyze pose
                    if pose_results.pose_landmarks:
                        result.pose_detections += 1
                        landmarks = pose_results.pose_landmarks.landmark
                        
                        # ...existing code for pose detection...
                        
                        frame_activities.append(activity)
                        activity_counts[activity] += 1
                    
                    # Analyze hands
                    if hand_results.multi_hand_landmarks:
                        result.hand_detections += len(hand_results.multi_hand_landmarks)
                        
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            
                            if wrist.y < 0.5:  # Upper half of frame
                                frame_activities.append("hands_raised")
                                activity_counts["hands_raised"] += 1
                            else:
                                frame_activities.append("hands_down")
                                activity_counts["hands_down"] += 1
                    
                    # Record activities using Pydantic model
                    if frame_activities:
                        result.activities.append(ActivityDetection(
                            frame=frame_count,
                            timestamp=round(timestamp, 2),
                            activities=frame_activities
                        ))
                    else:
                        # No activity detected
                        activity_counts["unknown"] += 1
                        result.anomalies.append(ActivityAnomaly(
                            frame=frame_count,
                            timestamp=round(timestamp, 2),
                            type="no_pose_detected"
                        ))
            
            finally:
                cap.release()
        
        result.frames_analyzed = analyzed_count
        result.activity_summary = activity_counts
        
        return result.model_dump_json(indent=2)
```

**What this does:**
- Uses **Pydantic models** for type-safe results
- Uses **MediaPipe Pose** for body skeleton detection
- Uses **MediaPipe Hands** for hand tracking
- Classifies poses: standing/sitting/moving
- Detects hand positions: raised/lowered
- Tracks movement between frames

**Key techniques:**
- **Landmark detection**: 33 body points tracked
- **Geometric analysis**: Torso length indicates pose
- **Motion detection**: Compare frames for movement
- **Hand position**: Y-coordinate determines raised/lowered
- **Type safety**: Pydantic validates all data

### Step 2: Create the Agent

**File**: `agents/activity_detector_agent.py`

```python
from crewai import Agent
from tools.activity_detector_tool import ActivityDetectorTool
from config.llm_config import llm_config


def create_activity_detector_agent():
    """Create agent specialized in human activity recognition."""
    
    return Agent(
        role="Human Activity Recognition Specialist",
        goal="Identify and classify human activities, poses, and gestures in video footage",
        backstory=(
            "You are a computer vision expert specializing in human activity "
            "recognition and pose estimation. You analyze body movements, gestures, "
            "and activities to understand what people are doing in videos. "
            "You provide detailed insights about physical activities, movement patterns, "
            "and behavioral analysis."
        ),
        tools=[ActivityDetectorTool()],
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )
```

### Step 3: Create Test

**File**: `tests/test_activity_agent.py`

```python
import os
from crewai import Crew, Task
from agents.activity_detector_agent import create_activity_detector_agent


def test_activity_detector_agent():
    """Test activity detection on a video."""
    
    video_path = "tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("💡 Update video_path to your actual video file")
        return
    
    print("🏃 Testing Activity Detector Agent\n")
    
    # Create agent
    agent = create_activity_detector_agent()
    print("✅ Agent created\n")
    
    # Create task
    task = Task(
        description=f"""
        Analyze the video at '{video_path}' for human activities and poses.
        
        Use the activity_detector tool with sample_rate=10 for faster processing.
        
        Provide a comprehensive report including:
        1. Total frames analyzed
        2. Activity breakdown (standing, sitting, moving percentages)
        3. Hand gesture patterns (raised/lowered hands)
        4. Timeline of major activities
        5. Movement patterns and insights
        6. Any anomalies (frames without pose detection)
        
        Format as a structured analysis report.
        """,
        agent=agent,
        expected_output="Detailed activity recognition report with statistics and timeline"
    )
    print("✅ Task created\n")
    
    # Execute
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    
    print("🚀 Analyzing activities...\n")
    print("⏱️  This may take 2-5 minutes\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*70)
    print("🏃 ACTIVITY DETECTION ANALYSIS")
    print("="*70)
    print(result)
    print("="*70)
    
    return result


if __name__ == "__main__":
    test_activity_detector_agent()
```

## 🧪 Testing

Run the test:
```bash
python tests/test_activity_agent.py
```

**Expected output:**
```
🏃 ACTIVITY DETECTION ANALYSIS
======================================================================
**Activity Analysis Summary**
- Frames Analyzed: 50
- Pose Detections: 48 (96%)
- Hand Detections: 35

**Activity Distribution**
- Standing: 30 (60%)
- Sitting: 10 (20%)
- Moving: 8 (16%)
- Unknown: 2 (4%)

**Hand Gesture Analysis**
- Hands Raised: 15 occurrences
- Hands Down: 20 occurrences

**Activity Timeline**
- 0:00-0:15: Predominantly standing
- 0:15-0:30: Mix of sitting and standing
- 0:30-0:45: Active movement detected

**Key Insights**
- Subject primarily in standing position
- Regular hand gestures indicate communication
- Minimal seated time suggests active engagement

**Anomalies**
- 2 frames without pose detection (likely occlusion)
======================================================================
```

## 🐛 Troubleshooting

### Error: "No module named 'mediapipe'"
**Fix**: Install MediaPipe
```bash
pip install mediapipe
```

### Low detection rate
**Causes**:
- Poor lighting
- Partial body visibility
- Person too far from camera

**Fix**: Use videos with full-body visibility

### "No pose detected" anomalies
**Normal**: Happens when:
- Person leaves frame
- Back turned to camera
- Body partially occluded

### Slow processing
**Fix**: Increase sample_rate
```python
# In task description
Use the activity_detector tool with sample_rate=20
```

## 💡 Understanding MediaPipe

### Pose Landmarks
MediaPipe detects 33 body points:
- **Face**: nose, eyes, ears
- **Upper body**: shoulders, elbows, wrists
- **Lower body**: hips, knees, ankles

### How Pose Classification Works

```python
# Standing: Torso is extended (large distance)
torso_length = hip_y - shoulder_y
if torso_length > 0.3:  # Standing

# Sitting: Torso is compressed (small distance)
if torso_length < 0.15:  # Sitting

# Moving: Shoulder position changes between frames
if abs(current_y - previous_y) > threshold:  # Moving
```

### Hand Detection
- **Wrist Y < 0.5**: Hand in upper half (raised)
- **Wrist Y > 0.5**: Hand in lower half (down)

## ✅ Verification Checklist

- [ ] Tool detects poses in video
- [ ] Activities are classified
- [ ] Hand gestures are tracked
- [ ] Agent generates report
- [ ] Timeline makes sense
- [ ] Statistics are reasonable

## 🎯 Activity Categories

### Body Poses
- **Standing**: Upright posture, extended torso
- **Sitting**: Compressed torso, bent at hips
- **Moving**: Position changes between frames

### Hand Gestures
- **Hands Raised**: Gesturing, waving, pointing
- **Hands Down**: Resting position, by sides

### Future Extensions
- Walking vs standing
- Specific gestures (waving, pointing)
- Multiple people tracking
- Action recognition (jumping, bending)

## 📝 Next Steps

**Module 4**: Build the Summarizer Agent to aggregate results!

**Enhancements to try:**
- Add more activity categories
- Detect specific gestures
- Track multiple people
- Generate activity heatmap

---

**Ready?** Move to `04-summarizer.md` when this works! 🚀

