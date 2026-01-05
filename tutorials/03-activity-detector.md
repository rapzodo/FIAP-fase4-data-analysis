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
models/activity_detection_models.py     # Create Pydantic models
tools/activity_detector_tool.py          # Create activity detector tool
tests/test_activity_detector_tool.py     # Create unit tests
config/tasks.yml                          # Update with output_pydantic
```

## 💻 Implementation

### Step 1: Create Activity Detection Models

**File**: `models/activity_detection_models.py` (NEW FILE)

Create a new file for all activity detection models:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# === Tool Input/Output Models ===

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
    """Complete activity detection tool result."""
    frames_analyzed: int
    activities: List[ActivityDetection]
    activity_summary: Dict[str, int]
    pose_detections: int
    hand_detections: int
    anomalies: List[ActivityAnomaly]


class ActivityDetectorError(BaseModel):
    """Error response."""
    error: str


# === Task Output Models ===

class ActivityData(BaseModel):
    """Single activity statistic for task output."""
    activity: str = Field(..., description="Activity name")
    percentage: float = Field(..., description="Percentage of frames")
    frame_count: int = Field(..., description="Number of frames")


class PoseData(BaseModel):
    """Single pose statistic for task output."""
    pose: str = Field(..., description="Pose name")
    percentage: float = Field(..., description="Percentage of frames")
    frame_count: int = Field(..., description="Number of frames")


class GestureData(BaseModel):
    """Single gesture statistic for task output."""
    gesture: str = Field(..., description="Gesture name")
    percentage: float = Field(..., description="Percentage of frames")
    frame_count: int = Field(..., description="Number of frames")


class ActivityAnalysisTaskOutput(BaseModel):
    """Task output: Aggregated activity, pose, and gesture analysis."""
    total_frames: int = Field(..., description="Total frames analyzed")
    activities: List[ActivityData] = Field(
        default_factory=list,
        description="Activities detected"
    )
    poses: List[PoseData] = Field(
        default_factory=list,
        description="Poses detected"
    )
    gestures: List[GestureData] = Field(
        default_factory=list,
        description="Hand gestures detected"
    )
    pose_detection_rate: float = Field(..., description="Pose detection rate %")
    anomalies_count: int = Field(..., description="Total anomalies")
    anomaly_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Anomaly type counts"
    )
```

**Why separate file for activity models?**
- ✅ **Clean organization**: All activity-related models in one place
- ✅ **Tool outputs**: Models for tool results
- ✅ **Task outputs**: Models for agent/task results
- ✅ **Reusability**: Can be imported where needed

---

### Step 1.5: Update tasks.yml

**File**: `config/tasks.yml`

Update the `detect_activities` task to use Pydantic output:

```yaml
detect_activities:
  description: |
    Analyze the video at '{video_path}' for human activities, poses and gestures.
    Use activity_detector_tool with sample_rate={frame_sample_rate}.
    
    Identify:
    1. Total frames analyzed
    2. Activities detected (playing, running, jumping, etc)
    3. poses detected (standing, sitting, laying, crouching, etc)
    4. hand gestures (raised, waiving)
    5. Activity timelines with timestamps
    6. Anomalies detected (no poses, no human)

  expected_output: "Activity detection analysis with pose, gestures and timeline"
  agent: activity_detector
  output_pydantic: ActivityAnalysisTaskOutput  # ✅ Add this line
```

**What this does:**
- Agent aggregates tool output into structured `ActivityAnalysisTaskOutput`
- Next task (summarizer) receives validated data
- Type-safe data flow

---

### Step 2: Create Activity Detector Tool

**File**: `tools/activity_detector_tool.py`

⚠️ **IMPORTANT**: This tool uses MediaPipe Tasks API, not the deprecated `solutions` API.

**Prerequisites:**
- MediaPipe 0.10.31 (only version available for Apple Silicon)
- Models will be auto-downloaded on first run:
  - `pose_landmarker_full.task` (~26 MB)
  - `hand_landmarker.task` (~9 MB)

```python
import os
import urllib.request
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Dict, Optional


class ActivityDetectorTool(BaseTool):
    name: str = "activity_detector"
    description: str = "Detects human poses, gestures, and activities in video using MediaPipe Tasks API"
    args_schema: type[BaseModel] = ActivityDetectorInput
    
    def _ensure_model(self, model_name: str, url: str) -> str:
        """Download model if not exists."""
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_name)
            print(f"{model_name} downloaded.")
        return model_name
    
    def _detect_activity_from_pose(self, pose_landmarks) -> List[str]:
        """Analyze pose landmarks to classify activity."""
        activities = []
        
        if not pose_landmarks or len(pose_landmarks) == 0:
            return activities
        
        landmarks = pose_landmarks[0]
        
        # Get key landmarks (MediaPipe's 33-landmark model)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        
        # Calculate average positions
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        # Detect standing vs sitting based on torso length
        torso_length = abs(hip_y - shoulder_y)
        if torso_length > 0.3:
            activities.append("standing")
        elif torso_length > 0.15:
            activities.append("sitting")
        
        # Detect hands raised (wrists above shoulders)
        if left_wrist.y < shoulder_y or right_wrist.y < shoulder_y:
            activities.append("hands_raised")
        else:
            activities.append("hands_down")
        
        # Detect movement (knees above hips)
        if left_knee.y < hip_y or right_knee.y < hip_y:
            activities.append("moving")
        
        return activities
    
    def _run(self, video_path: str, sample_rate: int = 5) -> str:
        """Analyze video for human activities using MediaPipe Tasks API."""
        
        # Ensure models are downloaded
        pose_model = self._ensure_model(
            'pose_landmarker_full.task',
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
        )
        
        hand_model = self._ensure_model(
            'hand_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        )
        
        # Initialize pose landmarker (NEW Tasks API)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Initialize hand landmarker (NEW Tasks API)
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Activity counters
        activity_counts = {
            "standing": 0,
            "sitting": 0,
            "moving": 0,
            "hands_raised": 0,
            "hands_down": 0,
            "unknown": 0
        }
        
        frame_idx = 0
        analyzed_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on sample_rate
                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    continue
                
                analyzed_count += 1
                timestamp = frame_idx / fps if fps > 0 else frame_idx
                
                # Convert to RGB and create MediaPipe Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Get timestamp in milliseconds (required for Tasks API)
                frame_timestamp_ms = int(timestamp * 1000)
                
                # Detect pose (NEW Tasks API method)
                pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                # Detect hands (NEW Tasks API method)
                hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                frame_activities = []
                
                # Track detections
                if pose_result.pose_landmarks:
                    result.pose_detections += len(pose_result.pose_landmarks)
                
                if hand_result.hand_landmarks:
                    result.hand_detections += len(hand_result.hand_landmarks)
                
                # Analyze activities from pose
                detected_activities = self._detect_activity_from_pose(pose_result.pose_landmarks)
                
                if detected_activities:
                    frame_activities.extend(detected_activities)
                    for activity in detected_activities:
                        activity_counts[activity] = activity_counts.get(activity, 0) + 1
                
                # Record activities using Pydantic model
                if frame_activities:
                    result.activities.append(ActivityDetection(
                        frame=frame_idx,
                        timestamp=round(timestamp, 2),
                        activities=frame_activities
                    ))
                else:
                    # No activity detected - log anomaly
                    activity_counts["unknown"] += 1
                    result.anomalies.append(ActivityAnomaly(
                        frame=frame_idx,
                        timestamp=round(timestamp, 2),
                        type="no_pose_detected",
                        details="No human pose detected in frame"
                    ))
                
                frame_idx += 1
        
        finally:
            # Cleanup resources (IMPORTANT!)
            cap.release()
            pose_landmarker.close()
            hand_landmarker.close()
        
        result.frames_analyzed = analyzed_count
        result.activity_summary = activity_counts
        
        return result.model_dump_json(indent=2)
```

**What this does:**
- Uses **Pydantic models** for type-safe results
- Uses **MediaPipe Tasks API** for pose and hand detection
- **Auto-downloads models** on first run (~35 MB total)
- Classifies poses: standing/sitting/moving
- Detects hand positions: raised/lowered
- Tracks activity statistics across frames
- Logs anomalies when no pose detected
- **Proper cleanup** with landmarker.close()

**Key techniques:**
- **Model bundles**: Pose detection + landmarking in single file
- **Landmark detection**: 33 body points + 21 hand points tracked
- **Geometric analysis**: Torso length indicates pose
- **Motion detection**: Knee position indicates movement
- **Hand position**: Y-coordinate relative to shoulders
- **Type safety**: Pydantic validates all data
- **Resource management**: Proper cleanup of video and landmarkers

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
**Fix**: Install MediaPipe 0.10.31
```bash
pip install mediapipe==0.10.31
```

### Error: "Cannot find reference 'solutions'"
**Cause**: Using deprecated `mp.solutions` API  
**Fix**: Use MediaPipe Tasks API as shown in this tutorial
```python
# ❌ Don't use
mp_pose = mp.solutions.pose

# ✅ Use instead
from mediapipe.tasks.python import vision
landmarker = vision.PoseLandmarker.create_from_options(options)
```

### Error: "Model file not found"
**Cause**: Model not downloaded  
**Fix**: The `_ensure_model()` method auto-downloads. Ensure:
- Internet connection available
- Write permissions in working directory
- ~35 MB free space for both models

### Error: "ImportError: cannot import name 'solutions'"
**Cause**: MediaPipe 0.10.30+ doesn't have `solutions` module  
**Fix**: This is expected! Use Tasks API throughout

### Low detection rate
**Causes**:
- Poor lighting
- Partial body visibility
- Person too far from camera

**Fix**: Use videos with full-body visibility and good lighting

### "No pose detected" anomalies
**Normal**: Happens when:
- Person leaves frame
- Back turned to camera
- Body partially occluded
- Fast movement (motion blur)

**Not normal**: If > 50% frames have no detection, check:
- Video quality
- Camera angle
- Lighting conditions

### Slow processing
**Fix**: Increase sample_rate
```python
# In task description
Use the activity_detector tool with sample_rate=20
```

**Alternative**: Use lighter model
```python
# In _ensure_model() call
pose_model = self._ensure_model(
    'pose_landmarker_lite.task',  # Faster but less accurate
    'https://...pose_landmarker_lite...'
)
```

### Memory issues
**Cause**: Processing too many frames  
**Fix**: 
- Increase sample_rate (process fewer frames)
- Process shorter video segments
- Close landmarkers after use (already implemented)

## 💡 Understanding MediaPipe Tasks API

### ⚠️ Important: API Migration

MediaPipe **deprecated the `solutions` API** in version 0.10.30+. The new **Tasks API** is required for:
- Apple Silicon (macOS ARM64) compatibility
- Modern MediaPipe features
- Better performance

**Old API (DEPRECATED):**
```python
# ❌ This no longer works on Apple Silicon
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
results = pose.process(frame)
```

**New API (REQUIRED):**
```python
# ✅ Use this instead
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.PoseLandmarkerOptions(...)
landmarker = vision.PoseLandmarker.create_from_options(options)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

### Model Bundles

MediaPipe pose landmarker models (`.task` files) include:
1. **Pose Detection Model** - Finds human bodies in frames
2. **Pose Landmarker Model** - Predicts 33 3D landmarks per body

Both packaged together based on:
- **BlazePose** architecture (CNN similar to MobileNetV2)
- **GHUM** pipeline (3D human shape modeling)

**Available models:**
- `pose_landmarker_lite.task` (~12 MB) - Fastest, basic accuracy
- `pose_landmarker_full.task` (~26 MB) - **Recommended balance**
- `pose_landmarker_heavy.task` (~50 MB) - Slowest, highest accuracy

### Pose Landmarks
MediaPipe detects 33 body points:
- **Face**: nose, eyes, ears (indices 0-10)
- **Upper body**: shoulders (11-12), elbows (13-14), wrists (15-16)
- **Hands**: pinky, index, thumb (17-22)
- **Lower body**: hips (23-24), knees (25-26), ankles (27-28)
- **Feet**: heel, foot index (29-32)

### How Pose Classification Works

```python
# Standing: Torso is extended (large distance)
torso_length = hip_y - shoulder_y
if torso_length > 0.3:  # Standing

# Sitting: Torso is compressed (small distance)
if torso_length < 0.15:  # Sitting

# Moving: Knee position indicates leg movement
if left_knee.y < hip_y or right_knee.y < hip_y:  # Moving
```

### Hand Detection

Hand landmarker detects 21 landmarks per hand:
- **Wrist** (index 0)
- **Thumb** (indices 1-4)
- **Fingers** (indices 5-20)

**Hand position detection:**
- **Wrist Y < shoulder Y**: Hand raised
- **Wrist Y > shoulder Y**: Hand down

### Key Differences: Old vs New API

| Aspect | Old (solutions) | New (Tasks) |
|--------|----------------|-------------|
| **Import** | `mp.solutions.pose` | `vision.PoseLandmarker` |
| **Initialize** | `Pose()` | `create_from_options()` |
| **Process** | `pose.process(frame)` | `landmarker.detect_for_video(mp_image, timestamp_ms)` |
| **Image format** | NumPy array (BGR) | `mp.Image` (RGB) |
| **Timestamp** | Not required | Milliseconds required |
| **Models** | Auto-loaded | Manual download required |
| **Cleanup** | Context manager | Call `.close()` |
| **Result** | `pose_landmarks` | `pose_landmarks` (same structure) |

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
