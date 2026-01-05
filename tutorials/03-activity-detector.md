# Module 3: Activity Detector Agent

**Time**: 45 minutes | **Difficulty**: Medium

## 🎯 What You'll Build

An agent that identifies human activities in videos:
- Body pose detection (standing, sitting, moving)
- Hand gesture recognition (raised, lowered)
- Activity timeline
- Movement patterns

## 📋 Files to Create/Update

```
models/activity_detection_models.py     # Activity detection Pydantic models
tools/activity_detection_tool.py        # Activity detector tool with MediaPipe
config/agents.yml                        # Agent configuration
config/tasks.yml                         # Task configuration with output_pydantic
tests/test_activity_detection_agent.py  # Integration test
```

## 💻 Implementation

### Step 1: Create Activity Detection Models

**File**: `models/activity_detection_models.py`

```python
from typing import Optional
from pydantic import BaseModel, Field


class BodyLandmarks:
    """Helper class to access body landmarks by name."""
    def __init__(self, landmarks):
        self.left_shoulder = landmarks[11]
        self.right_shoulder = landmarks[12]
        self.left_hip = landmarks[23]
        self.right_hip = landmarks[24]
        self.left_wrist = landmarks[15]
        self.right_wrist = landmarks[16]
        self.left_ankle = landmarks[27]
        self.right_ankle = landmarks[28]
        self.left_knee = landmarks[25]
        self.right_knee = landmarks[26]


# === Tool Input/Output Models ===

class ActivityDetectionInput(BaseModel):
    """Input for activity detection tool."""
    video_path: str = Field(description="Path to the video file")
    media_pipe_model: str = Field(description="MediaPipe model for pose detection. Valid values: 'LITE', 'FULL', 'HEAVY'")
    sample_rate: int = Field(description="Sample rate for frame processing")


class ActivityDetection(BaseModel):
    """Single activity detection result."""
    frame: int = Field(description="Frame number")
    timestamp: float = Field(description="Timestamp")
    activities: list[str] = Field(description="Activities detected", examples=["standing, hands_raised"])


class ActivityAnomaly(BaseModel):
    """Activity detection anomaly."""
    frame: int = Field(description="Frame number")
    timestamp: float = Field(description="Timestamp")
    type: str = Field(description="Anomaly type")
    details: Optional[str] = Field(None, description="Anomaly details")


class ActivityDetectionResult(BaseModel):
    """Complete activity detection tool result."""
    frames_analyzed: int = Field(description="Frames analyzed")
    activities: list[ActivityDetection] = Field(description="Activities detected")
    activity_summary: dict[str, int] = Field(description="Activity summary")
    pose_detections: int = Field(description="Total Poses detected")
    hands_detections: int = Field(description="Total Hands detected")
    anomalies: list[ActivityAnomaly] = Field(description="Anomalies detected")


# === Task Output Models ===

class ActivityData(BaseModel):
    """Single activity statistic for task output."""
    activity: str = Field(..., title="Activity name")
    percentage: float = Field(..., title="Activity frequency percentage")
    frame_count: int = Field(..., title="Number of frames with this activity")


class PoseData(BaseModel):
    """Single pose statistic for task output."""
    pose: str = Field(..., title="Pose name")
    percentage: float = Field(..., title="Pose frequency percentage")
    frame_count: int = Field(..., title="Number of frames with this pose")


class GestureData(BaseModel):
    """Single gesture statistic for task output."""
    gesture: str = Field(..., title="Gesture name")
    percentage: float = Field(..., title="Gesture frequency percentage")
    frame_count: int = Field(..., title="Number of frames with this gesture")
```

**Key changes:**
- ✅ **BodyLandmarks class**: Makes landmark access more readable
- ✅ **media_pipe_model parameter**: Supports LITE/FULL/HEAVY models
- ✅ **Separate tool and task models**: Clear separation of concerns

---

### Step 2: Create Activity Detector Tool

**File**: `tools/activity_detection_tool.py`

⚠️ **IMPORTANT**: This uses MediaPipe Tasks API with multiple model options.

```python
import os
import urllib
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
from crewai.tools import BaseTool
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions, PoseLandmarker

from models import ExecutionError
from models.activity_detection_models import (
    ActivityDetectionInput, ActivityDetectionResult, 
    BodyLandmarks, ActivityDetection, ActivityAnomaly
)

MEDIA_PIPE_MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/"


class MediaPipeModel(Enum):
    """Available MediaPipe pose detection models."""
    LITE = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", "pose_landmarker_lite.task"
    FULL = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", "pose_landmarker_full.task"
    HEAVY = MEDIA_PIPE_MODEL_BASE_URL + "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", "pose_landmarker_heavy.task"
    HANDS = MEDIA_PIPE_MODEL_BASE_URL + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "hand_landmarker.task"


class ActivityDetectionTool(BaseTool):
    name: str = "activity_detection"
    description: str = "Detects human activities, poses, and gestures in videos using MediaPipe. Requires video_path (string), media_pipe_model (string: 'LITE', 'FULL', or 'HEAVY'), and sample_rate (integer)."
    args_schema: type[ActivityDetectionInput] = ActivityDetectionInput

    def detect_activity_from_pose(self, pose_landmarks: list[list[NormalizedLandmark]]):
        """Analyze pose landmarks to classify activity."""
        activities = []
        if not pose_landmarks or len(pose_landmarks) == 0:
            return activities

        landmarks = pose_landmarks[0]
        body_landmarks = BodyLandmarks(landmarks)

        activities.append(self.detect_standing_or_sitting(body_landmarks))
        activities.append(self.detect_hand_position(body_landmarks))
        activities.append(self.detect_body_movement(body_landmarks))
        return activities

    def detect_standing_or_sitting(self, body_landmarks: BodyLandmarks):
        """Detect if person is standing or sitting based on torso length."""
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        torso_length = abs(hip_y - shoulder_y)
        if torso_length < 0.3:
            return "standing"
        elif torso_length < 0.15:
            return "sitting"
        return "unknown"

    def detect_hand_position(self, body_landmarks: BodyLandmarks):
        """Detect if hands are raised or lowered."""
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        if body_landmarks.left_wrist.y < shoulder_y or body_landmarks.right_wrist.y < shoulder_y:
            return "hands_raised"
        else:
            return "hands_down"

    def detect_body_movement(self, body_landmarks: BodyLandmarks):
        """Detect if person is moving based on knee position."""
        shoulder_y, hip_y = self.calculate_average_position(body_landmarks)
        if body_landmarks.left_knee.y < hip_y or body_landmarks.right_knee.y < hip_y:
            return "moving"
        else:
            return "standing"

    @staticmethod
    def calculate_average_position(body_landmarks: BodyLandmarks):
        """Calculate average shoulder and hip Y positions."""
        shoulder_y = (body_landmarks.left_shoulder.y + body_landmarks.right_shoulder.y) / 2
        hip_y = (body_landmarks.left_hip.y + body_landmarks.right_hip.y) / 2
        return shoulder_y, hip_y

    @staticmethod
    def download_model(pose_model: MediaPipeModel):
        """Download MediaPipe model if not already cached."""
        pose_model_url = pose_model.value[0]
        pose_model_name = pose_model.value[1]

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media_pipe', 'pose_models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, pose_model_name)

        if not os.path.exists(model_path):
            print(f'Downloading model file... {pose_model_name}')
            urllib.request.urlretrieve(pose_model_url, model_path)
            print(f'Model {pose_model_name} downloaded successfully.')

        return model_path

    def _run(self, video_path, media_pipe_model: str, sample_rate: int = 5) -> str:
        """Analyze video for human activities using MediaPipe Tasks API."""
        
        # Get project root and resolve video path
        project_root = Path(__file__).parent.parent
        video_path = str(project_root / video_path)

        # Download models if needed
        pose_model = self.download_model(MediaPipeModel[media_pipe_model])
        hands_model = self.download_model(MediaPipeModel.HANDS)

        # Initialize pose landmarker
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
        )
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        # Initialize hand landmarker
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hands_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        hands_landmarker = HandLandmarker.create_from_options(hand_options)

        # Initialize result model
        result = ActivityDetectionResult(
            frames_analyzed=0,
            activities=[],
            activity_summary={},
            pose_detections=0,
            hands_detections=0,
            anomalies=[]
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ExecutionError(error=f"Unable to open video source {video_path}.").model_dump_json(indent=2)

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        activity_counter = {
            "standing": 0,
            "sitting": 0,
            "hands_raised": 0,
            "hands_down": 0,
            "moving": 0,
            "unknown": 0
        }

        frame_number = 0
        analyzed_counter = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Sample frames based on sample_rate
                if frame_number % sample_rate != 0:
                    frame_number += 1
                    continue

                analyzed_counter += 1
                timestamp = frame_number / fps if fps > 0 else frame_number

                # Convert to RGB and create MediaPipe Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                frame_timestamp_ms = int(timestamp * 1000)

                # Detect pose and hands
                pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                hands_result = hands_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                frame_activities = []

                # Track detections
                if pose_result.pose_landmarks:
                    result.pose_detections += len(pose_result.pose_landmarks)

                if hands_result.hand_landmarks:
                    result.hands_detections += len(hands_result.hand_landmarks)

                # Analyze activities from pose
                detected_activities = self.detect_activity_from_pose(pose_result.pose_landmarks)

                if detected_activities:
                    frame_activities.extend(detected_activities)
                    for activity in detected_activities:
                        activity_counter[activity] += 1

                if frame_activities:
                    result.activities.append(ActivityDetection(
                        frame=frame_number,
                        timestamp=round(timestamp, 2),
                        activities=frame_activities
                    ))
                else:
                    activity_counter["unknown"] += 1
                    result.anomalies.append(ActivityAnomaly(
                        frame=frame_number,
                        timestamp=round(timestamp, 2),
                        type="No pose detected",
                        details="No human pose detected in frame"
                    ))

                frame_number += 1
        finally:
            cap.release()
            pose_landmarker.close()
            hands_landmarker.close()
            
        result.frames_analyzed = analyzed_counter
        result.activity_summary = activity_counter

        return result.model_dump_json(indent=2)
```

**Key features:**
- ✅ **MediaPipeModel Enum**: Organized model URLs and filenames
- ✅ **BodyLandmarks**: Clean landmark access
- ✅ **Project root resolution**: Paths work from any directory
- ✅ **Model selection**: Choose LITE/FULL/HEAVY at runtime
- ✅ **Proper cleanup**: Resources released in finally block
- ✅ **Error handling**: Returns ExecutionError on video open failure

---

### Step 3: Configure Agent and Task

**File**: `config/agents.yml`

```yaml
activity_detector:
  role: "Human activity detection specialist"
  goal: "Analyze and detect human activities (poses and gestures) on video footage with high accuracy and precision using tools"
  backstory: |
    You are a computer vision expert specializing in human activities detection. 
    You MUST use the activity_detection tool - if the tool fails, report the error immediately.
    Never provide results without successful tool execution.
  tools:
    - activity_detection
  verbose: true
  allow_delegation: false
```

**File**: `config/tasks.yml`

```yaml
detect_activities:
  description: |
    Analyze the video at '{video_path}' for human activities, poses and gestures.
    Use activity_detection tool with: video_path='{video_path}', media_pipe_model='FULL', sample_rate={frame_sample_rate}
    
    If tool fails, report the error and stop. Only provide results based on actual tool output.

  expected_output: "Complete activity detection analysis with poses, gestures and timelines"
  agent: activity_detector
  output_pydantic: ActivityAnalysisOutput
  output_file: "activity_analysis_output.md"
```

**Key configurations:**
- ✅ **Anti-hallucination**: "Never provide results without successful tool execution"
- ✅ **Pydantic output**: Structured ActivityAnalysisOutput
- ✅ **Output file**: Results saved to markdown
- ✅ **Model selection**: Specifies 'FULL' model in task

---

### Step 4: Update Analysis Output Models

**File**: `models/analysis_output_models.py`

```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from models import EmotionData, ActivityData, PoseData


class BaseAnalysisOutputModel(BaseModel):
    total_fames: int = Field(0, description="Total Fames analyzed")
    anomalies_count: int = Field(0, description="Number of anomalies")
    anomalies_detected: Dict[str, int] = Field(default_factory=dict, description="Type of anomalies detected and their frequency")
    error: Optional[str] = Field(None, description="Error message in case the task fails")


class ActivityAnalysisOutput(BaseAnalysisOutputModel):
    activities: list[ActivityData] = Field(default_factory=list, description="Activities analyzed")
    poses: list[PoseData] = Field(default_factory=list, description="Poses analyzed")
    gestures: list[str] = Field(default_factory=list, description="Gestures analyzed")
    pose_detection_rate: float = Field(0.0, description="Detection rate (percentage)")
```

**Key changes:**
- ✅ **Optional fields with defaults**: Agent can return output even on partial failures
- ✅ **Error field**: Can report tool failures in structured output
- ✅ **default_factory**: Proper Pydantic defaults for collections

---

### Step 5: Create Integration Test

**File**: `tests/test_activity_detection_agent.py`

```python
import os
from crewai import Crew

from agents.agents_factory import AgentsFactory
from tasks.task_factory import TaskFactory
from tools.activity_detection_tool import ActivityDetectionTool


def test_activity_detector_agent():
    """Test activity detection on a video."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(project_root, "tech-challenge", "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print(f"💡 Project root: {project_root}")
        return

    # Set environment variables for TaskFactory
    os.environ["VIDEO_PATH"] = video_path
    os.environ["FRAME_SAMPLE_RATE"] = "1"

    print("🏃 Testing Activity Detector Agent\n")

    # Create agent
    config_path = os.path.join(project_root, "config", "agents.yml")
    config_path_task = os.path.join(project_root, "config", "tasks.yml")
    tool = ActivityDetectionTool()
    agent = AgentsFactory(config_path).create_agent("activity_detector", tools={tool.name: tool})
    print("✅ Agent created\n")

    # Create task
    task = TaskFactory(config_path_task).create_task(task_name="detect_activities", agent=agent)
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

**Key features:**
- ✅ **Absolute path resolution**: Works from any directory
- ✅ **Environment variables**: Sets VIDEO_PATH and FRAME_SAMPLE_RATE
- ✅ **AgentsFactory and TaskFactory**: Uses YAML configurations
- ✅ **Tool registration**: Passes tool to agent by name

---

## 🎓 What You Learned

### MediaPipe Tasks API Migration
- ✅ **Enum for models**: MediaPipeModel enum for organized model management
- ✅ **Model selection**: Runtime choice of LITE/FULL/HEAVY models
- ✅ **Proper initialization**: BaseOptions and model-specific options
- ✅ **Video mode**: RunningMode.VIDEO for frame-by-frame processing

### Code Organization
- ✅ **BodyLandmarks class**: Readable landmark access
- ✅ **Separation of concerns**: Tool models vs task output models
- ✅ **Path management**: Project root resolution for reliable paths

### Error Handling
- ✅ **Optional fields**: Pydantic models with defaults
- ✅ **Error reporting**: Structured error in output models
- ✅ **Resource cleanup**: Proper finally blocks

### Agent Configuration
- ✅ **Anti-hallucination prompts**: Clear instructions to stop on tool failure
- ✅ **Pydantic outputs**: Type-safe task results
- ✅ **Output files**: Saved results for verification

## 🎯 Success Criteria

- ✅ Video analysis completes without errors
- ✅ Detects poses: standing/sitting/moving
- ✅ Detects hand positions: raised/lowered  
- ✅ Returns structured ActivityAnalysisOutput
- ✅ Creates activity_analysis_output.md file
- ✅ Works from any directory (project root resolution)

## 📦 Next Steps

Module 4: Learn to create summarizer agents that aggregate multi-agent results!        landmarks = pose_landmarks[0]
        
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
