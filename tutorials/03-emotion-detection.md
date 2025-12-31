# Module 3: Emotion Detection Tool

**Time**: 40 minutes | **Difficulty**: Medium

## 🎯 What You'll Build

A tool that **analyzes emotions** in pre-detected faces from Module 2.

**What it does:**
- Takes face locations from facial recognition tool
- Analyzes emotions using DeepFace
- Returns 7 emotion scores per face
- Tracks anomalies

**Requirements:**
- ✅ Must have completed Module 2 (Facial Recognition Tool)
- ✅ Faces must be pre-detected before emotion analysis

## 📋 Files to Update/Create

```
models/facial_recognition_models.py     # Add emotion models to existing file
tools/emotion_detection_tool.py         # Create emotion detection tool
tests/test_emotion_detection_tool.py    # Create unit tests
```

---

## 💻 Step 1: Add Emotion Detection Models

**File**: `models/facial_recognition_models.py` (ADD to existing file)

Add these models **after** the facial recognition models:

```python
from typing import Optional, Dict

# === Emotion Detection Models (ADD THESE) ===

class EmotionDetectionInput(BaseModel):
    """Input schema for emotion detection tool."""
    video_path: str = Field(..., description="Path to the video file")
    face_locations: List[DetectedFace] = Field(..., description="Pre-detected faces")


class EmotionScores(BaseModel):
    """All 7 emotion confidence scores."""
    angry: float = Field(..., ge=0, le=100)
    disgust: float = Field(..., ge=0, le=100)
    fear: float = Field(..., ge=0, le=100)
    happy: float = Field(..., ge=0, le=100)
    sad: float = Field(..., ge=0, le=100)
    surprise: float = Field(..., ge=0, le=100)
    neutral: float = Field(..., ge=0, le=100)


class FaceEmotion(BaseModel):
    """Emotion analysis for a single face."""
    frame: int
    timestamp: float
    face_id: int  # Links to DetectedFace.face_id from Module 2
    dominant_emotion: str
    confidence: float = Field(..., ge=0, le=100)
    emotion_scores: EmotionScores


class EmotionAnomaly(BaseModel):
    """Anomaly in emotion detection."""
    frame: int
    timestamp: float
    face_id: int
    type: str
    confidence: Optional[float] = None
    error: Optional[str] = None


class EmotionDetectionResult(BaseModel):
    """Complete emotion detection result."""
    faces_analyzed: int
    emotions_detected: List[FaceEmotion] = Field(default_factory=list)
    emotion_summary: Dict[str, int] = Field(default_factory=dict)
    anomalies: List[EmotionAnomaly] = Field(default_factory=list)


class EmotionDetectionError(BaseModel):
    """Error response."""
    error: str
```

**Key Points:**
- `FaceEmotion.face_id` links to `DetectedFace.face_id` from Module 2
- `EmotionScores` validates 0-100 range
- Models added to same file as facial recognition models

---

## 💻 Step 2: Create Emotion Detection Tool

**File**: `tools/emotion_detection_tool.py`

```python
from crewai.tools import BaseTool
import cv2
from deepface import DeepFace

from models.facial_recognition_models import (
    EmotionDetectionInput,
    EmotionScores,
    FaceEmotion,
    EmotionAnomaly,
    EmotionDetectionResult,
    EmotionDetectionError
)


class EmotionDetectionTool(BaseTool):
    name: str = 'emotion_detection'
    description: str = (
        'Analyzes emotions in pre-detected faces. '
        'Requires face locations from facial_recognition tool.'
    )
    args_schema: type[EmotionDetectionInput] = EmotionDetectionInput
    
    def _run(self, video_path: str, face_locations: list) -> str:
        """
        Detect emotions in pre-identified faces.
        
        Args:
            video_path: Path to video file
            face_locations: List of DetectedFace from Module 2
            
        Returns:
            JSON string with EmotionDetectionResult
        """
        # TODO: Initialize result
        result = EmotionDetectionResult(
            faces_analyzed=0,
            emotions_detected=[],
            emotion_summary={},
            anomalies=[]
        )
        
        # TODO: Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error = EmotionDetectionError(error=f"Cannot open: {video_path}")
            return error.model_dump_json(indent=2)
        
        # TODO: Initialize emotion counters
        emotion_counts = {
            "happy": 0, "sad": 0, "angry": 0,
            "neutral": 0, "surprise": 0, "fear": 0, "disgust": 0
        }
        
        # TODO: Group faces by frame for efficient processing
        faces_by_frame = {}
        for face in face_locations:
            # Handle both dict and Pydantic object
            frame_num = face.get('frame') if isinstance(face, dict) else face.frame
            if frame_num not in faces_by_frame:
                faces_by_frame[frame_num] = []
            faces_by_frame[frame_num].append(face)
        
        try:
            # TODO: For each frame with faces
            for frame_num, faces in faces_by_frame.items():
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                
                if not ret:
                    # Add anomalies for failed frame reads
                    for face in faces:
                        face_id = face.get('face_id') if isinstance(face, dict) else face.face_id
                        timestamp = face.get('timestamp') if isinstance(face, dict) else face.timestamp
                        result.anomalies.append(EmotionAnomaly(
                            frame=frame_num,
                            timestamp=timestamp,
                            face_id=face_id,
                            type="frame_read_failed"
                        ))
                    continue
                
                # TODO: Process each face in this frame
                for face in faces:
                    # Extract face data (handle dict or Pydantic)
                    face_id = face.get('face_id') if isinstance(face, dict) else face.face_id
                    timestamp = face.get('timestamp') if isinstance(face, dict) else face.timestamp
                    location = face.get('location') if isinstance(face, dict) else face.location
                    
                    # Get coordinates
                    if isinstance(location, dict):
                        top, left = location['top'], location['left']
                        right, bottom = location['right'], location['bottom']
                    else:
                        top, left = location.top, location.left
                        right, bottom = location.right, location.bottom
                    
                    # Extract face region
                    face_img = frame[top:bottom, left:right]
                    
                    if face_img.size == 0:
                        result.anomalies.append(EmotionAnomaly(
                            frame=frame_num,
                            timestamp=timestamp,
                            face_id=face_id,
                            type="invalid_face_region"
                        ))
                        continue
                    
                    try:
                        # TODO: Analyze emotion with DeepFace
                        analysis = DeepFace.analyze(
                            face_img,
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=True
                        )
                        
                        # Handle list or dict response
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        
                        dominant_emotion = analysis['dominant_emotion']
                        emotion_scores_raw = analysis['emotion']
                        confidence = max(emotion_scores_raw.values())
                        
                        # Create EmotionScores
                        emotion_scores = EmotionScores(
                            angry=round(emotion_scores_raw.get('angry', 0), 2),
                            disgust=round(emotion_scores_raw.get('disgust', 0), 2),
                            fear=round(emotion_scores_raw.get('fear', 0), 2),
                            happy=round(emotion_scores_raw.get('happy', 0), 2),
                            sad=round(emotion_scores_raw.get('sad', 0), 2),
                            surprise=round(emotion_scores_raw.get('surprise', 0), 2),
                            neutral=round(emotion_scores_raw.get('neutral', 0), 2)
                        )
                        
                        # Create FaceEmotion result
                        face_emotion = FaceEmotion(
                            frame=frame_num,
                            timestamp=timestamp,
                            face_id=face_id,
                            dominant_emotion=dominant_emotion,
                            confidence=round(confidence, 2),
                            emotion_scores=emotion_scores
                        )
                        
                        result.emotions_detected.append(face_emotion)
                        emotion_counts[dominant_emotion] += 1
                        result.faces_analyzed += 1
                        
                        # Check for low confidence
                        if confidence < 50:
                            result.anomalies.append(EmotionAnomaly(
                                frame=frame_num,
                                timestamp=timestamp,
                                face_id=face_id,
                                type="low_confidence",
                                confidence=round(confidence, 2)
                            ))
                    
                    except Exception as e:
                        result.anomalies.append(EmotionAnomaly(
                            frame=frame_num,
                            timestamp=timestamp,
                            face_id=face_id,
                            type="emotion_detection_failed",
                            error=str(e)
                        ))
        
        finally:
            cap.release()
        
        result.emotion_summary = emotion_counts
        
        return result.model_dump_json(indent=2)
```

**Key Points:**
- Imports models from same file as facial recognition
- Handles both dict and Pydantic face inputs
- Groups faces by frame for efficiency
- Links emotions to faces via face_id

---

## 🧪 Step 3: Create Unit Tests

**File**: `tests/test_emotion_detection_tool.py`

```python
import unittest
import json

from tools.emotion_detection_tool import EmotionDetectionTool


class TestEmotionDetectionTool(unittest.TestCase):
    """Test suite for EmotionDetectionTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tool = EmotionDetectionTool()
        self.mock_faces = [
            {
                "frame": 1,
                "timestamp": 0.03,
                "face_id": 0,
                "location": {"top": 10, "left": 5, "right": 20, "bottom": 30}
            }
        ]
    
    def test_tool_name(self):
        """Test tool has correct name."""
        self.assertEqual(self.tool.name, "emotion_detection")
    
    def test_tool_description(self):
        """Test tool has description."""
        self.assertIsNotNone(self.tool.description)
        self.assertIn("emotion", self.tool.description.lower())
    
    def test_invalid_video_returns_error(self):
        """Test invalid video path returns error."""
        result = self.tool._run("nonexistent.mp4", self.mock_faces)
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)
    
    def test_result_is_valid_json(self):
        """Test tool returns valid JSON."""
        result = self.tool._run("invalid.mp4", self.mock_faces)
        
        try:
            json.loads(result)
        except json.JSONDecodeError:
            self.fail("Tool output is not valid JSON")
    
    def test_empty_face_list(self):
        """Test with empty face list."""
        result = self.tool._run("video.mp4", [])
        result_dict = json.loads(result)
        
        self.assertIn("faces_analyzed", result_dict)
        self.assertEqual(result_dict["faces_analyzed"], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

---

## 🧪 Step 4: Test the Complete Pipeline

**File**: `tests/test_face_emotion_pipeline.py`

```python
import json
from tools.facial_recognition_tool import FacialRecognitionTool
from tools.emotion_detection_tool import EmotionDetectionTool


def test_pipeline():
    """Test face detection → emotion analysis pipeline."""
    
    print("🧪 Testing Complete Pipeline\n")
    
    # Step 1: Detect faces
    print("Step 1: Detecting faces...")
    facial_tool = FacialRecognitionTool()
    faces_json = facial_tool._run(
        video_path="tech-challenge/video.mp4",
        sample_rate=30
    )
    
    faces_result = json.loads(faces_json)
    print(f"✅ Found {faces_result['total_faces']} faces")
    
    if faces_result['total_faces'] == 0:
        print("⚠️  No faces detected")
        return
    
    # Step 2: Analyze emotions
    print("\nStep 2: Analyzing emotions...")
    emotion_tool = EmotionDetectionTool()
    emotions_json = emotion_tool._run(
        video_path="tech-challenge/video.mp4",
        face_locations=faces_result['faces_detected']
    )
    
    emotions_result = json.loads(emotions_json)
    print(f"✅ Analyzed {emotions_result['faces_analyzed']} faces")
    
    print(f"\n📊 Emotion Summary:")
    for emotion, count in emotions_result['emotion_summary'].items():
        if count > 0:
            print(f"   {emotion}: {count}")
    
    print("\n✅ Pipeline PASSED!")


if __name__ == "__main__":
    test_pipeline()
```

---

## 💡 Implementation Tips

### Handling Dict or Pydantic Input
```python
def get_value(obj, key):
    """Get value from dict or Pydantic object."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key)

face_id = get_value(face, 'face_id')
```

### DeepFace Analysis
```python
analysis = DeepFace.analyze(
    face_img,
    actions=['emotion'],
    enforce_detection=False,
    silent=True
)

if isinstance(analysis, list):
    analysis = analysis[0]
```

---

## ✅ Verification Checklist

- [ ] Emotion models added to `models/facial_recognition_models.py`
- [ ] Tool created in `tools/emotion_detection_tool.py`
- [ ] Tool imports from models file
- [ ] Unit tests pass (5 tests)
- [ ] Pipeline test works
- [ ] Emotion scores in 0-100 range
- [ ] face_id links Module 2 to Module 3
- [ ] NO emotion code in facial recognition tool

---

## 🎯 What You've Learned

1. **Two-Stage Processing** - Separate tools working together
2. **Data Pipeline** - Output of tool 1 → input of tool 2
3. **DeepFace Integration** - Emotion analysis
4. **Efficient Processing** - Group by frame, seek as needed
5. **Model Reuse** - Both tools use same models file

---

## 📝 Next Steps

**Module 4**: Build the Activity Detector Tool!

**Pipeline complete so far:**
1. ✅ Facial Recognition → Detect faces
2. ✅ Emotion Detection → Analyze emotions  
3. ⏭️ Activity Detection → Body poses
4. ⏭️ Summarizer → Aggregate results

---

**Ready?** Move to Module 4! 🚀

