# Module 2: Facial Recognition Tool (Face Detection Only)

**Time**: 30 minutes | **Difficulty**: Medium

## 🎯 What You'll Build

A tool that **only detects faces** in video frames - no emotion analysis. This follows the Single Responsibility Principle.

**What it does:**
- Detect faces in video frames
- Return face locations with unique IDs
- Fast and efficient
- Passes results to emotion detection tool (Module 3)

**What it does NOT do:**
- ❌ Emotion analysis (that's Module 3)
- ❌ Face recognition (identifying people)

## 📋 Files to Create

```
models/facial_recognition_models.py     # Pydantic models (facial only)
tools/facial_recognition_tool.py        # Face detection tool
tests/test_facial_recognition_tool.py   # Unit tests
```

---

## 💻 Step 1: Create Pydantic Models (Facial Recognition)

**File**: `models/facial_recognition_models.py`

Create models for facial recognition (face detection only):

```python
from pydantic import BaseModel, Field
from typing import List


class FacialRecognitionInput(BaseModel):
    """Input schema for facial recognition tool."""
    video_path: str = Field(..., description="Path to the video file")
    sample_rate: int = Field(5, description="Process every Nth frame (1=all frames)")


class FaceLocation(BaseModel):
    """Face bounding box coordinates."""
    top: int = Field(..., description="Top coordinate")
    left: int = Field(..., description="Left coordinate")
    right: int = Field(..., description="Right coordinate")
    bottom: int = Field(..., description="Bottom coordinate")


class DetectedFace(BaseModel):
    """Single detected face without emotion."""
    frame: int = Field(..., description="Frame number")
    timestamp: float = Field(..., description="Timestamp in seconds")
    face_id: int = Field(..., description="Unique face ID")
    location: FaceLocation = Field(..., description="Face bounding box")


class FacialRecognitionResult(BaseModel):
    """Complete facial recognition result."""
    frames_analyzed: int = Field(..., description="Number of frames analyzed")
    faces_detected: List[DetectedFace] = Field(default_factory=list)
    total_faces: int = Field(..., description="Total number of faces detected")


class FacialRecognitionError(BaseModel):
    """Error response for facial recognition."""
    error: str = Field(..., description="Error message")
```

**Key Points:**
- `DetectedFace` has NO emotion field - that's for Module 3
- `face_id` creates a unique identifier for each face
- Simple, focused on face location only
- Models are in separate file, not in tool

---

## 💻 Step 2: Create Facial Recognition Tool

**File**: `tools/facial_recognition_tool.py`

```python
from crewai.tools import BaseTool
import cv2
import numpy as np
import face_recognition

from models.facial_recognition_models import (
    FacialRecognitionInput,
    FaceLocation,
    DetectedFace,
    FacialRecognitionResult,
    FacialRecognitionError
)


class FacialRecognitionTool(BaseTool):
    name: str = 'facial_recognition'
    description: str = (
        'Detects faces in video frames. Returns face locations and metadata. '
        'Use this tool first, then pass results to emotion detection tool.'
    )
    args_schema: type[FacialRecognitionInput] = FacialRecognitionInput
    
    def _run(self, video_path: str, sample_rate: int = 5) -> str:
        """
        Detect faces in video frames.
        
        Returns:
            JSON string with FacialRecognitionResult or FacialRecognitionError
        """
        # TODO: Initialize result model
        result = FacialRecognitionResult(
            frames_analyzed=0,
            faces_detected=[],
            total_faces=0
        )
        
        # TODO: Open video with cv2.VideoCapture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error = FacialRecognitionError(error=f"Cannot open video: {video_path}")
            return error.model_dump_json(indent=2)
        
        # TODO: Get video metadata (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        analyzed_count = 0
        face_id_counter = 0
        
        try:
            while True:
                # TODO: Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # TODO: Sample frames (check if frame_count % sample_rate == 0)
                if frame_count % sample_rate != 0:
                    continue
                
                analyzed_count += 1
                
                # TODO: Calculate timestamp
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                # TODO: Resize frame for speed (0.25 scale)
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])
                
                # TODO: Detect faces using face_recognition.face_locations()
                face_locations = face_recognition.face_locations(rgb_small)
                
                # TODO: For each detected face:
                for face_loc in face_locations:
                    top, right, bottom, left = face_loc
                    
                    # Scale coordinates back to original size (multiply by 4)
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                    
                    # Create DetectedFace model
                    detected_face = DetectedFace(
                        frame=frame_count,
                        timestamp=round(timestamp, 2),
                        face_id=face_id_counter,
                        location=FaceLocation(
                            top=top,
                            left=left,
                            right=right,
                            bottom=bottom
                        )
                    )
                    
                    # Append to result.faces_detected
                    result.faces_detected.append(detected_face)
                    face_id_counter += 1
        
        finally:
            # TODO: Release video capture
            cap.release()
        
        # TODO: Set result.frames_analyzed and result.total_faces
        result.frames_analyzed = analyzed_count
        result.total_faces = len(result.faces_detected)
        
        # TODO: Return result.model_dump_json(indent=2)
        return result.model_dump_json(indent=2)
```

**Key Points:**
- ✅ Models imported from separate file
- ✅ Only face detection, no emotion analysis
- ✅ TODOs show where you need to complete code
- ✅ Returns face_id for each detected face

---

## 🧪 Step 3: Create Unit Tests

**File**: `tests/test_facial_recognition_tool.py`

```python
import unittest
import json
from unittest.mock import patch, MagicMock

from tools.facial_recognition_tool import FacialRecognitionTool
from models.facial_recognition_models import FacialRecognitionResult


class TestFacialRecognitionTool(unittest.TestCase):
    """Test suite for FacialRecognitionTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tool = FacialRecognitionTool()
    
    def test_tool_name(self):
        """Test tool has correct name."""
        self.assertEqual(self.tool.name, "facial_recognition")
    
    def test_tool_description(self):
        """Test tool has description."""
        self.assertIsNotNone(self.tool.description)
        self.assertIn("face", self.tool.description.lower())
    
    def test_invalid_video_returns_error(self):
        """Test that invalid video path returns error."""
        result = self.tool._run("nonexistent_video.mp4")
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)
        self.assertIn("Cannot open", result_dict["error"])
    
    def test_result_is_valid_json(self):
        """Test tool returns valid JSON."""
        result = self.tool._run("invalid.mp4")
        
        try:
            json.loads(result)
        except json.JSONDecodeError:
            self.fail("Tool output is not valid JSON")
    
    def test_result_has_correct_structure(self):
        """Test result structure on error."""
        result = self.tool._run("invalid.mp4")
        result_dict = json.loads(result)
        
        self.assertIsInstance(result_dict, dict)
    
    @patch('cv2.VideoCapture')
    def test_with_mocked_video(self, mock_video):
        """Test with mocked video capture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30  # 30 fps
        mock_cap.read.return_value = (False, None)  # No frames
        mock_video.return_value = mock_cap
        
        result = self.tool._run("mock.mp4", sample_rate=5)
        result_dict = json.loads(result)
        
        self.assertIn("frames_analyzed", result_dict)
        self.assertEqual(result_dict["frames_analyzed"], 0)
    
    def test_no_emotion_in_result(self):
        """Test that detected faces have NO emotion field."""
        # This tool should NOT have emotion analysis
        result = self.tool._run("invalid.mp4")
        result_dict = json.loads(result)
        
        # If there were faces, they should not have emotion
        if "faces_detected" in result_dict:
            for face in result_dict["faces_detected"]:
                self.assertNotIn("emotion", face)


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

---

## 🧪 Step 4: Test Your Implementation

Run the tests:
```bash
python -m unittest tests.test_facial_recognition_tool -v
```

**Expected Output:**
```
test_invalid_video_returns_error ... ok
test_no_emotion_in_result ... ok
test_result_has_correct_structure ... ok
test_result_is_valid_json ... ok
test_tool_description ... ok
test_tool_name ... ok
test_with_mocked_video ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.015s

OK
```

---

## 🧪 Step 5: Test with Real Video

Create a simple test script:

**File**: `tests/manual_test_facial.py`

```python
import json
from tools.facial_recognition_tool import FacialRecognitionTool

# Test with your video
tool = FacialRecognitionTool()
result_json = tool._run(
    video_path="tech-challenge/video.mp4",
    sample_rate=30  # High sample rate for speed
)

result = json.loads(result_json)

print(f"✅ Frames Analyzed: {result['frames_analyzed']}")
print(f"✅ Total Faces: {result['total_faces']}")
print(f"✅ Faces Detected: {len(result['faces_detected'])}")

if result['faces_detected']:
    print(f"\n📊 First Detection:")
    first = result['faces_detected'][0]
    print(f"   Frame: {first['frame']}")
    print(f"   Timestamp: {first['timestamp']}s")
    print(f"   Face ID: {first['face_id']}")
    print(f"   Location: {first['location']}")
    
    # Verify no emotion field
    if 'emotion' in first:
        print("   ⚠️  WARNING: Face has emotion field! Should not be in facial recognition tool!")
    else:
        print("   ✅ No emotion field (correct - that's Module 3)")
```

Run it:
```bash
python tests/manual_test_facial.py
```

---

## 💡 Implementation Tips

### 1. Video Reading
```python
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    # Return error immediately
    error = FacialRecognitionError(error=f"Cannot open video: {video_path}")
    return error.model_dump_json(indent=2)
    
fps = cap.get(cv2.CAP_PROP_FPS)
```

### 2. Frame Sampling
```python
# Only process every Nth frame
if frame_count % sample_rate != 0:
    continue

analyzed_count += 1
```

### 3. Face Detection
```python
# Resize for speed (4x faster)
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])

# Detect faces
face_locations = face_recognition.face_locations(rgb_small)

# Scale coordinates back (multiply by 4)
for (top, right, bottom, left) in face_locations:
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
```

### 4. Creating Result with Pydantic
```python
detected_face = DetectedFace(
    frame=frame_count,
    timestamp=round(frame_count / fps, 2),
    face_id=face_id_counter,
    location=FaceLocation(top=top, left=left, right=right, bottom=bottom)
)

result.faces_detected.append(detected_face)
```

---

## ✅ Verification Checklist

Before moving to Module 3:

- [ ] Models created in `models/facial_recognition_models.py` (separate file)
- [ ] Tool implemented in `tools/facial_recognition_tool.py`
- [ ] Tool imports models from models file (not defined in tool)
- [ ] Unit tests pass (7 tests)
- [ ] Tool returns valid JSON
- [ ] Tool detects faces in test video
- [ ] Face locations are reasonable
- [ ] **NO emotion analysis in this tool**
- [ ] Each face has a unique face_id
- [ ] Result includes frames_analyzed and total_faces

---

## 🎯 What You've Learned

1. **Single Responsibility** - Tool only detects faces
2. **Model Separation** - Pydantic models in separate file
3. **Clean Imports** - Tool imports from models module
4. **Video Processing** - OpenCV for frame reading
5. **Face Detection** - face_recognition library
6. **Frame Sampling** - Performance optimization
7. **Unit Testing** - Test with mocks and validation

---

## 🔑 Key Differences from Old Tutorial

| Old (Incorrect) | New (Correct) |
|----------------|---------------|
| Models IN tool file | Models in separate `models/` file |
| Face detection + emotion | Face detection ONLY |
| One big tool | Separated into 2 tools |
| Hard to test | Easy to test independently |
| Tight coupling | Loose coupling |

---

## 📝 Next Steps

**Module 3**: Build the Emotion Detection Tool that uses faces from this tool!

**The separation allows:**
- Fast face detection pass
- Targeted emotion analysis
- Reuse face data for other purposes
- Test tools independently
- Follow Single Responsibility Principle

---

**Ready?** Complete this module, verify all tests pass, then move to `03-emotion-detection.md`! 🚀

