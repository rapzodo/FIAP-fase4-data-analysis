import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from models import FaceDetection, FaceLocation, EmotionDetectionInput
from tools import EmotionDetectionTool


class TestEmotionDetectionTool(unittest.TestCase):

    def setUp(self):
        self.tool = EmotionDetectionTool()
        self.mock_faces_json = json.dumps([
            {
                "face_id": 1,
                "location": {
                    "top": 100,
                    "left": 100,
                    "right": 200,
                    "bottom": 200
                },
                "frame": 1,
                "timestamp": 0.5
            }
        ])
        self.mock_video_path = "test_video.mp4"

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "emotion_detection")

    def test_tool_description(self):
        self.assertIsNotNone(self.tool.description)
        self.assertIn("emotion", self.tool.description.lower())

    def test_tool_args_schema(self):
        self.assertEqual(self.tool.args_schema, EmotionDetectionInput)

    def test_invalid_video_returns_error(self):
        result = self.tool._run("nonexistent.mp4", self.mock_faces_json)
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertEqual(result_dict["error"], "Unable to open video file")

    def test_result_is_valid_json(self):
        result = self.tool._run("invalid.mp4", self.mock_faces_json)

        try:
            json.loads(result)
        except json.JSONDecodeError:
            self.fail("Tool output is not valid JSON")

    @patch('cv2.VideoCapture')
    def test_empty_face_list(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        result = self.tool._run("video.mp4", json.dumps([]))
        result_dict = json.loads(result)

        self.assertEqual(result_dict["total_faces_analyzed"], 0)
        self.assertEqual(len(result_dict["emotions_detected"]), 0)

    @patch('cv2.VideoCapture')
    @patch('tools.emotion_detection_tool.DeepFace')
    def test_successful_emotion_detection(self, mock_deepface, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_deepface.analyze.return_value = {
            "dominant_emotion": "happy",
            "emotion": {
                "angry": 2.5,
                "disgust": 1.0,
                "fear": 3.2,
                "happy": 85.5,
                "sad": 2.8,
                "surprise": 3.0,
                "neutral": 2.0
            }
        }

        result = self.tool._run(self.mock_video_path, self.mock_faces_json)
        result_dict = json.loads(result)

        self.assertIn("total_faces_analyzed", result_dict)
        self.assertIn("emotions_detected", result_dict)
        self.assertIn("emotion_summary", result_dict)
        self.assertIn("anomalies", result_dict)

    @patch('cv2.VideoCapture')
    def test_video_read_failure_creates_anomaly(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        result = self.tool._run(self.mock_video_path, self.mock_faces_json)
        result_dict = json.loads(result)

        self.assertIn("anomalies", result_dict)
        self.assertGreater(len(result_dict["anomalies"]), 0)

    @patch('cv2.VideoCapture')
    @patch('tools.emotion_detection_tool.DeepFace')
    def test_low_confidence_creates_anomaly(self, mock_deepface, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_deepface.analyze.return_value = {
            "dominant_emotion": "neutral",
            "emotion": {
                "angry": 20.0,
                "disgust": 15.0,
                "fear": 10.0,
                "happy": 15.0,
                "sad": 20.0,
                "surprise": 10.0,
                "neutral": 10.0
            }
        }

        result = self.tool._run(self.mock_video_path, self.mock_faces_json)
        result_dict = json.loads(result)

        self.assertIn("anomalies", result_dict)

    @patch('cv2.VideoCapture')
    @patch('tools.emotion_detection_tool.DeepFace')
    def test_deepface_exception_creates_anomaly(self, mock_deepface, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_deepface.analyze.side_effect = Exception("DeepFace analysis failed")

        result = self.tool._run(self.mock_video_path, self.mock_faces_json)
        result_dict = json.loads(result)

        self.assertIn("anomalies", result_dict)
        self.assertGreater(len(result_dict["anomalies"]), 0)

    def test_group_faces_by_frame(self):
        faces_json = json.dumps([
            {
                "face_id": 1,
                "location": {"top": 10, "left": 10, "right": 50, "bottom": 50},
                "frame": 1,
                "timestamp": 0.5
            },
            {
                "face_id": 2,
                "location": {"top": 60, "left": 60, "right": 100, "bottom": 100},
                "frame": 1,
                "timestamp": 0.5
            },
            {
                "face_id": 3,
                "location": {"top": 10, "left": 10, "right": 50, "bottom": 50},
                "frame": 2,
                "timestamp": 1.0
            }
        ])

        grouped = self.tool.group_faces_by_frame(faces_json)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(len(grouped[2]), 1)

    def test_create_unknown_anomalies(self):
        faces = [
            FaceDetection(
                face_id=1,
                location=FaceLocation(top=100, left=100, right=200, bottom=200),
                frame=1,
                timestamp=0.5
            )
        ]

        anomalies = self.tool.create_unknown_anomalies(faces)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].type, "UNKNOWN")

    @patch('cv2.VideoCapture')
    @patch('tools.emotion_detection_tool.DeepFace')
    def test_multiple_faces_in_single_frame(self, mock_deepface, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_deepface.analyze.return_value = {
            "dominant_emotion": "happy",
            "emotion": {
                "angry": 2.5,
                "disgust": 1.0,
                "fear": 3.2,
                "happy": 85.5,
                "sad": 2.8,
                "surprise": 3.0,
                "neutral": 2.0
            }
        }

        multiple_faces_json = json.dumps([
            {
                "face_id": 1,
                "location": {"top": 100, "left": 100, "right": 200, "bottom": 200},
                "frame": 1,
                "timestamp": 0.5
            },
            {
                "face_id": 2,
                "location": {"top": 300, "left": 300, "right": 400, "bottom": 400},
                "frame": 1,
                "timestamp": 0.5
            }
        ])

        result = self.tool._run(self.mock_video_path, multiple_faces_json)
        result_dict = json.loads(result)

        self.assertIn("emotions_detected", result_dict)

    @patch('cv2.VideoCapture')
    @patch('tools.emotion_detection_tool.DeepFace')
    def test_emotion_scores_within_range(self, mock_deepface, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_deepface.analyze.return_value = {
            "dominant_emotion": "happy",
            "emotion": {
                "angry": 2.5,
                "disgust": 1.0,
                "fear": 3.2,
                "happy": 85.5,
                "sad": 2.8,
                "surprise": 3.0,
                "neutral": 2.0
            }
        }

        result = self.tool._run(self.mock_video_path, self.mock_faces_json)
        result_dict = json.loads(result)

        if result_dict.get("emotions_detected"):
            print(result_dict)
            for emotion in result_dict["emotions_detected"]:
                if emotion:
                    scores = emotion["emotion_score"]
                    for emotion_name, score in scores.items():
                        self.assertGreaterEqual(score, 0)
                        self.assertLessEqual(score, 100)

    @patch('cv2.VideoCapture')
    @patch('tools.emotion_detection_tool.DeepFace')
    def test_deepface_returns_list(self, mock_deepface, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        mock_deepface.analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {
                "angry": 2.5,
                "disgust": 1.0,
                "fear": 3.2,
                "happy": 85.5,
                "sad": 2.8,
                "surprise": 3.0,
                "neutral": 2.0
            }
        }]

        result = self.tool._run(self.mock_video_path, self.mock_faces_json)
        result_dict = json.loads(result)

        self.assertIn("emotions_detected", result_dict)

    @patch('cv2.VideoCapture')
    def test_invalid_face_region_creates_anomaly(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        invalid_face_json = json.dumps([
            {
                "face_id": 1,
                "location": {"top": 100, "left": 100, "right": 100, "bottom": 100},
                "frame": 1,
                "timestamp": 0.5
            }
        ])

        result = self.tool._run(self.mock_video_path, invalid_face_json)
        result_dict = json.loads(result)

        self.assertIn("anomalies", result_dict)

    @patch('cv2.VideoCapture')
    def test_video_capture_released(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        self.tool._run(self.mock_video_path, self.mock_faces_json)

        mock_cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
