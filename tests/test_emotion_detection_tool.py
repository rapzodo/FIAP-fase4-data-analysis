import json
from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np
from tensorflow.python.ops.check_ops import assert_equal

from tools.emotion_detection_tool import EmotionDetectionTool


class TestEmotionDetectionTool(TestCase):

    def setUp(self):
        self.tool = EmotionDetectionTool()

    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_video_not_opened(self, mock_video_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        result = self.tool._run("invalid_video.mp4", 30)
        result_dict = json.loads(result)

        assert_equal(result_dict["error"], "Unable to open video file")
        mock_cap.release.assert_not_called()

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_with_valid_emotions(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=2)
        mock_analyze.return_value = [
            {"dominant_emotion": "happy", "face_confidence": 0.95}
        ]

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 1)
        assert_equal(result_dict["statistics"][0]["detection_name"], "dominant emotion - happy")
        assert_equal(result_dict["statistics"][0]["total_fames_appearances"], 2)
        mock_cap.release.assert_called_once()

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_with_low_confidence(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)
        mock_analyze.return_value = [
            {"dominant_emotion": "sad", "face_confidence": 0.2}
        ]

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 1)
        assert_equal(result_dict["statistics"][0]["detection_name"], "anomaly - low confidence")

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_with_zero_confidence(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)
        mock_analyze.return_value = [
            {"dominant_emotion": "angry", "face_confidence": 0.0}
        ]

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 0)

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_with_frame_sampling(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=10)
        mock_analyze.return_value = [
            {"dominant_emotion": "neutral", "face_confidence": 0.85}
        ]

        result = self.tool._run("test_video.mp4", 5)
        result_dict = json.loads(result)

        assert_equal(result_dict["statistics"][0]["total_fames_appearances"], 2)

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_with_multiple_faces(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)
        mock_analyze.return_value = [
            {"dominant_emotion": "happy", "face_confidence": 0.9},
            {"dominant_emotion": "sad", "face_confidence": 0.8}
        ]

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 2)
        emotions = {stat["detection_name"] for stat in result_dict["statistics"]}
        self.assertIn("dominant emotion - happy", emotions)
        self.assertIn("dominant emotion - sad", emotions)

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_with_exception(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)
        mock_analyze.side_effect = Exception("Analysis failed")

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 0)
        mock_cap.release.assert_called_once()

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_mixed_confidence_levels(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=3)
        mock_analyze.side_effect = [
            [{"dominant_emotion": "happy", "face_confidence": 0.9}],
            [{"dominant_emotion": "sad", "face_confidence": 0.2}],
            [{"dominant_emotion": "angry", "face_confidence": 0.5}]
        ]

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 3)

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_empty_analysis_result(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)
        mock_analyze.return_value = []

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 0)

    @patch('tools.emotion_detection_tool.tqdm')
    @patch('tools.emotion_detection_tool.DeepFace.analyze')
    @patch('tools.emotion_detection_tool.cv2.VideoCapture')
    def test_emotion_detection_boundary_confidence(self, mock_video_capture, mock_analyze, mock_tqdm):
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=2)
        mock_analyze.side_effect = [
            [{"dominant_emotion": "happy", "face_confidence": 0.3}],
            [{"dominant_emotion": "sad", "face_confidence": 0.29}]
        ]

        result = self.tool._run("test_video.mp4", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 2)
        emotions = [stat["detection_name"] for stat in result_dict["statistics"]]
        self.assertIn("dominant emotion - happy", emotions)
        self.assertIn("anomaly - low confidence", emotions)

    def _setup_mock_video_capture(self, mock_video_capture, total_frames):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: total_frames if prop == 7 else 1000.0

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, mock_frame)] * total_frames + [(False, None)]
        mock_video_capture.return_value = mock_cap
        return mock_cap
