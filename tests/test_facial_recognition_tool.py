import json
import unittest
from unittest.mock import patch, MagicMock

from polars import self_dtype

from tools.facial_recognition_tool import FacialRecognitionTool


class TestFacialRecognitionTool(unittest.TestCase):

    def setUp(self):
        self.tool = FacialRecognitionTool()

    def test_fail_on_invalid_video(self):
        result = self.tool._run("invalid_video.mp4")
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("Unable to open video", result_dict["error"])

    @patch("cv2.VideoCapture")
    def test_with_mock_video(self, mock_video):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30
        mock_cap.read.return_value = (False, None)
        mock_video.return_value = mock_cap

        result = self.tool._run("test_video.mp4", sample_rate=5)
        result_dict = json.loads(result)

        self.assertIn("frames_analyzed", result_dict)
        self.assertEqual(result_dict["frames_analyzed"], 0)
