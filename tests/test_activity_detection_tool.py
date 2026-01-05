import os
import unittest
from pathlib import Path

from tools.activity_detection_tool import ActivityDetectionTool, MediaPipeModel


class TestActivityDetectionTool(unittest.TestCase):

    def setUp(self):
        self.tool = ActivityDetectionTool()

    def test_download_model(self):
        self.tool.download_model(MediaPipeModel.FULL)
        model_path = Path(self.tool.get_model_path(MediaPipeModel.FULL))
        self.assertTrue(model_path.exists(), f"Model file not found at {model_path}")
        self.assertTrue(model_path.is_file(), f"Expected a file at {model_path}")

    def tearDown(self):
        os.remove(self.tool.get_model_path(MediaPipeModel.FULL))
