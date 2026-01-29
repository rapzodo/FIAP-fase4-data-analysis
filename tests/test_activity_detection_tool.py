import json
from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np
from tensorflow.python.ops.check_ops import assert_equal

from tools.activity_detection_tool import ActivityDetectionTool, MediaPipeModel


class TestActivityDetectionTool(TestCase):

    def setUp(self):
        self.tool = ActivityDetectionTool()

    @patch('tools.activity_detection_tool.cv2.VideoCapture')
    @patch('tools.activity_detection_tool.ActivityDetectionTool.download_model')
    @patch('tools.activity_detection_tool.PoseLandmarker.create_from_options')
    def test_activity_detection_video_not_opened(self, mock_pose_landmarker, mock_download, mock_video_capture):
        mock_download.return_value = "/fake/path/model.task"
        mock_landmarker_instance = MagicMock()
        mock_pose_landmarker.return_value = mock_landmarker_instance

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        result = self.tool._run("invalid_video.mp4", "LITE", 30)
        result_dict = json.loads(result)

        assert_equal(result_dict["error"], "Unable to open video source invalid_video.mp4.")
        mock_cap.release.assert_not_called()
        mock_landmarker_instance.close.assert_not_called()

    @patch('tools.activity_detection_tool.cv2.VideoCapture')
    @patch('tools.activity_detection_tool.ActivityDetectionTool.download_model')
    @patch('tools.activity_detection_tool.PoseLandmarker.create_from_options')
    def test_activity_detection_with_valid_poses(self, mock_pose_landmarker, mock_download, mock_video_capture):
        mock_download.return_value = "/fake/path/model.task"

        mock_landmarker_instance = MagicMock()
        mock_pose_result = self._create_mock_pose_result_with_landmarks()
        mock_landmarker_instance.detect_for_video.return_value = mock_pose_result
        mock_pose_landmarker.return_value = mock_landmarker_instance

        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=2)

        result = self.tool._run("test_video.mp4", "LITE", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 2)
        detection_names = {stat["detection_name"] for stat in result_dict["statistics"]}
        self.assertTrue(any("Pose activity" in name for name in detection_names))
        self.assertTrue(any("Hand activity" in name for name in detection_names))
        mock_cap.release.assert_called_once()
        mock_landmarker_instance.close.assert_called_once()

    @patch('tools.activity_detection_tool.cv2.VideoCapture')
    @patch('tools.activity_detection_tool.ActivityDetectionTool.download_model')
    @patch('tools.activity_detection_tool.PoseLandmarker.create_from_options')
    def test_activity_detection_with_no_pose_landmarks(self, mock_pose_landmarker, mock_download, mock_video_capture):
        mock_download.return_value = "/fake/path/model.task"

        mock_landmarker_instance = MagicMock()
        mock_pose_result = MagicMock()
        mock_pose_result.pose_landmarks = []
        mock_landmarker_instance.detect_for_video.return_value = mock_pose_result
        mock_pose_landmarker.return_value = mock_landmarker_instance

        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)

        result = self.tool._run("test_video.mp4", "LITE", 1)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 0)
        mock_cap.release.assert_called_once()
        mock_landmarker_instance.close.assert_called_once()

    @patch('tools.activity_detection_tool.cv2.VideoCapture')
    @patch('tools.activity_detection_tool.ActivityDetectionTool.download_model')
    @patch('tools.activity_detection_tool.PoseLandmarker.create_from_options')
    def test_activity_detection_with_frame_sampling(self, mock_pose_landmarker, mock_download, mock_video_capture):
        mock_download.return_value = "/fake/path/model.task"

        mock_landmarker_instance = MagicMock()
        mock_pose_result = self._create_mock_pose_result_with_landmarks()
        mock_landmarker_instance.detect_for_video.return_value = mock_pose_result
        mock_pose_landmarker.return_value = mock_landmarker_instance

        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=10)

        result = self.tool._run("test_video.mp4", "LITE", 5)
        result_dict = json.loads(result)

        assert_equal(len(result_dict["statistics"]), 2)
        for stat in result_dict["statistics"]:
            assert_equal(stat["total_fames_appearances"], 2)

    @patch('tools.activity_detection_tool.cv2.VideoCapture')
    @patch('tools.activity_detection_tool.ActivityDetectionTool.download_model')
    @patch('tools.activity_detection_tool.PoseLandmarker.create_from_options')
    def test_activity_detection_with_exception(self, mock_pose_landmarker, mock_download, mock_video_capture):
        mock_download.return_value = "/fake/path/model.task"

        mock_landmarker_instance = MagicMock()
        mock_landmarker_instance.detect_for_video.side_effect = Exception("Detection failed")
        mock_pose_landmarker.return_value = mock_landmarker_instance

        mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)

        result = self.tool._run("test_video.mp4", "LITE", 1)
        result_dict = json.loads(result)

        assert_equal(result_dict["error"], "Detection failed")
        mock_cap.release.assert_called_once()
        mock_landmarker_instance.close.assert_called_once()

    @patch('tools.activity_detection_tool.cv2.VideoCapture')
    @patch('tools.activity_detection_tool.ActivityDetectionTool.download_model')
    @patch('tools.activity_detection_tool.PoseLandmarker.create_from_options')
    def test_activity_detection_different_models(self, mock_pose_landmarker, mock_download, mock_video_capture):
        mock_download.return_value = "/fake/path/model.task"

        mock_landmarker_instance = MagicMock()
        mock_pose_result = self._create_mock_pose_result_with_landmarks()
        mock_landmarker_instance.detect_for_video.return_value = mock_pose_result
        mock_pose_landmarker.return_value = mock_landmarker_instance

        for model in ["LITE", "FULL", "HEAVY"]:
            mock_cap = self._setup_mock_video_capture(mock_video_capture, total_frames=1)

            result = self.tool._run("test_video.mp4", model, 1)
            result_dict = json.loads(result)

            assert_equal(len(result_dict["statistics"]), 2)
            mock_download.assert_called_with(MediaPipeModel[model])

    def test_detect_activity_from_pose_with_no_landmarks(self):
        result = self.tool.detect_activity_from_pose([])
        self.assertIsNone(result)

    def test_detect_activity_from_pose_with_none(self):
        result = self.tool.detect_activity_from_pose(None)
        self.assertIsNone(result)

    def test_detect_hand_position_hands_raised(self):
        landmarks = self._create_mock_landmarks(
            left_wrist_y=0.2, right_wrist_y=0.2,
            left_shoulder_y=0.3, right_shoulder_y=0.3
        )
        from models.activity_detection_models import BodyLandmarks
        body_landmarks = BodyLandmarks(landmarks)

        result = self.tool.detect_hand_position(body_landmarks)
        assert_equal(result, "hands_raised")

    def test_detect_hand_position_hands_down(self):
        landmarks = self._create_mock_landmarks(
            left_wrist_y=0.6, right_wrist_y=0.6,
            left_shoulder_y=0.3, right_shoulder_y=0.3
        )
        from models.activity_detection_models import BodyLandmarks
        body_landmarks = BodyLandmarks(landmarks)

        result = self.tool.detect_hand_position(body_landmarks)
        assert_equal(result, "hands_down")

    def test_detect_body_movement_standing(self):
        landmarks = self._create_mock_landmarks(
            left_knee_y=0.7, right_knee_y=0.7,
            left_hip_y=0.5, right_hip_y=0.5
        )
        from models.activity_detection_models import BodyLandmarks
        body_landmarks = BodyLandmarks(landmarks)

        result = self.tool.detect_body_movement(body_landmarks)
        assert_equal(result, "standing")

    def _setup_mock_video_capture(self, mock_video_capture, total_frames):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 1000.0

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        read_results = [(True, mock_frame)] * total_frames + [(False, None)]
        mock_cap.read.side_effect = read_results
        mock_video_capture.return_value = mock_cap
        return mock_cap

    def _create_mock_pose_result_with_landmarks(self):
        mock_pose_result = MagicMock()
        landmarks = self._create_mock_landmarks()
        mock_pose_result.pose_landmarks = [[landmarks[i] for i in range(33)]]
        return mock_pose_result

    def _create_mock_landmarks(self, left_wrist_y=0.5, right_wrist_y=0.5,
                               left_shoulder_y=0.3, right_shoulder_y=0.3,
                               left_hip_y=0.5, right_hip_y=0.5,
                               left_knee_y=0.7, right_knee_y=0.7,
                               left_wrist_x=0.3, right_wrist_x=0.7,
                               left_elbow_x=0.35, right_elbow_x=0.65):
        landmarks = []
        landmark_positions = {
            11: (left_shoulder_y, 0.3),
            12: (right_shoulder_y, 0.7),
            13: (0.4, left_elbow_x),
            14: (0.4, right_elbow_x),
            15: (left_wrist_y, left_wrist_x),
            16: (right_wrist_y, right_wrist_x),
            23: (left_hip_y, 0.3),
            24: (right_hip_y, 0.7),
            25: (left_knee_y, 0.3),
            26: (right_knee_y, 0.7),
        }

        for i in range(33):
            mock_landmark = MagicMock()
            if i in landmark_positions:
                mock_landmark.y = landmark_positions[i][0]
                mock_landmark.x = landmark_positions[i][1]
            else:
                mock_landmark.y = 0.5
                mock_landmark.x = 0.5
            landmarks.append(mock_landmark)

        return landmarks

    @patch('os.path.exists')
    @patch('urllib.request.urlretrieve')
    @patch('os.makedirs')
    def test_download_model_when_not_exists(self, mock_makedirs, mock_urlretrieve, mock_exists):
        mock_exists.return_value = False

        result = self.tool.download_model(MediaPipeModel.LITE)

        mock_makedirs.assert_called_once()
        mock_urlretrieve.assert_called_once()
        self.assertTrue(result.endswith('pose_landmarker_lite.task'))

    @patch('os.path.exists')
    @patch('urllib.request.urlretrieve')
    def test_download_model_when_exists(self, mock_urlretrieve, mock_exists):
        mock_exists.return_value = True

        result = self.tool.download_model(MediaPipeModel.LITE)

        mock_urlretrieve.assert_not_called()
        self.assertTrue(result.endswith('pose_landmarker_lite.task'))
