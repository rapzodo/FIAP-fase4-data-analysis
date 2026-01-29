from unittest import TestCase
from unittest.mock import Mock

from utils.helper_functions import (
    format_frame_timestamp,
    capture_statistics,
    reset_crew_memory,
    clean_detection_tools_input
)


class TestFormatFrameTimestamp(TestCase):

    def test_format_frame_timestamp_zero(self):
        result = format_frame_timestamp(0.0)
        self.assertEqual(result, "0:00:00")

    def test_format_frame_timestamp_milliseconds(self):
        result = format_frame_timestamp(500.0)
        self.assertEqual(result, "0:00:00.500000")

    def test_format_frame_timestamp_seconds(self):
        result = format_frame_timestamp(5000.0)
        self.assertEqual(result, "0:00:05")

    def test_format_frame_timestamp_minutes(self):
        result = format_frame_timestamp(65000.0)
        self.assertEqual(result, "0:01:05")

    def test_format_frame_timestamp_hours(self):
        result = format_frame_timestamp(3665000.0)
        self.assertEqual(result, "1:01:05")

    def test_format_frame_timestamp_negative(self):
        result = format_frame_timestamp(-1000.0)
        self.assertTrue(result.startswith("-"))

    def test_format_frame_timestamp_large_value(self):
        result = format_frame_timestamp(86400000.0)
        self.assertIn("1 day", result)


class TestCaptureStatistics(TestCase):

    def test_capture_statistics_new_activity(self):
        stats = {}
        capture_statistics(stats, "activity_1", 1000.0)

        self.assertEqual(len(stats), 1)
        self.assertIn("activity_1", stats)
        self.assertEqual(stats["activity_1"].detection_name, "activity_1")
        self.assertEqual(stats["activity_1"].total_fames_appearances, 1)
        self.assertEqual(len(stats["activity_1"].timestamps), 1)

    def test_capture_statistics_existing_activity(self):
        stats = {}
        capture_statistics(stats, "activity_1", 1000.0)
        capture_statistics(stats, "activity_1", 2000.0)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats["activity_1"].total_fames_appearances, 2)
        self.assertEqual(len(stats["activity_1"].timestamps), 2)

    def test_capture_statistics_multiple_activities(self):
        stats = {}
        capture_statistics(stats, "activity_1", 1000.0)
        capture_statistics(stats, "activity_2", 2000.0)
        capture_statistics(stats, "activity_3", 3000.0)

        self.assertEqual(len(stats), 3)
        self.assertIn("activity_1", stats)
        self.assertIn("activity_2", stats)
        self.assertIn("activity_3", stats)

    def test_capture_statistics_timestamp_format(self):
        stats = {}
        capture_statistics(stats, "activity_1", 5000.0)

        self.assertEqual(stats["activity_1"].timestamps[0], "0:00:05")

    def test_capture_statistics_empty_dict(self):
        stats = {}
        capture_statistics(stats, "test", 0.0)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats["test"].total_fames_appearances, 1)

    def test_capture_statistics_incremental_count(self):
        stats = {}
        for i in range(5):
            capture_statistics(stats, "activity", float(i * 1000))

        self.assertEqual(stats["activity"].total_fames_appearances, 5)
        self.assertEqual(len(stats["activity"].timestamps), 5)

    def test_capture_statistics_different_activities_independent(self):
        stats = {}
        capture_statistics(stats, "activity_1", 1000.0)
        capture_statistics(stats, "activity_1", 2000.0)
        capture_statistics(stats, "activity_2", 3000.0)

        self.assertEqual(stats["activity_1"].total_fames_appearances, 2)
        self.assertEqual(stats["activity_2"].total_fames_appearances, 1)


class TestResetCrewMemory(TestCase):

    def test_reset_crew_memory_all(self):
        mock_crew = Mock()
        reset_crew_memory(mock_crew, "all")

        mock_crew.reset_memories.assert_called_once_with("all")

    def test_reset_crew_memory_short_term(self):
        mock_crew = Mock()
        reset_crew_memory(mock_crew, "short_term")

        mock_crew.reset_memories.assert_called_once_with("short_term")

    def test_reset_crew_memory_long_term(self):
        mock_crew = Mock()
        reset_crew_memory(mock_crew, "long_term")

        mock_crew.reset_memories.assert_called_once_with("long_term")

    def test_reset_crew_memory_default_parameter(self):
        mock_crew = Mock()
        reset_crew_memory(mock_crew)

        mock_crew.reset_memories.assert_called_once_with("all")

    def test_reset_crew_memory_custom_value(self):
        mock_crew = Mock()
        reset_crew_memory(mock_crew, "custom")

        mock_crew.reset_memories.assert_called_once_with("custom")


class TestCleanDetectionToolsInput(TestCase):

    def test_clean_detection_tools_input_with_properties(self):
        mock_context = Mock()
        mock_context.tool_name = "test_tool"
        mock_context.tool_input = {
            'properties': {
                'video_path': '/path/to/video.mp4',
                'frame_rate': 30
            }
        }

        clean_detection_tools_input(mock_context)

        self.assertEqual(mock_context.tool_input['video_path'], '/path/to/video.mp4')
        self.assertEqual(mock_context.tool_input['frame_rate'], 30)
        self.assertNotIn('properties', mock_context.tool_input)

    def test_clean_detection_tools_input_with_media_pipe_model(self):
        mock_context = Mock()
        mock_context.tool_name = "activity_detection"
        mock_context.tool_input = {
            'properties': {
                'video_path': '/path/to/video.mp4',
                'frame_rate': 30,
                'media_pipe_model': 'LITE'
            }
        }

        clean_detection_tools_input(mock_context)

        self.assertEqual(mock_context.tool_input['video_path'], '/path/to/video.mp4')
        self.assertEqual(mock_context.tool_input['frame_rate'], 30)
        self.assertEqual(mock_context.tool_input['media_pipe_model'], 'LITE')
        self.assertNotIn('properties', mock_context.tool_input)

    def test_clean_detection_tools_input_without_properties(self):
        mock_context = Mock()
        mock_context.tool_name = "test_tool"
        mock_context.tool_input = {
            'video_path': '/path/to/video.mp4',
            'frame_rate': 30
        }

        clean_detection_tools_input(mock_context)

        self.assertEqual(mock_context.tool_input['video_path'], '/path/to/video.mp4')
        self.assertEqual(mock_context.tool_input['frame_rate'], 30)

    def test_clean_detection_tools_input_without_media_pipe_model(self):
        mock_context = Mock()
        mock_context.tool_name = "emotion_detection"
        mock_context.tool_input = {
            'properties': {
                'video_path': '/path/to/video.mp4',
                'frame_rate': 15
            }
        }

        clean_detection_tools_input(mock_context)

        self.assertEqual(mock_context.tool_input['video_path'], '/path/to/video.mp4')
        self.assertEqual(mock_context.tool_input['frame_rate'], 15)
        self.assertNotIn('media_pipe_model', mock_context.tool_input)
        self.assertNotIn('properties', mock_context.tool_input)

    def test_clean_detection_tools_input_empty_properties(self):
        mock_context = Mock()
        mock_context.tool_name = "test_tool"
        mock_context.tool_input = {
            'properties': {}
        }

        with self.assertRaises(KeyError):
            clean_detection_tools_input(mock_context)

    def test_clean_detection_tools_input_preserves_other_fields(self):
        mock_context = Mock()
        mock_context.tool_name = "test_tool"
        mock_context.tool_input = {
            'properties': {
                'video_path': '/path/to/video.mp4',
                'frame_rate': 30
            },
            'other_field': 'value'
        }

        clean_detection_tools_input(mock_context)

        self.assertEqual(mock_context.tool_input['video_path'], '/path/to/video.mp4')
        self.assertEqual(mock_context.tool_input['frame_rate'], 30)
        self.assertEqual(mock_context.tool_input['other_field'], 'value')
        self.assertNotIn('properties', mock_context.tool_input)

    def test_clean_detection_tools_input_all_models(self):
        for model in ['LITE', 'FULL', 'HEAVY']:
            mock_context = Mock()
            mock_context.tool_name = "activity_detection"
            mock_context.tool_input = {
                'properties': {
                    'video_path': '/test.mp4',
                    'frame_rate': 30,
                    'media_pipe_model': model
                }
            }

            clean_detection_tools_input(mock_context)

            self.assertEqual(mock_context.tool_input['media_pipe_model'], model)

    def test_clean_detection_tools_input_different_frame_rates(self):
        for rate in [1, 5, 10, 30, 60]:
            mock_context = Mock()
            mock_context.tool_name = "test_tool"
            mock_context.tool_input = {
                'properties': {
                    'video_path': '/test.mp4',
                    'frame_rate': rate
                }
            }

            clean_detection_tools_input(mock_context)

            self.assertEqual(mock_context.tool_input['frame_rate'], rate)

    def test_clean_detection_tools_input_different_video_paths(self):
        paths = [
            '/path/to/video.mp4',
            'relative/path/video.mp4',
            'C:\\Windows\\video.mp4',
            './local_video.mp4'
        ]

        for path in paths:
            mock_context = Mock()
            mock_context.tool_name = "test_tool"
            mock_context.tool_input = {
                'properties': {
                    'video_path': path,
                    'frame_rate': 30
                }
            }

            clean_detection_tools_input(mock_context)

            self.assertEqual(mock_context.tool_input['video_path'], path)
