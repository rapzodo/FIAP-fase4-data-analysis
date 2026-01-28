from unittest import TestCase
from unittest.mock import Mock

from crewai.hooks import LLMCallHookContext

from video_analysis_summary_crew import VideoAnalysisSummaryCrew


class TestCleanReasoningText(TestCase):

    def _create_mock_context(self, response_text):
        mock_context = Mock(spec=LLMCallHookContext)
        mock_context.response = response_text
        return mock_context

    def test_clean_reasoning_text_with_reasoning_plan(self):
        response = 'Aggregate the activities detection data and create a statistic report. Use timestamps to create a timeline for activities detected"   Reasoning Plan: ## Strategic Plan: Human Activity Data Analysis Report'
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "## Strategic Plan: Human Activity Data Analysis Report"

    def test_clean_reasoning_text_with_strategic_plan(self):
        response = "Some introduction text Strategic Plan: The actual content we want"
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "The actual content we want"

    def test_clean_reasoning_text_with_analysis_plan(self):
        response = "Preliminary thoughts Analysis Plan: Final analysis here"
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "Final analysis here"

    def test_clean_reasoning_text_with_plan(self):
        response = "Random text Plan: Actual plan details"
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "Actual plan details"

    def test_clean_reasoning_text_with_final_answer(self):
        response = """Okay, I understand. The previous attempt failed validation because the output was not in Portuguese. I need to translate the provided markdown reports into Portuguese (pt-br) and ensure the formatting is correct. I will use the markdown syntax as requested.

Final Answer:
```markdown"""
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "```markdown"

    def test_clean_reasoning_text_no_marker(self):
        response = "This text has no reasoning markers"
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "This text has no reasoning markers"

    def test_clean_reasoning_text_empty_string(self):
        response = ""
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == ""

    def test_clean_reasoning_text_none(self):
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(None)
        assert result == "None"

    def test_clean_reasoning_text_multiple_markers(self):
        response = "Text Reasoning Plan: First marker Plan: Second marker"
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "First marker Plan: Second marker"

    def test_clean_reasoning_text_with_whitespace(self):
        response = "Some text   Reasoning Plan:   Content with spaces   "
        mock_context = self._create_mock_context(response)
        result = VideoAnalysisSummaryCrew.clean_reasoning_text(mock_context)
        assert result == "Content with spaces"
