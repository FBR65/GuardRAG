"""
Basic tests for GuardRAG components
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.input_guardrail import InputGuardrail, ValidationResult
from src.output_guardrail import OutputGuardrail, OutputValidationResult
from src.colpali_integration import DocumentPage


class TestInputGuardrail:
    """Test cases for input guardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a mock input guardrail."""
        return InputGuardrail(
            llm_endpoint="http://mock-endpoint",
            llm_api_key="mock-key",
            llm_model="mock-model",
            enable_llm_validation=False,  # Disable LLM for testing
        )

    def test_basic_validation_empty_query(self, guardrail):
        """Test basic validation with empty query."""
        result = guardrail._basic_validation("")
        assert result.result == ValidationResult.REJECTED
        assert "leer" in result.reason.lower()

    def test_basic_validation_short_query(self, guardrail):
        """Test basic validation with too short query."""
        result = guardrail._basic_validation("hi")
        assert result.result == ValidationResult.REJECTED
        assert "kurz" in result.reason.lower()

    def test_basic_validation_long_query(self, guardrail):
        """Test basic validation with too long query."""
        long_query = "x" * 1001
        result = guardrail._basic_validation(long_query)
        assert result.result == ValidationResult.REJECTED
        assert "lang" in result.reason.lower()

    def test_basic_validation_valid_query(self, guardrail):
        """Test basic validation with valid query."""
        result = guardrail._basic_validation(
            "What is the methodology used in this study?"
        )
        assert result.result == ValidationResult.ACCEPTED

    def test_keyword_validation_banned_words(self, guardrail):
        """Test keyword validation with banned words."""
        result = guardrail._keyword_validation("How to hack the system?")
        assert result.result == ValidationResult.REJECTED
        assert "hack" in result.reason

    def test_keyword_validation_scientific_words(self, guardrail):
        """Test keyword validation with scientific words."""
        result = guardrail._keyword_validation("What research methodology was used?")
        assert result.result == ValidationResult.ACCEPTED
        assert "research" in result.reason or "methodology" in result.reason

    def test_add_banned_keyword(self, guardrail):
        """Test adding banned keywords."""
        initial_count = len(guardrail.banned_keywords)
        guardrail.add_banned_keyword("testword")
        assert len(guardrail.banned_keywords) == initial_count + 1
        assert "testword" in guardrail.banned_keywords


class TestOutputGuardrail:
    """Test cases for output guardrail."""

    @pytest.fixture
    def guardrail(self):
        """Create a mock output guardrail."""
        with (
            patch("src.output_guardrail.OpenAIProvider"),
            patch("src.output_guardrail.OpenAIModel"),
            patch("src.output_guardrail.Agent"),
        ):
            return OutputGuardrail(
                llm_endpoint="http://mock-endpoint",
                llm_api_key="mock-key",
                llm_model="mock-model",
            )

    def test_basic_validation_empty_response(self, guardrail):
        """Test basic validation with empty response."""
        result = guardrail._basic_validation("")
        assert result.result == OutputValidationResult.REJECTED
        assert "leer" in result.reason.lower()

    def test_basic_validation_short_response(self, guardrail):
        """Test basic validation with too short response."""
        result = guardrail._basic_validation("Yes.")
        assert result.result == OutputValidationResult.REJECTED
        assert "kurz" in result.reason.lower()

    def test_basic_validation_error_indicators(self, guardrail):
        """Test basic validation with error indicators."""
        result = guardrail._basic_validation("Error: Could not process the request")
        assert result.result == OutputValidationResult.REJECTED
        assert "fehler" in result.reason.lower()

    def test_basic_validation_valid_response(self, guardrail):
        """Test basic validation with valid response."""
        response = "Based on the provided documents, the study used a quantitative methodology..."
        result = guardrail._basic_validation(response)
        assert result.result == OutputValidationResult.APPROVED

    def test_pattern_validation_toxic_content(self, guardrail):
        """Test pattern validation with toxic content."""
        response = "You should kill the process immediately"
        result = guardrail._pattern_validation(response)
        assert result.result == OutputValidationResult.REJECTED

    def test_pattern_validation_hallucination_indicators(self, guardrail):
        """Test pattern validation with hallucination indicators."""
        response = "Ich glaube, dass die Studie möglicherweise eine qualitative Methodik verwendet hat, aber ich bin nicht sicher."
        result = guardrail._pattern_validation(response)
        assert result.result == OutputValidationResult.WARNING
        assert len(result.warnings) > 0


class TestDocumentPage:
    """Test cases for DocumentPage dataclass."""

    def test_document_page_creation(self):
        """Test creating a DocumentPage instance."""
        from PIL import Image

        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="white")

        page = DocumentPage(
            page_number=1,
            image=test_image,
            text_content="Sample text content",
            metadata={"source": "test.pdf"},
        )

        assert page.page_number == 1
        assert page.text_content == "Sample text content"
        assert page.metadata["source"] == "test.pdf"
        assert page.image.size == (100, 100)


@pytest.mark.asyncio
async def test_async_input_validation():
    """Test async input validation workflow."""
    with (
        patch("src.input_guardrail.OpenAIProvider"),
        patch("src.input_guardrail.OpenAIModel"),
        patch("src.input_guardrail.Agent") as mock_agent,
    ):
        # Mock the agent response
        mock_response = Mock()
        mock_response.data.is_appropriate = True
        mock_response.data.is_relevant = True
        mock_response.data.is_safe = True
        mock_response.data.reason = "Test validation passed"
        mock_response.data.confidence = 0.9
        mock_response.data.suggestions = None

        mock_agent_instance = Mock()
        mock_agent_instance.run = AsyncMock(return_value=mock_response)
        mock_agent.return_value = mock_agent_instance

        guardrail = InputGuardrail(
            llm_endpoint="http://mock-endpoint",
            llm_api_key="mock-key",
            llm_model="mock-model",
            enable_llm_validation=True,
        )

        result = await guardrail.validate_query("What is the research methodology?")
        assert result.result == ValidationResult.ACCEPTED
        assert result.confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
