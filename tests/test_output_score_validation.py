"""
Test Output Guardrail Score Validation

Tests whether the output guardrail correctly blocks responses
when all Qdrant scores are above the threshold.
"""

import sys
import asyncio
import pytest

# Add the parent directory to the path to import src modules
sys.path.append("..")

from src.output_guardrail import OutputGuardrail, OutputValidationResult


@pytest.mark.asyncio
async def test_output_guardrail_score_threshold():
    """Test that output guardrail blocks when all scores are above threshold."""

    # Initialize Output Guardrail
    guardrail = OutputGuardrail(
        llm_endpoint="http://localhost:11434/v1",
        llm_api_key="dummy-key",
        llm_model="qwen2.5:latest",
        enable_factual_check=False,  # Disable LLM-based checks for testing
        enable_toxicity_check=False,
        enable_completeness_check=False,
        enable_score_check=True,  # Enable score checking
        relevance_score_threshold=0.2,  # Set threshold for testing
    )

    print("✅ Output Guardrail initialized")

    # Test case 1: All scores above threshold (should be blocked)
    query = "What are the ingredients of pesticide X?"
    response = "I found some information about that topic."
    high_scores = [0.3, 0.4, 0.5, 0.6]  # All above 0.2 threshold

    # Create mock retrieval results with high scores
    from src.colpali_integration import RetrievalResult, DocumentPage
    from PIL import Image
    import numpy as np

    # Create mock document page
    mock_image = Image.fromarray(
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    )
    mock_page = DocumentPage(
        page_number=1,
        image=mock_image,
        text_content="Mock document content",
        metadata={},
    )

    # Create mock retrieval results
    mock_results = []
    for score in high_scores:
        mock_results.append(
            RetrievalResult(
                page=mock_page, score=score, query=query, explanation="Mock explanation"
            )
        )

    result = await guardrail.validate_response(
        response=response, query=query, retrieval_results=mock_results
    )

    assert result.result == OutputValidationResult.REJECTED, (
        f"Expected REJECTED, got {result.result}"
    )
    assert (
        "ICH HABE KEINE INFORMATIONEN ZU IHRER ANFRAGE VORLIEGEN." in result.reason
        or result.revised_response
        == "ICH HABE KEINE INFORMATIONEN ZU IHRER ANFRAGE VORLIEGEN."
    ), f"Expected standard blocked message, got: {result.reason}"

    print("✅ Test 1 passed: High scores correctly blocked")

    # Test case 2: Some scores below threshold (should be allowed)
    low_scores = [0.1, 0.05, 0.3, 0.15]  # Mix of low and high scores

    mock_results_low = []
    for score in low_scores:
        mock_results_low.append(
            RetrievalResult(
                page=mock_page, score=score, query=query, explanation="Mock explanation"
            )
        )

    result = await guardrail.validate_response(
        response=response, query=query, retrieval_results=mock_results_low
    )

    assert result.result != OutputValidationResult.REJECTED, (
        f"Expected not REJECTED, got {result.result}"
    )

    print("✅ Test 2 passed: Mixed scores correctly allowed")

    print("🎉 All output score validation tests passed!")


if __name__ == "__main__":
    asyncio.run(test_output_guardrail_score_threshold())
