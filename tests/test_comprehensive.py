"""
Comprehensive Test Suite for GuardRAG System
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np

# Test configuration
TEST_CONFIG = {
    "llm_endpoint": "http://mock-endpoint",
    "llm_api_key": "mock-key",
    "llm_model": "mock-model",
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_pdf_file():
    """Create a mock PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"Mock PDF content")
        yield Path(tmp.name)
    os.unlink(tmp.name)


@pytest.fixture
def mock_image():
    """Create a mock PIL image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def mock_document_page(mock_image):
    """Create a mock DocumentPage."""
    from src.colpali_integration import DocumentPage

    return DocumentPage(
        page_number=1,
        image=mock_image,
        text_content="This is a test document page with sample content.",
        metadata={"source": "test"},
    )


@pytest.fixture
def mock_qdrant_config():
    """Create a mock Qdrant configuration."""
    from src.qdrant_integration import QdrantConfig

    return QdrantConfig(
        host=TEST_CONFIG["qdrant_host"],
        port=TEST_CONFIG["qdrant_port"],
        collection_name="test_collection",
    )


class TestQdrantIntegration:
    """Test Qdrant vector store functionality."""

    @patch("src.qdrant_integration.QdrantClient")
    def test_qdrant_client_initialization(self, mock_client, mock_qdrant_config):
        """Test Qdrant client initialization."""
        from src.qdrant_integration import QdrantVectorStore

        # Mock client methods
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client.return_value = mock_client_instance

        vector_store = QdrantVectorStore(mock_qdrant_config)
        assert vector_store.config == mock_qdrant_config
        mock_client.assert_called_once()

    @patch("src.qdrant_integration.QdrantClient")
    def test_add_embeddings(self, mock_client, mock_qdrant_config, mock_document_page):
        """Test adding embeddings to Qdrant."""
        from src.qdrant_integration import QdrantVectorStore

        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.upsert.return_value = None
        mock_client.return_value = mock_client_instance

        vector_store = QdrantVectorStore(mock_qdrant_config)

        # Test data
        embeddings = [np.random.rand(128)]
        pages = [mock_document_page]
        document_id = "test_doc_123"

        # Test adding embeddings
        point_ids = vector_store.add_embeddings(embeddings, pages, document_id)

        assert len(point_ids) == 1
        mock_client_instance.upsert.assert_called_once()

    @patch("src.qdrant_integration.QdrantClient")
    def test_search_embeddings(self, mock_client, mock_qdrant_config):
        """Test searching embeddings in Qdrant."""
        from src.qdrant_integration import QdrantVectorStore

        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_search_result = [
            Mock(
                id="test_id",
                score=0.9,
                payload={"page_number": 1, "text_content": "test"},
            )
        ]
        mock_client_instance.search.return_value = mock_search_result
        mock_client.return_value = mock_client_instance

        vector_store = QdrantVectorStore(mock_qdrant_config)

        # Test search
        query_embedding = np.random.rand(128)
        results = vector_store.search(query_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]["score"] == 0.9
        mock_client_instance.search.assert_called_once()


class TestCOLPALIIntegration:
    """Test COLPALI document processing."""

    @patch("src.colpali_integration.ColQwen2")
    @patch("src.colpali_integration.ColQwen2Processor")
    @patch("src.qdrant_integration.QdrantVectorStore")
    def test_colpali_initialization(
        self, mock_vector_store, mock_processor, mock_model, mock_qdrant_config
    ):
        """Test COLPALI processor initialization."""
        from src.colpali_integration import COLPALIProcessor

        # Setup mocks
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        mock_vector_store.return_value = Mock()

        processor = COLPALIProcessor(qdrant_config=mock_qdrant_config)

        assert processor.model_name == "vidore/colqwen2.5-v0.2"
        mock_model.from_pretrained.assert_called_once()
        mock_processor.from_pretrained.assert_called_once()

    @patch("src.colpali_integration.pdf2image")
    @patch("src.colpali_integration.fitz")
    def test_pdf_to_pages(self, mock_fitz, mock_pdf2image, mock_pdf_file):
        """Test PDF to pages conversion."""
        from src.colpali_integration import COLPALIProcessor

        # Mock PDF processing
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample text content"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.__len__ = Mock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        mock_images = [Image.new("RGB", (100, 100))]
        mock_pdf2image.convert_from_path.return_value = mock_images

        with (
            patch.object(COLPALIProcessor, "_load_model"),
            patch.object(COLPALIProcessor, "__init__", lambda x: None),
        ):
            processor = COLPALIProcessor()
            pages = processor.pdf_to_pages(mock_pdf_file)

            assert len(pages) == 1
            assert pages[0].page_number == 1
            assert "Sample text content" in pages[0].text_content

    @patch("src.colpali_integration.ColQwen2")
    @patch("src.colpali_integration.ColQwen2Processor")
    @patch("src.qdrant_integration.QdrantVectorStore")
    def test_search_functionality(
        self, mock_vector_store, mock_processor_class, mock_model, mock_qdrant_config
    ):
        """Test document search functionality."""
        from src.colpali_integration import COLPALIProcessor

        # Setup mocks
        mock_model.from_pretrained.return_value = Mock()
        mock_processor = Mock()
        mock_processor.process_queries.return_value = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_vector_store_instance = Mock()
        mock_search_results = [
            {
                "id": "test_id",
                "score": 0.9,
                "payload": {
                    "page_number": 1,
                    "text_content": "Test content",
                    "metadata": {},
                },
            }
        ]
        mock_vector_store_instance.search.return_value = mock_search_results
        mock_vector_store.return_value = mock_vector_store_instance

        with (
            patch("torch.no_grad"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            processor = COLPALIProcessor(qdrant_config=mock_qdrant_config)

            results = processor.search("test query", top_k=5)

            assert len(results) == 1
            assert results[0].score == 0.9
            assert results[0].page.page_number == 1


class TestInputGuardrails:
    """Test input validation guardrails."""

    def test_basic_validation_empty_query(self):
        """Test validation with empty query."""
        from src.input_guardrail import InputGuardrail

        guardrail = InputGuardrail(**TEST_CONFIG, enable_llm_validation=False)

        result = guardrail._basic_validation("")
        assert result.result.value == "REJECTED"
        assert "leer" in result.reason.lower()

    def test_basic_validation_valid_query(self):
        """Test validation with valid query."""
        from src.input_guardrail import InputGuardrail

        guardrail = InputGuardrail(**TEST_CONFIG, enable_llm_validation=False)

        result = guardrail._basic_validation("What is the research methodology?")
        assert result.result.value == "ACCEPTED"

    @pytest.mark.asyncio
    @patch("src.input_guardrail.Agent")
    async def test_llm_validation(self, mock_agent):
        """Test LLM-based validation."""
        from src.input_guardrail import InputGuardrail

        # Mock agent response
        mock_response = Mock()
        mock_response.data.is_appropriate = True
        mock_response.data.is_relevant = True
        mock_response.data.is_safe = True
        mock_response.data.reason = "Valid query"
        mock_response.data.confidence = 0.9

        mock_agent_instance = Mock()
        mock_agent_instance.run = AsyncMock(return_value=mock_response)
        mock_agent.return_value = mock_agent_instance

        guardrail = InputGuardrail(**TEST_CONFIG, enable_llm_validation=True)

        result = await guardrail.validate_query("What is the research methodology?")
        assert result.result.value == "ACCEPTED"


class TestOutputGuardrails:
    """Test output validation guardrails."""

    @pytest.mark.asyncio
    @patch("src.output_guardrail.Agent")
    async def test_response_validation(self, mock_agent, mock_document_page):
        """Test response validation."""
        from src.output_guardrail import OutputGuardrail
        from src.colpali_integration import RetrievalResult

        # Mock agent response
        mock_response = Mock()
        mock_response.data.is_factual = True
        mock_response.data.is_grounded = True
        mock_response.data.is_safe = True
        mock_response.data.reason = "Valid response"
        mock_response.data.confidence = 0.9

        mock_agent_instance = Mock()
        mock_agent_instance.run = AsyncMock(return_value=mock_response)
        mock_agent.return_value = mock_agent_instance

        guardrail = OutputGuardrail(**TEST_CONFIG)

        retrieval_results = [
            RetrievalResult(
                page=mock_document_page,
                score=0.9,
                query="test query",
                explanation="test explanation",
            )
        ]

        result = await guardrail.validate_response(
            response="This is a test response",
            query="test query",
            retrieval_results=retrieval_results,
        )

        assert result.result.value == "ACCEPTED"


class TestRAGAgent:
    """Test main RAG agent functionality."""

    @patch("src.rag_agent.COLPALIProcessor")
    @patch("src.rag_agent.InputGuardrail")
    @patch("src.rag_agent.OutputGuardrail")
    def test_rag_agent_initialization(
        self, mock_output_guardrail, mock_input_guardrail, mock_colpali
    ):
        """Test RAG agent initialization."""
        from src.rag_agent import GuardRAGAgent

        # Setup mocks
        mock_colpali.return_value = Mock()
        mock_input_guardrail.return_value = Mock()
        mock_output_guardrail.return_value = Mock()

        with patch.object(GuardRAGAgent, "_init_generation_agent"):
            agent = GuardRAGAgent(**TEST_CONFIG)

            assert agent.llm_endpoint == TEST_CONFIG["llm_endpoint"]
            assert agent.llm_model == TEST_CONFIG["llm_model"]
            mock_colpali.assert_called_once()

    @patch("src.rag_agent.COLPALIProcessor")
    @patch("src.rag_agent.InputGuardrail")
    @patch("src.rag_agent.OutputGuardrail")
    def test_document_loading(
        self,
        mock_output_guardrail,
        mock_input_guardrail,
        mock_colpali,
        mock_pdf_file,
        mock_document_page,
    ):
        """Test document loading functionality."""
        from src.rag_agent import GuardRAGAgent

        # Setup mocks
        mock_colpali_instance = Mock()
        mock_colpali_instance.process_document.return_value = [mock_document_page]
        mock_colpali.return_value = mock_colpali_instance
        mock_input_guardrail.return_value = Mock()
        mock_output_guardrail.return_value = Mock()

        with patch.object(GuardRAGAgent, "_init_generation_agent"):
            agent = GuardRAGAgent(**TEST_CONFIG)

            result = agent.load_document(mock_pdf_file)

            assert result["success"] is True
            assert result["pages_processed"] == 1
            mock_colpali_instance.process_document.assert_called_once_with(
                mock_pdf_file
            )

    @pytest.mark.asyncio
    @patch("src.rag_agent.COLPALIProcessor")
    @patch("src.rag_agent.InputGuardrail")
    @patch("src.rag_agent.OutputGuardrail")
    async def test_query_processing(
        self,
        mock_output_guardrail,
        mock_input_guardrail,
        mock_colpali,
        mock_document_page,
    ):
        """Test query processing pipeline."""
        from src.rag_agent import GuardRAGAgent
        from src.colpali_integration import RetrievalResult
        from src.input_guardrail import ValidationResult
        from src.output_guardrail import OutputValidationResult

        # Setup mocks
        mock_colpali_instance = Mock()
        mock_retrieval_result = RetrievalResult(
            page=mock_document_page,
            score=0.9,
            query="test query",
            explanation="test explanation",
        )
        mock_colpali_instance.search.return_value = [mock_retrieval_result]
        mock_colpali.return_value = mock_colpali_instance

        # Mock input guardrail
        mock_input_result = Mock()
        mock_input_result.result = ValidationResult.ACCEPTED
        mock_input_result.reason = "Valid query"
        mock_input_result.confidence = 0.9
        mock_input_result.suggestions = []
        mock_input_guardrail_instance = Mock()
        mock_input_guardrail_instance.validate_query = AsyncMock(
            return_value=mock_input_result
        )
        mock_input_guardrail.return_value = mock_input_guardrail_instance

        # Mock output guardrail
        mock_output_result = Mock()
        mock_output_result.result = OutputValidationResult.APPROVED
        mock_output_result.reason = "Valid response"
        mock_output_result.confidence = 0.9
        mock_output_result.warnings = []
        mock_output_guardrail_instance = Mock()
        mock_output_guardrail_instance.validate_response = AsyncMock(
            return_value=mock_output_result
        )
        mock_output_guardrail.return_value = mock_output_guardrail_instance

        with (
            patch.object(GuardRAGAgent, "_init_generation_agent"),
            patch.object(GuardRAGAgent, "_generate_response") as mock_generate,
        ):
            # Mock generation response
            mock_generation_response = Mock()
            mock_generation_response.answer = "Test answer"
            mock_generation_response.confidence = 0.9
            mock_generate.return_value = mock_generation_response

            agent = GuardRAGAgent(**TEST_CONFIG)

            result = await agent.process_query("test query")

            assert result.answer == "Test answer"
            assert len(result.sources) == 1
            assert result.confidence == 0.9


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_system_health_check(self):
        """Test overall system health check."""
        # This would test the main.py health endpoint
        pass

    def test_file_upload_workflow(self):
        """Test complete file upload and processing workflow."""
        # This would test the FastAPI upload endpoint
        pass

    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test various error conditions
        pass

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        # Test system under load
        pass


# Performance and load tests
class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.slow
    def test_large_document_processing(self):
        """Test processing of large documents."""
        pass

    @pytest.mark.slow
    def test_concurrent_search_performance(self):
        """Test search performance under concurrent load."""
        pass


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
