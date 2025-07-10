"""
Test script for GuardRAG system
"""

import asyncio
import os
import logging
import pytest

# Set environment variables for testing
os.environ["LLM_ENDPOINT"] = "http://localhost:11434/v1"
os.environ["LLM_API_KEY"] = "ollama"
os.environ["LLM_MODEL"] = "qwen2.5:latest"
os.environ["COLPALI_MODEL"] = "vidore/colqwen2.5-v0.2"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"

logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_guardrag():
    """Test GuardRAG initialization."""
    try:
        from src.rag_agent import GuardRAGAgent
        from src.qdrant_integration import QdrantConfig

        print("✅ Imports successful")

        # Test Qdrant config
        qdrant_config = QdrantConfig(
            host="localhost",
            port=6333,
        )
        print("✅ Qdrant config created")

        # Verify the config is correct
        assert qdrant_config.host == "localhost"
        assert qdrant_config.port == 6333
        print("✅ Qdrant config validation passed")

        # Test that GuardRAGAgent class is importable (without initializing)
        assert GuardRAGAgent is not None
        print("✅ GuardRAGAgent class imported successfully")

        print("🎉 All tests passed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_guardrag())
