"""
Tests for DocumentProcessor functionality.
"""

import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock

from rag_system.document_processor import DocumentProcessor
from rag_system.config import OpenRouterConfig
from rag_system.data_models import ContextualChunk, EmbeddedChunk


@pytest.fixture
def mock_config():
    """Create a mock OpenRouter configuration."""
    return OpenRouterConfig(
        api_base="https://openrouter.ai/api/",
        api_key="sk-test-key",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo"
    )


@pytest.fixture
def document_processor(mock_config):
    """Create a DocumentProcessor instance with mocked LLM integration."""
    processor = DocumentProcessor(mock_config)

    # Mock the LLM integration
    processor.llm_integration = MagicMock()
    processor.llm_integration.generate_text = AsyncMock(return_value="This chunk is from a test document.")
    processor.llm_integration.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    return processor


def test_read_text_file_valid(document_processor):
    """Test reading a valid text file."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.")
        temp_path = f.name

    try:
        content = document_processor._read_text_file(temp_path, "test.txt")
        assert content == "This is a test document.\nIt has multiple lines."
    finally:
        os.unlink(temp_path)


def test_read_text_file_unsupported_format(document_processor):
    """Test reading an unsupported file format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            document_processor._read_text_file(temp_path, "test.pdf")
    finally:
        os.unlink(temp_path)


def test_read_text_file_empty(document_processor):
    """Test reading an empty file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="empty or contains only whitespace"):
            document_processor._read_text_file(temp_path, "empty.txt")
    finally:
        os.unlink(temp_path)


def test_chunk_document(document_processor):
    """Test document chunking functionality."""
    text = "This is a test document. " * 100  # Create a long text
    chunks = document_processor._chunk_document(text)

    assert len(chunks) > 1  # Should create multiple chunks
    assert all(len(chunk) <= document_processor.chunk_size + 100 for chunk in chunks)  # Chunks should be reasonable size

    # Verify all content is preserved
    combined = " ".join(chunks)
    assert "This is a test document." in combined


def test_chunk_document_empty(document_processor):
    """Test chunking empty document."""
    chunks = document_processor._chunk_document("")
    assert chunks == []


def test_chunk_document_boundary_handling(document_processor):
    """Test that chunking respects sentence and word boundaries."""
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    chunks = document_processor._chunk_document(text)

    # Should preserve sentence boundaries when possible
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.strip()  # No empty chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
