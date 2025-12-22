"""
Data models for the Discord RAG System.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class ContextualChunk:
    """A document chunk enhanced with contextual information."""

    original_text: str
    contextual_text: str  # Original text + generated context
    chunk_index: int
    document_filename: str

    def __post_init__(self):
        """Validate chunk data after initialization."""
        if not self.original_text.strip():
            raise ValueError("Original text cannot be empty")
        if not self.contextual_text.strip():
            raise ValueError("Contextual text cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")
        if not self.document_filename.strip():
            raise ValueError("Document filename cannot be empty")


@dataclass
class EmbeddedChunk:
    """A contextual chunk with its embedding vector."""

    contextual_chunk: ContextualChunk
    embedding: List[float]

    def __post_init__(self):
        """Validate embedded chunk data after initialization."""
        if not self.embedding:
            raise ValueError("Embedding cannot be empty")
        if not all(isinstance(x, (int, float)) for x in self.embedding):
            raise ValueError("Embedding must contain only numeric values")


@dataclass
class ProcessedDocument:
    """A document that has been processed and chunked."""

    filename: str
    original_text: str
    chunks: List[EmbeddedChunk]
    upload_timestamp: datetime

    def __post_init__(self):
        """Validate processed document data after initialization."""
        if not self.filename.strip():
            raise ValueError("Filename cannot be empty")
        if not self.original_text.strip():
            raise ValueError("Original text cannot be empty")
        if not self.chunks:
            raise ValueError("Document must have at least one chunk")
        if not isinstance(self.upload_timestamp, datetime):
            raise ValueError("Upload timestamp must be a datetime object")


@dataclass
class RelevantChunk:
    """A chunk with its similarity score for a query."""

    chunk: EmbeddedChunk
    similarity_score: float

    def __post_init__(self):
        """Validate relevant chunk data after initialization."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class QueryResponse:
    """Response to a user query with sources."""

    answer: str
    source_chunks: List[RelevantChunk]
    confidence_score: float

    def __post_init__(self):
        """Validate query response data after initialization."""
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        # source_chunks can be empty if no relevant content found