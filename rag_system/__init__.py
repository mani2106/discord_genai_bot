"""
Discord RAG System - Retrieval-Augmented Generation for Discord Bot

This module provides RAG functionality for processing and querying text documents
uploaded to Discord, using LlamaIndex and OpenRouter for embeddings and LLM responses.
"""

from .config import OpenRouterConfig
from .data_models import (
    ContextualChunk,
    EmbeddedChunk,
    ProcessedDocument,
    RelevantChunk,
    QueryResponse
)
from .document_processor import DocumentProcessor
from .document_storage import DocumentStorage
from .query_engine import QueryEngine
from .rag_system import DiscordRAGSystem
from .llama_integration import OpenRouterLLMIntegration, create_openrouter_integration

__all__ = [
    "OpenRouterConfig",
    "ContextualChunk",
    "EmbeddedChunk",
    "ProcessedDocument",
    "RelevantChunk",
    "QueryResponse",
    "DocumentProcessor",
    "DocumentStorage",
    "QueryEngine",
    "DiscordRAGSystem",
    "OpenRouterLLMIntegration",
    "create_openrouter_integration"
]