"""
LlamaIndex integration with OpenRouter for embeddings and LLM functionality.
"""

from typing import List, Optional
import asyncio
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
import requests
import json

from .config import OpenRouterConfig


class OpenRouterBgeM3Embedder(BaseEmbedding):
    """
    LlamaIndex-compatible embedding class for bge-m3 via OpenRouter API.
    """

    # Declare fields for Pydantic
    api_key: str
    site_url: Optional[str] = None
    site_name: Optional[str] = None

    base_url: str = "https://openrouter.ai/api/v1/embeddings"

    def _post(self, inputs: List[str], encoding_format: str = "float") -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "input": inputs,
            "encoding_format": encoding_format,
        }

        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        result = self._post(texts)
        return [item["embedding"] for item in result["data"]]

    # --- Required abstract methods (sync) ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    # --- Required abstract methods (async) ---
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


class OpenRouterLLMIntegration:
    """Integration class for OpenRouter LLM and embedding services."""

    def __init__(self, config: OpenRouterConfig):
        """Initialize OpenRouter integration.

        Args:
            config: OpenRouter configuration
        """
        self.config = config
        self._llm: Optional[LLM] = None
        self._embedding_model: Optional[BaseEmbedding] = None

    def get_llm(self) -> LLM:
        """Get configured LLM instance.

        Returns:
            LLM: Configured OpenRouter LLM instance
        """
        if self._llm is None:
            self._llm = OpenAILike(
                model=self.config.llm_model,
                api_base=self.config.api_base + "v1",
                api_key=self.config.api_key,
                is_chat_model=True,
                temperature=0.1,
                max_tokens=2048
            )
        return self._llm

    def get_embedding_model(self) -> BaseEmbedding:
        """Get configured embedding model instance.

        Returns:
            BaseEmbedding: Configured OpenRouter embedding model
        """
        if self._embedding_model is None:
            # self._embedding_model = OpenAIEmbedding(
            #     model=self.config.embedding_model,
            #     api_base=self.config.api_base + "v1",
            #     api_key=self.config.api_key,
            #     embed_batch_size=10,
            #     max_retries=3
            # )
            self._embedding_model = OpenRouterBgeM3Embedder(
                api_key=self.config.api_key,
                embed_batch_size=5,
                model_name="baai/bge-m3"
            )
        return self._embedding_model

    def setup_global_settings(self) -> None:
        """Configure global LlamaIndex settings."""
        Settings.llm = self.get_llm()
        Settings.embed_model = self.get_embedding_model()
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

    async def generate_text(self, prompt: str) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: Input prompt for text generation

        Returns:
            str: Generated text response

        Raises:
            Exception: If text generation fails
        """
        try:
            llm = self.get_llm()
            response = await llm.acomplete(prompt)
            return str(response)
        except Exception as e:
            raise Exception(f"Failed to generate text: {str(e)}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        try:
            embedding_model = self.get_embedding_model()
            embeddings = await embedding_model.aget_text_embedding_batch(texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        try:
            embedding_model = self.get_embedding_model()
            embedding = await embedding_model.aget_text_embedding(text)
            return embedding
        except Exception as e:
            raise Exception(f"Failed to generate single embedding: {str(e)}")


def create_openrouter_integration(config: Optional[OpenRouterConfig] = None) -> OpenRouterLLMIntegration:
    """Create and configure OpenRouter integration.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        OpenRouterLLMIntegration: Configured integration instance

    Raises:
        ValueError: If configuration is invalid
    """
    if config is None:
        config = OpenRouterConfig.from_env()

    config.validate()
    integration = OpenRouterLLMIntegration(config)
    integration.setup_global_settings()

    return integration