"""
Configuration management for OpenRouter API integration.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API integration."""

    api_base: str
    api_key: str
    embedding_model: str
    llm_model: str

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        """Create configuration from environment variables.

        Returns:
            OpenRouterConfig: Configuration loaded from environment variables

        Raises:
            ValueError: If required environment variables are missing
        """
        api_base = os.getenv("OPENROUTER_API_BASE")
        api_key = os.getenv("OPENROUTER_API_KEY")
        embedding_model = os.getenv("OPENROUTER_EMBED_MODEL")
        llm_model = os.getenv("OPENROUTER_MODEL")

        missing_vars = []
        if not api_base:
            missing_vars.append("OPENROUTER_API_BASE")
        if not api_key:
            missing_vars.append("OPENROUTER_API_KEY")
        if not embedding_model:
            missing_vars.append("OPENROUTER_EMBED_MODEL")
        if not llm_model:
            missing_vars.append("OPENROUTER_MODEL")

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        return cls(
            api_base=api_base,
            api_key=api_key,
            embedding_model=embedding_model,
            llm_model=llm_model
        )

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.api_base.startswith(("http://", "https://")):
            raise ValueError("OPENROUTER_API_BASE must be a valid URL")

        if not self.api_key.startswith("sk-"):
            raise ValueError("OPENROUTER_API_KEY must start with 'sk-'")

        if not self.embedding_model:
            raise ValueError("OPENROUTER_EMBED_MODEL cannot be empty")

        if not self.llm_model:
            raise ValueError("OPENROUTER_MODEL cannot be empty")