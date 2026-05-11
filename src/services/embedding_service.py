"""
Embedding generation service for semantic search.

This service provides a clean abstraction for generating embeddings
from text using various embedding models/APIs.
"""

from typing import List, Optional
import os
from enum import Enum
import re
import math
import hashlib

import anthropic
import requests

from src.utils.logging import get_logger
from src.config import settings

logger = get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    ANTHROPIC = "anthropic"  # Claude embeddings via Voyage AI
    OPENAI = "openai"        # OpenAI text-embedding-3-small
    LOCAL = "local"          # Local sentence-transformers model
    OLLAMA = "ollama"        # Local Ollama embeddings (e.g., llama3)


# Embedding dimensions for different models
EMBEDDING_DIMENSIONS = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "all-MiniLM-L6-v2": 384,
    # Ollama models: we will fallback to detected dim if not set
    "llama3": 4096,
    "llama3.2": 4096,
    "llama3.2:3b": 3072,
}


class EmbeddingService:
    """
    Service for generating text embeddings.

    Supports multiple embedding providers with a unified interface.
    The embedding model can be swapped via environment variables.
    """

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider (default from settings)
            model: Model name (default from settings)
            api_key: API key (default from settings)
        """
        # Import here to avoid circular dependency
        from src.config import settings as app_settings

        self.provider = provider or EmbeddingProvider(
            app_settings.embedding_provider
        )
        self.model = model or app_settings.embedding_model
        self.api_key = api_key or app_settings.embedding_api_key

        # Dimension for this model
        self.dimension = EMBEDDING_DIMENSIONS.get(self.model, 1536)

        logger.info(
            f"Initialized EmbeddingService: provider={self.provider}, "
            f"model={self.model}, dimension={self.dimension}"
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        # Truncate very long text (most models have token limits)
        max_chars = 8000
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars]

        if self.provider == EmbeddingProvider.OPENAI:
            return self._get_openai_embedding(text)
        elif self.provider == EmbeddingProvider.ANTHROPIC:
            return self._get_anthropic_embedding(text)
        elif self.provider == EmbeddingProvider.LOCAL:
            try:
                return self._get_local_embedding(text)
            except Exception as e:
                logger.warning(f"Local embedding unavailable, using deterministic fallback embedding: {e}")
                return self._get_hash_embedding(text)
        elif self.provider == EmbeddingProvider.OLLAMA:
            return self._get_ollama_embedding(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []

        # For now, process sequentially
        # TODO: Implement true batch API calls for efficiency
        embeddings = []
        for text in texts:
            try:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                embeddings.append(self._get_hash_embedding(text))

        return embeddings

    def _get_ollama_embedding(self, text: str) -> List[float]:
        """
        Get embedding from local Ollama server (e.g., llama3 embeddings).

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not embedding:
                raise RuntimeError("No embedding returned from Ollama")

            # Update dimension dynamically if needed
            if self.dimension != len(embedding):
                logger.info(
                    f"Ollama embedding dimension detected: {len(embedding)} "
                    f"(was {self.dimension}), updating dimension."
                )
                self.dimension = len(embedding)

            return embedding

        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}") from e

    def _get_openai_embedding(self, text: str) -> List[float]:
        """
        Get embedding from OpenAI API.

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector
        """
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key not set. Set EMBEDDING_API_KEY environment variable."
            )

        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            return embedding

        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}") from e

    def _get_anthropic_embedding(self, text: str) -> List[float]:
        """
        Get embedding from Anthropic/Voyage AI.

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector
        """
        if not self.api_key:
            raise RuntimeError(
                "Voyage API key not set. Set EMBEDDING_API_KEY environment variable."
            )

        try:
            # Voyage AI is Anthropic's recommended embedding provider
            response = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": [text]
                },
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            return embedding

        except Exception as e:
            raise RuntimeError(f"Voyage embedding failed: {e}") from e

    def _get_local_embedding(self, text: str) -> List[float]:
        """
        Get embedding from local sentence-transformers model.

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Lazy load model
            if not hasattr(self, "_local_model"):
                logger.info(f"Loading local embedding model: {self.model}")
                self._local_model = SentenceTransformer(self.model)

            embedding = self._local_model.encode(text)
            return embedding.tolist()

        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Local embedding failed: {e}") from e

    def _get_hash_embedding(self, text: str) -> List[float]:
        """Deterministic lightweight fallback embedding when model backends fail."""
        dim = max(1, int(self.dimension))
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", (text or "").lower())
        for tok in tokens:
            digest = hashlib.sha1(tok.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = -1.0 if (digest[4] & 1) else 1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0:
            vec[0] = 1e-12
            return vec
        return [v / norm for v in vec]


# Global singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.

    Returns:
        EmbeddingService: The embedding service
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
