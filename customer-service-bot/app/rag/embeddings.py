"""Embedding provider abstraction for RAG."""

from typing import Optional

from langchain_core.embeddings import Embeddings

from app.config import get_config, EmbeddingsConfig
from app.utils.logging import get_logger

logger = get_logger("embeddings")


class EmbeddingProvider:
    """
    Embedding provider that supports multiple backends.
    """
    
    def __init__(self, config: Optional[EmbeddingsConfig] = None):
        """
        Initialize the embedding provider.
        
        Args:
            config: Embeddings configuration (uses default if not provided)
        """
        if config is None:
            config = get_config().providers.embeddings
        self.config = config
        self._embeddings: Optional[Embeddings] = None
    
    def _create_embeddings(self) -> Embeddings:
        """Create the embeddings instance based on configuration."""
        if self.config.provider == "sentence-transformers":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            logger.info(f"Using sentence-transformers model: {self.config.model}")
            return HuggingFaceEmbeddings(
                model_name=self.config.model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        
        elif self.config.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            
            settings = get_config().settings
            logger.info(f"Using OpenAI embeddings model: {self.config.model}")
            return OpenAIEmbeddings(
                model=self.config.model,
                api_key=settings.openai_api_key,
            )
        
        else:
            raise ValueError(f"Unknown embeddings provider: {self.config.provider}")
    
    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance, creating it if necessary."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)


# Global embedding provider instance
_embedding_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the global embedding provider instance.
    
    Returns:
        EmbeddingProvider instance
    """
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = EmbeddingProvider()
    return _embedding_provider
