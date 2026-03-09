"""Vector store interface using ChromaDB."""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config import get_config, VectorStoreConfig, BASE_DIR
from app.rag.embeddings import get_embedding_provider
from app.utils.logging import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """
    Vector store using ChromaDB for document retrieval.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store configuration
        """
        if config is None:
            config = get_config().providers.vector_store
        self.config = config
        
        # Resolve persist directory relative to project root
        persist_path = Path(config.persist_directory)
        if not persist_path.is_absolute():
            persist_path = BASE_DIR / persist_path
        self.persist_directory = persist_path
        
        self._collection = None
        self._client = None
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        
        return self._client
    
    def _get_collection(self):
        """Get or create the document collection."""
        if self._collection is None:
            client = self._get_client()
            embedding_provider = get_embedding_provider()
            
            # Get or create collection
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Using collection: {self.config.collection_name}")
        
        return self._collection
    
    def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        collection = self._get_collection()
        embedding_provider = get_embedding_provider()
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = embedding_provider.embed_documents(texts)
        
        # Generate unique IDs
        existing_count = collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (Document, score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        collection = self._get_collection()
        embedding_provider = get_embedding_provider()
        
        # Check if collection has documents
        if collection.count() == 0:
            logger.debug("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = embedding_provider.embed_query(query)
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Process results
        documents_with_scores = []
        
        if results["documents"] and results["documents"][0]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity (ChromaDB returns L2 distance for cosine)
                # For cosine distance, similarity = 1 - distance
                similarity = 1 - distance
                
                if similarity >= threshold:
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata or {},
                    )
                    documents_with_scores.append((doc, similarity))
        
        logger.debug(f"Found {len(documents_with_scores)} relevant documents for query")
        return documents_with_scores
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        client = self._get_client()
        
        # Delete and recreate collection
        try:
            client.delete_collection(self.config.collection_name)
            logger.info(f"Cleared collection: {self.config.collection_name}")
        except Exception:
            pass
        
        # Reset cached collection
        self._collection = None
    
    def count(self) -> int:
        """Get the number of documents in the store."""
        collection = self._get_collection()
        return collection.count()


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
