"""Knowledge Hub combining FAQ matching with RAG retrieval."""

from dataclasses import dataclass
from typing import Optional

from app.config import get_config, FAQItem
from app.rag.document_loader import DocumentLoader
from app.rag.vector_store import get_vector_store
from app.utils.logging import get_logger

logger = get_logger("knowledge")


@dataclass
class KnowledgeResult:
    """Result from knowledge retrieval."""
    
    content: str
    source: str  # "faq", "document", or "combined"
    confidence: float
    metadata: dict


class FAQMatcher:
    """
    Matches user queries against FAQ entries using keyword and fuzzy matching.
    """
    
    def __init__(self):
        """Initialize the FAQ matcher."""
        self._config = get_config()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return text.lower().strip()
    
    def _calculate_keyword_score(self, query: str, faq: FAQItem) -> float:
        """
        Calculate a match score based on keyword overlap.
        
        Args:
            query: User query
            faq: FAQ item to match against
            
        Returns:
            Score between 0 and 1
        """
        query_normalized = self._normalize_text(query)
        query_words = set(query_normalized.split())
        
        # Check keyword matches
        keyword_matches = 0
        for keyword in faq.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in query_normalized:
                keyword_matches += 1
        
        if not faq.keywords:
            return 0.0
        
        return keyword_matches / len(faq.keywords)
    
    def _calculate_question_similarity(self, query: str, faq: FAQItem) -> float:
        """
        Calculate similarity between query and FAQ question.
        
        Args:
            query: User query
            faq: FAQ item to match against
            
        Returns:
            Score between 0 and 1
        """
        query_words = set(self._normalize_text(query).split())
        question_words = set(self._normalize_text(faq.question).split())
        
        # Remove common stop words
        stop_words = {"what", "is", "your", "how", "do", "i", "the", "a", "an", "to", "can", "you"}
        query_words -= stop_words
        question_words -= stop_words
        
        if not query_words or not question_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & question_words
        union = query_words | question_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_match(self, query: str, threshold: float = 0.3) -> Optional[tuple[FAQItem, float]]:
        """
        Find the best matching FAQ for a query.
        
        Args:
            query: User query
            threshold: Minimum score threshold
            
        Returns:
            Tuple of (FAQItem, score) or None if no match
        """
        faqs = self._config.faqs.faqs
        
        if not faqs:
            return None
        
        best_match = None
        best_score = 0.0
        
        for faq in faqs:
            # Combine keyword and question similarity scores
            keyword_score = self._calculate_keyword_score(query, faq)
            question_score = self._calculate_question_similarity(query, faq)
            
            # Weighted combination (keywords are more reliable)
            combined_score = (keyword_score * 0.6) + (question_score * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = faq
        
        if best_match and best_score >= threshold:
            logger.debug(f"FAQ match found with score {best_score:.2f}")
            return (best_match, best_score)
        
        return None


class KnowledgeHub:
    """
    Central knowledge hub that combines FAQ matching with RAG retrieval.
    """
    
    def __init__(self):
        """Initialize the knowledge hub."""
        self.faq_matcher = FAQMatcher()
        self.vector_store = get_vector_store()
        self.document_loader = DocumentLoader()
        self._config = get_config()
    
    def index_documents(self) -> int:
        """
        Index all documents from the documents directory.
        
        Returns:
            Number of documents indexed
        """
        # Load documents
        documents = self.document_loader.load_directory()
        
        if not documents:
            logger.info("No documents to index")
            return 0
        
        # Add to vector store
        count = self.vector_store.add_documents(documents)
        logger.info(f"Indexed {count} document chunks")
        
        return count
    
    def reindex(self) -> int:
        """
        Clear and re-index all documents.
        
        Returns:
            Number of documents indexed
        """
        self.vector_store.clear()
        return self.index_documents()
    
    def get_context(
        self,
        query: str,
        include_faqs: bool = True,
        include_documents: bool = True,
    ) -> list[KnowledgeResult]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            include_faqs: Whether to search FAQs
            include_documents: Whether to search documents
            
        Returns:
            List of KnowledgeResult objects
        """
        results = []
        
        # Check FAQs first
        if include_faqs:
            faq_match = self.faq_matcher.find_match(query)
            if faq_match:
                faq, score = faq_match
                results.append(KnowledgeResult(
                    content=f"Q: {faq.question}\nA: {faq.answer}",
                    source="faq",
                    confidence=score,
                    metadata={"question": faq.question},
                ))
        
        # Search documents
        if include_documents:
            doc_results = self.vector_store.search(query)
            for doc, score in doc_results:
                results.append(KnowledgeResult(
                    content=doc.page_content,
                    source="document",
                    confidence=score,
                    metadata=doc.metadata,
                ))
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def get_formatted_context(self, query: str) -> str:
        """
        Get formatted context string for use in prompts.
        
        Args:
            query: User query
            
        Returns:
            Formatted context string
        """
        results = self.get_context(query)
        
        if not results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source_label = "FAQ" if result.source == "faq" else "Document"
            context_parts.append(
                f"[{source_label} - Relevance: {result.confidence:.0%}]\n{result.content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def check_escalation_topic(self, query: str) -> Optional[str]:
        """
        Check if the query matches an escalation topic.
        
        Args:
            query: User query
            
        Returns:
            Matched escalation topic or None
        """
        query_lower = query.lower()
        
        for topic in self._config.company.escalation_topics:
            if topic.lower() in query_lower:
                return topic
        
        return None


# Global knowledge hub instance
_knowledge_hub: Optional[KnowledgeHub] = None


def get_knowledge_hub() -> KnowledgeHub:
    """
    Get the global knowledge hub instance.
    
    Returns:
        KnowledgeHub instance
    """
    global _knowledge_hub
    if _knowledge_hub is None:
        _knowledge_hub = KnowledgeHub()
    return _knowledge_hub
