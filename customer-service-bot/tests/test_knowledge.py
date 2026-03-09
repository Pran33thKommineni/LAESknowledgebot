"""Tests for the knowledge hub and FAQ matching."""

import pytest
from unittest.mock import MagicMock, patch

from app.config import FAQItem
from app.core.knowledge import FAQMatcher, KnowledgeHub, KnowledgeResult


class TestFAQMatcher:
    """Tests for FAQ matching logic."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config with test FAQs."""
        mock = MagicMock()
        mock.faqs.faqs = [
            FAQItem(
                question="What is your return policy?",
                answer="We offer 30-day returns on all items.",
                keywords=["return", "refund", "exchange", "money back"]
            ),
            FAQItem(
                question="How do I track my order?",
                answer="Visit our tracking page with your order number.",
                keywords=["track", "tracking", "order status", "shipping status"]
            ),
            FAQItem(
                question="How do I rack mount the ICX 8200?",
                answer="Use the included rack mount brackets.",
                keywords=["rack", "mount", "mounting", "install", "bracket"]
            ),
            FAQItem(
                question="What are the power requirements?",
                answer="100-240V AC, 50/60 Hz.",
                keywords=["power", "voltage", "watts", "electrical", "ac"]
            ),
        ]
        return mock
    
    @patch('app.core.knowledge.get_config')
    def test_exact_keyword_match(self, mock_get_config, mock_config):
        """Test matching with exact keyword."""
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        # Use query with multiple matching keywords for better score
        result = matcher.find_match("I want a return and refund")
        
        assert result is not None
        faq, score = result
        assert "return" in faq.question.lower()
        assert score > 0.0
    
    @patch('app.core.knowledge.get_config')
    def test_multiple_keyword_match(self, mock_get_config, mock_config):
        """Test matching with multiple keywords."""
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        result = matcher.find_match("Can I get a refund or exchange?")
        
        assert result is not None
        faq, score = result
        assert "return" in faq.question.lower()
        # Multiple keyword matches should give reasonable score
        assert score >= 0.3
    
    @patch('app.core.knowledge.get_config')
    def test_question_similarity_match(self, mock_get_config, mock_config):
        """Test matching based on question similarity."""
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        result = matcher.find_match("How do I mount the ICX 8200 in a rack?")
        
        assert result is not None
        faq, score = result
        assert "rack" in faq.question.lower()
    
    @patch('app.core.knowledge.get_config')
    def test_no_match_below_threshold(self, mock_get_config, mock_config):
        """Test that queries below threshold return None."""
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        result = matcher.find_match("What is the weather today?")
        
        # Should not match any FAQ
        assert result is None
    
    @patch('app.core.knowledge.get_config')
    def test_empty_faqs(self, mock_get_config):
        """Test handling of empty FAQ list."""
        mock_config = MagicMock()
        mock_config.faqs.faqs = []
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        result = matcher.find_match("Any question")
        
        assert result is None
    
    @patch('app.core.knowledge.get_config')
    def test_case_insensitive_matching(self, mock_get_config, mock_config):
        """Test that matching is case insensitive."""
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        # Use queries with actual keywords that will match
        result1 = matcher.find_match("RETURN and REFUND")
        result2 = matcher.find_match("return and refund")
        result3 = matcher.find_match("Return and Refund")
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
    
    @patch('app.core.knowledge.get_config')
    def test_power_requirements_match(self, mock_get_config, mock_config):
        """Test matching power-related queries."""
        mock_get_config.return_value = mock_config
        matcher = FAQMatcher()
        
        # Use query with actual keywords from the FAQ
        result = matcher.find_match("What are the power voltage requirements?")
        
        assert result is not None
        faq, score = result
        assert "power" in faq.question.lower()


class TestKnowledgeResult:
    """Tests for KnowledgeResult dataclass."""
    
    def test_create_faq_result(self):
        """Test creating a FAQ knowledge result."""
        result = KnowledgeResult(
            content="Q: Test?\nA: Yes.",
            source="faq",
            confidence=0.85,
            metadata={"question": "Test?"}
        )
        
        assert result.source == "faq"
        assert result.confidence == 0.85
        assert "Test?" in result.content
    
    def test_create_document_result(self):
        """Test creating a document knowledge result."""
        result = KnowledgeResult(
            content="Document content here.",
            source="document",
            confidence=0.72,
            metadata={"filename": "guide.pdf", "page": 5}
        )
        
        assert result.source == "document"
        assert result.metadata["filename"] == "guide.pdf"


class TestEscalationTopics:
    """Tests for escalation topic detection."""
    
    @pytest.fixture
    def mock_config_with_escalation(self):
        """Create mock config with escalation topics."""
        mock = MagicMock()
        mock.company.escalation_topics = [
            "warranty claims",
            "RMA requests",
            "pricing information",
            "billing disputes"
        ]
        mock.faqs.faqs = []
        return mock
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_detect_warranty_escalation(self, mock_vs, mock_get_config, mock_config_with_escalation):
        """Test detection of warranty-related escalation."""
        mock_get_config.return_value = mock_config_with_escalation
        mock_vs.return_value = MagicMock()
        
        hub = KnowledgeHub()
        result = hub.check_escalation_topic("I need to file a warranty claims for my switch")
        
        assert result == "warranty claims"
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_detect_pricing_escalation(self, mock_vs, mock_get_config, mock_config_with_escalation):
        """Test detection of pricing-related escalation."""
        mock_get_config.return_value = mock_config_with_escalation
        mock_vs.return_value = MagicMock()
        
        hub = KnowledgeHub()
        result = hub.check_escalation_topic("What is the pricing information for ICX 8200?")
        
        assert result == "pricing information"
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_no_escalation_for_normal_query(self, mock_vs, mock_get_config, mock_config_with_escalation):
        """Test that normal queries don't trigger escalation."""
        mock_get_config.return_value = mock_config_with_escalation
        mock_vs.return_value = MagicMock()
        
        hub = KnowledgeHub()
        result = hub.check_escalation_topic("How do I rack mount the switch?")
        
        assert result is None
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_case_insensitive_escalation(self, mock_vs, mock_get_config, mock_config_with_escalation):
        """Test that escalation detection is case insensitive."""
        mock_get_config.return_value = mock_config_with_escalation
        mock_vs.return_value = MagicMock()
        
        hub = KnowledgeHub()
        result = hub.check_escalation_topic("I have BILLING DISPUTES to resolve")
        
        assert result == "billing disputes"


class TestKnowledgeHubContext:
    """Tests for knowledge hub context retrieval."""
    
    @pytest.fixture
    def mock_config_with_faqs(self):
        """Create mock config with FAQs."""
        mock = MagicMock()
        mock.faqs.faqs = [
            FAQItem(
                question="How do I connect to console?",
                answer="Use a console cable with 9600 baud.",
                keywords=["console", "serial", "terminal", "baud"]
            ),
        ]
        mock.company.escalation_topics = []
        return mock
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_get_context_with_faq_match(self, mock_vs, mock_get_config, mock_config_with_faqs):
        """Test getting context when FAQ matches."""
        mock_get_config.return_value = mock_config_with_faqs
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        mock_vs.return_value = mock_vector_store
        
        hub = KnowledgeHub()
        # Use query with keywords that will match the FAQ
        results = hub.get_context("console serial terminal connection")
        
        assert len(results) >= 1
        assert any(r.source == "faq" for r in results)
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_get_formatted_context(self, mock_vs, mock_get_config, mock_config_with_faqs):
        """Test formatted context output."""
        mock_get_config.return_value = mock_config_with_faqs
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        mock_vs.return_value = mock_vector_store
        
        hub = KnowledgeHub()
        formatted = hub.get_formatted_context("console connection")
        
        assert "[FAQ" in formatted or formatted == ""
    
    @patch('app.core.knowledge.get_config')
    @patch('app.core.knowledge.get_vector_store')
    def test_empty_context_for_no_match(self, mock_vs, mock_get_config):
        """Test empty context when nothing matches."""
        mock_config = MagicMock()
        mock_config.faqs.faqs = []
        mock_config.company.escalation_topics = []
        mock_get_config.return_value = mock_config
        
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        mock_vs.return_value = mock_vector_store
        
        hub = KnowledgeHub()
        formatted = hub.get_formatted_context("random unrelated query xyz123")
        
        assert formatted == ""
