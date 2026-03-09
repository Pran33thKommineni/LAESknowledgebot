"""Tests for configuration management."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.config import (
    CompanyConfig,
    FAQItem,
    FAQConfig,
    PromptsConfig,
    LLMProviderConfig,
    EmbeddingsConfig,
    VectorStoreConfig,
    ProvidersConfig,
    Settings,
    load_yaml_config,
    ConfigManager,
)


class TestCompanyConfig:
    """Tests for company configuration."""
    
    def test_default_values(self):
        """Test default company config values."""
        config = CompanyConfig()
        
        assert config.name == "Acme Corp"
        assert config.industry == "E-commerce"
        assert config.tone == "friendly and professional"
    
    def test_custom_values(self):
        """Test custom company config values."""
        config = CompanyConfig(
            name="Ruckus Networks",
            industry="Network Equipment",
            tone="technical and precise",
            escalation_email="support@ruckus.com"
        )
        
        assert config.name == "Ruckus Networks"
        assert config.industry == "Network Equipment"
        assert config.escalation_email == "support@ruckus.com"
    
    def test_escalation_topics(self):
        """Test escalation topics list."""
        config = CompanyConfig(
            escalation_topics=["warranty", "billing", "legal"]
        )
        
        assert len(config.escalation_topics) == 3
        assert "warranty" in config.escalation_topics


class TestFAQConfig:
    """Tests for FAQ configuration."""
    
    def test_empty_faqs(self):
        """Test empty FAQ list."""
        config = FAQConfig()
        
        assert config.faqs == []
    
    def test_faq_items(self):
        """Test FAQ items."""
        config = FAQConfig(faqs=[
            FAQItem(
                question="Test question?",
                answer="Test answer.",
                keywords=["test", "example"]
            )
        ])
        
        assert len(config.faqs) == 1
        assert config.faqs[0].question == "Test question?"
        assert "test" in config.faqs[0].keywords


class TestFAQItem:
    """Tests for individual FAQ items."""
    
    def test_create_faq_item(self):
        """Test creating FAQ item."""
        item = FAQItem(
            question="How do I return an item?",
            answer="Visit our returns portal.",
            keywords=["return", "refund"]
        )
        
        assert item.question == "How do I return an item?"
        assert len(item.keywords) == 2
    
    def test_faq_item_empty_keywords(self):
        """Test FAQ item with no keywords."""
        item = FAQItem(
            question="Test?",
            answer="Answer."
        )
        
        assert item.keywords == []


class TestPromptsConfig:
    """Tests for prompts configuration."""
    
    def test_default_prompts(self):
        """Test default prompt values."""
        config = PromptsConfig()
        
        assert config.system_prompt == ""
        assert config.greeting_prompt == ""
    
    def test_custom_prompts(self):
        """Test custom prompt values."""
        config = PromptsConfig(
            system_prompt="You are a helpful assistant.",
            greeting_prompt="Hello! How can I help?"
        )
        
        assert "helpful assistant" in config.system_prompt
        assert "Hello" in config.greeting_prompt


class TestLLMProviderConfig:
    """Tests for LLM provider configuration."""
    
    def test_required_model(self):
        """Test that model is required."""
        config = LLMProviderConfig(model="gpt-4o")
        
        assert config.model == "gpt-4o"
    
    def test_default_values(self):
        """Test default provider config values."""
        config = LLMProviderConfig(model="test")
        
        assert config.enabled is True
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
    
    def test_custom_values(self):
        """Test custom provider config values."""
        config = LLMProviderConfig(
            model="llama-3.3-70b-versatile",
            enabled=True,
            temperature=0.3,
            max_tokens=2048
        )
        
        assert config.temperature == 0.3
        assert config.max_tokens == 2048


class TestEmbeddingsConfig:
    """Tests for embeddings configuration."""
    
    def test_default_values(self):
        """Test default embeddings config."""
        config = EmbeddingsConfig()
        
        assert config.provider == "sentence-transformers"
        assert config.model == "all-MiniLM-L6-v2"
    
    def test_openai_embeddings(self):
        """Test OpenAI embeddings config."""
        config = EmbeddingsConfig(
            provider="openai",
            model="text-embedding-3-small"
        )
        
        assert config.provider == "openai"


class TestVectorStoreConfig:
    """Tests for vector store configuration."""
    
    def test_default_values(self):
        """Test default vector store config."""
        config = VectorStoreConfig()
        
        assert config.collection_name == "customer_service_docs"
        assert config.top_k == 3
        assert config.similarity_threshold == 0.7
    
    def test_custom_values(self):
        """Test custom vector store config."""
        config = VectorStoreConfig(
            collection_name="icx8200_docs",
            top_k=5,
            similarity_threshold=0.5
        )
        
        assert config.collection_name == "icx8200_docs"
        assert config.top_k == 5


class TestProvidersConfig:
    """Tests for providers configuration."""
    
    def test_default_provider(self):
        """Test default provider setting."""
        config = ProvidersConfig()
        
        assert config.default_provider == "openai"
    
    def test_multiple_providers(self):
        """Test multiple provider configs."""
        config = ProvidersConfig(
            default_provider="groq",
            providers={
                "openai": LLMProviderConfig(model="gpt-4o"),
                "groq": LLMProviderConfig(model="llama-3.3-70b-versatile"),
            }
        )
        
        assert config.default_provider == "groq"
        assert "openai" in config.providers
        assert "groq" in config.providers


class TestSettings:
    """Tests for application settings."""
    
    def test_default_settings(self):
        """Test default settings values."""
        with patch.dict('os.environ', {}, clear=True):
            settings = Settings()
            
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.debug is False
            assert settings.log_level == "INFO"
    
    def test_api_keys_from_env(self):
        """Test loading API keys from environment."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'GROQ_API_KEY': 'test-groq-key',
        }):
            settings = Settings()
            
            assert settings.openai_api_key == 'test-openai-key'
            assert settings.anthropic_api_key == 'test-anthropic-key'
            assert settings.groq_api_key == 'test-groq-key'


class TestLoadYamlConfig:
    """Tests for YAML config loading."""
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file returns empty dict."""
        result = load_yaml_config("nonexistent.yaml", tmp_path)
        
        assert result == {}
    
    def test_load_valid_yaml(self, tmp_path):
        """Test loading valid YAML file."""
        yaml_content = """
company:
  name: Test Corp
  industry: Testing
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        result = load_yaml_config("test.yaml", tmp_path)
        
        assert result["company"]["name"] == "Test Corp"
    
    def test_auto_append_yaml_extension(self, tmp_path):
        """Test that .yaml extension is auto-appended."""
        yaml_content = "key: value"
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        
        result = load_yaml_config("config", tmp_path)
        
        assert result["key"] == "value"


class TestConfigManager:
    """Tests for ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory with test files."""
        # Create company.yaml
        (tmp_path / "company.yaml").write_text("""
company:
  name: Test Company
  industry: Testing
  tone: friendly
  escalation_email: test@test.com
  business_hours: 9-5
""")
        
        # Create faqs.yaml
        (tmp_path / "faqs.yaml").write_text("""
faqs:
  - question: Test question?
    answer: Test answer.
    keywords:
      - test
""")
        
        # Create prompts.yaml
        (tmp_path / "prompts.yaml").write_text("""
system_prompt: You are a test assistant.
greeting_prompt: Hello!
""")
        
        # Create providers.yaml
        (tmp_path / "providers.yaml").write_text("""
default_provider: groq
providers:
  groq:
    enabled: true
    model: llama-3.3-70b-versatile
    temperature: 0.3
    max_tokens: 2048
""")
        
        return tmp_path
    
    def test_load_company_config(self, temp_config_dir):
        """Test loading company configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        assert manager.company.name == "Test Company"
        assert manager.company.industry == "Testing"
    
    def test_load_faq_config(self, temp_config_dir):
        """Test loading FAQ configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        assert len(manager.faqs.faqs) == 1
        assert manager.faqs.faqs[0].question == "Test question?"
    
    def test_load_prompts_config(self, temp_config_dir):
        """Test loading prompts configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        assert "test assistant" in manager.prompts.system_prompt
    
    def test_load_providers_config(self, temp_config_dir):
        """Test loading providers configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        assert manager.providers.default_provider == "groq"
        assert "groq" in manager.providers.providers
    
    def test_reload_config(self, temp_config_dir):
        """Test reloading configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        original_name = manager.company.name
        
        # Modify config file
        (temp_config_dir / "company.yaml").write_text("""
company:
  name: Updated Company
  industry: Testing
  tone: friendly
  escalation_email: test@test.com
  business_hours: 9-5
""")
        
        manager.reload()
        
        assert manager.company.name == "Updated Company"
        assert manager.company.name != original_name
