"""Tests for LLM provider abstraction."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.config import LLMProviderConfig
from app.core.llm_providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GroqProvider,
    LLMProviderFactory,
    get_llm_factory,
)


class TestLLMProviderConfig:
    """Tests for LLM provider configuration."""
    
    def test_create_config(self):
        """Test creating provider config."""
        config = LLMProviderConfig(
            enabled=True,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1024
        )
        
        assert config.enabled is True
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
    
    def test_config_defaults(self):
        """Test config default values."""
        config = LLMProviderConfig(model="test-model")
        
        assert config.enabled is True
        assert config.temperature == 0.7
        assert config.max_tokens == 1024


class TestOpenAIProvider:
    """Tests for OpenAI provider."""
    
    def test_create_provider(self):
        """Test creating OpenAI provider."""
        config = LLMProviderConfig(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048
        )
        
        provider = OpenAIProvider(config, api_key="test-key")
        
        assert provider.config.model == "gpt-4o"
        assert provider.api_key == "test-key"
    
    @patch('langchain_openai.ChatOpenAI')
    def test_model_creation(self, mock_chat):
        """Test that model is created with correct params."""
        config = LLMProviderConfig(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048
        )
        
        provider = OpenAIProvider(config, api_key="test-key")
        _ = provider.model  # Trigger model creation
        
        mock_chat.assert_called_once_with(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
            api_key="test-key"
        )


class TestAnthropicProvider:
    """Tests for Anthropic provider."""
    
    def test_create_provider(self):
        """Test creating Anthropic provider."""
        config = LLMProviderConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=1024
        )
        
        provider = AnthropicProvider(config, api_key="test-key")
        
        assert provider.config.model == "claude-3-5-sonnet-20241022"
    
    @patch('langchain_anthropic.ChatAnthropic')
    def test_model_creation(self, mock_chat):
        """Test that model is created with correct params."""
        config = LLMProviderConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=1024
        )
        
        provider = AnthropicProvider(config, api_key="test-key")
        _ = provider.model
        
        mock_chat.assert_called_once()


class TestGroqProvider:
    """Tests for Groq provider."""
    
    def test_create_provider(self):
        """Test creating Groq provider."""
        config = LLMProviderConfig(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048
        )
        
        provider = GroqProvider(config, api_key="test-groq-key")
        
        assert provider.config.model == "llama-3.3-70b-versatile"
        assert provider.api_key == "test-groq-key"
    
    @patch('langchain_groq.ChatGroq')
    def test_model_creation(self, mock_chat):
        """Test that Groq model is created with correct params."""
        config = LLMProviderConfig(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048
        )
        
        provider = GroqProvider(config, api_key="test-groq-key")
        _ = provider.model
        
        mock_chat.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
            api_key="test-groq-key"
        )


class TestLLMProviderFactory:
    """Tests for LLM provider factory."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        mock = MagicMock()
        mock.providers.default_provider = "groq"
        mock.providers.providers = {
            "openai": LLMProviderConfig(model="gpt-4o", enabled=True),
            "anthropic": LLMProviderConfig(model="claude-3-5-sonnet-20241022", enabled=True),
            "groq": LLMProviderConfig(model="llama-3.3-70b-versatile", enabled=True),
        }
        mock.settings.openai_api_key = "test-openai-key"
        mock.settings.anthropic_api_key = "test-anthropic-key"
        mock.settings.groq_api_key = "test-groq-key"
        return mock
    
    @patch('app.core.llm_providers.get_config')
    def test_get_default_provider(self, mock_get_config, mock_config):
        """Test getting default provider."""
        mock_get_config.return_value = mock_config
        
        factory = LLMProviderFactory()
        provider = factory.get_provider()
        
        assert isinstance(provider, GroqProvider)
    
    @patch('app.core.llm_providers.get_config')
    def test_get_specific_provider(self, mock_get_config, mock_config):
        """Test getting a specific provider by name."""
        mock_get_config.return_value = mock_config
        
        factory = LLMProviderFactory()
        provider = factory.get_provider("openai")
        
        assert isinstance(provider, OpenAIProvider)
    
    @patch('app.core.llm_providers.get_config')
    def test_provider_caching(self, mock_get_config, mock_config):
        """Test that providers are cached."""
        mock_get_config.return_value = mock_config
        
        factory = LLMProviderFactory()
        provider1 = factory.get_provider("groq")
        provider2 = factory.get_provider("groq")
        
        assert provider1 is provider2
    
    @patch('app.core.llm_providers.get_config')
    def test_list_available_providers(self, mock_get_config, mock_config):
        """Test listing available providers."""
        mock_get_config.return_value = mock_config
        
        factory = LLMProviderFactory()
        available = factory.list_available_providers()
        
        assert "openai" in available
        assert "anthropic" in available
        assert "groq" in available
    
    @patch('app.core.llm_providers.get_config')
    def test_disabled_provider_not_available(self, mock_get_config):
        """Test that disabled providers raise error."""
        mock_config = MagicMock()
        mock_config.providers.default_provider = "openai"
        mock_config.providers.providers = {
            "openai": LLMProviderConfig(model="gpt-4o", enabled=False),
        }
        mock_get_config.return_value = mock_config
        
        factory = LLMProviderFactory()
        
        with pytest.raises(ValueError, match="not enabled"):
            factory.get_provider("openai")
    
    @patch('app.core.llm_providers.get_config')
    def test_unknown_provider_raises_error(self, mock_get_config, mock_config):
        """Test that unknown provider raises error."""
        mock_get_config.return_value = mock_config
        
        factory = LLMProviderFactory()
        
        with pytest.raises(ValueError, match="not configured"):
            factory.get_provider("unknown-provider")


class TestProviderGeneration:
    """Tests for LLM response generation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        mock = AsyncMock()
        mock.ainvoke.return_value = MagicMock(content="Test response")
        return mock
    
    @pytest.mark.asyncio
    async def test_generate_with_messages(self, mock_model):
        """Test generating response with messages."""
        config = LLMProviderConfig(model="test-model")
        
        with patch.object(GroqProvider, '_create_model', return_value=mock_model):
            provider = GroqProvider(config, api_key="test-key")
            
            messages = [
                {"role": "user", "content": "Hello"},
            ]
            
            response = await provider.generate(messages)
            
            assert response == "Test response"
            mock_model.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, mock_model):
        """Test generating response with system prompt."""
        config = LLMProviderConfig(model="test-model")
        
        with patch.object(GroqProvider, '_create_model', return_value=mock_model):
            provider = GroqProvider(config, api_key="test-key")
            
            messages = [{"role": "user", "content": "Hello"}]
            system_prompt = "You are a helpful assistant."
            
            response = await provider.generate(messages, system_prompt)
            
            assert response == "Test response"
            # Verify system message was included
            call_args = mock_model.ainvoke.call_args[0][0]
            assert len(call_args) == 2  # System + User message
