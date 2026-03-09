"""Multi-provider LLM abstraction layer."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import get_config, LLMProviderConfig
from app.utils.logging import get_logger

logger = get_logger("llm_providers")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMProviderConfig, api_key: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            config: Provider configuration
            api_key: API key for the provider
        """
        self.config = config
        self.api_key = api_key
        self._model: Optional[BaseChatModel] = None
    
    @abstractmethod
    def _create_model(self) -> BaseChatModel:
        """Create and return the LangChain model instance."""
        pass
    
    @property
    def model(self) -> BaseChatModel:
        """Get the LangChain model instance, creating it if necessary."""
        if self._model is None:
            self._model = self._create_model()
        return self._model
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated response text
        """
        langchain_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        # Convert messages to LangChain format
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
        
        # Generate response
        try:
            response = await self.model.ainvoke(langchain_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def _create_model(self) -> BaseChatModel:
        """Create OpenAI chat model."""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.api_key,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def _create_model(self) -> BaseChatModel:
        """Create Anthropic chat model."""
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.api_key,
        )


class GroqProvider(LLMProvider):
    """Groq LLM provider for fast inference."""
    
    def _create_model(self) -> BaseChatModel:
        """Create Groq chat model."""
        from langchain_groq import ChatGroq
        
        return ChatGroq(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.api_key,
        )


class LLMProviderFactory:
    """
    Factory for creating and managing LLM providers.
    
    Supports runtime switching between providers.
    """
    
    # Registry of available provider classes
    _provider_classes: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "groq": GroqProvider,
    }
    
    def __init__(self):
        """Initialize the factory with cached provider instances."""
        self._providers: dict[str, LLMProvider] = {}
        self._config = get_config()
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """
        Register a custom provider class.
        
        Args:
            name: Provider name identifier
            provider_class: Provider class to register
        """
        cls._provider_classes[name] = provider_class
        logger.info(f"Registered custom provider: {name}")
    
    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        """
        Get an LLM provider by name.
        
        Args:
            name: Provider name (uses default if not specified)
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider is not found or not enabled
        """
        if name is None:
            name = self._config.providers.default_provider
        
        # Return cached provider if available
        if name in self._providers:
            return self._providers[name]
        
        # Check if provider is configured
        if name not in self._config.providers.providers:
            raise ValueError(f"Provider '{name}' is not configured")
        
        provider_config = self._config.providers.providers[name]
        
        if not provider_config.enabled:
            raise ValueError(f"Provider '{name}' is not enabled")
        
        # Check if provider class exists
        if name not in self._provider_classes:
            raise ValueError(f"Provider class for '{name}' not found")
        
        # Get API key
        api_key = self._get_api_key(name)
        
        # Create provider instance
        provider_class = self._provider_classes[name]
        provider = provider_class(provider_config, api_key)
        
        # Cache and return
        self._providers[name] = provider
        logger.info(f"Created provider instance: {name}")
        
        return provider
    
    def _get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get the API key for a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            API key string or None
        """
        settings = self._config.settings
        
        if provider_name == "openai":
            return settings.openai_api_key
        elif provider_name == "anthropic":
            return settings.anthropic_api_key
        elif provider_name == "groq":
            return settings.groq_api_key
        
        return None
    
    def list_available_providers(self) -> list[str]:
        """
        List all available and enabled providers.
        
        Returns:
            List of provider names
        """
        available = []
        for name, config in self._config.providers.providers.items():
            if config.enabled and name in self._provider_classes:
                available.append(name)
        return available
    
    def get_default_provider(self) -> LLMProvider:
        """
        Get the default LLM provider.
        
        Returns:
            Default LLMProvider instance
        """
        return self.get_provider(self._config.providers.default_provider)


# Global factory instance
_factory: Optional[LLMProviderFactory] = None


def get_llm_factory() -> LLMProviderFactory:
    """
    Get the global LLM provider factory.
    
    Returns:
        LLMProviderFactory instance
    """
    global _factory
    if _factory is None:
        _factory = LLMProviderFactory()
    return _factory


async def generate_response(
    messages: list[dict[str, str]],
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Convenience function to generate a response using the specified provider.
    
    Args:
        messages: List of message dicts
        system_prompt: Optional system prompt
        provider: Optional provider name (uses default if not specified)
        
    Returns:
        Generated response text
    """
    factory = get_llm_factory()
    llm_provider = factory.get_provider(provider)
    return await llm_provider.generate(messages, system_prompt)
