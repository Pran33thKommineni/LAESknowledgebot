"""Configuration management with Pydantic models and YAML loading."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Base paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DOCUMENTS_DIR = BASE_DIR / "documents"


class CompanyConfig(BaseModel):
    """Company-specific configuration."""
    
    name: str = "Acme Corp"
    industry: str = "E-commerce"
    tone: str = "friendly and professional"
    escalation_email: str = "support@example.com"
    business_hours: str = "9am-5pm EST"
    website: str = ""
    description: str = ""
    escalation_topics: list[str] = Field(default_factory=list)


class FAQItem(BaseModel):
    """Single FAQ entry."""
    
    question: str
    answer: str
    keywords: list[str] = Field(default_factory=list)


class FAQConfig(BaseModel):
    """FAQ knowledge base configuration."""
    
    faqs: list[FAQItem] = Field(default_factory=list)


class PromptsConfig(BaseModel):
    """System prompts configuration."""
    
    system_prompt: str = ""
    context_prompt: str = ""
    escalation_prompt: str = ""
    greeting_prompt: str = ""
    fallback_prompt: str = ""


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    
    enabled: bool = True
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    
    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    
    collection_name: str = "customer_service_docs"
    persist_directory: str = "./chroma_db"
    top_k: int = 3
    similarity_threshold: float = 0.7


class ProvidersConfig(BaseModel):
    """LLM providers configuration."""
    
    default_provider: str = "openai"
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    
    # Paths
    config_dir: Path = CONFIG_DIR
    documents_dir: Path = DOCUMENTS_DIR
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def load_yaml_config(filename: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        filename: Name of the YAML file (with or without .yaml extension)
        config_dir: Directory containing config files
        
    Returns:
        Dictionary containing the configuration
    """
    if not filename.endswith(".yaml"):
        filename = f"{filename}.yaml"
    
    filepath = config_dir / filename
    
    if not filepath.exists():
        return {}
    
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


class ConfigManager:
    """
    Central configuration manager that loads and provides access to all configs.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Optional custom config directory path
        """
        self.config_dir = config_dir or CONFIG_DIR
        self._settings: Optional[Settings] = None
        self._company: Optional[CompanyConfig] = None
        self._faqs: Optional[FAQConfig] = None
        self._prompts: Optional[PromptsConfig] = None
        self._providers: Optional[ProvidersConfig] = None
        
        # Load all configurations
        self.reload()
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._settings = Settings()
        self._load_company_config()
        self._load_faq_config()
        self._load_prompts_config()
        self._load_providers_config()
    
    def _load_company_config(self) -> None:
        """Load company configuration."""
        data = load_yaml_config("company", self.config_dir)
        company_data = data.get("company", {})
        self._company = CompanyConfig(**company_data)
    
    def _load_faq_config(self) -> None:
        """Load FAQ configuration."""
        data = load_yaml_config("faqs", self.config_dir)
        self._faqs = FAQConfig(**data)
    
    def _load_prompts_config(self) -> None:
        """Load prompts configuration."""
        data = load_yaml_config("prompts", self.config_dir)
        self._prompts = PromptsConfig(**data)
    
    def _load_providers_config(self) -> None:
        """Load providers configuration."""
        data = load_yaml_config("providers", self.config_dir)
        
        # Parse providers into proper config objects
        providers_dict = {}
        for name, config in data.get("providers", {}).items():
            if config:
                providers_dict[name] = LLMProviderConfig(**config)
        
        self._providers = ProvidersConfig(
            default_provider=data.get("default_provider", "openai"),
            providers=providers_dict,
            embeddings=EmbeddingsConfig(**data.get("embeddings", {})),
            vector_store=VectorStoreConfig(**data.get("vector_store", {})),
        )
    
    @property
    def settings(self) -> Settings:
        """Get application settings."""
        if self._settings is None:
            self._settings = Settings()
        return self._settings
    
    @property
    def company(self) -> CompanyConfig:
        """Get company configuration."""
        if self._company is None:
            self._load_company_config()
        return self._company  # type: ignore
    
    @property
    def faqs(self) -> FAQConfig:
        """Get FAQ configuration."""
        if self._faqs is None:
            self._load_faq_config()
        return self._faqs  # type: ignore
    
    @property
    def prompts(self) -> PromptsConfig:
        """Get prompts configuration."""
        if self._prompts is None:
            self._load_prompts_config()
        return self._prompts  # type: ignore
    
    @property
    def providers(self) -> ProvidersConfig:
        """Get providers configuration."""
        if self._providers is None:
            self._load_providers_config()
        return self._providers  # type: ignore
    
    def get_formatted_system_prompt(self) -> str:
        """
        Get the system prompt with company variables substituted.
        
        Returns:
            Formatted system prompt string
        """
        return self.prompts.system_prompt.format(
            company_name=self.company.name,
            industry=self.company.industry,
            tone=self.company.tone,
            company_description=self.company.description,
            business_hours=self.company.business_hours,
            escalation_email=self.company.escalation_email,
        )
    
    def get_formatted_greeting(self) -> str:
        """Get the greeting prompt with company name substituted."""
        return self.prompts.greeting_prompt.format(
            company_name=self.company.name,
        )


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reload_config() -> ConfigManager:
    """
    Reload all configurations and return the updated manager.
    
    Returns:
        Updated ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    else:
        _config_manager.reload()
    return _config_manager
