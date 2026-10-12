"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.

"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os

class Settings(BaseSettings):
    
    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None
    embeddings_provider : str = "ollama"
    primary_model: str = "gemma4:cloud"
    fallback_model: str = "gemma4:cloud"

    # Embeddings
    embeddings_model: str = "chroma/all-minilm-l6-v2-f32:latest"
    embeddings_base_url: str | None = None
    embeddings_api_key: str | None = None

    aws_region: str = "eu-central-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    @property
    def resolved_embeddings_base_url(self) -> str:
        return self.embeddings_base_url or self.ollama_base_url

    @property
    def resolved_embeddings_api_key(self) -> str | None:
        return self.embeddings_api_key or self.ollama_api_key

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str | None = None
    langchain_project: str = "production-api"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    max_retries: int = 3
    cache_ttl_seconds: int = 300
    
    # Supabase
    collection_name: str = "production_docs"
    supabase_database_url: str | None = None
    supabase_url: str=""
    supabase_publishable_key: str | None = None
    supabase_secret_key: str | None = None
    supabase_jwks_url: str | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere"""
    return Settings()







