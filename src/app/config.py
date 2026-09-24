"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.

"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    
    # LLM Configuration
    ollama_api_key: str = "3d3603d23f7e43529a6a65b0a2023f30.nUyGdHS0ROzRVO-nsc9vjrn7"
    model_provider: str = "ollama"
    primary_model: str = "gemma4:cloud"
    fallback_model: str = "gemma4:cloud"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "production-api"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    max_retries: int = 3
    cache_ttl_seconds: int = 300
    
    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere"""
    return Settings()







