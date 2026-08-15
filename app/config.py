"""Application settings, overridable via env vars prefixed LLAMASWAP_."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLAMASWAP_", env_file=".env", extra="ignore"
    )

    # Proxy (OpenAI-compatible) listener
    host: str = "0.0.0.0"
    port: int = 11434

    # Model registry
    backend_dir: str = "backend"

    # Internal llama-server binding (only one instance runs at a time)
    internal_host: str = "127.0.0.1"

    # Process lifecycle
    startup_timeout: float = 600.0
    stop_timeout: float = 30.0
    health_interval: float = 1.0
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
