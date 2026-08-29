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

    # Process lifecycle
    startup_timeout: float = 600.0
    stop_timeout: float = 30.0
    health_interval: float = 1.0
    # Unload the loaded LLM / audio server after this many seconds with no
    # requests. 0 disables idle unload (default).
    idle_unload_seconds: float = 300
    # Where the proxy stages uploaded audio files for backends whose
    # transcription API takes a server-side path (audio.cpp). Must be
    # writable by llamaswap AND readable by the backend process.
    audio_tmp_dir: str = "/tmp/llamaswap-audio"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
