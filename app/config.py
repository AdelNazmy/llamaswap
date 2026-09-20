"""Application settings, overridable via env vars prefixed LLAMASWAP_."""

from functools import lru_cache
from typing import Optional

from pydantic import field_validator
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
    # When both audio roles (tts AND asr) are loaded, only the smallest
    # chat LLM (smallest weights file) may be served — TTS + ASR + a big
    # LLM may not fit on one GPU. The requested chat model is substituted
    # with the smallest one and the swap happens as usual; set false to
    # disable.
    audio_vram_guard: bool = True
    # Inverse guard: while a "big" chat LLM (any LLM other than the
    # smallest by weights-file size) is loaded, TTS/ASR requests are
    # rejected with 409 so audio cannot stack on top of a large model.
    # Set false to allow audio alongside a big LLM.
    block_audio_on_big_llm: bool = True
    # When a "big" chat LLM (any LLM other than the smallest by
    # weights-file size) is requested, stop the running TTS/ASR servers
    # first to free VRAM (the embedding server stays up). Set false to
    # keep audio loaded instead — then audio_vram_guard may downgrade
    # the request.
    unload_audio_on_big_llm: bool = True
    # Image-generation VRAM guard. Diffusion models (e.g. FLUX) can use
    # most of the GPU, so by default:
    #   * before loading the image server, stop the chat LLM, TTS/ASR,
    #     and embedding servers first to free the whole GPU
    #   * before serving any chat request, stop the image server
    #   * before serving any TTS/ASR request, stop the image server
    unload_on_image: bool = True
    unload_image_on_llm: bool = True
    unload_image_on_audio: bool = True
    # The embedding server (role: embedding, e.g. octen-embedding) is
    # launched automatically at boot and kept running for the lifetime of
    # the proxy (only stopped to free VRAM for an LLM or the image server).
    # Set false to launch it on first use instead — like the chat / TTS /
    # ASR / image endpoints — so nothing boots until a request names it.
    embedding_auto_start: bool = True
    # Where the proxy stages uploaded audio files for backends whose
    # transcription API takes a server-side path (audio.cpp). Must be
    # writable by llamaswap AND readable by the backend process.
    audio_tmp_dir: str = "/tmp/llamaswap-audio"
    log_level: str = "INFO"

    # Per-role idle-unload overrides. None inherits idle_unload_seconds;
    # 0 disables idle unload for that role (chat always uses
    # idle_unload_seconds). Useful because a diffusion server or TTS server
    # wants a much shorter residency than a chat LLM.
    idle_unload_audio_seconds: Optional[float] = None
    idle_unload_image_seconds: Optional[float] = None

    @field_validator(
        "idle_unload_audio_seconds", "idle_unload_image_seconds", mode="before"
    )
    @classmethod
    def _empty_string_to_none(cls, value):
        # docker-compose passes ``${VAR:-}`` as an empty string when the
        # variable is unset; treat that the same as "not set" (None).
        if value is None or value == "":
            return None
        return value

    # Optional authentication. When api_key is set, every /v1/* request (and
    # the admin dashboard at /) must present it as
    # ``Authorization: Bearer <key>`` or ``X-Api-Key: <key>``. /health and
    # /metrics stay open for health-checking and scraping.
    api_key: str = ""
    # Admin key gating the destructive/admin endpoints (/v1/reset,
    # /v1/registry/reload, /v1/models/{model}/load|unload). When unset it
    # falls back to api_key; if both are unset those endpoints are open.
    admin_key: str = ""

    # Global request backpressure: maximum concurrently in-flight requests
    # before the proxy answers 429 + Retry-After. 0 disables the limit.
    max_concurrency: int = 0

    # Emit structured JSON logs (one JSON object per line) instead of the
    # default human-readable format.
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
