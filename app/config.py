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
    # Where the proxy stages uploaded audio files for backends whose
    # transcription API takes a server-side path (audio.cpp). Must be
    # writable by llamaswap AND readable by the backend process.
    audio_tmp_dir: str = "/tmp/llamaswap-audio"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
