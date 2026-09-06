"""llamaswap — an OpenAI-compatible proxy in front of llama-server.

Model registry lives in backend/*.yaml; the requested model is launched
(or swapped in) transparently on demand. Only the embedding server is
persistent; everything else is on-demand and idle-unloaded like the chat
LLM:

  * ``embedding`` — dedicated embedding llama-server (EmbeddingManager),
    started at boot and kept running (only stopped to free VRAM for an LLM)
  * ``chat`` — one LLM at a time (ProcessManager), swapped on request and
    unloaded after ``idle_unload_seconds`` with no requests
  * ``tts`` / ``asr`` — audio servers (RoleServerManager) that boot on
    first use, swap between backends per request (e.g. qwen3-asr vs
    whisper-server) and idle-unload like the chat LLM; proxied through
    the OpenAI audio endpoints (/v1/audio/speech,
    /v1/audio/transcriptions, ...).
  * ``image`` — an image-generation server (RoleServerManager) that boots
    on first use, swaps between image models per request, and idle-unloads
    like the chat LLM; proxied through the OpenAI image endpoints
    (/v1/images/generations, /v1/images/edits).

The OpenAI Responses API (``/v1/responses``) is proxied straight through to
llama-server's responses route, alongside ``/v1/chat/completions`` and
``/v1/completions``. Two Ollama-compatible read-only endpoints are also
served: ``/api/tags`` (registry model list) and ``/api/ps`` (currently
resident backends).

While tts AND asr are both loaded, a VRAM guard substitutes the smallest
chat LLM for any requested chat model (see Settings.audio_vram_guard).
Requesting a "big" chat LLM (any LLM other than the smallest by
weights-file size) unloads the running TTS/ASR servers first to free VRAM,
leaving the embedding server up (see Settings.unload_audio_on_big_llm);
conversely, while a big chat LLM is loaded, TTS/ASR requests are rejected
with 409 (see Settings.block_audio_on_big_llm).

Image generation is the heaviest workload, so it is guarded the same way:
before it loads, the chat LLM, TTS/ASR servers, and the embedding server
are stopped to free the whole GPU (see Settings.unload_on_image); before
any chat request the image server is stopped
(Settings.unload_image_on_llm); and any TTS/ASR request stops the image
server too (Settings.unload_image_on_audio).
"""

import asyncio
import hashlib
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .events import bus
from .metrics import metrics
from .server_manager import RoleServerLoadError, RoleServerManager
from .config import Settings, get_settings
from .embedding_manager import EmbeddingLoadError, EmbeddingManager
from .process_manager import ModelLoadError, ProcessManager
from .proxy import (
    extract_model,
    extract_response_format,
    extract_speech_format,
    extract_stream_flag,
    extract_text_field,
    format_transcription,
    json_inject_field,
    json_remove_key,
    multipart_inject_field,
    proxy_json,
    proxy_raw,
    proxy_raw_stream,
    proxy_stream,
    remove_audio_upload,
    save_audio_upload,
    transcode_audio,
)
from .registry import ModelConfig, Registry, RegistryError, UnknownModelError

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("llamaswap")


class _JsonFormatter(logging.Formatter):
    """One JSON object per log line (LLAMASWAP_LOG_JSON=true)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _error(status: int, message: str, etype: str = "invalid_request_error"):
    return status, {"error": {"message": message, "type": etype}}


def _fwd_headers(request: Request) -> dict[str, str]:
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "accept-encoding")
    }


def _apply_pin(request: Request, *managers) -> None:
    """Honour ``X-Pin-Seconds`` by suppressing idle unload on the given
    managers for the requested number of seconds."""
    raw = request.headers.get("x-pin-seconds")
    if not raw:
        return
    try:
        seconds = float(raw)
    except ValueError:
        return
    for mgr in managers:
        mgr.pin(seconds)


def _runtime_state(request: Request, entry: dict) -> dict:
    """Attach ``state``/``loaded`` to a ``/v1/models`` entry from live
    manager status."""
    name = entry["id"]
    role = entry["role"]
    state = "stopped"
    if role == "llm":
        st = request.app.state.manager.status()
        if st.get("model") == name:
            state = st.get("state", "stopped")
    elif role == "embedding":
        emb: Optional[EmbeddingManager] = request.app.state.embedding_manager
        if emb is not None and emb.name == name:
            state = emb.status().get("state", "stopped")
    elif role in ("tts", "asr"):
        st = request.app.state.audio_managers[role].status()
        if st.get("model") == name:
            state = st.get("state", "stopped")
    elif role == "image":
        st = request.app.state.image_manager.status()
        if st.get("model") == name:
            state = st.get("state", "stopped")
    entry["state"] = state
    entry["loaded"] = state == "ready"
    return entry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    if settings.log_json:
        formatter: logging.Formatter = _JsonFormatter()
        for handler in logging.root.handlers:
            handler.setFormatter(formatter)
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        for handler in logging.root.handlers:
            handler.setFormatter(formatter)
    backend_dir = Path(settings.backend_dir)
    if not backend_dir.is_absolute():
        backend_dir = ROOT / backend_dir
    app.state.settings = settings
    app.state.registry = Registry(backend_dir)
    app.state.manager = ProcessManager(
        startup_timeout=settings.startup_timeout,
        stop_timeout=settings.stop_timeout,
        health_interval=settings.health_interval,
        idle_unload_seconds=settings.idle_unload_seconds,
    )
    # Audio (tts/asr) managers: one per role, on-demand like the chat
    # LLM — nothing boots here, the first request launches a server
    # (see _route_audio and RoleServerManager.ensure_model).
    audio_idle = (
        settings.idle_unload_audio_seconds
        if settings.idle_unload_audio_seconds is not None
        else settings.idle_unload_seconds
    )
    app.state.audio_managers: dict[str, RoleServerManager] = {}
    for role in ("tts", "asr"):
        manager = RoleServerManager(
            role,
            startup_timeout=settings.startup_timeout,
            stop_timeout=settings.stop_timeout,
            health_interval=settings.health_interval,
            idle_unload_seconds=audio_idle,
            config_dir=settings.audio_tmp_dir,
        )
        manager.configure(app.state.registry.role_configs(role))
        app.state.audio_managers[role] = manager

    # Image-generation manager: one on-demand server (swapped among the
    # model configs of role "image"), routed through the OpenAI image
    # endpoints. Nothing boots here; the first /v1/images/* request
    # launches it (see _route_image).
    image_idle = (
        settings.idle_unload_image_seconds
        if settings.idle_unload_image_seconds is not None
        else settings.idle_unload_seconds
    )
    app.state.image_manager = RoleServerManager(
        "image",
        startup_timeout=settings.startup_timeout,
        stop_timeout=settings.stop_timeout,
        health_interval=settings.health_interval,
        idle_unload_seconds=image_idle,
        config_dir=settings.audio_tmp_dir,
    )
    app.state.image_manager.configure(app.state.registry.role_configs("image"))

    # Persistent embedding server: launched at boot, kept running for the
    # lifetime of the proxy (only stopped to free VRAM for an LLM).
    app.state.embedding_manager = None
    emb_cfg = app.state.registry.embedding_config()
    if emb_cfg is not None:
        emb_manager = EmbeddingManager(
            emb_cfg,
            startup_timeout=settings.startup_timeout,
            stop_timeout=settings.stop_timeout,
            health_interval=settings.health_interval,
        )
        app.state.embedding_manager = emb_manager
        try:
            await emb_manager.start()
        except EmbeddingLoadError as exc:
            logger.error(
                "embedding server failed to start at boot: %s", exc
            )
    logger.info(
        "llamaswap ready on %s:%d (%d models)",
        settings.host, settings.port, len(app.state.registry.models),
    )
    try:
        yield
    finally:
        for manager in app.state.audio_managers.values():
            await manager.shutdown()
        await app.state.image_manager.shutdown()
        if app.state.embedding_manager is not None:
            await app.state.embedding_manager.shutdown()
        await app.state.manager.shutdown()


app = FastAPI(title="llamaswap", version="0.1.0", lifespan=lifespan)


def _health_snapshot(request: Request) -> dict:
    """Full proxy status: chat LLM, embedding, TTS/ASR, and image managers,
    plus registry-level warnings and degraded conditions."""
    embedding_manager: Optional[EmbeddingManager] = (
        request.app.state.embedding_manager
    )
    registry: Registry = request.app.state.registry
    degraded = any(
        mgr.status().get("state") == "failed"
        for mgr in list(request.app.state.audio_managers.values())
        + [request.app.state.image_manager, request.app.state.manager]
    ) or bool(registry.degraded)
    return {
        "status": "degraded" if degraded else "ok",
        "current_model": request.app.state.manager.status(),
        "embedding": (
            embedding_manager.status()
            if embedding_manager is not None
            else None
        ),
        "audio": {
            role: manager.status()
            for role, manager in request.app.state.audio_managers.items()
        },
        "image": request.app.state.image_manager.status(),
        "registry": {
            "degraded": registry.degraded,
            "warnings": registry.warnings,
        },
    }


@app.get("/health")
async def health(request: Request):
    return _health_snapshot(request)


@app.get("/v1/models")
async def list_models(request: Request):
    registry: Registry = request.app.state.registry
    data = [
        _runtime_state(request, entry)
        for entry in registry.list_openai()
    ]
    return {"object": "list", "data": data}


@app.get("/v1/models/{model}")
async def get_model(model: str, request: Request):
    registry: Registry = request.app.state.registry
    try:
        data = registry.to_openai(model)
    except UnknownModelError:
        status, body = _error(404, f"model '{model}' not found")
        return JSONResponse(status_code=status, content=body)
    return _runtime_state(request, data)


@app.post("/v1/registry/reload")
async def reload_registry(request: Request):
    registry: Registry = request.app.state.registry
    try:
        registry.reload()
    except RegistryError as exc:
        status, body = _error(500, str(exc), "server_error")
        return JSONResponse(status_code=status, content=body)
    # Refresh per-role configs only — servers boot on first use and
    # are idle-unloaded, so never pre-start them here.
    for role, manager in request.app.state.audio_managers.items():
        manager.configure(registry.role_configs(role))
    request.app.state.image_manager.configure(registry.role_configs("image"))
    return {
        "reloaded": True,
        "models": request.app.state.registry.names(),
        "warnings": registry.warnings,
        "degraded": registry.degraded,
    }


@app.get("/v1/registry/info")
async def registry_info(request: Request):
    """Registry summary: models by role, load warnings, degraded conditions,
    and the VRAM ranking used by the guards."""
    registry: Registry = request.app.state.registry
    return {
        "models": registry.names(),
        "roles": {
            role: [c.name for c in registry.role_configs(role)]
            for role in ("llm", "embedding", "tts", "asr", "image")
            if registry.role_configs(role)
        },
        "smallest_llm": registry.smallest_llm(),
        "llm_vram_mb": registry.llm_vram_mb(),
        "warnings": registry.warnings,
        "degraded": registry.degraded,
    }


@app.post("/v1/reset")
async def reset(request: Request):
    """Unload every loaded backend: chat LLM, TTS/ASR audio servers, the
    image server, and the persistent embedding server. Everything boots
    again on its next request (the embedding server also best-effort
    relaunches after a chat request, per its persistent role). Returns
    the full post-reset /health snapshot."""
    chat_manager = request.app.state.manager
    audio_managers = request.app.state.audio_managers
    image_manager = request.app.state.image_manager
    embedding_manager = request.app.state.embedding_manager

    await chat_manager.unload()
    for mgr in audio_managers.values():
        await mgr.unload()
    await image_manager.unload()
    if embedding_manager is not None:
        await embedding_manager.stop()

    # Respond with the full post-reset /health snapshot.
    return _health_snapshot(request)


async def _ensure_and_route(
    request: Request, path: str,
):
    registry: Registry = request.app.state.registry
    manager: ProcessManager = request.app.state.manager
    body = await request.body()
    try:
        model = extract_model(body)
    except Exception:  # noqa: BLE001
        model = None
    if model is None:
        status, payload = _error(400, "'model' is required in the request body")
        return JSONResponse(status_code=status, content=payload)
    try:
        cfg = registry.get(model)
    except UnknownModelError as exc:
        known = ", ".join(exc.known)
        status, payload = _error(
            404, f"model '{exc.name}' not found; available: {known}"
        )
        return JSONResponse(status_code=status, content=payload)
    if cfg.role in ("embedding", "tts", "asr"):
        status, payload = _error(
            400,
            f"model '{model}' is a {cfg.role} model; "
            "use /v1/embeddings, /v1/audio/speech or "
            "/v1/audio/transcriptions instead",
        )
        return JSONResponse(status_code=status, content=payload)
    settings: Settings = request.app.state.settings
    smallest = registry.smallest_llm()
    # Big-model request: stop TTS/ASR first to free VRAM (the embedding
    # server stays up). The big model then loads instead of being
    # downgraded by the audio_vram_guard below. Disable via
    # LLAMASWAP_UNLOAD_AUDIO_ON_BIG_LLM=false.
    if settings.unload_audio_on_big_llm and smallest is not None \
            and model != smallest:
        logger.info("big model '%s' requested; unloading TTS/ASR", model)
        await asyncio.gather(*(
            mgr.unload() for mgr in request.app.state.audio_managers.values()
        ))
    # Image generation is the heaviest workload: stop the image server
    # before any chat request so the LLM fits (disable via
    # LLAMASWAP_UNLOAD_IMAGE_ON_LLM=false).
    if settings.unload_image_on_llm:
        await request.app.state.image_manager.unload()
    # VRAM guard: while BOTH audio roles are loaded, only the smallest
    # chat LLM may be served (TTS + ASR + a big LLM may not fit on one
    # GPU). The requested model is transparently substituted and the swap
    # happens as usual; disable via LLAMASWAP_AUDIO_VRAM_GUARD=false.
    if settings.audio_vram_guard:
        tts_mgr = request.app.state.audio_managers["tts"]
        asr_mgr = request.app.state.audio_managers["asr"]
        if tts_mgr.is_running and asr_mgr.is_running:
            if smallest is not None and smallest != model:
                logger.warning(
                    "audio_vram_guard: tts+asr loaded; serving smallest "
                    "chat LLM '%s' instead of requested '%s'",
                    smallest, model,
                )
                model = smallest
                cfg = registry.get(smallest)
    embedding_manager: Optional[EmbeddingManager] = (
        request.app.state.embedding_manager
    )
    try:
        _, port = await manager.ensure_model(model, registry)
    except ModelLoadError as exc:
        # The model may not fit in VRAM while the embedding server is
        # running: free its VRAM and give the LLM launch one retry.
        if embedding_manager is not None and embedding_manager.is_running:
            logger.info(
                "model load failed for '%s'; stopping embedding server to "
                "free resources and retrying", model,
            )
            await embedding_manager.stop_for_resources()
            try:
                _, port = await manager.ensure_model(model, registry)
            except ModelLoadError as retry_exc:
                logger.error("model load failed: %s", retry_exc)
                status, payload = _error(
                    503,
                    f"failed to load model '{model}': {retry_exc}",
                    "server_error",
                )
                return JSONResponse(status_code=status, content=payload)
        else:
            logger.error("model load failed: %s", exc)
            status, payload = _error(
                503, f"failed to load model '{model}': {exc}", "server_error"
            )
            return JSONResponse(status_code=status, content=payload)
    # The LLM is ready. Best-effort bring the embedding server back up if
    # it was stopped to make room for an LLM; never block the response.
    if embedding_manager is not None and not embedding_manager.is_running:
        asyncio.get_running_loop().create_task(
            embedding_manager.ensure_running()
        )
    _apply_pin(request, manager)
    headers = _fwd_headers(request)
    if extract_stream_flag(body):
        gen = proxy_stream(
            port, cfg.host, path, body, model, headers
        )
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    status, data = await proxy_json(port, cfg.host, path, body, model, headers)
    return JSONResponse(status_code=status, content=data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _ensure_and_route(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _ensure_and_route(request, "/v1/completions")


@app.post("/v1/responses")
async def responses(request: Request):
    return await _ensure_and_route(request, "/v1/responses")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    embedding_manager: Optional[EmbeddingManager] = (
        request.app.state.embedding_manager
    )
    if embedding_manager is not None:
        # Fast path: the dedicated embedding server never goes through the
        # LLM swap path, so the loaded chat model is left untouched.
        if not embedding_manager.is_running:
            # Temporarily down (e.g. stopped to free VRAM for an LLM):
            # try to bring it back before answering.
            if not await embedding_manager.ensure_running():
                status, payload = _error(
                    503, "embedding server is not running", "server_error"
                )
                return JSONResponse(status_code=status, content=payload)
        body = await request.body()
        content_type = request.headers.get("content-type", "")
        try:
            model = extract_model(body, content_type)
        except Exception:  # noqa: BLE001
            model = None
        if model is None:
            model = embedding_manager.name
        cfg = embedding_manager.config
        headers = _fwd_headers(request)
        if extract_stream_flag(body):
            gen = proxy_stream(
                cfg.port, cfg.host, "/v1/embeddings", body, model, headers
            )
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
        status, data = await proxy_json(
            cfg.port, cfg.host, "/v1/embeddings", body, model, headers
        )
        return JSONResponse(status_code=status, content=data)
    # No dedicated embedding server configured: fall back to the legacy
    # behaviour (swap an embedding-capable model in as the LLM).
    return await _ensure_and_route(request, "/v1/embeddings")


# ---------------------------------------------------------------------------
# OpenAI audio API (TTS / ASR)
# ---------------------------------------------------------------------------

def _apply_clone_reference(cfg: ModelConfig, body: bytes) -> bytes:
    """Inject backend clone-reference fields for voice-cloning TTS models.

    Clone-capable audio.cpp backends (``qwen3_tts`` base, ``outetts``, ...)
    synthesize from a reference WAV plus an optional transcript
    (``voice_ref`` / ``reference_text``). Those are upstream request fields,
    not OpenAI ``/v1/audio/speech`` parameters, so llamaswap keeps them off
    the public surface and supplies them internally from the model config
    (``meta.reference_voice`` / ``meta.reference_text``), only for models
    declaring the ``clone`` capability. Preset-voice TTS models never get
    them.

    Returns ``body`` unchanged when the model is not clone-capable or has no
    reference configured (or the client already supplied the field).
    """
    capabilities = getattr(cfg.meta, "capabilities", None) or []
    if "clone" not in capabilities:
        return body
    reference_voice = getattr(cfg.meta, "reference_voice", None)
    if reference_voice:
        voice_path = Path(reference_voice)
        voice_ref = str(
            voice_path if voice_path.is_absolute() else ROOT / voice_path
        )
        body = json_inject_field(body, "voice_ref", voice_ref)
    reference_text = getattr(cfg.meta, "reference_text", None)
    if reference_text:
        body = json_inject_field(body, "reference_text", reference_text)
    return body


async def _route_audio(
    request: Request, role: str, backend_path: str, *,
    expect_body_model: bool = True,
):
    """Route an OpenAI audio request to the on-demand ``role`` manager.

    ``role`` is "tts" or "asr". The request body may be JSON (speech) or
    multipart/form-data (transcriptions). The model field (JSON or form)
    selects which backend of the role to load; llamaswap starts the
    server on first use (or swaps the running one when the model differs)
    and idle-unloads it after ``idle_unload_seconds`` with no requests.
    """
    registry: Registry = request.app.state.registry
    manager: RoleServerManager = request.app.state.audio_managers[role]
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    try:
        model = extract_model(body, content_type)
    except Exception:  # noqa: BLE001
        model = None
    if model is None:
        if expect_body_model:
            status, payload = _error(
                400, "'model' is required in the request body"
            )
            return JSONResponse(status_code=status, content=payload)
        # Single configured model of this role: fall back to it.
        configs = registry.role_configs(role)
        if not configs:
            status, payload = _error(
                404, f"no {role} model configured", "server_error"
            )
            return JSONResponse(status_code=status, content=payload)
        model = configs[0].name
    try:
        cfg = registry.get(model)
    except UnknownModelError as exc:
        status, payload = _error(
            404, f"model '{exc.name}' not found; available: {', '.join(exc.known)}"
        )
        return JSONResponse(status_code=status, content=payload)
    if cfg.role != role:
        status, payload = _error(
            400,
            f"model '{model}' is not a {role} model "
            f"(role: {cfg.role})",
        )
        return JSONResponse(status_code=status, content=payload)
    # Inverse VRAM guard: while a "big" chat LLM (anything other than the
    # smallest by weights-file size) is loaded — or mid-load — refuse audio
    # so TTS/ASR cannot stack on top of a large model. Disable via
    # LLAMASWAP_BLOCK_AUDIO_ON_BIG_LLM=false.
    if request.app.state.settings.block_audio_on_big_llm:
        smallest = registry.smallest_llm()
        chat_status = request.app.state.manager.status()
        loaded = chat_status.get("model")
        state = chat_status.get("state")
        if (smallest is not None and loaded is not None
                and state in ("loading", "ready") and loaded != smallest):
            status, payload = _error(
                409,
                f"audio {role} is blocked while big chat model '{loaded}' "
                f"is loaded; unload it (or wait for idle unload) before "
                f"using {role}",
                "server_error",
            )
            return JSONResponse(status_code=status, content=payload)
    # VRAM guard for image: a TTS/ASR request stops the image server
    # first so audio never stacks on top of a diffusion model (FLUX etc.
    # use most of the GPU). Disable via LLAMASWAP_UNLOAD_IMAGE_ON_AUDIO=false.
    if request.app.state.settings.unload_image_on_audio:
        await request.app.state.image_manager.unload()
    try:
        cfg = await manager.ensure_model(model, registry)
    except RoleServerLoadError as exc:
        status, payload = _error(
            503, f"failed to load {role} model '{model}': {exc}",
            "server_error",
        )
        return JSONResponse(status_code=status, content=payload)
    except UnknownModelError as exc:
        status, payload = _error(
            404, f"model '{exc.name}' not found; available: {', '.join(exc.known)}"
        )
        return JSONResponse(status_code=status, content=payload)

    _apply_pin(request, manager)
    headers = _fwd_headers(request)

    # Backends whose transcription API expects a server-side file path
    # (audio.cpp) instead of the OpenAI multipart upload: translate.
    translated: Optional[str] = None
    out_body = body
    if backend_path == "/v1/audio/transcriptions" \
            and cfg.meta.request_format == "json_path":
        if "multipart/form-data" in content_type.lower():
            saved = save_audio_upload(body, content_type,
                                      request.app.state.settings.audio_tmp_dir)
            if saved is None:
                status, payload = _error(
                    400, "multipart request is missing a 'file' upload"
                )
                return JSONResponse(status_code=status, content=payload)
            translated = saved
            out_body = json.dumps({
                "model": model,
                "file": saved,
            }).encode()
            headers["Content-Type"] = "application/json"
        elif "application/json" in content_type.lower():
            # Client already sent the audio.cpp JSON dialect; pass through.
            pass

    # /v1/audio/translations: OpenAI semantics = transcribe + translate to
    # English. No backend exposes a dedicated translations route; those that
    # can translate accept a flag on their transcription route instead, so
    # rewrite the request to that route (meta.translation_target) with the
    # flag set. Backends without translation support get a clear error.
    if backend_path == "/v1/audio/translations":
        # pydantic extra="allow": unset extras raise AttributeError on
        # attribute access, so go through getattr.
        target = getattr(cfg.meta, "translation_target", None)
        if not target:
            status, payload = _error(
                400,
                f"model '{model}' does not support translation "
                f"(backend '{cfg.meta.family or cfg.name}' has no "
                "translate-capable route)",
            )
            return JSONResponse(status_code=status, content=payload)
        if "multipart/form-data" in content_type.lower():
            out_body = multipart_inject_field(
                body, content_type, "translate", "true"
            )
        elif "application/json" in content_type.lower():
            # JSON-dialect backend that translates: flag it in the body.
            try:
                obj = json.loads(body)
                obj["translate"] = True
                out_body = json.dumps(obj).encode()
            except (ValueError, TypeError):
                pass
        backend_path = target

    # audio.cpp clone-capable TTS backends clone a speaker from a reference
    # WAV + transcript. Those are upstream request fields, not part of the
    # OpenAI /v1/audio/speech surface, so they are injected internally here
    # from the model config — and only for models that declare the `clone`
    # capability (e.g. qwen3-tts, outetts). Preset-voice TTS models
    # (e.g. supertonic-3) never receive them.
    if backend_path in ("/v1/audio/speech", "/v1/audio/speech/stream") \
            and "application/json" in content_type.lower():
        out_body = _apply_clone_reference(cfg, out_body)

    try:
        if backend_path == "/v1/audio/speech/stream":
            # Live chunked audio: stream bytes through without buffering.
            # ``response_format`` transcoding is a buffered operation, so a
            # non-wav format is ignored on the streaming path and the
            # backend's native codec is relayed as-is.
            out_body = json_remove_key(out_body, "response_format")
            gen = proxy_raw_stream(
                cfg.port, cfg.host, backend_path, out_body, headers
            )
            return StreamingResponse(
                gen, media_type="application/octet-stream"
            )
        if backend_path == "/v1/audio/speech":
            # ``response_format`` is an OpenAI-only construct: audio.cpp emits
            # WAV and does not understand it, so honour it locally via ffmpeg
            # instead of forwarding it upstream.
            speech_format = extract_speech_format(out_body)
            out_body = json_remove_key(out_body, "response_format")
            status, data, ctype = await proxy_raw(
                cfg.port, cfg.host, backend_path, out_body, headers
            )
            if status != 200:
                return JSONResponse(
                    status_code=status,
                    content=_maybe_json(data),
                )
            if speech_format and speech_format != "wav":
                transcoded = await transcode_audio(data, speech_format)
                if transcoded is not None:
                    data, ctype = transcoded
            return Response(content=data, media_type=ctype)
        # transcriptions / translations → normalise to the requested format
        status, data = await proxy_json(
            cfg.port, cfg.host, backend_path, out_body, model, headers
        )
        if status != 200:
            return JSONResponse(status_code=status, content=data)
        fmt = extract_response_format(body, content_type)
        language = extract_text_field(body, content_type, "language")
        content, ctype = format_transcription(data, fmt, language=language)
        if ctype == "application/json":
            return JSONResponse(status_code=status, content=content)
        return Response(
            content=content, media_type=ctype, status_code=status
        )
    finally:
        remove_audio_upload(translated)


def _maybe_json(data: bytes):
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        return {"error": {"message": data.decode(errors="replace")[:500]}}


@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    return await _route_audio(request, "tts", "/v1/audio/speech")


@app.post("/v1/audio/speech/stream")
async def audio_speech_stream(request: Request):
    return await _route_audio(request, "tts", "/v1/audio/speech/stream")


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    return await _route_audio(request, "asr", "/v1/audio/transcriptions")


@app.post("/v1/audio/translations")
async def audio_translations(request: Request):
    # OpenAI semantics: transcribe *and* translate to English. Backends
    # don't expose a /v1/audio/translations route directly; models that can
    # translate (meta.translation_target set, e.g. whisper.cpp) get the
    # request rewritten onto their transcription route with the translate
    # flag set; models without translation support get a clear 400.
    return await _route_audio(request, "asr", "/v1/audio/translations")


@app.get("/v1/audio/voices")
async def audio_voices(request: Request):
    """List voices from the TTS backend, booting the first configured TTS
    model on demand so the endpoint is immediately useful."""
    manager: RoleServerManager = request.app.state.audio_managers["tts"]
    cfg = manager.current_config
    if not manager.is_running or cfg is None:
        configs = request.app.state.registry.role_configs("tts")
        if not configs:
            status, payload = _error(
                404, "no tts model configured", "server_error"
            )
            return JSONResponse(status_code=status, content=payload)
        try:
            cfg = await manager.ensure_model(
                configs[0].name, request.app.state.registry
            )
        except (RoleServerLoadError, UnknownModelError) as exc:
            status, payload = _error(
                503, f"failed to load tts server: {exc}", "server_error"
            )
            return JSONResponse(status_code=status, content=payload)
    headers = _fwd_headers(request)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0)
        ) as client:
            resp = await client.get(
                f"http://{cfg.host}:{cfg.port}/v1/audio/voices",
                headers=headers,
            )
    except httpx.HTTPError as exc:
        status, payload = _error(502, f"voices lookup failed: {exc}")
        return JSONResponse(status_code=status, content=payload)
    return Response(content=resp.content, media_type=resp.headers.get(
        "content-type", "application/json"), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# OpenAI image API (text-to-image / image edit)
# ---------------------------------------------------------------------------

async def _route_image(request: Request, path: str):
    """Route an OpenAI image request to the on-demand image server.

    The ``model`` field (JSON body, or multipart form for /v1/images/edits)
    selects which image backend to load; llamaswap starts the server on
    first use (or swaps the running one when the model differs) and
    idle-unloads it like the other on-demand roles. sd-server speaks the
    OpenAI image dialect natively, so this is a near pass-through.
    """
    registry: Registry = request.app.state.registry
    manager: RoleServerManager = request.app.state.image_manager
    settings: Settings = request.app.state.settings
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    try:
        model = extract_model(body, content_type)
    except Exception:  # noqa: BLE001
        model = None
    if model is None:
        configs = registry.role_configs("image")
        if not configs:
            status, payload = _error(
                404, "no image model configured", "server_error"
            )
            return JSONResponse(status_code=status, content=payload)
        model = configs[0].name
    try:
        cfg = registry.get(model)
    except UnknownModelError as exc:
        status, payload = _error(
            404, f"model '{exc.name}' not found; available: {', '.join(exc.known)}"
        )
        return JSONResponse(status_code=status, content=payload)
    if cfg.role != "image":
        status, payload = _error(
            400, f"model '{model}' is not an image model (role: {cfg.role})"
        )
        return JSONResponse(status_code=status, content=payload)

    # VRAM guard: a diffusion model wants the whole GPU, so stop the chat
    # LLM, TTS/ASR, and the embedding server first — FLUX gets the full
    # card. Disable via LLAMASWAP_UNLOAD_ON_IMAGE=false.
    if settings.unload_on_image:
        await asyncio.gather(
            request.app.state.manager.unload(),
            *(mgr.unload() for mgr in request.app.state.audio_managers.values()),
        )
        embedding_manager = request.app.state.embedding_manager
        if embedding_manager is not None and embedding_manager.is_running:
            await embedding_manager.stop_for_resources(reason="the image model")
    try:
        cfg = await manager.ensure_model(model, registry)
    except RoleServerLoadError as exc:
        status, payload = _error(
            503, f"failed to load image model '{model}': {exc}",
            "server_error",
        )
        return JSONResponse(status_code=status, content=payload)
    except UnknownModelError as exc:
        status, payload = _error(
            404, f"model '{exc.name}' not found; available: {', '.join(exc.known)}"
        )
        return JSONResponse(status_code=status, content=payload)

    _apply_pin(request, manager)
    headers = _fwd_headers(request)
    # The backend's schema doesn't carry the OpenAI `model` selector
    # (llamaswap uses it only to pick the backend), so drop it first.
    out_body = body
    if "application/json" in content_type.lower():
        out_body = json_remove_key(body, "model")
    status, data, ctype = await proxy_raw(
        cfg.port, cfg.host, path, out_body, headers
    )
    if status != 200:
        return JSONResponse(status_code=status, content=_maybe_json(data))
    return Response(content=data, media_type=ctype)


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    return await _route_image(request, "/v1/images/generations")


@app.post("/v1/images/edits")
async def images_edits(request: Request):
    return await _route_image(request, "/v1/images/edits")


# ---------------------------------------------------------------------------
# Ollama compatibility surface (/api/tags, /api/ps)
# ---------------------------------------------------------------------------
# llamaswap listens on Ollama's default port, so it also serves the two most
# commonly probed Ollama endpoints. /api/tags lists the registry (one entry
# per backend/*.yaml model, Ollama-shaped); /api/ps lists the backends that
# are actually resident right now (chat LLM, embedding, TTS/ASR, image).

# Go's zero time: what Ollama emits for a model that never expires.
_NEVER_EXPIRES = "0001-01-01T00:00:00Z"

_PARAM_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*[bB]\b")
_QUANT_RE = re.compile(r"(?i)\b(Q[2-8](?:_[A-Z0-9]+)*|F16|F32|BF16)\b")


def _parameter_size(name: str) -> str:
    """Best-effort parameter-size label from a model name (e.g. "27B")."""
    m = _PARAM_RE.search(name)
    return f"{m.group(1)}B" if m else ""


def _quantization_level(name: str) -> str:
    """Quantization token from a model name (Q4_K_M, Q8_0, F16, ...), if any."""
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else ""


def _model_digest(name: str) -> str:
    # Stable synthetic digest. Ollama digests are a real hash over the blob;
    # hashing multi-GB GGUF weights on every list call would be far too slow,
    # so llamaswap derives a deterministic digest from the model name instead.
    return "sha256:" + hashlib.sha256(name.encode()).hexdigest()


def _model_file_info(registry: Registry, cfg: ModelConfig) -> tuple[int, str]:
    """(size_bytes, modified_at_rfc3339) for a model's weights file."""
    size = 0
    modified_at = "1970-01-01T00:00:00Z"
    path = registry.model_path(cfg.name)
    if path:
        try:
            st = Path(path).stat()
            size = st.st_size
            modified_at = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")
        except OSError:
            pass
    return size, modified_at


def _ollama_details(cfg: ModelConfig) -> dict:
    family = cfg.meta.family or cfg.name
    return {
        "parent_model": "",
        "format": "gguf",
        "family": family,
        "families": [family],
        "parameter_size": _parameter_size(cfg.name),
        "quantization_level": _quantization_level(cfg.name),
    }


def _ollama_tag_entry(registry: Registry, cfg: ModelConfig) -> dict:
    """One Ollama /api/tags entry (a registry model)."""
    size, modified_at = _model_file_info(registry, cfg)
    return {
        "name": cfg.name,
        "model": cfg.name,
        "modified_at": modified_at,
        "size": size,
        "digest": _model_digest(cfg.name),
        "details": _ollama_details(cfg),
    }


def _ollama_process_entry(registry: Registry, cfg: ModelConfig,
                          expires_at: str) -> dict:
    """One Ollama /api/ps entry (a currently resident model)."""
    size, _ = _model_file_info(registry, cfg)
    return {
        "name": cfg.name,
        "model": cfg.name,
        "size": size,
        "size_vram": size,
        "digest": _model_digest(cfg.name),
        "details": _ollama_details(cfg),
        "expires_at": expires_at,
    }


def _expires_at(status: dict) -> str:
    """Idle-unload deadline from a manager status dict, Ollama-expiry style.

    Returns Go-zero-time (never expires) when the manager has no idle unload
    configured or is already past its deadline.
    """
    idle_unload = status.get("idle_unload_seconds")
    if idle_unload:
        idle = status.get("idle_seconds", 0.0)
        remaining = float(idle_unload) - float(idle)
        if remaining > 0:
            expiry = datetime.now(timezone.utc) + timedelta(seconds=remaining)
            return expiry.isoformat().replace("+00:00", "Z")
    return _NEVER_EXPIRES


@app.get("/api/tags")
async def api_tags(request: Request):
    """Ollama-style model list: one entry per registry model."""
    registry: Registry = request.app.state.registry
    models = [
        _ollama_tag_entry(registry, cfg)
        for cfg in sorted(registry.models.values(), key=lambda c: c.name)
    ]
    return {"models": models}


@app.get("/api/ps")
async def api_ps(request: Request):
    """Ollama-style running-processes list: every backend resident now."""
    registry: Registry = request.app.state.registry
    models: list[dict] = []

    def add(name: Optional[str], status: dict) -> None:
        if not name:
            return
        cfg = registry.models.get(name)
        if cfg is None:
            return
        models.append(_ollama_process_entry(registry, cfg, _expires_at(status)))

    st = request.app.state.manager.status()
    if st.get("state") == "ready":
        add(st.get("model"), st)

    emb: Optional[EmbeddingManager] = request.app.state.embedding_manager
    if emb is not None and emb.is_running:
        add(emb.name, {})

    for mgr in request.app.state.audio_managers.values():
        st = mgr.status()
        if st.get("state") == "ready":
            add(st.get("model"), st)

    st = request.app.state.image_manager.status()
    if st.get("state") == "ready":
        add(st.get("model"), st)

    return {"models": models}


# ---------------------------------------------------------------------------
# Ops surface: metrics, events, model preload/unload, admin dashboard
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def get_metrics(request: Request):
    """Prometheus text exposition (unauthenticated, scrape-friendly)."""
    registry: Registry = request.app.state.registry
    metrics.set_gauge("llamaswap_registered_models", float(len(registry.models)))
    degraded = bool(registry.degraded) or any(
        m.status().get("state") == "failed"
        for m in list(request.app.state.audio_managers.values())
        + [request.app.state.image_manager, request.app.state.manager]
    )
    metrics.set_gauge("llamaswap_status", 1.0 if degraded else 0.0)
    return Response(
        metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/events")
async def events_stream(request: Request):
    """Server-sent events: model_loaded / model_unloaded / swap / load_failed."""
    async def gen():
        q = bus.subscribe()
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            bus.unsubscribe(q)
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/models/{model}/load")
async def load_model(model: str, request: Request):
    """Preload/warm a model in the background (admin)."""
    registry: Registry = request.app.state.registry
    try:
        cfg = registry.get(model)
    except UnknownModelError as exc:
        status, body = _error(
            404, f"model not found; available: {', '.join(exc.known)}"
        )
        return JSONResponse(status_code=status, content=body)
    try:
        if cfg.role == "llm":
            await request.app.state.manager.ensure_model(model, registry)
        elif cfg.role == "embedding":
            emb: Optional[EmbeddingManager] = request.app.state.embedding_manager
            if emb is None or emb.name != model:
                status, body = _error(
                    400, f"model '{model}' is not the configured embedding server"
                )
                return JSONResponse(status_code=status, content=body)
            if not await emb.ensure_running():
                status, body = _error(
                    503, f"embedding server '{model}' failed to start",
                    "server_error",
                )
                return JSONResponse(status_code=status, content=body)
        elif cfg.role in ("tts", "asr"):
            await request.app.state.audio_managers[cfg.role].ensure_model(
                model, registry
            )
        elif cfg.role == "image":
            await request.app.state.image_manager.ensure_model(model, registry)
    except (ModelLoadError, RoleServerLoadError) as exc:
        status, body = _error(
            503, f"failed to load model '{model}': {exc}", "server_error"
        )
        return JSONResponse(status_code=status, content=body)
    return _health_snapshot(request)


@app.post("/v1/models/{model}/unload")
async def unload_model(model: str, request: Request):
    """Unload the model currently resident for this model's role (admin)."""
    registry: Registry = request.app.state.registry
    try:
        cfg = registry.get(model)
    except UnknownModelError as exc:
        status, body = _error(
            404, f"model not found; available: {', '.join(exc.known)}"
        )
        return JSONResponse(status_code=status, content=body)
    if cfg.role == "llm":
        await request.app.state.manager.unload()
    elif cfg.role == "embedding":
        emb: Optional[EmbeddingManager] = request.app.state.embedding_manager
        if emb is not None and emb.name == model:
            await emb.stop()
        else:
            status, body = _error(
                400, f"model '{model}' is not the configured embedding server"
            )
            return JSONResponse(status_code=status, content=body)
    elif cfg.role in ("tts", "asr"):
        await request.app.state.audio_managers[cfg.role].unload()
    elif cfg.role == "image":
        await request.app.state.image_manager.unload()
    return _health_snapshot(request)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llamaswap</title>
<style>
:root { color-scheme: dark; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       background: #101418; color: #e6edf3; margin: 0; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; }
.card h2 { font-size: 14px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .06em; color: #8b949e; }
.row { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; }
.badge { padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.ready { background: #11351f; color: #3fb950; }
.loading { background: #2e2a0f; color: #d29922; }
.stopped { background: #21262d; color: #8b949e; }
.failed { background: #4d1d24; color: #f85149; }
input { background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
        border-radius: 6px; padding: 6px 8px; font-size: 13px; margin-bottom: 12px; width: 260px; }
button { background: #238636; border: 0; color: #fff; border-radius: 6px; padding: 6px 10px;
         font-size: 12px; cursor: pointer; margin-right: 6px; }
button.off { background: #21262d; color: #c9d1d9; }
a { color: #79c0ff; }
#log { font: 12px/1.5 ui-monospace, monospace; color: #8b949e; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>llamaswap</h1>
<div class="sub">OpenAI-compatible model proxy · <a href="/health">/health</a> · <a href="/metrics">/metrics</a> · <a href="/docs">/docs</a> · <a href="/v1/events">/v1/events</a></div>
<input id="key" type="password" placeholder="admin/api key (if enabled)" />
<div class="grid" id="grid"></div>
<pre id="log"></pre>
<script>
const $ = (id) => document.getElementById(id);
function badge(state) { return '<span class="badge ' + (state||'stopped') + '">' + (state||'stopped') + '</span>'; }
async function refresh() {
  const key = $('key').value;
  const hdr = key ? { 'X-Api-Key': key, 'X-Admin-Key': key } : {};
  let health;
  try { health = await (await fetch('/health')).json(); } catch (e) { $('grid').innerHTML = 'unreachable'; return; }
  const snap = {
    'chat LLM': health.current_model,
    'embedding': health.embedding,
    'TTS': health.audio && health.audio.tts,
    'ASR': health.audio && health.audio.asr,
    'image': health.image,
  };
  let html = '';
  for (const [label, st] of Object.entries(snap)) {
    if (!st) continue;
    const state = st.state || 'stopped';
    html += '<div class="card"><h2>' + label + '</h2>' +
      '<div class="row"><span>state</span>' + badge(state) + '</div>' +
      '<div class="row"><span>model</span><span>' + (st.model || '—') + '</span></div>' +
      (st.port ? '<div class="row"><span>port</span><span>' + st.port + '</span></div>' : '') +
      (st.idle_seconds != null ? '<div class="row"><span>idle</span><span>' + st.idle_seconds + 's</span></div>' : '') +
      (st.detail ? '<div class="row"><span>detail</span><span>' + st.detail + '</span></div>' : '') +
      '</div>';
  }
  $('grid').innerHTML = html;
  try {
    const models = await (await fetch('/v1/models', { headers: hdr })).json();
    let rows = '<div class="card"><h2>models</h2>';
    for (const m of (models.data || [])) {
      rows += '<div class="row"><span>' + m.id + ' <i style="color:#6e7681">(' + m.role + ')</i></span>' +
        '<span>' + badge(m.state) + ' ' +
        '<button onclick="act(\'' + m.id + '\', \'load\')">load</button>' +
        '<button class="off" onclick="act(\'' + m.id + '\', \'unload\')">unload</button></span></div>';
    }
    rows += '</div>';
    $('grid').insertAdjacentHTML('beforeend', rows);
  } catch (e) { $('log').textContent = 'set the API/admin key above to manage models.'; }
}
async function act(model, op) {
  const key = $('key').value;
  const hdr = key ? { 'X-Api-Key': key, 'X-Admin-Key': key } : {};
  try {
    await fetch('/v1/models/' + model + '/' + op, { method: 'POST', headers: hdr });
  } catch (e) {}
  refresh();
}
refresh(); setInterval(refresh, 3000);
</script>
</body>
</html>"""


@app.get("/")
async def dashboard(request: Request):
    """Tiny admin dashboard (HTML)."""
    return Response(_DASHBOARD_HTML, media_type="text/html")


@app.exception_handler(UnknownModelError)
async def unknown_model_handler(request: Request, exc: UnknownModelError):
    status, payload = _error(
        404, f"model '{exc.name}' not found; available: {', '.join(exc.known)}"
    )
    return JSONResponse(status_code=status, content=payload)


@app.exception_handler(ModelLoadError)
async def model_load_handler(request: Request, exc: ModelLoadError):
    status, payload = _error(503, str(exc), "server_error")
    return JSONResponse(status_code=status, content=payload)


@app.exception_handler(RoleServerLoadError)
async def role_server_load_handler(request: Request, exc: RoleServerLoadError):
    status, payload = _error(503, str(exc), "server_error")
    return JSONResponse(status_code=status, content=payload)


# ---------------------------------------------------------------------------
# Gateway middleware (auth + backpressure + request metrics)
# ---------------------------------------------------------------------------

class _GatewayMiddleware:
    """Pure-ASGI wrapper (not Starlette BaseHTTPMiddleware, so SSE and
    streaming responses pass through untouched) that enforces optional API/
    admin key auth, caps concurrent in-flight requests, and counts requests
    for /metrics.
    """

    def __init__(self, app) -> None:
        self.app = app
        self._inflight = 0

    @staticmethod
    def _headers(scope: dict) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in scope.get("headers") or []:
            out.setdefault(k.decode("latin-1").lower(), v.decode("latin-1"))
        return out

    @staticmethod
    def _is_admin(path: str) -> bool:
        return (
            path in ("/v1/reset", "/v1/registry/reload")
            or (path.startswith("/v1/models/")
                and (path.endswith("/load") or path.endswith("/unload")))
        )

    @staticmethod
    def _provided_key(headers: dict[str, str]) -> str:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return (
            headers.get("x-api-key")
            or headers.get("x-admin-key")
            or ""
        ).strip()

    def _authorized(self, settings, path: str, headers: dict[str, str]) -> bool:
        if self._is_admin(path):
            required = settings.admin_key or settings.api_key
        else:
            required = settings.api_key
        if not required:
            return True
        provided = self._provided_key(headers)
        return bool(provided) and hmac.compare_digest(provided, required)

    async def _json(self, send, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = get_settings()
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        headers = self._headers(scope)
        status_holder = {"status": 200}

        # Auth (health + metrics stay open for scrapers/probes).
        if path not in ("/health", "/metrics") \
                and not self._authorized(settings, path, headers):
            status_holder["status"] = 401
            _, body = _error(
                401, "unauthorized: missing or invalid API key",
                "authentication_error",
            )
            await self._json(send, 401, body)
            metrics.inc_request(method, path, 401)
            return

        # Backpressure: reject fast when over the concurrency cap.
        limited = False
        if settings.max_concurrency > 0:
            if self._inflight >= settings.max_concurrency:
                limited = True
            else:
                self._inflight += 1
        if limited:
            status_holder["status"] = 429
            _, body = _error(
                429,
                "too many concurrent requests; retry shortly",
                "rate_limit_error",
            )
            await self._json(send, 429, body)
            metrics.inc_request(method, path, 429)
            return

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message.get("status", 200)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if settings.max_concurrency > 0 and not limited:
                self._inflight -= 1
            metrics.inc_request(method, path, status_holder["status"])


# Wrap the ASGI app with the gateway middleware. (Everything above was
# registered on the inner FastAPI app; this wrapper sits in front of it.)
app = _GatewayMiddleware(app)
