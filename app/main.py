"""llamaswap — an OpenAI-compatible proxy in front of llama-server.

Model registry lives in backend/*.yaml; the requested model is launched
(or swapped in) transparently on demand. Only the embedding server is
persistent; everything else is on-demand and idle-unloaded like the chat
LLM:

  * ``embedding`` — dedicated embedding llama-server (EmbeddingManager),
    started at boot and kept running (only stopped to free VRAM for an LLM)
  * ``chat`` — one LLM at a time (ProcessManager), swapped on request and
    unloaded after ``idle_unload_seconds`` with no requests
  * ``tts`` / ``asr`` — audio servers (AudioManager) that boot on first
    use, swap between backends per request (e.g. qwen3-asr vs
    whisper-server) and idle-unload like the chat LLM; proxied through
    the OpenAI audio endpoints (/v1/audio/speech,
    /v1/audio/transcriptions, ...).

While tts AND asr are both loaded, a VRAM guard substitutes the smallest
chat LLM for any requested chat model (see Settings.audio_vram_guard).
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .audio_manager import AudioLoadError, AudioManager
from .config import Settings, get_settings
from .embedding_manager import EmbeddingLoadError, EmbeddingManager
from .process_manager import ModelLoadError, ProcessManager
from .proxy import (
    extract_model,
    extract_stream_flag,
    multipart_inject_field,
    proxy_json,
    proxy_raw,
    proxy_stream,
    remove_audio_upload,
    save_audio_upload,
)
from .registry import Registry, RegistryError, UnknownModelError

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("llamaswap")


def _error(status: int, message: str, etype: str = "invalid_request_error"):
    return status, {"error": {"message": message, "type": etype}}


def _fwd_headers(request: Request) -> dict[str, str]:
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "accept-encoding")
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
    # (see _route_audio and AudioManager.ensure_model).
    app.state.audio_managers: dict[str, AudioManager] = {}
    for role in ("tts", "asr"):
        manager = AudioManager(
            role,
            startup_timeout=settings.startup_timeout,
            stop_timeout=settings.stop_timeout,
            health_interval=settings.health_interval,
            idle_unload_seconds=settings.idle_unload_seconds,
            config_dir=settings.audio_tmp_dir,
        )
        manager.configure(app.state.registry.role_configs(role))
        app.state.audio_managers[role] = manager

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
        if app.state.embedding_manager is not None:
            await app.state.embedding_manager.shutdown()
        await app.state.manager.shutdown()


app = FastAPI(title="llamaswap", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    embedding_manager: Optional[EmbeddingManager] = (
        request.app.state.embedding_manager
    )
    return {
        "status": "ok",
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
    }


@app.get("/v1/models")
async def list_models(request: Request):
    registry: Registry = request.app.state.registry
    return {"object": "list", "data": registry.list_openai()}


@app.get("/v1/models/{model}")
async def get_model(model: str, request: Request):
    registry: Registry = request.app.state.registry
    try:
        data = registry.to_openai(model)
    except UnknownModelError:
        status, body = _error(404, f"model '{model}' not found")
        return JSONResponse(status_code=status, content=body)
    return data


@app.post("/v1/registry/reload")
async def reload_registry(request: Request):
    registry: Registry = request.app.state.registry
    try:
        registry.reload()
    except RegistryError as exc:
        status, body = _error(500, str(exc), "server_error")
        return JSONResponse(status_code=status, content=body)
    # Refresh per-role audio configs only — servers boot on first use and
    # are idle-unloaded, so never pre-start them here.
    for role, manager in request.app.state.audio_managers.items():
        manager.configure(registry.role_configs(role))
    return {"reloaded": True, "models": request.app.state.registry.names()}


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
    # VRAM guard: while BOTH audio roles are loaded, only the smallest
    # chat LLM may be served (TTS + ASR + a big LLM may not fit on one
    # GPU). The requested model is transparently substituted and the swap
    # happens as usual; disable via LLAMASWAP_AUDIO_VRAM_GUARD=false.
    settings: Settings = request.app.state.settings
    if settings.audio_vram_guard:
        tts_mgr = request.app.state.audio_managers["tts"]
        asr_mgr = request.app.state.audio_managers["asr"]
        if tts_mgr.is_running and asr_mgr.is_running:
            smallest = registry.smallest_llm()
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
    manager: AudioManager = request.app.state.audio_managers[role]
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
    try:
        cfg = await manager.ensure_model(model, registry)
    except AudioLoadError as exc:
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

    try:
        if backend_path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
            status, data, ctype = await proxy_raw(
                cfg.port, cfg.host, backend_path, out_body, headers
            )
            if status != 200:
                return JSONResponse(
                    status_code=status,
                    content=_maybe_json(data),
                )
            return Response(content=data, media_type=ctype)
        # transcriptions / translations → JSON
        status, data = await proxy_json(
            cfg.port, cfg.host, backend_path, out_body, model, headers
        )
        return JSONResponse(status_code=status, content=data)
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
    """List voices from the TTS backend, if it exposes such an endpoint."""
    manager: AudioManager = request.app.state.audio_managers["tts"]
    if not manager.is_running:
        status, payload = _error(
            503, "tts server is not running", "server_error"
        )
        return JSONResponse(status_code=status, content=payload)
    cfg = manager.current_config
    if cfg is None:
        status, payload = _error(503, "tts server is not ready", "server_error")
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


@app.exception_handler(AudioLoadError)
async def audio_load_handler(request: Request, exc: AudioLoadError):
    status, payload = _error(503, str(exc), "server_error")
    return JSONResponse(status_code=status, content=payload)
