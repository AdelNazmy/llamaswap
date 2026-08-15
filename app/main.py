"""llamaswap — an OpenAI-compatible proxy in front of llama-server.

Model registry lives in backend/*.yaml; the requested model is launched
(or swapped in) transparently on demand.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .embedding_manager import EmbeddingLoadError, EmbeddingManager
from .process_manager import ModelLoadError, ProcessManager
from .proxy import (
    extract_model,
    extract_stream_flag,
    proxy_json,
    proxy_stream,
)
from .registry import Registry, RegistryError, UnknownModelError

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("llamaswap")


def _error(status: int, message: str, etype: str = "invalid_request_error"):
    return status, {"error": {"message": message, "type": etype}}


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
    app.state.registry = Registry(backend_dir)
    app.state.manager = ProcessManager(
        startup_timeout=settings.startup_timeout,
        stop_timeout=settings.stop_timeout,
        health_interval=settings.health_interval,
    )
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
    try:
        request.app.state.registry.reload()
    except RegistryError as exc:
        status, body = _error(500, str(exc), "server_error")
        return JSONResponse(status_code=status, content=body)
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
    if cfg.role == "embedding":
        status, payload = _error(
            400,
            f"model '{model}' is an embedding model; "
            "use /v1/embeddings instead",
        )
        return JSONResponse(status_code=status, content=payload)
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
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "accept-encoding")
    }
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
        try:
            model = extract_model(body)
        except Exception:  # noqa: BLE001
            model = None
        if model is None:
            model = embedding_manager.name
        cfg = embedding_manager.config
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "accept-encoding")
        }
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
