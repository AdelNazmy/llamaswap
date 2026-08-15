"""custom_ollama — an OpenAI-compatible proxy in front of llama-server.

Model registry lives in backend/*.yaml; the requested model is launched
(or swapped in) transparently on demand.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .config import get_settings
from .process_manager import ModelLoadError, ProcessManager
from .proxy import (
    extract_model,
    extract_stream_flag,
    proxy_json,
    proxy_stream,
)
from .registry import Registry, RegistryError, UnknownModelError

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("custom_ollama")


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
    logger.info(
        "custom_ollama ready on %s:%d (%d models)",
        settings.host, settings.port, len(app.state.registry.models),
    )
    try:
        yield
    finally:
        await app.state.manager.shutdown()


app = FastAPI(title="custom_ollama", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "current_model": request.app.state.manager.status(),
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
) -> Response:
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
        _, port = await manager.ensure_model(model, registry)
    except UnknownModelError as exc:
        known = ", ".join(exc.known)
        status, payload = _error(
            404, f"model '{exc.name}' not found; available: {known}"
        )
        return JSONResponse(status_code=status, content=payload)
    except ModelLoadError as exc:
        logger.error("model load failed: %s", exc)
        status, payload = _error(
            503, f"failed to load model '{model}': {exc}", "server_error"
        )
        return JSONResponse(status_code=status, content=payload)

    cfg = registry.get(model)
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
