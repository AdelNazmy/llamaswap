"""Async reverse proxy from the OpenAI API surface to llama-server.

llama-server already speaks the OpenAI protocol, so this is a thin
pass-through that:
  * rewrites the response `model` field to the registry name,
  * relays SSE streams chunk-by-chunk,
  * translates upstream failures into OpenAI-style error envelopes.
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger("custom_ollama.proxy")

UPSTREAM_TIMEOUT = 600.0


class UpstreamError(Exception):
    """Upstream returned a non-2xx response."""

    def __init__(self, status: int, payload: Any, raw: bytes):
        self.status = status
        self.payload = payload
        self.raw = raw
        super().__init__(f"upstream returned {status}")


def _model_field(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"]
        for key in ("choices", "data"):
            val = obj.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val[0].get("model")
    return None


async def proxy_stream(
    port: int, host: str, path: str, body: Any,
    request_model: str, headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Stream the response body from llama-server, rewriting the model field."""
    url = f"http://{host}:{port}{path}"
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=10.0)
        ) as client:
            async with client.stream(
                "POST", url, content=body, headers=fwd_headers
            ) as resp:
                if resp.status_code != 200:
                    raw = await resp.aread()
                    yield _error_bytes(resp.status_code, raw)
                    return
                async for chunk in resp.aiter_raw():
                    if chunk:
                        yield _rewrite_chunk(chunk, request_model)
    except httpx.ConnectError:
        yield _error_bytes(503, b"backend llama-server is not reachable")
    except httpx.HTTPError as exc:
        yield _error_bytes(502, str(exc).encode())


def _rewrite_chunk(chunk: bytes, request_model: str) -> bytes:
    parts = chunk.split(b"\n")
    out = bytearray()
    for idx, raw_line in enumerate(parts):
        line = raw_line
        if line.startswith(b"data: "):
            data = line[6:].strip()
            if data and data != b"[DONE]":
                try:
                    obj = json.loads(data)
                    m = _model_field(obj)
                    if m is not None and m != request_model:
                        obj["model"] = request_model
                        if isinstance(obj.get("choices"), list):
                            for c in obj["choices"]:
                                if isinstance(c, dict) and "model" not in c:
                                    c["model"] = request_model
                        data = json.dumps(obj, separators=(",", ":"),
                                          ensure_ascii=False).encode()
                    line = b"data: " + data
                except (ValueError, TypeError):
                    pass
        out += line
        if idx < len(parts) - 1:
            out += b"\n"
    return bytes(out)


def _error_bytes(status: int, body: bytes) -> bytes:
    logger.warning("upstream error %s: %s", status, body[:300])
    try:
        err = json.loads(body)
    except (ValueError, TypeError):
        err = {"error": {"message": body.decode(errors="replace")[:500]}}
    return f"event: error\ndata: {json.dumps({'error': err}, default=str)}\n\n".encode()


async def proxy_json(
    port: int, host: str, path: str, body: Any,
    request_model: str, headers: dict[str, str],
) -> tuple[int, Any]:
    """Non-streaming proxy; returns (status_code, parsed_json_or_error_body)."""
    url = f"http://{host}:{port}{path}"
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=10.0)
        ) as client:
            resp = await client.post(url, content=body, headers=fwd_headers)
    except httpx.ConnectError:
        return 503, {"error": {"message": "backend llama-server is not reachable"}}
    except httpx.HTTPError as exc:
        return 502, {"error": {"message": str(exc)}}
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            return 200, resp.text
        m = _model_field(data)
        if isinstance(data, dict) and m is not None and m != request_model:
            data["model"] = request_model
        return 200, data
    return resp.status_code, resp.json() if _is_json(resp.content) else resp.text


def _is_json(content: bytes) -> bool:
    head = content.lstrip()[:1]
    return head in (b"{", b"[")


def extract_stream_flag(body: bytes) -> bool:
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return False
    return bool(obj.get("stream"))


def extract_model(body: bytes) -> Optional[str]:
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None
    return obj.get("model")
