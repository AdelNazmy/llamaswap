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
import os
import re
import shutil
from typing import (
    Any,
    AsyncIterator,
    Optional,
    Tuple,
)
from uuid import uuid4

import httpx

from .config import get_settings
from .metrics import metrics

logger = logging.getLogger("llamaswap.proxy")

UPSTREAM_TIMEOUT = 600.0

# boundary=... inside a multipart/form-data Content-Type header
_MULTIPART_RE = re.compile(r'boundary="?([^";]+)"?', re.IGNORECASE)


def _model_field(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"]
        for key in ("choices", "data"):
            val = obj.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val[0].get("model")
    return None


# --------------------------------------------------------------------------
# Context-window overflow trim
# --------------------------------------------------------------------------
# Upstream chat backends (llama.cpp / strata) refuse any prompt that leaves
# no room for an answer: "request exceeds the context window: prompt (N
# tokens) leaves no room to answer in the context (M); requests are never
# truncated". A long-running agent conversation that grows past n_ctx then
# fails on every turn forever. When the error names both counts, llamaswap
# drops the oldest droppable message turns and retries upstream, so stale
# history ages out instead of dead-ending the client. The upstream's own
# token counts (authoritative) decide HOW MUCH to drop; the chars/4 estimate
# below only decides WHICH (oldest) turns. Disable via
# LLAMASWAP_CONTEXT_TRIM=false.

OVERFLOW_RE = re.compile(
    r"prompt \((\d+) tokens\).*?context(?: window)? \((\d+)\)", re.IGNORECASE,
)

_TRIM_ATTEMPTS = 4  # upstream calls per request at most: 1 + 3 trim retries


def overflow_info(err_text: Any) -> Optional[Tuple[int, int]]:
    """Parse an upstream overflow error into (prompt_tokens, ctx_tokens).

    Accepts a decoded string or an already-parsed error JSON body.
    """
    if isinstance(err_text, (bytes, bytearray)):
        err_text = err_text.decode(errors="replace")
    elif not isinstance(err_text, str):
        err_text = json.dumps(err_text, default=str)
    m = OVERFLOW_RE.search(err_text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def estimate_tokens(value: Any) -> int:
    """Cheap chars/4 estimate of the token cost of a JSON payload piece.

    Only used to pick WHICH turns to drop; the upstream token counts in the
    overflow error decide HOW MUCH must go, so an underestimate just means
    another (bounded) retry round.
    """
    if isinstance(value, str):
        return len(value) // 4 + 1
    if isinstance(value, dict):
        return sum(
            estimate_tokens(k) + estimate_tokens(v) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(estimate_tokens(v) for v in value)
    return 1


def _message_tokens(msg: Any) -> int:
    if not isinstance(msg, dict):
        return estimate_tokens(msg)
    total = 8  # role/separator/chat-template overhead per message
    for key in ("content", "name", "tool_calls", "function_call"):
        if msg.get(key) is not None:
            total += estimate_tokens(msg[key])
    return total


def _turn_groups(messages: list) -> list:
    """Group messages into indivisible removal units.

    An assistant message carrying tool_calls is grouped together with the
    consecutive 'tool' messages answering it, so a dropped turn never leaves
    orphaned tool results (upstream rejects those).
    """
    groups = []
    i, n = 0, len(messages)
    while i < n:
        group = [i]
        msg = messages[i]
        j = i + 1
        if (isinstance(msg, dict) and msg.get("tool_calls")
                and msg.get("role") != "tool"):
            while j < n and isinstance(messages[j], dict) \
                    and messages[j].get("role") == "tool":
                group.append(j)
                j += 1
        groups.append(group)
        i = j
    return groups


def trim_messages(
    body_obj: dict, prompt_tokens: int, ctx_tokens: int, margin: int,
) -> Optional[Tuple[dict, int]]:
    """Drop oldest droppable turns until the prompt fits ctx_tokens - margin.

    Keeps the leading system block, the last message, and the newest turns
    that fit; drops whole tool-call turns atomically. Inserts one short
    system marker noting the drop. Returns None when nothing is droppable,
    in which case the caller must surface the original error unchanged.
    """
    messages = body_obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    # Leading system block (agents pin it) is never dropped.
    keep_head = 0
    while (keep_head < len(messages) - 1
           and isinstance(messages[keep_head], dict)
           and messages[keep_head].get("role") == "system"):
        keep_head += 1
    # The last message (the live request) is never dropped either.
    tail_start = len(messages) - 1

    droppable = [
        g for g in _turn_groups(messages)
        if g[0] >= keep_head and g[-1] < tail_start
    ]
    if not droppable:
        return None

    marker = "[llamaswap: dropped {} earlier messages to fit the context window]"
    marker_tokens = 64  # placeholder count for the marker line itself
    budget = ctx_tokens - margin - marker_tokens
    # How much upstream says must go; fall back to our own estimate when the
    # estimate exceeds the budget but upstream's number does not (estimates
    # can also overcount for non-token-like payloads).
    needed = prompt_tokens - (ctx_tokens - margin)
    total_est = sum(_message_tokens(m) for m in messages)
    needed = max(needed, total_est - (ctx_tokens - margin))
    if needed <= 0:
        return None

    dropped = 0
    keep_from = len(messages)  # everything below keep_from is kept
    for group in droppable:
        if dropped >= needed:
            break
        dropped += sum(_message_tokens(messages[idx]) for idx in group)
        keep_from = group[-1] + 1
    n_dropped = keep_from - keep_head
    if n_dropped <= 0:
        return None
    body = dict(body_obj)
    body["messages"] = (
        messages[:keep_head]
        + [{"role": "system",
            "content": marker.format(n_dropped)}]
        + messages[keep_from:]
    )
    return body, n_dropped


def _trimmed_body(
    body: Any, err_text: Any, request_model: str,
) -> Optional[bytes]:
    """Trim a chat request body after an upstream overflow error.

    Returns the re-serialized (trimmed) body bytes, or None when trimming
    is disabled, the error is not an overflow, the body carries no chat
    ``messages`` array, or nothing can be dropped.
    """
    settings = get_settings()
    if not settings.context_trim:
        return None
    info = overflow_info(err_text)
    if info is None:
        return None
    prompt_tokens, ctx_tokens = info
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    margin = max(
        256,
        int(obj.get("max_tokens") or obj.get("max_completion_tokens")
            or settings.context_trim_margin),
    )
    trimmed = trim_messages(obj, prompt_tokens, ctx_tokens, margin)
    if trimmed is None:
        return None
    body_new, n_dropped = trimmed
    logger.warning(
        "upstream context overflow: prompt=%d ctx=%d dropped=%d messages, retrying",
        prompt_tokens, ctx_tokens, n_dropped,
    )
    metrics.inc_context_trim(request_model, "trimmed")
    return json.dumps(body_new, separators=(",", ":"),
                      ensure_ascii=False).encode()


async def proxy_stream(
    port: int, host: str, path: str, body: Any,
    request_model: str, headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Stream the response body from llama-server, rewriting the model field.

    A non-200 is buffered (``aread``) before any live chunk is yielded, so a
    context-window overflow can be trimmed and the upstream call retried here
    without the client ever seeing the failed attempt.
    """
    url = f"http://{host}:{port}{path}"
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=10.0)
        ) as client:
            for attempt in range(_TRIM_ATTEMPTS):
                async with client.stream(
                    "POST", url, content=body, headers=fwd_headers
                ) as resp:
                    if resp.status_code != 200:
                        raw = await resp.aread()
                        if attempt < _TRIM_ATTEMPTS - 1:
                            trimmed = _trimmed_body(body, raw, request_model)
                            if trimmed is not None:
                                body = trimmed
                                continue
                        yield _error_bytes(resp.status_code, raw)
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield _rewrite_chunk(chunk, request_model)
                return
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
                        line = b"data: " + json.dumps(
                            obj, separators=(",", ":"), ensure_ascii=False
                        ).encode()
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
            for attempt in range(_TRIM_ATTEMPTS):
                resp = await client.post(url, content=body, headers=fwd_headers)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        return 200, resp.text
                    m = _model_field(data)
                    if isinstance(data, dict) and m is not None \
                            and m != request_model:
                        data["model"] = request_model
                    return 200, data
                if attempt < _TRIM_ATTEMPTS - 1:
                    trimmed = _trimmed_body(body, resp.text, request_model)
                    if trimmed is not None:
                        body = trimmed
                        continue
                return resp.status_code, (
                    resp.json() if _is_json(resp.content) else resp.text
                )
    except httpx.ConnectError:
        return 503, {"error": {"message": "backend llama-server is not reachable"}}
    except httpx.HTTPError as exc:
        return 502, {"error": {"message": str(exc)}}


async def proxy_raw(
    port: int, host: str, path: str, body: Any,
    headers: dict[str, str],
) -> tuple[int, bytes, str]:
    """Non-streaming binary pass-through (e.g. audio/wav TTS output).

    Returns (status_code, body_bytes, content_type).
    """
    url = f"http://{host}:{port}{path}"
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=10.0)
        ) as client:
            resp = await client.post(url, content=body, headers=fwd_headers)
    except httpx.ConnectError:
        return 503, json.dumps({
            "error": {"message": "backend audio server is not reachable"}
        }).encode(), "application/json"
    except httpx.HTTPError as exc:
        return 502, json.dumps({
            "error": {"message": str(exc)}
        }).encode(), "application/json"
    return (
        resp.status_code,
        resp.content,
        resp.headers.get("content-type", "application/octet-stream"),
    )


async def proxy_raw_stream(
    port: int, host: str, path: str, body: Any,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Streaming binary pass-through (e.g. chunked TTS audio output).

    Unlike ``proxy_raw`` (which buffers the whole body), this yields raw
    bytes as they arrive so a backend exposing a chunked speech endpoint
    can be relayed live. No SSE rewriting — the payload is opaque audio.
    """
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
                    yield json.dumps({
                        "error": {
                            "message": raw.decode(errors="replace")[:500]
                            if not _is_json(raw) else json.loads(raw),
                        }
                    }).encode()
                    return
                async for chunk in resp.aiter_raw():
                    if chunk:
                        yield chunk
    except httpx.ConnectError:
        yield json.dumps({
            "error": {"message": "backend audio server is not reachable"}
        }).encode()
    except httpx.HTTPError as exc:
        yield json.dumps({"error": {"message": str(exc)}}).encode()


async def proxy_get_json(
    port: int, host: str, path: str, headers: dict[str, str],
) -> tuple[int, Any]:
    """GET forward for llama-server's native read-only endpoints
    (/props, /slots, /lora-adapters); returns (status, parsed_json_or_text).
    """
    url = f"http://{host}:{port}{path}"
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=10.0)
        ) as client:
            resp = await client.get(url, headers=fwd_headers)
    except httpx.ConnectError:
        return 503, {"error": {"message": "backend llama-server is not reachable"}}
    except httpx.HTTPError as exc:
        return 502, {"error": {"message": str(exc)}}
    if _is_json(resp.content):
        try:
            return resp.status_code, resp.json()
        except ValueError:
            pass
    return resp.status_code, resp.text


def _is_json(content: bytes) -> bool:
    head = content.lstrip()[:1]
    return head in (b"{", b"[")


def extract_stream_flag(body: bytes) -> bool:
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    return bool(obj.get("stream"))


def json_inject_field(body: bytes, key: str, value: Any) -> bytes:
    """Inject ``key=value`` into a JSON request body (no-op if already set).

    Returns the re-serialized body, or the original bytes unchanged when the
    body is not a JSON object or already carries ``key``.
    """
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return body
    if not isinstance(obj, dict) or key in obj:
        return body
    obj[key] = value
    return json.dumps(obj).encode()


def extract_model(body: bytes, content_type: str = "") -> Optional[str]:
    """Model id from a JSON body or a multipart/form-data upload."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        pass
    else:
        return obj.get("model") if isinstance(obj, dict) else None
    if "multipart/form-data" in content_type.lower():
        return multipart_field(body, content_type, "model")
    return None


def _multipart_parts(body: bytes, content_type: str) -> list[tuple[bytes, bytes]]:
    """Split a multipart body into (headers, payload) per part."""
    m = _MULTIPART_RE.search(content_type or "")
    if not m:
        return []
    boundary = f"--{m.group(1)}".encode()
    parts: list[tuple[bytes, bytes]] = []
    for raw in body.split(boundary):
        head, sep, payload = raw.partition(b"\r\n\r\n")
        if not sep:
            continue
        head = head.strip(b"\r\n -")
        if not head:
            continue
        parts.append((head, payload))
    return parts


def multipart_field(body: bytes, content_type: str, name: str) -> Optional[str]:
    """Value of a named text field in a multipart/form-data body."""
    for head, payload in _multipart_parts(body, content_type):
        if f'name="{name}"'.encode() in head:
            # Form field values are terminated by CRLF (or boundary): take
            # the first line and strip any trailing CR/LF.
            value = payload.strip(b"\r\n").split(b"\r\n", 1)[0].strip()
            if value:
                return value.decode(errors="replace")
    return None


def multipart_inject_field(body: bytes, content_type: str,
                           name: str, value: str) -> bytes:
    """Return ``body`` with a text form field ("name=value") appended.

    The field becomes a new multipart part just before the closing
    boundary; if a part with the same name already exists (or the body is
    not a parseable multipart payload), the body is returned unchanged.
    """
    m = _MULTIPART_RE.search(content_type or "")
    if not m:
        return body
    if multipart_field(body, content_type, name) is not None:
        return body
    boundary = f"--{m.group(1)}".encode()
    closing = boundary + b"--"
    if closing not in body:
        return body
    part = (
        boundary + b"\r\n"
        + f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        + value.encode() + b"\r\n" + closing
    )
    return body.replace(closing, part)


def multipart_file(body: bytes, content_type: str) -> Optional[tuple[str, bytes]]:
    """Filename and content of the uploaded 'file' part, or None."""
    for head, payload in _multipart_parts(body, content_type):
        if b'name="file"' not in head:
            continue
        fn = "upload"
        m = re.search(r'filename="([^"]*)"', head.decode(errors="replace"))
        if m:
            fn = m.group(1) or "upload"
        # Strip the trailing CRLF that precedes the closing boundary.
        payload = payload.rstrip(b"\r\n")
        return fn, payload
    return None


def save_audio_upload(body: bytes, content_type: str,
                      tmp_dir: str) -> Optional[str]:
    """Save the uploaded audio to ``tmp_dir`` and return its absolute path.

    Used for backends whose transcription API takes a server-side file path
    (audio.cpp ``/v1/audio/transcriptions``) instead of the OpenAI
    multipart upload. Returns None if there is no file part.
    """
    part = multipart_file(body, content_type)
    if part is None:
        return None
    filename, content = part
    suffix = os.path.splitext(filename)[1] or ".wav"
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"{uuid4().hex}{suffix}")
    try:
        with open(path, "wb") as fh:
            fh.write(content)
    except OSError as exc:
        logger.warning("failed to save audio upload: %s", exc)
        return None
    return path


def remove_audio_upload(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def extract_text_field(body: bytes, content_type: str, name: str) -> Optional[str]:
    """Value of a request field from a JSON body or a multipart upload."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        pass
    else:
        if isinstance(obj, dict):
            val = obj.get(name)
            if val is not None:
                return str(val)
    if "multipart/form-data" in content_type.lower():
        return multipart_field(body, content_type, name)
    return None


def extract_response_format(body: bytes, content_type: str = "") -> str:
    """Transcription ``response_format`` from JSON/multipart (default "json").

    Accepts the legacy ``format`` alias too.
    """
    return (
        extract_text_field(body, content_type, "response_format")
        or extract_text_field(body, content_type, "format")
        or "json"
    ).strip().lower()


def extract_speech_format(body: bytes) -> Optional[str]:
    """Speech ``response_format`` from a JSON body (None if unspecified)."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict):
        val = obj.get("response_format")
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def json_remove_key(body: bytes, key: str) -> bytes:
    """Return ``body`` with ``key`` removed from a JSON object (no-op otherwise)."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return body
    if isinstance(obj, dict) and key in obj:
        obj.pop(key)
        return json.dumps(obj).encode()
    return body


# --------------------------------------------------------------------------
# ASR response-format normalisation
# --------------------------------------------------------------------------
# OpenAI's /v1/audio/transcriptions accepts ``response_format`` =
# json | text | srt | verbose_json | vtt. whisper-server renders those
# natively; audio.cpp backends only ever return ``{"text": ...}``. The proxy
# normalises every ASR backend to the requested format so clients always get
# plain-OpenAI behaviour regardless of which backend is loaded.

_SUBTITLE_CT = {"srt": "application/x-subrip", "vtt": "text/vtt"}


def format_transcription(
    data: Any, fmt: str, language: Optional[str] = None,
) -> tuple[Any, str]:
    """Normalise an ASR backend result into the OpenAI-requested format.

    ``data`` is whatever ``proxy_json`` decoded from the backend (a dict for
    JSON backends, or a plain `str` for backends that already returned
    text/srt/vtt — e.g. whisper-server). Returns ``(content, content_type)``
    where ``content`` is a dict for JSON responses and a `str` otherwise.
    """
    fmt = (fmt or "json").strip().lower()
    if fmt not in ("json", "text", "srt", "vtt", "verbose_json"):
        fmt = "json"

    if isinstance(data, str):
        # Backend already returned a non-JSON body (whisper text/srt/vtt).
        if fmt in ("srt", "vtt"):
            return data, _SUBTITLE_CT[fmt]
        if fmt == "text":
            return data, "text/plain"
        if fmt == "verbose_json":
            return _verbose_json(data, language), "application/json"
        return {"text": data}, "application/json"

    if not isinstance(data, dict):
        return {"text": str(data)}, "application/json"

    text = data.get("text") or ""
    if not isinstance(text, str):
        text = str(text)

    if fmt == "text":
        return text.rstrip("\n") + "\n", "text/plain"
    if fmt == "verbose_json":
        return _verbose_dict(data, text, language), "application/json"
    if fmt in ("srt", "vtt"):
        segments = data.get("segments") or []
        if segments:
            return _render_subtitles(segments, fmt), _SUBTITLE_CT[fmt]
        # No timing info: fall back to a single untimed cue only when the
        # backend provided a duration, otherwise degrade to plain text.
        duration = _as_float(data.get("duration"))
        if duration > 0:
            return _render_subtitles(
                [{"start": 0.0, "end": duration, "text": text}], fmt
            ), _SUBTITLE_CT[fmt]
        return text.rstrip("\n") + "\n", "text/plain"
    # json (default): keep the backend's JSON payload as-is.
    return data, "application/json"


def _verbose_dict(data: dict, text: str, language: Optional[str]) -> dict:
    obj: dict[str, Any] = {
        "task": data.get("task", "transcribe"),
        "language": data.get("language") or language or "",
        "duration": _as_float(data.get("duration")),
        "text": text,
        "segments": data.get("segments") or [],
    }
    if data.get("words") is not None:
        obj["words"] = data["words"]
    return obj


def _verbose_json(text: str, language: Optional[str]) -> dict:
    return {
        "task": "transcribe",
        "language": language or "",
        "duration": 0.0,
        "text": text,
        "segments": [],
    }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _render_subtitles(segments: list, fmt: str) -> str:
    lines: list[str] = []
    if fmt == "vtt":
        lines.extend(("WEBVTT", ""))
    for i, seg in enumerate(segments, start=1):
        if not isinstance(seg, dict):
            continue
        start = _as_float(seg.get("start"))
        end = _as_float(seg.get("end"))
        if end <= start:
            end = start
        text = str(seg.get("text", "")).strip()
        if fmt == "srt":
            lines.extend((
                str(i),
                f"{_ts_srt(start)} --> {_ts_srt(end)}",
                text,
                "",
            ))
        else:
            lines.extend((
                f"{_ts_vtt(start)} --> {_ts_vtt(end)}",
                text,
                "",
            ))
    return "\n".join(lines)


def _ts_srt(t: float) -> str:
    t = max(0.0, t)
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _ts_vtt(t: float) -> str:
    t = max(0.0, t)
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


# --------------------------------------------------------------------------
# Speech response-format transcoding (WAV → mp3/opus/aac/flac/pcm)
# --------------------------------------------------------------------------
# audio.cpp backends emit WAV only. OpenAI's /v1/audio/speech accepts
# ``response_format``; the proxy honours it locally via ffmpeg so clients
# requesting mp3/opus/aac/flac/pcm get the requested codec instead of a
# mislabelled WAV.

# destination codec -> (content-type, extra ffmpeg argv past `-i pipe:0`)
_SPEECH_FORMATS: dict[str, tuple[str, list[str]]] = {
    "mp3": ("audio/mpeg", ["-f", "mp3"]),
    "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus"]),
    "aac": ("audio/aac", ["-f", "adts", "-c:a", "aac"]),
    "flac": ("audio/x-flac", ["-f", "flac"]),
    "pcm": ("audio/x-pcm", ["-f", "s16le", "-acodec", "pcm_s16le"]),
}


async def transcode_audio(data: bytes, dst: str) -> Optional[tuple[bytes, str]]:
    """Transcode ``data`` (expected WAV) to OpenAI speech format ``dst``.

    Returns ``(bytes, content_type)`` on success or ``None`` when the
    conversion cannot be performed (ffmpeg missing, unknown format, or a
    failed encode) — callers then pass the source bytes through unchanged.
    """
    dst = (dst or "wav").strip().lower()
    if dst == "wav":
        return data, "audio/wav"
    spec = _SPEECH_FORMATS.get(dst)
    if spec is None:
        return None
    ctype, args = spec
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; returning %s audio untranscoded", dst)
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", *args, "-y", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        logger.warning("failed to spawn ffmpeg: %s", exc)
        return None
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(data), timeout=120.0
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    if proc.returncode != 0 or not out:
        logger.warning(
            "ffmpeg transcode to %s failed: %s", dst, err[:300]
        )
        return None
    return out, ctype
