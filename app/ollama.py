"""Ollama request/response translation.

The Ollama API is not OpenAI-shaped: it uses ``/api/chat`` and
``/api/generate`` with NDJSON streaming, a different metrics envelope,
and ``/api/embed`` for embeddings. These helpers translate between that
dialect and the OpenAI surface llamaswap already proxies, keeping the
route handlers in ``main.py`` small.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ndjson(payload: dict) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode() + b"\n"


def _image_data_url(value: str) -> str:
    if value.startswith("data:"):
        return value
    if value.startswith("iVBORw0KGgo"):
        mime = "image/png"
    elif value.startswith("/9j/"):
        mime = "image/jpeg"
    elif value.startswith("R0lGOD"):
        mime = "image/gif"
    elif value.startswith("UklGR"):
        mime = "image/webp"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{value}"


def convert_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list):
        return []
    out: list[dict] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role") or "user"
        content = raw.get("content")
        if content is None:
            content = ""
        images = raw.get("images") or []
        if images:
            parts: list[dict] = []
            if content:
                parts.append({"type": "text", "text": str(content)})
            for image in images:
                if isinstance(image, str) and image:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                    })
            content = parts
        msg: dict[str, Any] = {"role": role, "content": content}
        if raw.get("tool_calls") is not None:
            msg["tool_calls"] = raw["tool_calls"]
        if raw.get("tool_call_id") is not None:
            msg["tool_call_id"] = raw["tool_call_id"]
        out.append(msg)
    return out


def _map_format(fmt: Any) -> dict:
    if fmt == "json":
        return {"response_format": {"type": "json_object"}}
    if isinstance(fmt, dict):
        return {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt},
        }}
    return {}


def _map_options(options: Any) -> dict:
    if not isinstance(options, dict):
        return {}
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "typical_p": "typical_p",
        "seed": "seed",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repeat_penalty": "repeat_penalty",
        "repeat_last_n": "repeat_last_n",
    }
    out: dict[str, Any] = {}
    for src, dst in mapping.items():
        if options.get(src) is not None:
            out[dst] = options[src]
    stop = options.get("stop")
    if isinstance(stop, str):
        out["stop"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop"] = stop
    num_predict = options.get("num_predict")
    if isinstance(num_predict, (int, float)) and num_predict > 0:
        out["max_tokens"] = int(num_predict)
    return out


def openai_chat_body(obj: dict) -> tuple[str, dict, bool]:
    model = str(obj.get("model") or "")
    stream = bool(obj.get("stream", True))
    body: dict[str, Any] = {
        "model": model,
        "messages": convert_messages(obj.get("messages")),
        "stream": stream,
    }
    body.update(_map_options(obj.get("options")))
    body.update(_map_format(obj.get("format")))
    if obj.get("tools"):
        body["tools"] = obj["tools"]
    if stream:
        body["stream_options"] = {"include_usage": True}
    return model, body, stream


def openai_generate_body(obj: dict) -> tuple[str, dict, bool, str]:
    model = str(obj.get("model") or "")
    stream = bool(obj.get("stream", True))
    prompt = str(obj.get("prompt") or "")
    system = obj.get("system")
    images = obj.get("images") or []
    fmt = obj.get("format")
    raw = bool(obj.get("raw")) and not system and not images and not fmt

    options = _map_options(obj.get("options"))
    if raw:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        body.update(options)
        if stream:
            body["stream_options"] = {"include_usage": True}
        return model, body, stream, "/v1/completions"

    content: Any = prompt
    if images:
        parts: list[dict] = []
        if prompt:
            parts.append({"type": "text", "text": prompt})
        for image in images:
            if isinstance(image, str) and image:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image)},
                })
        content = parts
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": content})
    body = {"model": model, "messages": messages, "stream": stream}
    body.update(options)
    body.update(_map_format(fmt))
    if stream:
        body["stream_options"] = {"include_usage": True}
    return model, body, stream, "/v1/chat/completions"


def _openai_text(data: dict) -> tuple[str, str, Any]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "", "", None
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = choice.get("text")
    if content is None:
        content = ""
    thinking = (
        message.get("reasoning_content")
        or message.get("thinking")
        or choice.get("reasoning_content")
        or ""
    )
    return str(content), str(thinking), message.get("tool_calls")


def _done_reason(data: dict) -> str:
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if reason == "length":
            return "length"
        if reason:
            return str(reason)
    return "stop"


def _tool_calls_to_ollama(tool_calls: Any) -> list[dict]:
    out: list[dict] = []
    if not isinstance(tool_calls, list):
        return out
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass
        if name:
            out.append({"function": {"name": name, "arguments": args}})
    return out


def _metrics(
    data: dict, *, load_ns: int, total_ns: int,
) -> dict[str, int]:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    prompt_duration = int(prompt_ms * 1_000_000)
    eval_duration = int(predicted_ms * 1_000_000)
    total = total_ns or (load_ns + prompt_duration + eval_duration)
    return {
        "total_duration": int(total),
        "load_duration": int(load_ns),
        "prompt_eval_count": int(
            usage.get("prompt_tokens") or timings.get("prompt_n") or 0
        ),
        "prompt_eval_duration": prompt_duration,
        "eval_count": int(
            usage.get("completion_tokens") or timings.get("predicted_n") or 0
        ),
        "eval_duration": eval_duration,
    }


def chat_response(
    data: dict, model: str, *, created_at: str,
    load_ns: int, total_ns: int,
) -> dict:
    content, thinking, tool_calls = _openai_text(data)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    calls = _tool_calls_to_ollama(tool_calls)
    if calls:
        message["tool_calls"] = calls
    payload = {
        "model": model,
        "created_at": created_at,
        "message": message,
        "done": True,
        "done_reason": _done_reason(data),
    }
    payload.update(_metrics(data, load_ns=load_ns, total_ns=total_ns))
    return payload


def generate_response(
    data: dict, model: str, *, created_at: str,
    load_ns: int, total_ns: int,
) -> dict:
    content, thinking, _ = _openai_text(data)
    payload: dict[str, Any] = {
        "model": model,
        "created_at": created_at,
        "response": content,
        "done": True,
        "done_reason": _done_reason(data),
    }
    if thinking:
        payload["thinking"] = thinking
    payload.update(_metrics(data, load_ns=load_ns, total_ns=total_ns))
    return payload


def _merge_tool_call(acc: dict[int, dict], call: Any) -> None:
    if not isinstance(call, dict):
        return
    index = call.get("index")
    if not isinstance(index, int):
        index = len(acc)
    item = acc.setdefault(index, {"name": "", "arguments": ""})
    fn = call.get("function") or {}
    if isinstance(fn, dict):
        if fn.get("name"):
            item["name"] = fn["name"]
        args = fn.get("arguments")
        if isinstance(args, str):
            item["arguments"] += args
        elif isinstance(args, dict):
            item["arguments"] = json.dumps(args)


def _accumulated_tool_calls(acc: dict[int, dict]) -> list[dict]:
    out: list[dict] = []
    for index in sorted(acc):
        item = acc[index]
        if not item.get("name"):
            continue
        args: Any = item.get("arguments") or ""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass
        out.append({"function": {"name": item["name"], "arguments": args}})
    return out


async def stream_response(
    sse: AsyncIterator[bytes],
    model: str,
    *,
    generate: bool,
    created_at: str,
    started: float,
    load_ns: int,
) -> AsyncIterator[bytes]:
    buffer = b""
    finished = False
    finish_reason = "stop"
    usage: dict = {}
    timings: dict = {}
    tools: dict[int, dict] = {}

    def final(payload: dict) -> dict:
        total = int((time.perf_counter() - started) * 1e9)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        predicted_ms = float(timings.get("predicted_ms") or 0.0)
        prompt_duration = int(prompt_ms * 1_000_000)
        eval_duration = int(predicted_ms * 1_000_000)
        eval_count = int(usage.get("completion_tokens") or 0)
        if not eval_duration and eval_count:
            eval_duration = max(0, total - int(load_ns))
        metrics = {
            "total_duration": total,
            "load_duration": int(load_ns),
            "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
            "prompt_eval_duration": prompt_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
        }
        payload.update(metrics)
        return payload

    async for chunk in sse:
        buffer = (buffer + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            data_lines = [
                line[5:].strip()
                for line in event.split(b"\n")
                if line.startswith(b"data:")
            ]
            for data in data_lines:
                if not data:
                    continue
                if data == b"[DONE]":
                    if not finished:
                        finished = True
                        tone = _accumulated_tool_calls(tools)
                        message: dict[str, Any] = {
                            "role": "assistant", "content": "",
                        }
                        if tone:
                            message["tool_calls"] = tone
                        if generate:
                            payload = {
                                "model": model,
                                "created_at": created_at,
                                "response": "",
                                "done": True,
                                "done_reason": finish_reason,
                            }
                        else:
                            payload = {
                                "model": model,
                                "created_at": created_at,
                                "message": message,
                                "done": True,
                                "done_reason": finish_reason,
                            }
                        yield _ndjson(final(payload))
                    continue
                try:
                    obj = json.loads(data)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict) and obj.get("error") is not None:
                    yield _ndjson({"error": obj["error"]})
                    return
                if not isinstance(obj, dict):
                    continue
                current_usage = obj.get("usage")
                if isinstance(current_usage, dict):
                    usage = current_usage
                current_timings = obj.get("timings")
                if isinstance(current_timings, dict):
                    timings = current_timings
                choices = obj.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = {}
                content = delta.get("content")
                if content is None:
                    content = choice.get("text")
                thinking = (
                    delta.get("reasoning_content")
                    or delta.get("thinking")
                    or choice.get("reasoning_content")
                )
                tool_calls = delta.get("tool_calls") or choice.get("tool_calls")
                if tool_calls:
                    for call in tool_calls:
                        _merge_tool_call(tools, call)
                reason = choice.get("finish_reason")
                if reason:
                    finish_reason = "length" if reason == "length" else str(reason)
                if content:
                    if generate:
                        yield _ndjson({
                            "model": model,
                            "created_at": created_at,
                            "response": str(content),
                            "done": False,
                        })
                    else:
                        yield _ndjson({
                            "model": model,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": str(content),
                            },
                            "done": False,
                        })
                if thinking:
                    if generate:
                        yield _ndjson({
                            "model": model,
                            "created_at": created_at,
                            "response": "",
                            "thinking": str(thinking),
                            "done": False,
                        })
                    else:
                        yield _ndjson({
                            "model": model,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "thinking": str(thinking),
                            },
                            "done": False,
                        })
        if finished and not buffer.strip():
            return

    if not finished:
        tone = _accumulated_tool_calls(tools)
        message = {"role": "assistant", "content": ""}
        if tone:
            message["tool_calls"] = tone
        if generate:
            payload = {
                "model": model,
                "created_at": created_at,
                "response": "",
                "done": True,
                "done_reason": finish_reason,
            }
        else:
            payload = {
                "model": model,
                "created_at": created_at,
                "message": message,
                "done": True,
                "done_reason": finish_reason,
            }
        yield _ndjson(final(payload))
