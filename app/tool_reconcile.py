"""Recovery of tool calls the backend parsed but refused to honour.

A chat backend that parses ``<tool_call>`` blocks checks the written name against
the tools the request declared. When the model writes a name the request
never declared — the usual cause being a client that namespaces MCP tools
with the server name (``unity`` + ``unity_list_instances`` is declared as
``unity_unity_list_instances``) while the project's own docs tell the model
the bare name — the backend leaves the block in the answer as plain text
and, on Strata, reports it as ``strata_rejected_tool_calls`` on the response
and counts it in its own ``/metrics``. The agent loop then sees text, no
call, and spends the step re-reading the same files.

When the intended tool is unambiguous this module re-emits the call as a
normal ``tool_calls`` entry, with outcomes:

  * ``reconciled`` — exactly one declared name fits; the call is emitted.
  * ``ambiguous``  — several declared names fit; nothing is emitted.
  * ``unmatched``  — no declared name fits; nothing is emitted.

The assistant text is never rewritten, so whatever the client displayed
stays consistent with what it executes. Nothing is invented: a call is only
emitted for a name the model itself wrote. A refused call keeps the model's
arguments: Strata reports refusals as names only, while the original
``<tool_call>`` block — arguments included — stays in the content as text,
so the two are paired up before the call is re-emitted.
"""

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("llamaswap.tool_reconcile")

# Strata names the field after its own vendor prefix; other backends that
# carry the same idea use a plain one.
_REJECT_KEYS = ("strata_rejected_tool_calls", "rejected_tool_calls", "invalid_tool_calls")

# <tool_call>{"name": "x", "arguments": {...}}</tool_call> and <function=x>...</function>
_JSON_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*<\/?tool_call>", re.S)
_XML_BLOCK_RE = re.compile(r"<function\s*=\s*([A-Za-z0-9_.\-]+)\s*>(.*?)</function>", re.S)

RECONCILED = "reconciled"
AMBIGUOUS = "ambiguous"
UNMATCHED = "unmatched"


def _load(body: Any) -> dict:
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return {}
    return body if isinstance(body, dict) else {}


def declared_tool_names(body: Any) -> list[str]:
    """Names the client declared in this request (OpenAI tool schema)."""
    obj = _load(body)
    names: list[str] = []
    for tool in obj.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def match_name(
    name: str, declared: list[str], min_ratio: float = 0.75,
) -> tuple[Optional[str], str]:
    """The declared tool the model meant, if exactly one fits."""
    target = _norm(name)
    if not target or not declared:
        return None, UNMATCHED
    exact = [d for d in declared if _norm(d) == target]
    if len(exact) == 1:
        return exact[0], RECONCILED
    # Prefix drift: a server-prefixed client name contains the documented
    # bare name (or the reverse). Only accept it when one tool fits.
    fits = [d for d in declared
            if target.endswith(_norm(d)) or _norm(d).endswith(target)]
    if len(fits) == 1:
        return fits[0], RECONCILED
    if fits:
        return None, AMBIGUOUS
    ratios = [(d, SequenceMatcher(None, _norm(d), target).ratio()) for d in declared]
    good = [d for d, ratio in ratios if ratio >= min_ratio]
    if len(good) == 1:
        return good[0], RECONCILED
    return None, AMBIGUOUS if good else UNMATCHED


def rejected_calls(data: dict) -> list[dict]:
    """Calls the backend reported as rejected, wherever it attached them."""
    found: list[dict] = []
    seen: set[str] = set()

    def take(holder: Any) -> None:
        if not isinstance(holder, dict):
            return
        for key in _REJECT_KEYS:
            value = holder.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                call = item if isinstance(item, dict) else {"name": item}
                if not isinstance(call.get("name"), str):
                    continue
                key_id = json.dumps(
                    {"n": call["name"], "a": _arguments(call)}, sort_keys=True
                )
                if key_id in seen:
                    continue
                seen.add(key_id)
                found.append(call)

    take(data)
    for choice in data.get("choices") or []:
        take(choice)
        if isinstance(choice, dict):
            take(choice.get("message"))
    return found


def _arguments(call: dict) -> str:
    raw = call.get("arguments", call.get("parameters", "{}"))
    if isinstance(raw, dict):
        return json.dumps(raw)
    if isinstance(raw, str):
        try:
            return json.dumps(json.loads(raw))
        except (ValueError, TypeError):
            return "{}"
    return "{}"


def text_tool_calls(content: str) -> list[dict]:
    """Calls left behind as text — the shape a client actually sees."""
    found: list[dict] = []
    for match in _JSON_BLOCK_RE.finditer(content):
        try:
            payload = json.loads(match.group(1))
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            args = payload.get("arguments", payload.get("parameters"))
            found.append({"name": payload["name"], "arguments": _arguments({"arguments": args})})
    for match in _XML_BLOCK_RE.finditer(content):
        found.append({"name": match.group(1), "arguments": _arguments({"arguments": match.group(2)})})
    return found


def _assistant_message(data: dict) -> Optional[dict]:
    for choice in data.get("choices") or []:
        if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
            return choice["message"]
    return None


def _merge_arguments(
    rejected: list[dict], text_calls: list[dict],
) -> list[dict]:
    """Arguments for refused calls, taken from the matching text block.

    Strata reports a refusal as names only (``strata_rejected_tool_calls``
    carries ``{"name": ..., "reason": "undeclared"}``) while the original
    ``<tool_call>`` block — the model's actual arguments — stays in the
    content as text. Pair each refused name with its text block, in order;
    a refused call the backend itself reported with arguments keeps them.
    """
    remaining = list(text_calls)
    merged: list[dict] = []
    for call in rejected:
        if _arguments(call) == "{}":
            name = _norm(str(call.get("name") or ""))
            for i, tc in enumerate(remaining):
                if _norm(str(tc.get("name") or "")) == name:
                    call = {**call, "arguments": tc.get("arguments", "{}")}
                    del remaining[i]
                    break
        merged.append(call)
    return merged


def apply_reconcile(
    data: dict, declared: list[str], min_ratio: float = 0.75,
) -> list[tuple[str, Optional[str], str]]:
    """Emit reconciled tool_calls on the response; report every outcome."""
    message = _assistant_message(data)
    if message is None or not declared:
        return []
    content = message.get("content")
    text_calls = (
        text_tool_calls(content)
        if isinstance(content, str) and content else []
    )
    calls = rejected_calls(data)
    calls = _merge_arguments(calls, text_calls) if calls else text_calls
    executed = {
        call.get("function", {}).get("name")
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    }
    records: list[tuple[str, Optional[str], str]] = []
    for call in calls:
        name = str(call.get("name") or "")
        matched, outcome = match_name(name, declared, min_ratio)
        records.append((name, matched, outcome))
        if outcome != RECONCILED or matched in executed:
            continue
        existing = message.get("tool_calls")
        if not isinstance(existing, list):
            existing = []
            message["tool_calls"] = existing
        existing.append({
            "id": f"call_{uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": matched, "arguments": _arguments(call)},
        })
        executed.add(matched)
    return records


class StreamReconciler:
    """Passes an SSE stream through untouched, appending the reconciled delta.

    Only the events that end the stream (the one carrying ``finish_reason``
    and ``[DONE]``) are held back, so reasoning and content keep streaming
    at full speed; the synthesized chunk is inserted just before them.
    """

    def __init__(
        self, declared: list[str], min_ratio: float = 0.75,
    ) -> None:
        self._declared = declared
        self._min_ratio = min_ratio
        self._residual = b""
        self._text: list[str] = []
        self._tail: list[bytes] = []
        self._template: Optional[dict] = None
        # Highest tool_calls index the backend streamed itself, so a
        # reconciled delta continues after them instead of colliding with
        # them (OpenAI clients merge streamed deltas by index).
        self._next_index = 0
        self._records: list[tuple[str, Optional[str], str]] = []
        self.done = False

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        buf = self._residual + chunk
        lines = buf.split(b"\n")
        self._residual = lines.pop()
        out: list[bytes] = []
        for line in lines:
            self._route(line, out)
        return b"".join(out)

    def finish(self) -> bytes:
        self.done = True
        out: list[bytes] = []
        if self._residual:
            self._route(self._residual, out)
            self._residual = b""
        out.extend(self._synthesized())
        return b"".join(out)

    def _route(self, line: bytes, out: list[bytes]) -> None:
        if not line.startswith(b"data: "):
            out.append(line + b"\n")
            return
        payload = line[6:].strip()
        if payload == b"[DONE]":
            self._tail.append(line + b"\n")
            return
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            out.append(line + b"\n")
            return
        if not isinstance(obj, dict):
            out.append(line + b"\n")
            return
        self._template = obj
        text: list[str] = []
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    text.append(content)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    idx = call.get("index")
                    if not isinstance(idx, int) or idx < 0:
                        idx = 0
                    self._next_index = max(self._next_index, idx + 1)
        self._text.extend(text)
        final = any(
            isinstance(choice, dict) and choice.get("finish_reason") is not None
            for choice in obj.get("choices") or []
        )
        if final or rejected_calls(obj):
            self._tail.append(line + b"\n")
        else:
            out.append(line + b"\n")

    def _synthesized(self) -> list[bytes]:
        text = "".join(self._text)
        text_calls = text_tool_calls(text) if text else []
        calls: list[dict] = []
        for line in self._tail:
            if line.startswith(b"data: "):
                try:
                    obj = json.loads(line[6:].strip())
                except (ValueError, TypeError):
                    continue
                calls.extend(rejected_calls(obj))
        calls = _merge_arguments(calls, text_calls) if calls else text_calls
        out: list[bytes] = []
        delta_calls: list[dict] = []
        emitted = 0
        for call in calls:
            name = str(call.get("name") or "")
            matched, outcome = match_name(name, self._declared, self._min_ratio)
            self._records.append((name, matched, outcome))
            if outcome != RECONCILED:
                continue
            delta_calls.append({
                "index": self._next_index + emitted,
                "id": f"call_{uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": matched, "arguments": _arguments(call)},
            })
            emitted += 1
        if delta_calls:
            template = self._template or {"object": "chat.completion.chunk"}
            event = {
                "id": template.get("id") or f"chatcmpl-{uuid4().hex[:20]}",
                "object": template.get("object") or "chat.completion.chunk",
                "created": template.get("created") or 0,
                "model": template.get("model") or "",
                "choices": [{"index": 0, "delta": {"tool_calls": delta_calls},
                             "finish_reason": None}],
            }
            out.append(
                b"data: " + json.dumps(event, separators=(",", ":"),
                                       ensure_ascii=False).encode() + b"\n\n"
            )
        out.extend(self._tail)
        self._tail = []
        return out

    @property
    def records(self) -> list[tuple[str, Optional[str], str]]:
        return self._records
