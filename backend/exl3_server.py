"""Minimal OpenAI-compatible server for exllamav3 (EXL3) models.

Runs the same inference stack that tabbyAPI uses (exllamav3 + torch + Triton)
but as a plain subprocess so llamaswap can launch/stop it like a llama-server
backend: SIGTERM from the ProcessManager unloads the model and frees VRAM.

Feature scope (mirrors ~/tabbyapi-config/config.yml for
Qwen3.8-27B-EXL3-3.5bpw):
  * EXL3 weights via exllamav3, Q4 KV cache, MTP speculative drafting
  * sysmem tiers for recurrent state + KV overflow (WSL 16 GB card)
  * /v1/chat/completions and /v1/completions, streaming and not
  * tool calling: OpenAI tools/tool_choice are rendered into the chat
    template and Qwen-style tool-call blocks in the output are parsed
    into OpenAI tool_calls (streaming included)
  * text-only: tabby's vision mode (vision_offload streaming of mtmd weights)
    is NOT implemented — this endpoint advertises capability "chat" only.

Endpoints the llamaswap proxy actually needs:
  GET  /health                       — ProcessManager readiness probe
  POST /v1/chat/completions          — OpenAI chat dialect ("json" format)
  POST /v1/completions               — legacy completions
"""

import argparse
import contextlib
import json
import re
import sys
import time
import typing
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2.exceptions import TemplateError

from exllamav3 import model_init
from exllamav3.generator import AsyncGenerator, AsyncJob
from exllamav3.generator.sampler.custom import (
    CustomSampler,
    SS_Argmax,
    SS_MinP,
    SS_PresFreqP,
    SS_RepP,
    SS_Sample,
    SS_Temperature,
    SS_TopK,
    SS_TopP,
)

MODEL_NAME = "exl3"
DEFAULT_PENALTY_RANGE = 1024


def log(msg: str) -> None:
    # stderr: llamaswap tails the backend's stderr into its own log
    print(f"[exl3-server] {msg}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Standard exllamav3 loading args (-m/-gs/-cs/-cq/--mtp/-chunk_size/
    # -ambs/-no_warmup/...). add_draft_model_args enables --mtp; init()
    # then wires the model's own MTP head as the drafter.
    model_init.add_args(
        parser,
        cache=True,
        default_cache_size=61440,
        add_draft_model_args=True,
        default_chunk_size=512,
        default_autosplit_max_batch_size=2,
    )
    # Server args. --host/--port are appended by llamaswap's build_argv()
    # and always win over anything the YAML passes.
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    # Name reported in OpenAI responses (llamaswap rewrites it anyway).
    # NOT spelled --model-name: that would make argparse's prefix matching
    # ambiguous between --model_dir and --model-name when llamaswap passes
    # bare --model (which registry._model_path also scans for).
    parser.add_argument("--display-name", type=str, default=MODEL_NAME)
    # System-memory tiers, in MiB (tabby: memory.sysmem_recurrent_cache /
    # memory.sysmem_kv_cache).
    parser.add_argument("--sysmem-recurrent-cache-mb", type=int, default=8192)
    parser.add_argument("--sysmem-kv-cache-mb", type=int, default=2176)
    # Per-device load reserve in MiB (tabby default_reserve = 96 MB). The
    # 14.3 GB model is within ~100 MB of the card limit, so this margin is
    # the difference between fitting and "Insufficient VRAM in split".
    parser.add_argument("--reserve-mb", type=int, default=96)
    return parser


def build_sampler(req: dict):
    """OAI request params -> exllamav3 sampler stack (tabbyapi-style)."""
    temperature = req.get("temperature", 0.8)
    if temperature is None:
        temperature = 0.8
    top_k = req.get("top_k") or 0
    top_p = req.get("top_p")
    if top_p is None:
        top_p = 1.0
    min_p = req.get("min_p")
    if min_p is None:
        min_p = 0.08
    rep_p = req.get("repetition_penalty") or 1.0
    pres_p = req.get("presence_penalty") or 0.0
    freq_p = req.get("frequency_penalty") or 0.0

    if temperature == 0:
        return CustomSampler([SS_Argmax()])

    steps = []
    if rep_p != 1.0:
        steps.append(SS_RepP(rep_p, DEFAULT_PENALTY_RANGE, 0))
    if pres_p != 0.0 or freq_p != 0.0:
        steps.append(SS_PresFreqP(pres_p, freq_p, DEFAULT_PENALTY_RANGE, 0))
    steps.append(SS_Temperature(temperature))
    if top_k > 0:
        steps.append(SS_TopK(top_k))
    if top_p < 1.0:
        steps.append(SS_TopP(top_p))
    if min_p > 0.0:
        steps.append(SS_MinP(min_p))
    steps.append(SS_Sample())
    return CustomSampler(steps)


class Engine:
    """Loads model on startup, owns the AsyncGenerator."""

    def __init__(self, args):
        self.args = args
        self.model_name = args.display_name
        self.cache_size = args.cache_size
        self.generator: AsyncGenerator | None = None
        self.tokenizer = None
        self.config = None

    def load(self) -> None:
        a = self.args
        log(f"loading {a.model_dir} (cache {a.cache_size} tok, "
            f"chunk {a.chunk_size}, mtp={a.mtp})")
        t0 = time.time()
        # tabbyAPI lowers exllamav3's default 0.5 GB-per-device load reserve
        # to 96 MB (its default_reserve = [96/1024] GiB). Without this the
        # 14.3 GB model misses the single-device threshold by ~0.4 GB and
        # falls into autosplit, which then fails with "Insufficient VRAM in
        # split for model and cache". Forwarded verbatim to Model.load().
        result = model_init.init(
            a, quiet=True, progress=False,
            reserve_per_device=[a.reserve_mb / 1024],
        )
        if a.mtp:  # add_draft_model_args enabled -> 7-tuple
            model, config, cache, tokenizer, draft_model, _, draft_cache = result
        else:
            model, config, cache, tokenizer = result
            draft_model = draft_cache = None
        self.config = config
        self.tokenizer = tokenizer
        # (model, config, cache, tokenizer, draft_model, draft_cache) — the
        # AsyncGenerator is built later by create_generator(), which must run
        # inside the event loop (its constructor calls asyncio.create_task).
        self._loaded = (model, config, cache, tokenizer, draft_model, draft_cache)
        log(f"model loaded in {time.time() - t0:.1f}s")

    def create_generator(self) -> None:
        a = self.args
        model, config, cache, tokenizer, draft_model, draft_cache = self._loaded
        self.generator = AsyncGenerator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
            max_batch_size=a.autosplit_max_batch_size,
            max_chunk_size=a.chunk_size,
            recurrent_cache_size=a.sysmem_recurrent_cache_mb * 1024 * 1024,
            cpu_cache_size=a.sysmem_kv_cache_mb * 1024 * 1024,
        )
        log("generator ready")


@contextlib.asynccontextmanager
async def lifespan(app):
    # Running inside the loop now — AsyncGenerator's constructor spawns its
    # iteration task, which requires an active event loop.
    engine.create_generator()
    yield
    generator, engine.generator = engine.generator, None
    await generator.close()


engine: Engine | None = None
app = FastAPI(lifespan=lifespan)


def oai_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=status,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": engine.model_name,
            "object": "model",
            "owned_by": "exllamav3",
        }],
    }


ROLE_ALIASES = {"developer": "system", "function": "tool"}

# Qwen-style tool-call markup. Built from concatenated literals so this
# file itself carries no raw tool-call tags (they trip doc renderers).
_TOOL_OPEN = "<tool" "_call>"
_TOOL_CLOSE = "</tool" "_call>"
TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_OPEN) + r"\s*(.*?)\s*" + re.escape(_TOOL_CLOSE),
    re.DOTALL,
)
# Qwen3.8 tool-call blocks are XML-ish: inside the tool_call tags sits a
# function block whose parameters are each wrapped in their own tags.
# (Adjacent literals again, to keep raw tags out of this file's source.)
_FUNC_NAME_RE = re.compile(r"<" r"function=([^>]+)>")
_PARAM_RE = re.compile(
    r"<" r"parameter=([^>]+)>\n?(.*?)\n?</" r"parameter>",
    re.DOTALL,
)
# Reasoning tags: with thinking enabled the template pre-fills the opening
# tag, so generation starts inside the thinking block and reasoning must
# be split out of the content at the closing tag.
_THINK_OPEN = "<" + "think>"
_THINK_CLOSE = "</" + "think>"


def _thinking_enabled(body: dict) -> bool:
    """Resolved enable_thinking flag (chat_template_kwargs wins over top)."""
    kwargs = body.get("chat_template_kwargs") or {}
    return kwargs.get("enable_thinking", body.get("enable_thinking", True)) \
        is not False


def _tools_for_template(body: dict) -> list | None:
    """OpenAI `tools`/`tool_choice` -> tool list for the chat template.

    None means render no tools: either the request had none, tool_choice
    was "none", or a named tool_choice matched nothing. A specific
    tool_choice filters the list down to that one function; "auto" and
    "required" pass everything through ("required" is best-effort — the
    model is allowed to call, there is no grammar forcing it).
    """
    tools = body.get("tools")
    if not tools:
        return None
    choice = body.get("tool_choice")
    if choice == "none":
        return None
    if isinstance(choice, dict) and choice.get("type") == "function":
        wanted = (choice.get("function") or {}).get("name")
        if wanted:
            filtered = [t for t in tools
                        if (t.get("function") or {}).get("name") == wanted]
            if filtered:
                return filtered
    return list(tools)


def _parse_tool_call_payload(block: str) -> dict | None:
    """Qwen3.8 tool-call block -> {"name", "arguments" (JSON str)}, or None.

    Block shape (the text between the tool_call tags): a function tag
    carrying the call name, followed by one tag per parameter with the
    raw value as its body. Parameter values that parse as JSON are kept
    as JSON values (numbers, lists, nested objects); anything else stays
    a string. The result matches the OpenAI schema where arguments is a
    JSON-encoded string.
    """
    m = _FUNC_NAME_RE.search(block)
    if not m:
        return None
    name = m.group(1).strip()
    args: dict = {}
    for pname, pvalue in _PARAM_RE.findall(block):
        value = pvalue.strip()
        try:
            args[pname.strip()] = json.loads(value)
        except json.JSONDecodeError:
            args[pname.strip()] = value
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def _args_str_to_dict(arguments):
    """OpenAI JSON-string arguments -> dict (pass-through unless a string)."""
    if not isinstance(arguments, str) or arguments == "":
        return arguments
    try:
        obj = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _normalize_tool_call_args(tool_call: dict) -> dict:
    """Copy of a tool_call whose arguments are a dict, not a JSON string.

    The OpenAI spec carries arguments as a JSON *string*, but strict
    templates (Qwen3.8 etc.) re-render assistant history with an items
    filter over arguments, which requires a mapping. Handles both the
    nested OpenAI shape (function.name/arguments) and the flat shape.
    """
    if not isinstance(tool_call, dict):
        return tool_call
    out = dict(tool_call)
    if isinstance(out.get("function"), dict):
        fn = dict(out["function"])
        if "arguments" in fn:
            fn["arguments"] = _args_str_to_dict(fn["arguments"])
        out["function"] = fn
    elif "arguments" in out:
        out["arguments"] = _args_str_to_dict(out["arguments"])
    return out


def _normalize_messages_for_template(messages: list) -> list:
    """Copy messages so assistant tool_calls carry dict arguments."""
    return [
        {**m, "tool_calls": [
            _normalize_tool_call_args(tc) for tc in m["tool_calls"]
        ]}
        if isinstance(m, dict) and m.get("tool_calls") else m
        for m in messages
    ]


def _oai_tool_call(index: int, parsed: dict) -> dict:
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": parsed,
    }


def _prompt_from_chat(body: dict) -> str:
    messages = body.get("messages") or []
    # Normalize roles some OpenAI clients emit that HF chat templates don't
    # know (e.g. the o-series "developer" role) — otherwise the template
    # raises "Unexpected message role".
    messages = [
        {**m, "role": ROLE_ALIASES.get(m.get("role"), m.get("role"))}
        if isinstance(m, dict) else m
        for m in messages
    ]
    # Assistant history carries arguments as JSON strings per the OpenAI
    # spec; the template needs dicts or rendering raises a TypeError.
    messages = _normalize_messages_for_template(messages)
    template_kwargs = dict(body.get("chat_template_kwargs") or {})
    # tabbyAPI-style top-level "enable_thinking"; request-level
    # chat_template_kwargs wins if both are present.
    if "enable_thinking" in body and "enable_thinking" not in template_kwargs:
        template_kwargs["enable_thinking"] = body["enable_thinking"]
    # Tool definitions go through the template the same way (Qwen3 and most
    # modern templates render them into the prompt). apply_chat_template
    # puts unknown kwargs in the render context, so templates without tool
    # support simply ignore them.
    tools = _tools_for_template(body)
    if tools is not None:
        template_kwargs["tools"] = tools
    return engine.tokenizer.hf_chat_template(
        messages, add_generation_prompt=True, **template_kwargs
    )


def _stop_conditions(body: dict) -> list:
    # EOS token ids (config + tokenizer), mirroring tabbyapi's
    # stop_conditions construction, plus the request's stop strings.
    stop: list = list(getattr(engine.config, "eos_token_id_list", None) or [])
    if engine.tokenizer.eos_token_id is not None:
        stop.append(engine.tokenizer.eos_token_id)
    req_stop = body.get("stop")
    if req_stop:
        stop += [req_stop] if isinstance(req_stop, str) else list(req_stop)
    return list(set(stop)) or None


def _max_new_tokens(body: dict, prompt_ids) -> int | None:
    mt = body.get("max_completion_tokens") or body.get("max_tokens")
    if mt is None:
        return None
    # Cap against cache capacity so an oversized request degrades to a
    # shorter answer instead of failing job validation.
    try:
        prompt_len = int(prompt_ids.shape[1]) if hasattr(prompt_ids, "shape") \
            else len(prompt_ids)
    except Exception:
        prompt_len = 0
    room = max(1, engine.cache_size - prompt_len)
    return max(1, min(int(mt), room))


async def run_job(body: dict):
    """Yield (text, eos_reason_or_None, usage_or_None) chunks for one request.

    ``usage`` is populated only on the final chunk, from the generator's own
    timing stats: prompt/completion token counts plus prefill/generate
    seconds (exllamav3 measures these server-side, wall-clock-accurate).
    """
    if body.get("prompt") is not None:  # /v1/completions
        input_ids = engine.tokenizer.encode(
            body["prompt"], encode_special_tokens=True
        )
    else:
        input_ids = _prompt_from_chat(body)
    prompt_tokens = int(input_ids.shape[1]) if hasattr(input_ids, "shape") \
        else len(input_ids)

    job = AsyncJob(
        engine.generator,
        input_ids=input_ids,
        max_new_tokens=_max_new_tokens(body, input_ids),
        sampler=build_sampler(body),
        seed=body.get("seed"),
        stop_conditions=_stop_conditions(body),
        decode_special_tokens=True,
    )
    try:
        async for result in job:
            if result.get("stage") == "streaming":
                eos = result.get("eos")
                eos_reason = result.get("eos_reason") if eos else None
                usage = None
                if eos:
                    new_tokens = result.get("new_tokens") or 0
                    t_prefill = result.get("time_prefill") or 0.0
                    t_gen = result.get("time_generate") or 0.0
                    t_queue = result.get("time_enqueued") or 0.0
                    # llama.cpp-style timing lines — same format llama-server
                    # prints ("slot print_timing"), so llamaswap's logs look
                    # identical for exl3 and llama.cpp backends.
                    serial = result.get("serial")
                    prefill_tps = prompt_tokens / t_prefill if t_prefill else 0.0
                    gen_tps = new_tokens / t_gen if t_gen else 0.0
                    log(f"slot print_timing: id {serial} | task 0 |    "
                        f"prompt eval time = {t_prefill * 1000:9.2f} ms /"
                        f" {prompt_tokens:6d} tokens ({prefill_tps:8.2f} tokens per second)")
                    log(f"slot print_timing: id {serial} | task 0 |    "
                        f" eval time = {t_gen * 1000:9.2f} ms /"
                        f" {new_tokens:6d} tokens ({gen_tps:8.2f} tokens per second)")
                    log(f"slot print_timing: id {serial} | task 0 |   "
                        f" total time = {(t_queue + t_prefill + t_gen) * 1000:9.2f} ms /"
                        f" {prompt_tokens + new_tokens:6d} tokens")
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": new_tokens,
                        "total_tokens": prompt_tokens + new_tokens,
                        "prompt_tokens_per_sec":
                            prompt_tokens / t_prefill if t_prefill else None,
                        "completion_tokens_per_sec":
                            new_tokens / t_gen if t_gen else None,
                        "timings": {
                            "prompt_ms": round(t_prefill * 1000),
                            "first_token_ms":
                                round((t_queue + t_prefill) * 1000),
                            "generation_ms": round(t_gen * 1000),
                        },
                    }
                yield result.get("text") or "", eos_reason, usage
    except BaseException:
        # Client disconnect / cancellation: release cache pages.
        try:
            await engine.generator.cancel(job)
        except Exception:
            pass
        raise


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _generate(body: dict, request: Request):
    stream = bool(body.get("stream"))
    chunk_id = f"cmpl-exl3-{int(time.time() * 1e6)}"
    created = int(time.time())
    common = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": body.get("model") or engine.model_name,
    }

    if not stream:
        parts: list[str] = []
        eos_reason = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0}
        try:
            async for text, reason, u in run_job(body):
                parts.append(text)
                if reason:
                    eos_reason = reason
                if u:
                    usage = u
        except ValueError as e:
            return oai_error(400, str(e))
        except TemplateError as e:
            return oai_error(400, f"chat template error: {e}")
        except Exception as e:
            log(f"request failed: {e!r}")
            return oai_error(500, str(e))
        content = "".join(parts)
        message: dict = {"role": "assistant", "content": content}
        finish_reason = "length" if eos_reason == "max_new_tokens" else "stop"
        # With thinking enabled the generation starts inside the thinking
        # block: split reasoning out of the visible content. No closing tag
        # (usually truncation) means everything is reasoning.
        reasoning_content = None
        if _thinking_enabled(body):
            stripped = content.lstrip()
            if stripped.startswith(_THINK_OPEN):
                content = stripped[len(_THINK_OPEN):]
            i = content.find(_THINK_CLOSE)
            if i != -1:
                reasoning_content = content[:i].strip()
                content = content[i + len(_THINK_CLOSE):].lstrip("\n")
            else:
                reasoning_content, content = content, ""
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
        message["content"] = content
        # Tool calls arrive as markup in the generated text (Qwen format):
        # lift them into message.tool_calls and strip the markup from the
        # visible content.
        tool_calls = [
            _oai_tool_call(i, parsed)
            for i, parsed in enumerate(
                p for p in (
                    _parse_tool_call_payload(m)
                    for m in TOOL_CALL_RE.findall(content)
                ) if p
            )
        ]
        if tool_calls:
            message["content"] = TOOL_CALL_RE.sub("", content).strip()
            message["tool_calls"] = tool_calls
            if finish_reason == "stop":
                finish_reason = "tool_calls"
        return {
            "id": chunk_id,
            "object": "chat.completion",
            "created": created,
            "model": common["model"],
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    async def sse():
        eos_reason = None
        # Tool-call streaming state: in "content" mode deltas are emitted
        # while holding back any suffix that could be the start of the tool
        # call opening tag (which may arrive split across chunks); once the
        # tag appears, "tools" mode buffers until each closing tag completes
        # a call, which is emitted as an OpenAI tool_calls delta immediately.
        mode = "content"
        carry = ""   # content mode: held-back partial opening tag
        buf = ""     # tools mode: accumulation between/inside call blocks
        n_tool_calls = 0
        # Reasoning phase: with thinking enabled the generation starts
        # inside the think block; stream reasoning deltas until the
        # closing tag, then switch to content/tool-call handling.
        phase = "think" if _thinking_enabled(body) else "content"
        think_seen = False  # any reasoning emitted yet (for final flush)
        # True right after the think block closes: strip the blank lines
        # between reasoning and the answer even when they arrive as
        # separate chunks after the transition.
        strip_lead_nl = _thinking_enabled(body)

        def feed_think(text: str) -> list[str]:
            nonlocal phase, carry, think_seen
            data = carry + text
            if not think_seen:
                # Safety: skip a literal opening tag if the model emits one
                # (normally the template pre-fills it into the prompt).
                ls = data.lstrip()
                if ls.startswith(_THINK_OPEN):
                    data = ls[len(_THINK_OPEN):]
            i = data.find(_THINK_CLOSE)
            if i == -1:
                # Hold back a suffix that could be the closing tag split
                # across chunks.
                hold = 0
                for n in range(min(len(data), len(_THINK_CLOSE) - 1), 0, -1):
                    if data.endswith(_THINK_CLOSE[:n]):
                        hold = n
                        break
                carry = data[len(data) - hold:]
                emit = data[:len(data) - hold]
                think_seen = think_seen or bool(emit)
                return [chunk({"reasoning_content": emit})] if emit else []
            carry = ""
            phase = "content"
            reasoning = data[:i]
            rest = data[i + len(_THINK_CLOSE):].lstrip("\n")
            out = [chunk({"reasoning_content": reasoning})] if reasoning else []
            return out + feed_content(rest)

        def feed(text: str) -> list[str]:
            if phase == "think":
                return feed_think(text)
            return feed_content(text)

        def feed_content(text: str) -> list[str]:
            nonlocal mode, carry, buf, strip_lead_nl
            if mode == "tools":
                return feed_tools(text)
            data = carry + text
            if strip_lead_nl:
                # Post-think blank lines are cosmetic; drop them wherever
                # they surface (same buffer as the close tag or later).
                data = data.lstrip("\n")
                if not data:
                    return []
                strip_lead_nl = False
            i = data.find(_TOOL_OPEN)
            if i == -1:
                # Hold back a suffix that could be the opening tag split
                # across chunks; emit everything before it.
                hold = 0
                for n in range(min(len(data), len(_TOOL_OPEN) - 1), 0, -1):
                    if data.endswith(_TOOL_OPEN[:n]):
                        hold = n
                        break
                carry = data[len(data) - hold:]
                emit = data[:len(data) - hold]
                return [chunk({"content": emit})] if emit else []
            mode = "tools"
            buf = data[i:]
            rest = data[:i]
            return ([chunk({"content": rest})] if rest else []) \
                + feed_tools("")

        def chunk(delta: dict, finish: str | None = None) -> str:
            return _sse({**common, "choices": [{
                "index": 0, "delta": delta, "finish_reason": finish,
            }]})

        def feed_tools(text: str) -> list[str]:
            nonlocal n_tool_calls, buf
            buf += text
            out: list[str] = []
            while True:
                start = buf.find(_TOOL_OPEN)
                end = buf.find(_TOOL_CLOSE)
                if start != -1 and (end == -1 or start < end):
                    # Skip anything before an opening tag (the whitespace
                    # between back-to-back calls) and the tag itself.
                    buf = buf[start + len(_TOOL_OPEN):]
                    continue
                if end == -1:
                    break  # inside a call, waiting for the closing tag
                payload = buf[:end].strip()
                buf = buf[end + len(_TOOL_CLOSE):]
                parsed = _parse_tool_call_payload(payload)
                if parsed:
                    out.append(chunk({"tool_calls": [
                        _oai_tool_call(n_tool_calls, parsed)]}))
                    n_tool_calls += 1
                else:
                    log(f"unparseable tool call: {payload[:120]}")
            return out

        try:
            async for text, reason, u in run_job(body):
                if reason:
                    eos_reason = reason
                if text:
                    for piece in feed(text):
                        yield piece
        except Exception as e:
            # Never let an exception escape mid-stream: the connection would
            # be dropped without a complete body (client-visible 502).
            log(f"request failed: {e!r}")
            yield _sse({"error": {"message": str(e),
                                  "type": "server_error"}})
            yield "data: [DONE]\n\n"
            return
        if phase == "think":
            # Truncated mid-thinking: surface the tail as reasoning.
            if carry:
                yield chunk({"reasoning_content": carry})
        elif mode == "content" and carry:
            yield chunk({"content": carry})
        elif mode == "tools" and buf.strip():
            if n_tool_calls == 0:
                # A block that never closed (usually truncation): surface it
                # as text rather than dropping it silently.
                yield chunk({"content": buf.strip()})
            else:
                log(f"unconsumed tail after tool call blocks: "
                    f"{buf.strip()[:80]}")
        finish_reason = "length" if eos_reason == "max_new_tokens" else "stop"
        if n_tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        yield chunk({}, finish_reason)
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await _generate(body, request)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    if not body.get("prompt"):
        return oai_error(400, "'prompt' is required")
    body["prompt"] = body["prompt"]
    # /v1/completions returns 'text' not 'content'; handled by callers that
    # use this endpoint — keep it simple by reusing the chat path and
    # converting at the end.
    result = await _generate(body, request)
    if isinstance(result, dict):
        for ch in result.get("choices", []):
            if "message" in ch:
                ch["text"] = ch["message"]["content"]
                ch.pop("message", None)
        result["object"] = "text_completion"
    return result


def main() -> None:
    global engine
    args = build_parser().parse_args()
    engine = Engine(args)
    engine.load()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
