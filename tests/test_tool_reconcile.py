import json

from app.tool_reconcile import (
    AMBIGUOUS,
    RECONCILED,
    UNMATCHED,
    StreamReconciler,
    apply_reconcile,
    match_name,
    text_tool_calls,
)


def chunk(delta, finish=None, extra=None):
    obj = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra:
        obj.update(extra)
    return obj


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


BLOCK = '<tool_call>{"name": "unity_read", "arguments": {"path": "/a/b"}}</tool_call>'


def test_match_name_prefix_drift():
    declared = ["unity_unity_list_instances"]
    assert match_name("unity_list_instances", declared) == (
        "unity_unity_list_instances", RECONCILED)
    assert match_name("unity_unity_list_instances", ["unity_list_instances"]) == (
        "unity_list_instances", RECONCILED)


def test_match_name_ambiguous_and_unmatched():
    declared = ["get_files", "list_get_files"]
    matched, outcome = match_name("files", declared)
    assert matched is None and outcome == AMBIGUOUS
    matched, outcome = match_name("totally_other_thing", declared)
    assert matched is None and outcome == UNMATCHED


def test_apply_reconcile_keeps_arguments_from_text():
    data = {
        "id": "x",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"Let me look.\n{BLOCK}",
            },
            "finish_reason": "stop",
        }],
        "strata_rejected_tool_calls": [{"name": "unity_read", "reason": "undeclared"}],
    }
    records = apply_reconcile(data, ["unity_unity_read"])
    assert records == [("unity_read", "unity_unity_read", RECONCILED)]
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "unity_unity_read"
    assert json.loads(call["function"]["arguments"]) == {"path": "/a/b"}


def test_apply_reconcile_text_fallback_without_rejected_list():
    data = {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": BLOCK},
            "finish_reason": "stop",
        }],
    }
    records = apply_reconcile(data, ["unity_unity_read"])
    assert records == [("unity_read", "unity_unity_read", RECONCILED)]
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"path": "/a/b"}


def test_apply_reconcile_no_duplicate_of_executed_call():
    data = {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": BLOCK,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "unity_unity_read", "arguments": "{}"},
                }],
            },
            "finish_reason": "stop",
        }],
        "strata_rejected_tool_calls": [{"name": "unity_read", "reason": "undeclared"}],
    }
    apply_reconcile(data, ["unity_unity_read"])
    assert len(data["choices"][0]["message"]["tool_calls"]) == 1


def test_stream_strata_shape_recovers_arguments():
    declared = ["unity_unity_read"]
    r = StreamReconciler(declared)
    out = r.feed(
        b": keep-alive\n\n"
        + sse(chunk({"content": "Let me look.\n"}))
        + sse(chunk({"content": BLOCK}))
        + sse(chunk({}, finish="stop", extra={
            "strata_rejected_tool_calls": [
                {"name": "unity_read", "reason": "undeclared"}]}))
        + b"data: [DONE]\n\n"
    )
    out += r.finish()
    text = out.decode()
    events = [json.loads(line[6:]) for line in text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    synthesized = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
    assert len(synthesized) == 1
    call = synthesized[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "unity_unity_read"
    assert json.loads(call["function"]["arguments"]) == {"path": "/a/b"}
    final = [e for e in events if e["choices"][0]["finish_reason"] is not None]
    assert len(final) == 1
    assert text.index("delta") < text.index('"finish_reason": "stop"') \
        or text.index(json.dumps(synthesized[0])) < text.rindex("finish_reason")
    assert r.records == [("unity_read", "unity_unity_read", RECONCILED)]


def test_stream_index_continues_after_backend_calls():
    declared = ["declared_tool", "unity_unity_read"]
    proper = chunk({}, extra={})
    proper["choices"][0]["delta"] = {
        "tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                        "function": {"name": "declared_tool", "arguments": "{}"}}]}
    r = StreamReconciler(declared)
    out = r.feed(
        sse(proper)
        + sse(chunk({"content": BLOCK}))
        + sse(chunk({}, finish="tool_calls", extra={
            "strata_rejected_tool_calls": [
                {"name": "unity_read", "reason": "undeclared"}]}))
        + b"data: [DONE]\n\n"
    )
    out += r.finish()
    events = [json.loads(line[6:]) for line in out.decode().splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    calls = [c for e in events for c in e["choices"][0]["delta"].get("tool_calls") or []]
    assert [c["index"] for c in calls] == [0, 1]
    assert calls[1]["function"]["name"] == "unity_unity_read"
    assert calls[1]["function"]["arguments"] == '{"path": "/a/b"}'
    assert r.records == [("unity_read", "unity_unity_read", RECONCILED)]


def test_stream_healthy_call_makes_no_records():
    declared = ["declared_tool"]
    proper = chunk({}, extra={})
    proper["choices"][0]["delta"] = {
        "tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                        "function": {"name": "declared_tool", "arguments": "{}"}}]}
    r = StreamReconciler(declared)
    out = r.feed(
        sse(chunk({"content": "working"}))
        + sse(proper)
        + sse(chunk({}, finish="tool_calls"))
        + b"data: [DONE]\n\n"
    )
    out += r.finish()
    assert b"tool_calls" in out
    assert r.records == []


def test_text_tool_calls_parses_json_block():
    calls = text_tool_calls(f"before {BLOCK} after")
    assert calls == [{"name": "unity_read", "arguments": '{"path": "/a/b"}'}]
