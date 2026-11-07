import json

from app.proxy import _trimmed_body, overflow_info, trim_messages

STRATA_NO_ROOM = (
    'request exceeds the context window: prompt (2500 tokens) leaves no room '
    'to answer in the context (3000); requests are never truncated'
)
STRATA_MAX = (
    'request exceeds the context window: prompt (7800 tokens) + max tokens '
    '(2000) exceeds the context (8192); requests are never truncated. Send a '
    'smaller max_tokens (at most 392 here)'
)
LLAMA_INPUT = (
    'input (2500 tokens) is larger than the max context size (3000 tokens). skipping'
)
LLAMA_REQUEST = (
    'request (2500 tokens) exceeds the available context size (3000 tokens), '
    'try increasing it'
)


def test_overflow_info_strata_variants():
    assert overflow_info(STRATA_NO_ROOM) == (2500, 3000)
    assert overflow_info(STRATA_MAX) == (7800, 8192)


def test_overflow_info_llama_variants():
    assert overflow_info(LLAMA_INPUT) == (2500, 3000)
    assert overflow_info(LLAMA_REQUEST) == (2500, 3000)


def test_overflow_info_json_and_bytes():
    body = json.dumps({"error": {"type": "invalid_request_error",
                                 "message": STRATA_NO_ROOM}})
    assert overflow_info(body) == (2500, 3000)
    assert overflow_info(body.encode()) == (2500, 3000)
    assert overflow_info("some other error") is None


def long_text(n):
    return "x" * n


def body_with_history():
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": long_text(4000)},
            {"role": "assistant", "tool_calls": [
                {"id": "1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "content": long_text(1600)},
            {"role": "user", "content": "LIVE"},
        ],
    }


def test_trim_messages_pins_system_and_last():
    body, n = trim_messages(body_with_history(), 2500, 3000, 1024)
    assert n == 1
    msgs = body["messages"]
    assert msgs[0]["content"] == "SYSTEM"
    assert "dropped 1 earlier messages" in msgs[1]["content"]
    assert msgs[-1]["content"] == "LIVE"
    assert msgs[2]["tool_calls"] and msgs[3]["role"] == "tool"


def test_trim_messages_drops_tool_turn_atomically():
    body = body_with_history()
    body["messages"][1]["content"] = "u"
    body, n = trim_messages(body, 2500, 3000, 1024)
    msgs = body["messages"]
    assert n == 3
    assert msgs[0]["content"] == "SYSTEM"
    assert msgs[-1]["content"] == "LIVE"
    assert not any(m.get("role") == "tool" for m in msgs)
    assert not any(m.get("tool_calls") for m in msgs)


def test_trim_messages_nothing_to_drop():
    assert trim_messages({"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
    ]}, 2500, 3000, 1024) is None
    assert trim_messages(body_with_history(), 2500, 30000, 1024) is None
    assert trim_messages({"messages": []}, 2500, 3000, 1024) is None
    assert trim_messages({}, 2500, 3000, 1024) is None


def test_trimmed_body_end_to_end_strata():
    body = json.dumps(body_with_history()).encode()
    err = json.dumps({"error": {"type": "invalid_request_error",
                                "message": STRATA_NO_ROOM}}).encode()
    out = _trimmed_body(body, err, "m")
    assert out is not None
    msgs = json.loads(out)["messages"]
    assert msgs[0]["content"] == "SYSTEM"
    assert msgs[-1]["content"] == "LIVE"


def test_trimmed_body_end_to_end_llama():
    body = json.dumps(body_with_history()).encode()
    out = _trimmed_body(body, LLAMA_INPUT.encode(), "m")
    assert out is not None
    assert "dropped" in json.loads(out)["messages"][1]["content"]


def test_trimmed_body_respects_request_max_tokens():
    body_obj = body_with_history()
    body_obj["max_tokens"] = 512
    body = json.dumps(body_obj).encode()
    out = _trimmed_body(body, STRATA_MAX.encode(), "m")
    assert out is not None


def test_trimmed_body_ignores_other_errors_and_disabled():
    body = json.dumps(body_with_history()).encode()
    assert _trimmed_body(body, b"model is loading", "m") is None
    import app.proxy as proxy
    orig = proxy.get_settings
    try:
        proxy.get_settings = lambda: type("S", (), {"context_trim": False})()
        assert _trimmed_body(body, STRATA_NO_ROOM.encode(), "m") is None
    finally:
        proxy.get_settings = orig
