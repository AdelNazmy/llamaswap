import json

import pytest

from app import proxy
from app.metrics import Metrics
from app.proxy import StreamTokenMeter, _count_response_tokens


class FakeStore:
    def __init__(self):
        self.added = []

    def add(self, model, direction, tokens):
        self.added.append((model, direction, tokens))


@pytest.fixture
def recorded(monkeypatch):
    m = Metrics()
    store = FakeStore()
    monkeypatch.setattr(proxy, "metrics", m)
    monkeypatch.setattr(proxy, "token_store", store)
    return m, store


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def chunk(delta, finish=None, usage=None):
    obj = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "model": "backend-name",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


CHAT_BODY = json.dumps({
    "model": "m",
    "messages": [{"role": "user", "content": "hi"}],
}).encode()


def test_meter_exact_usage_from_final_chunk(recorded):
    m, store = recorded
    meter = StreamTokenMeter("m", CHAT_BODY)
    meter.feed(
        sse(chunk({"content": "hello "}))
        + sse(chunk({"content": "world"}))
        + sse(chunk({}, finish="stop", usage={
            "prompt_tokens": 1000, "completion_tokens": 200}))
        + b"data: [DONE]\n\n"
    )
    meter.finish()
    assert m.tokens == {("m", "input"): 1000, ("m", "output"): 200}
    assert ("m", "input", 1000) in store.added
    assert ("m", "output", 200) in store.added


def test_meter_estimates_without_usage(recorded):
    m, _ = recorded
    body = json.dumps({
        "messages": [{"role": "user", "content": "word " * 400}],
    }).encode()
    meter = StreamTokenMeter("m", body)
    meter.feed(sse(chunk({"content": "y" * 400})) + b"data: [DONE]\n\n")
    meter.finish()
    inp = m.tokens[("m", "input")]
    out = m.tokens[("m", "output")]
    assert 400 <= inp <= 600
    assert 90 <= out <= 120


def test_meter_buffers_lines_across_chunks(recorded):
    m, _ = recorded
    meter = StreamTokenMeter("m", CHAT_BODY)
    event = sse(chunk({}, finish="stop", usage={
        "prompt_tokens": 42, "completion_tokens": 7}))
    cut = event.index(b"{") + 10
    meter.feed(b"data: " + event[6:cut])
    meter.feed(event[cut:] + b"data: [DONE]\n\n")
    meter.finish()
    assert m.tokens[("m", "input")] == 42
    assert m.tokens[("m", "output")] == 7


def test_meter_counts_tool_call_arguments(recorded):
    m, _ = recorded
    meter = StreamTokenMeter("m", CHAT_BODY)
    delta = {"tool_calls": [{"index": 0, "function": {
        "name": "f", "arguments": "x" * 400}}]}
    meter.feed(sse(chunk(delta)) + b"data: [DONE]\n\n")
    meter.finish()
    assert 90 <= m.tokens[("m", "output")] <= 120


def test_non_streaming_exact_usage(recorded):
    m, store = recorded
    data = {
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 33, "completion_tokens": 4},
    }
    _count_response_tokens("m", CHAT_BODY, data)
    assert m.tokens == {("m", "input"): 33, ("m", "output"): 4}
    assert store.added == [("m", "input", 33), ("m", "output", 4)]


def test_non_streaming_estimate_fallback(recorded):
    m, _ = recorded
    data = {"choices": [{"message": {"role": "assistant",
                                     "content": "z" * 400}}]}
    _count_response_tokens("m", CHAT_BODY, data)
    assert m.tokens[("m", "input")] > 0
    assert 90 <= m.tokens[("m", "output")] <= 200


def test_non_chat_bodies_not_counted(recorded):
    m, store = recorded
    embed_body = json.dumps({"model": "m", "input": "hello"}).encode()
    _count_response_tokens("m", embed_body, {
        "usage": {"prompt_tokens": 10, "completion_tokens": 0}})
    _count_response_tokens("m", b"not json at all", {
        "usage": {"prompt_tokens": 10}})
    _count_response_tokens("m", CHAT_BODY, "plain text")
    assert m.tokens == {}
    assert store.added == []


def test_completion_prompt_body_counted(recorded):
    m, _ = recorded
    body = json.dumps({"model": "m", "prompt": "word " * 100}).encode()
    data = {"choices": [{"text": "ok"}]}
    _count_response_tokens("m", body, data)
    assert 80 <= m.tokens[("m", "input")] <= 140
    assert m.tokens[("m", "output")] > 0


def test_render_session_and_install_metrics():
    m = Metrics()
    m.token_baseline[("m", "input")] = 500
    m.inc_tokens("m", "input", 20)
    m.inc_tokens("m", "output", 5)
    text = m.render()
    session = {}
    install = {}
    for line in text.splitlines():
        if line.startswith("llamaswap_tokens_session_total{"):
            label, value = line.rsplit(" ", 1)
            direction = "input" if 'direction="input"' in label else "output"
            session[direction] = int(value)
        elif line.startswith("llamaswap_tokens_total{"):
            label, value = line.rsplit(" ", 1)
            direction = "input" if 'direction="input"' in label else "output"
            install[direction] = int(value)
    assert session == {"input": 20, "output": 5}
    assert install == {"input": 520, "output": 5}
    assert "# TYPE llamaswap_tokens_total counter" in text
    assert "# TYPE llamaswap_tokens_session_total counter" in text
