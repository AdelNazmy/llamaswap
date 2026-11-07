"""Prometheus exposition completeness of the rendered /metrics text.

Complements test_token_meter.py (which covers the session/install token
labeling): every sample line here must belong to a family that declared
both a # HELP and a # TYPE line first — what scrapers and promtool expect.
"""

from app.metrics import Metrics


def _parse(text: str):
    types: dict[str, str] = {}
    helps: set[str] = set()
    samples: list[str] = []
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ", 3)
            types[name] = kind
        elif line.startswith("# HELP "):
            helps.add(line.split(" ", 2)[2].split(" ", 1)[0])
        elif line.startswith("#"):
            continue
        else:
            samples.append(line.split("{", 1)[0].split(" ", 1)[0])
    return types, helps, samples


def _populated_metrics() -> Metrics:
    m = Metrics()
    m.observe_load("llm", "m", 1.5)
    m.inc_load_failure("llm", "m2")
    m.inc_unload("llm", "m")
    m.inc_swap("llm", "a", "b")
    m.inc_request("GET", "/health", 200)
    m.inc_tool_call_reconcile("m", "reconciled")
    m.inc_context_trim("m", "trimmed")
    m.inc_tokens("m", "input", 10)
    m.token_baseline[("m", "output")] = 7
    m.set_gauge("llamaswap_registered_models", 2.0)
    m.set_gauge("llamaswap_status", 0.0)
    return m


def test_every_sample_family_has_type_and_help():
    text = _populated_metrics().render()
    types, helps, samples = _parse(text)
    for name in set(samples):
        assert name in types, f"{name} sampled without a # TYPE line"
        assert name in helps, f"{name} sampled without a # HELP line"


def test_gauges_and_uptime_declared():
    text = _populated_metrics().render()
    types, _, _ = _parse(text)
    assert types["llamaswap_uptime_seconds"] == "gauge"
    assert types["llamaswap_registered_models"] == "gauge"
    assert types["llamaswap_status"] == "gauge"
    assert "# HELP llamaswap_uptime_seconds" in text
    assert "# HELP llamaswap_registered_models" in text
    assert "# HELP llamaswap_status" in text


def test_unknown_gauge_still_renders():
    m = Metrics()
    m.set_gauge("llamaswap_something_new", 1.0)
    text = m.render()
    assert "# TYPE llamaswap_something_new gauge" in text
    assert "llamaswap_something_new 1.0" in text


def test_session_and_install_series_coexist():
    m = Metrics()
    m.token_baseline[("m", "input")] = 500
    m.inc_tokens("m", "input", 20)
    lines = m.render().splitlines()
    assert ('llamaswap_tokens_session_total{model="m",direction="input"} 20') in lines
    assert ('llamaswap_tokens_total{model="m",direction="input"} 520') in lines
