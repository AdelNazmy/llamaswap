"""In-process Prometheus metrics for llamaswap.

No external client: counters live in memory (single event loop, so plain
dict increments are safe) and are rendered on demand as the Prometheus text
exposition format for ``GET /metrics``.
"""

import time
from collections import defaultdict
from typing import DefaultDict, Tuple


def _label(*parts: str) -> str:
    return "{" + ",".join(parts) + "}"


# HELP text for the dynamic gauges set via set_gauge() (main.py); a gauge
# without an entry still renders, just without a HELP line.
_GAUGE_HELP = {
    "llamaswap_registered_models": "Models currently in the registry.",
    "llamaswap_status": "1.0 when the registry or any backend manager is degraded, 0.0 otherwise.",
}


class Metrics:
    """Thread-unsafe (event-loop-only) metrics collection."""

    def __init__(self) -> None:
        self._started = time.time()
        # (role, model) -> count / sum / count
        self.loads: DefaultDict[Tuple[str, str], float] = defaultdict(float)
        self.load_seconds: DefaultDict[Tuple[str, str], float] = defaultdict(float)
        self.load_seconds_count: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.load_failures: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.unloads: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        # role -> count
        self.swaps: DefaultDict[str, int] = defaultdict(int)
        # (method, route, status) -> count
        self.requests: DefaultDict[Tuple[str, str, int], int] = defaultdict(int)
        # (model, outcome) -> count for recovered/undeclared tool calls
        self.tool_calls: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        # (model, outcome) -> count for context-window overflow trims
        self.context_trims: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        # (model, direction) -> tokens this process ("session" scope;
        # resets on restart). Install-lifetime totals are rendered on top
        # of token_baseline, loaded from the SQLite store at startup.
        self.tokens: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.token_baseline: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        # gauges (set externally by main.py)
        self.gauges: dict[str, float] = {}

    # -- event counters ---------------------------------------------------

    def observe_load(self, role: str, model: str, seconds: float) -> None:
        key = (role, model)
        self.loads[key] += 1
        self.load_seconds[key] += seconds
        self.load_seconds_count[key] += 1

    def inc_load_failure(self, role: str, model: str) -> None:
        self.load_failures[(role, model)] += 1

    def inc_unload(self, role: str, model: str) -> None:
        self.unloads[(role, model)] += 1

    def inc_swap(self, role: str, from_model: str, to_model: str) -> None:
        self.swaps[role] += 1

    def inc_request(self, method: str, route: str, status: int) -> None:
        self.requests[(method, route, status)] += 1

    def inc_tool_call_reconcile(self, model: str, outcome: str) -> None:
        self.tool_calls[(model, outcome)] += 1

    def inc_context_trim(self, model: str, outcome: str) -> None:
        self.context_trims[(model, outcome)] += 1

    def inc_tokens(self, model: str, direction: str, tokens: int) -> None:
        if tokens > 0:
            self.tokens[(model, direction)] += tokens

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        out: list[str] = []
        family = []  # type: list[str]

        def flush(name: str, help_: str, type_: str, lines: list[str]) -> None:
            if not lines:
                return
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {type_}")
            out.extend(lines)

        family = [
            f'llamaswap_model_loads_total{{role="{r}",model="{m}"}} {int(c)}'
            for (r, m), c in sorted(self.loads.items())
        ]
        flush("llamaswap_model_loads_total",
              "Successful backend server loads.", "counter", family)

        family = [
            f'llamaswap_model_load_duration_seconds_total{{role="{r}",model="{m}"}} {s:.3f}'
            for (r, m), s in sorted(self.load_seconds.items())
        ]
        flush("llamaswap_model_load_duration_seconds_total",
              "Total time spent loading backend servers.", "counter", family)

        family = [
            f'llamaswap_model_load_failures_total{{role="{r}",model="{m}"}} {c}'
            for (r, m), c in sorted(self.load_failures.items())
        ]
        flush("llamaswap_model_load_failures_total",
              "Backend server load failures.", "counter", family)

        family = [
            f'llamaswap_model_unloads_total{{role="{r}",model="{m}"}} {c}'
            for (r, m), c in sorted(self.unloads.items())
        ]
        flush("llamaswap_model_unloads_total",
              "Backend server unloads (swap / idle / explicit).", "counter", family)

        family = [
            f'llamaswap_model_swaps_total{{role="{r}"}} {c}'
            for r, c in sorted(self.swaps.items())
        ]
        flush("llamaswap_model_swaps_total",
              "Model swaps within a single role's manager.", "counter", family)

        family = [
            f'llamaswap_http_requests_total{{method="{m}",route="{r}",status="{s}"}} {c}'
            for (m, r, s), c in sorted(self.requests.items())
        ]
        flush("llamaswap_http_requests_total",
              "HTTP requests received by the proxy.", "counter", family)

        family = [
            f'llamaswap_tool_call_reconciliations_total{{model="{m}",outcome="{o}"}} {c}'
            for (m, o), c in sorted(self.tool_calls.items())
        ]
        flush("llamaswap_tool_call_reconciliations_total",
              "Tool calls the backend refused: outcome=reconciled were re-emitted, "
              "ambiguous/unmatched were left to the agent.", "counter", family)

        family = [
            f'llamaswap_context_trims_total{{model="{m}",outcome="{o}"}} {c}'
            for (m, o), c in sorted(self.context_trims.items())
        ]
        flush("llamaswap_context_trims_total",
              "Chat requests retried after an upstream context-window overflow: "
              "outcome=trimmed dropped the oldest turns from the history.",
              "counter", family)

        family = [
            f'llamaswap_tokens_session_total{{model="{m}",direction="{d}"}} {c}'
            for (m, d), c in sorted(self.tokens.items()) if c > 0
        ]
        flush("llamaswap_tokens_session_total",
              "Chat/completion tokens proxied since proxy start (resets on "
              "restart): direction=input (prompt) or output (completion).",
              "counter", family)

        combined: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        for key, value in self.token_baseline.items():
            combined[key] += value
        for key, value in self.tokens.items():
            combined[key] += value
        family = [
            f'llamaswap_tokens_total{{model="{m}",direction="{d}"}} {c}'
            for (m, d), c in sorted(combined.items()) if c > 0
        ]
        flush("llamaswap_tokens_total",
              "Chat/completion tokens proxied since installation (persisted "
              "in SQLite; includes the current session): direction=input "
              "(prompt) or output (completion).",
              "counter", family)

        # gauges (set externally by main.py)
        for name, value in sorted(self.gauges.items()):
            help_ = _GAUGE_HELP.get(name)
            if help_:
                out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} gauge")
            out.append(f"{name} {value}")

        flush(
            "llamaswap_uptime_seconds",
            "Seconds since the proxy process started.",
            "gauge",
            [f"llamaswap_uptime_seconds {time.time() - self._started:.1f}"],
        )
        return "\n".join(out) + "\n"


# Process-wide singleton.
metrics = Metrics()
