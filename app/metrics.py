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

        for name, value in sorted(self.gauges.items()):
            out.append(f"# TYPE {name} gauge")
            out.append(f"{name} {value}")

        out.append(f"llamaswap_uptime_seconds {time.time() - self._started:.1f}")
        return "\n".join(out) + "\n"


# Process-wide singleton.
metrics = Metrics()
