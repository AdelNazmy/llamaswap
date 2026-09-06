"""Minimal in-process pub/sub for the ``GET /v1/events`` SSE stream.

Managers emit lifecycle events (model_loaded / model_unloaded / swap /
load_failed); the SSE endpoint subscribes with a bounded queue so a slow
client can never grow unbounded memory — the oldest event is dropped first.
"""

import asyncio
import json
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def emit(self, event: str, **data: Any) -> None:
        payload = json.dumps({"event": event, "data": data}, default=str)
        for q in tuple(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest event, keep the newest.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass


# Process-wide singleton.
bus = EventBus()
