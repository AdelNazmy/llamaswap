"""Lives/monitors the single llama-server subprocess.

Only one model is loaded at a time: ensure_model() swaps the running
server when a different model is requested.
"""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import httpx

from .registry import ModelConfig, UnknownModelError

logger = logging.getLogger("llamaswap.process_manager")


class ModelState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class RunningModel:
    name: str
    config: ModelConfig
    process: asyncio.subprocess.Process
    state: ModelState
    detail: str = ""


class ModelLoadError(Exception):
    """Raised when a model could not be loaded in time."""


class ProcessManager:
    """Starts, monitors and swaps the llama-server subprocess."""

    def __init__(self, startup_timeout: float, stop_timeout: float,
                 health_interval: float, idle_unload_seconds: float = 0.0):
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._health_interval = health_interval
        self._idle_unload_seconds = idle_unload_seconds
        self._lock = asyncio.Lock()
        self._current: Optional[RunningModel] = None
        self._last_activity: float = 0.0
        self._idle_task: Optional[asyncio.Task] = None
        if idle_unload_seconds > 0:
            self._idle_task = asyncio.get_running_loop().create_task(
                self._idle_watcher()
            )

    @property
    def current_name(self) -> Optional[str]:
        cur = self._current
        return cur.name if cur and cur.state is ModelState.READY else None

    def status(self) -> dict[str, Any]:
        cur = self._current
        if cur is None:
            return {"state": "stopped", "model": None, "detail": ""}
        info: dict[str, Any] = {
            "state": cur.state.value, "model": cur.name, "detail": cur.detail,
        }
        if cur.state is ModelState.READY and self._idle_unload_seconds > 0:
            elapsed = asyncio.get_running_loop().time() - self._last_activity
            info["idle_seconds"] = max(0.0, round(elapsed, 1))
            info["idle_unload_seconds"] = self._idle_unload_seconds
        return info

    def _health_url(self, cfg: ModelConfig) -> str:
        return f"http://{cfg.host}:{cfg.port}/health"

    def _stderr_line(self, entry: "RunningModel", chunk: bytes) -> None:
        for line in chunk.decode(errors="replace").splitlines():
            line = line.strip()
            if line:
                logger.info("llama-server[%s]: %s", entry.name, line)

    async def _wait_healthy(self, cfg: ModelConfig,
                            proc: asyncio.subprocess.Process) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if proc.returncode is not None:
                    raise ModelLoadError(
                        f"llama-server for '{cfg.name}' exited (code "
                        f"{proc.returncode}) during startup; see logs above"
                    )
                try:
                    r = await client.get(self._health_url(cfg))
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() > deadline:
                    raise ModelLoadError(
                        f"llama-server for '{cfg.name}' did not become "
                        f"healthy within {self._startup_timeout:.0f}s"
                    )
                await asyncio.sleep(self._health_interval)

    async def _stop_current(self) -> None:
        cur = self._current
        self._current = None
        if cur is None or cur.process.returncode is not None:
            return
        proc = cur.process
        logger.info("stopping llama-server for '%s' (pid %s)", cur.name, proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout)
        except asyncio.TimeoutError:
            logger.warning("llama-server for '%s' did not exit, killing", cur.name)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        logger.info("llama-server for '%s' stopped", cur.name)

    async def ensure_model(self, name: str,
                           registry: "Registry") -> tuple[str, int]:
        """Make sure `name` is the loaded model; returns (name, port)."""
        cfg = registry.get(name)  # raises UnknownModelError
        async with self._lock:
            self._last_activity = asyncio.get_running_loop().time()
            cur = self._current
            if cur is not None and cur.name == name and cur.state is ModelState.READY:
                return name, cfg.port
            if cur is not None:
                await self._stop_current()
            await self._launch(cfg)
            return name, cfg.port

    async def _launch(self, cfg: ModelConfig) -> None:
        argv = cfg.build_argv()
        logger.info(
            "launching llama-server for '%s': %s", cfg.name, " ".join(argv)
        )
        env = {**os.environ, **cfg.command.env}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ModelLoadError(f"failed to spawn llama-server: {exc}") from exc
        entry = RunningModel(name=cfg.name, config=cfg, process=proc,
                             state=ModelState.LOADING)
        self._current = entry
        # Drain stderr immediately so the pipe never fills up during the
        # (potentially long) model load.
        asyncio.get_running_loop().create_task(self._tail_stderr(entry))
        try:
            await self._wait_healthy(cfg, proc)
        except BaseException:
            await self._stop_current()
            raise
        entry.state = ModelState.READY
        entry.detail = ""
        logger.info("model '%s' ready on port %d", cfg.name, cfg.port)

    async def _tail_stderr(self, entry: RunningModel) -> None:
        assert entry.process.stderr is not None
        try:
            async for chunk in entry.process.stderr:
                if chunk:
                    self._stderr_line(entry, chunk)
        except asyncio.CancelledError:
            raise
        rc = await entry.process.wait()
        if entry.state is ModelState.READY and rc not in (0, -signal.SIGTERM):
            entry.state = ModelState.FAILED
            entry.detail = f"llama-server exited with code {rc}"
            logger.error(
                "llama-server for '%s' exited unexpectedly (code %s)",
                entry.name, rc,
            )

    async def _idle_watcher(self) -> None:
        """Unload the loaded model once it has been idle past the timeout."""
        try:
            while True:
                await asyncio.sleep(self._health_interval)
                if self._current is None:
                    continue
                elapsed = asyncio.get_running_loop().time() - self._last_activity
                if elapsed < self._idle_unload_seconds:
                    continue
                async with self._lock:
                    cur = self._current
                    if cur is None or cur.state is not ModelState.READY:
                        continue
                    elapsed = asyncio.get_running_loop().time() - \
                        self._last_activity
                    if elapsed < self._idle_unload_seconds:
                        continue
                    logger.info(
                        "unloading model '%s' after %.0fs idle",
                        cur.name, elapsed,
                    )
                    await self._stop_current()
        except asyncio.CancelledError:
            raise

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            await self._stop_current()
