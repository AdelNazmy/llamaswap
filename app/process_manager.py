"""Lives/monitors the single chat LLM subprocess.

Only one model is loaded at a time: ensure_model() swaps the running
server when a different model is requested. The backend is normally
llama-server, but any engine that speaks the OpenAI protocol and takes
``--host``/``--port`` works (e.g. Strata's ``strata serve``); backends
launched from a structured config file declare ``command.config_json``,
which this manager renders into ``config_dir`` before launch — the same
mechanism RoleServerManager uses for the audio/image tenants.
"""

import asyncio
import json
import logging
import os
import re
import signal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from .events import bus
from .metrics import metrics
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
    config_file: Optional[str] = None


class ModelLoadError(Exception):
    """Raised when a model could not be loaded in time."""


class ProcessManager:
    """Starts, monitors and swaps the llama-server subprocess."""

    def __init__(self, startup_timeout: float, stop_timeout: float,
                 health_interval: float, idle_unload_seconds: float = 0.0,
                 config_dir: str | Path = "/tmp/llamaswap-audio"):
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._health_interval = health_interval
        self._idle_unload_seconds = idle_unload_seconds
        self._config_dir = Path(config_dir)
        self._lock = asyncio.Lock()
        self._current: Optional[RunningModel] = None
        self._last_activity: float = 0.0
        self._idle_task: Optional[asyncio.Task] = None
        # Timestamp until which idle unload is suppressed (X-Pin-Seconds).
        self._pinned_until: float = 0.0
        # The watcher always runs: a model may override the idle window via
        # ``meta.idle_unload_seconds`` even when the global one is 0 (a
        # threshold of 0 for the *loaded* model still means "never unload").
        self._idle_task = asyncio.get_running_loop().create_task(
            self._idle_watcher()
        )

    @property
    def current_name(self) -> Optional[str]:
        cur = self._current
        return cur.name if cur and cur.state is ModelState.READY else None

    def _idle_threshold(self, cfg: Optional[ModelConfig] = None) -> float:
        """Idle window in force: the loaded model's ``meta.idle_unload_seconds``
        override when it declares one, else the manager's global value. 0
        disables idle unload for that model."""
        if cfg is not None and cfg.meta.idle_unload_seconds is not None:
            return cfg.meta.idle_unload_seconds
        return self._idle_unload_seconds

    def status(self) -> dict[str, Any]:
        cur = self._current
        if cur is None:
            return {"state": "stopped", "model": None, "detail": ""}
        info: dict[str, Any] = {
            "state": cur.state.value, "model": cur.name, "detail": cur.detail,
        }
        threshold = self._idle_threshold(cur.config)
        if cur.state is ModelState.READY and threshold > 0:
            elapsed = asyncio.get_running_loop().time() - self._last_activity
            info["idle_seconds"] = max(0.0, round(elapsed, 1))
            info["idle_unload_seconds"] = threshold
        return info

    def _health_url(self, cfg: ModelConfig) -> str:
        return cfg.health_url()

    def _render_config(self, cfg: ModelConfig) -> Optional[str]:
        """Write the per-model config JSON for a structured-config backend
        (Strata's ``strata serve --profile``), if the YAML declares one."""
        if not cfg.command.config_json:
            return None
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # The file stem doubles as the backend's profile name, and strata
        # only accepts [a-z0-9_-] there ("qwen3.8-..." would be refused for
        # its dot) — so sanitize instead of using the model name verbatim.
        stem = re.sub(r"[^a-z0-9_-]", "-", cfg.name.lower())
        path = self._config_dir / f"llm-{stem}.json"
        payload = cfg.render_config_json(str(path))
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def _log_lines(self, entry: RunningModel, chunk: bytes) -> None:
        for line in chunk.decode(errors="replace").splitlines():
            line = line.strip()
            if line:
                logger.info("llama-server[%s]: %s", entry.name, line)

    async def _tail(self, entry: RunningModel) -> None:
        """Drain both pipes of the subprocess for its whole lifetime.

        Chat engines report differently: llama-server writes progress to
        stderr, Strata's server to stdout (``print(..., flush=True)``). A
        full, unread pipe would wedge either of them mid-load, so both are
        drained and logged under the ``llama-server[<model>]`` tag.
        """
        proc = entry.process
        await asyncio.gather(
            *(self._drain(entry, stream)
              for stream in (proc.stdout, proc.stderr) if stream),
            return_exceptions=True,
        )
        rc = await proc.wait()
        if entry.state is ModelState.READY and rc not in (0, -signal.SIGTERM):
            entry.state = ModelState.FAILED
            entry.detail = f"llama-server exited with code {rc}"
            logger.error(
                "llama-server for '%s' exited unexpectedly (code %s)",
                entry.name, rc,
            )

    async def _drain(self, entry: RunningModel, stream) -> None:
        try:
            while True:
                chunk = await stream.readline()
                if not chunk:
                    return
                self._log_lines(entry, chunk)
        except asyncio.CancelledError:
            raise
        except (ValueError, OSError):
            # A line longer than StreamReader's 64 KiB limit (an engine
            # dumping a progress bar) ends this stream's tail; the server
            # itself is unaffected, it just stops being echoed.
            logger.warning(
                "log stream for '%s' hit an over-long line; tail stopped",
                entry.name,
            )

    def _load_reported(self, payload: bytes) -> Optional[bool]:
        """A backend's own "is the weights loaded yet?" answer, if it gives
        one. llama-server's health endpoint is a plain-text 200, so this
        returns None for it. Strata answers ``{"status": "ok", "loaded":
        false}`` as soon as the port is bound and before the model is
        resident; treating that as READY would let the proxy forward requests
        into a backend that replies 503 "the model is not loaded"."""
        try:
            info = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(info, dict) or "loaded" not in info:
            return None
        return bool(info["loaded"])

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
                    if r.status_code == 200 and self._load_reported(
                            r.content) is not False:
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
        if cur is None:
            return
        was_ready = cur.state is ModelState.READY
        if cur.process.returncode is not None:
            _cleanup_config(cur.config_file)
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
        _cleanup_config(cur.config_file)
        if was_ready:
            metrics.inc_unload("llm", cur.name)
            bus.emit("model_unloaded", role="llm", model=cur.name)

    async def ensure_model(self, name: str,
                           registry: "Registry") -> tuple[str, int]:
        """Make sure `name` is the loaded model; returns (name, port)."""
        cfg = registry.get(name)  # raises UnknownModelError
        async with self._lock:
            self._last_activity = asyncio.get_running_loop().time()
            cur = self._current
            if cur is not None and cur.name == name and cur.state is ModelState.READY:
                return name, cfg.port
            prev = cur.name if cur is not None and cur.state is ModelState.READY else None
            if cur is not None:
                await self._stop_current()
            await self._launch(cfg)
            if prev is not None and prev != name:
                metrics.inc_swap("llm", prev, name)
                bus.emit("swap", role="llm", from_model=prev, to_model=name)
            return name, cfg.port

    async def unload(self) -> bool:
        """Stop the currently loaded chat LLM, if any.

        Non-destructive: the idle watcher stays armed, so a later request
        simply boots the model again. Returns True if a model was running
        (or loading) and is now stopped.
        """
        async with self._lock:
            had = self._current is not None
            await self._stop_current()
            return had

    def pin(self, seconds: float) -> None:
        """Suppress idle unload for the next ``seconds`` (X-Pin-Seconds)."""
        if seconds > 0:
            self._pinned_until = asyncio.get_running_loop().time() + seconds

    async def _launch(self, cfg: ModelConfig) -> None:
        config_file = self._render_config(cfg)
        argv = cfg.build_argv(config_path=config_file)
        logger.info(
            "launching llama-server for '%s': %s", cfg.name, " ".join(argv)
        )
        env = cfg.launch_env(os.environ)
        start = asyncio.get_running_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            _cleanup_config(config_file)
            metrics.inc_load_failure("llm", cfg.name)
            bus.emit("load_failed", role="llm", model=cfg.name, error=str(exc))
            raise ModelLoadError(f"failed to spawn llama-server: {exc}") from exc
        entry = RunningModel(name=cfg.name, config=cfg, process=proc,
                             state=ModelState.LOADING,
                             config_file=config_file)
        self._current = entry
        # Drain both pipes immediately so neither fills up during the
        # (potentially long) model load. Engines that report to stdout
        # (Strata's server logs through ``print(..., flush=True)``) would
        # otherwise fail a load with nothing but "did not become healthy".
        asyncio.get_running_loop().create_task(self._tail(entry))
        try:
            await self._wait_healthy(cfg, proc)
        except BaseException:
            await self._stop_current()
            metrics.inc_load_failure("llm", cfg.name)
            bus.emit(
                "load_failed", role="llm", model=cfg.name,
                error="did not become healthy",
            )
            raise
        entry.state = ModelState.READY
        entry.detail = ""
        elapsed = asyncio.get_running_loop().time() - start
        metrics.observe_load("llm", cfg.name, elapsed)
        bus.emit(
            "model_loaded", role="llm", model=cfg.name,
            seconds=round(elapsed, 2),
        )
        logger.info("model '%s' ready on port %d", cfg.name, cfg.port)

    async def _idle_watcher(self) -> None:
        """Unload the loaded model once it has been idle past the timeout."""
        try:
            while True:
                await asyncio.sleep(self._health_interval)
                if self._current is None:
                    continue
                now = asyncio.get_running_loop().time()
                if now < self._pinned_until:
                    continue
                # 0 (global or per-model override) means "never unload".
                threshold = self._idle_threshold(self._current.config)
                if threshold <= 0:
                    continue
                elapsed = now - self._last_activity
                if elapsed < threshold:
                    continue
                async with self._lock:
                    cur = self._current
                    if cur is None or cur.state is not ModelState.READY:
                        continue
                    elapsed = asyncio.get_running_loop().time() - \
                        self._last_activity
                    if elapsed < threshold:
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


def _cleanup_config(path: Optional[str]) -> None:
    """Remove the rendered config file of a stopped backend.

    Structured-config backends (Strata's ``--profile``) get a fresh file per
    launch; leaving it behind after an unload would let a stale profile
    outlive the model it describes.
    """
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
